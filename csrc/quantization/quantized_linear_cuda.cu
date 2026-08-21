// Forward-only scalar teaching kernels for explicit FP8 E4M3FN and INT8 semantics.
// These kernels intentionally make no Tensor Core or performance claim.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kQuantizeThreads = 256;
constexpr int kLinearTile = 16;
constexpr float kInt8Bound = 127.0F;
constexpr float kFP8E4M3FNBound = 448.0F;

__device__ __forceinline__ uint8_t encode_fp8_e4m3fn(float value) {
  const uint8_t sign = signbit(value) ? 0x80 : 0x00;
  const float magnitude = fminf(fabsf(value), kFP8E4M3FNBound);
  if (magnitude == 0.0F) {
    return sign;
  }

  // E4M3FN bias is 7. Subnormals have a 2^-9 mantissa quantum.
  if (magnitude < 0.015625F) {
    const int mantissa = __float2int_rn(magnitude * 512.0F);
    return mantissa >= 8 ? static_cast<uint8_t>(sign | 0x08)
                         : static_cast<uint8_t>(sign | mantissa);
  }

  int exponent = static_cast<int>(floorf(log2f(magnitude)));
  int encoded_exponent = exponent + 7;
  int mantissa = __float2int_rn((ldexpf(magnitude, -exponent) - 1.0F) * 8.0F);
  if (mantissa == 8) {
    ++encoded_exponent;
    mantissa = 0;
  }
  if (encoded_exponent > 15 || (encoded_exponent == 15 && mantissa > 6)) {
    return static_cast<uint8_t>(sign | 0x7E);
  }
  return static_cast<uint8_t>(sign | (encoded_exponent << 3) | mantissa);
}

__device__ __forceinline__ float decode_fp8_e4m3fn(uint8_t value) {
  const bool negative = (value & 0x80) != 0;
  const uint8_t magnitude = value & 0x7F;
  const int exponent = magnitude >> 3;
  const int mantissa = magnitude & 0x07;
  float decoded = 0.0F;
  if (exponent == 0) {
    decoded = ldexpf(static_cast<float>(mantissa), -9);
  } else {
    decoded = ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
  }
  return negative ? -decoded : decoded;
}

template <typename Output, bool kFP8>
__global__ void quantize_per_row_kernel(
    const float* __restrict__ input,
    Output* __restrict__ values,
    float* __restrict__ scales,
    int64_t columns) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const float* row_input = input + row * columns;
  Output* row_output = values + row * columns;
  float local_maximum = 0.0F;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    local_maximum = fmaxf(local_maximum, fabsf(row_input[column]));
  }

  __shared__ float reduction[kQuantizeThreads];
  __shared__ float row_scale;
  reduction[threadIdx.x] = local_maximum;
  __syncthreads();
  for (int stride = kQuantizeThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float bound = kFP8 ? kFP8E4M3FNBound : kInt8Bound;
    row_scale = reduction[0] == 0.0F
        ? 1.0F
        : fmaxf(reduction[0] / bound, FLT_MIN);
    scales[row] = row_scale;
  }
  __syncthreads();

  const float scale = row_scale;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const float normalized = row_input[column] / scale;
    if constexpr (kFP8) {
      row_output[column] = encode_fp8_e4m3fn(normalized);
    } else {
      const int rounded = __float2int_rn(normalized);
      const int clamped = rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
      row_output[column] = static_cast<int8_t>(clamped);
    }
  }
}

template <typename Payload, bool kFP8>
__global__ void dequantized_linear_kernel(
    const Payload* __restrict__ activation_values,
    const float* __restrict__ activation_scales,
    const Payload* __restrict__ weight_values,
    const float* __restrict__ weight_scales,
    float* __restrict__ output,
    int64_t rows,
    int64_t output_channels,
    int64_t reduction) {
  const int64_t output_channel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= rows || output_channel >= output_channels) {
    return;
  }

  const float activation_scale = activation_scales[row];
  const float weight_scale = weight_scales[output_channel];
  float accumulator = 0.0F;
  for (int64_t inner = 0; inner < reduction; ++inner) {
    float activation;
    float weight;
    if constexpr (kFP8) {
      activation = decode_fp8_e4m3fn(
          static_cast<uint8_t>(activation_values[row * reduction + inner]));
      weight = decode_fp8_e4m3fn(
          static_cast<uint8_t>(weight_values[output_channel * reduction + inner]));
    } else {
      activation = static_cast<float>(activation_values[row * reduction + inner]);
      weight = static_cast<float>(weight_values[output_channel * reduction + inner]);
    }
    accumulator = fmaf(
        activation * activation_scale,
        weight * weight_scale,
        accumulator);
  }
  output[row * output_channels + output_channel] = accumulator;
}

void check_quantize_input(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must use float32");
  TORCH_CHECK(input.is_contiguous(), "input must be row-major contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be a rank-2 matrix");
  TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0, "input dimensions must be positive");
  TORCH_CHECK(!input.requires_grad(), "native quantization operators are forward-only");
}

template <typename Output, bool kFP8>
std::tuple<at::Tensor, at::Tensor> quantize_per_row_cuda(const at::Tensor& input) {
  check_quantize_input(input);
  const c10::cuda::CUDAGuard device_guard(input.device());
  // This eager precheck deliberately rejects NaN/Inf instead of producing a
  // payload. Capture/replay is supported by the already-quantized linear op.
  TORCH_CHECK(
      at::isfinite(input).all().item<bool>(),
      "input must contain only finite values");

  const int64_t rows = input.size(0);
  const int64_t columns = input.size(1);
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      rows <= static_cast<int64_t>(properties->maxGridSize[0]),
      "input rows exceed the one-block-per-row CUDA grid limit");
  const at::ScalarType value_type = kFP8 ? at::kByte : at::kChar;
  auto values = at::empty(input.sizes(), input.options().dtype(value_type));
  auto scales = at::empty({rows}, input.options().dtype(at::kFloat));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  quantize_per_row_kernel<Output, kFP8><<<
      static_cast<unsigned int>(rows), kQuantizeThreads, 0, stream>>>(
      input.const_data_ptr<float>(),
      values.mutable_data_ptr<Output>(),
      scales.mutable_data_ptr<float>(),
      columns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {values, scales};
}

void check_linear_tensor(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    int64_t dimensions) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == dimensions, name, " has an unsupported rank");
  TORCH_CHECK(!tensor.requires_grad(), "native dequantized linear operators are forward-only");
}

template <typename Payload, bool kFP8>
at::Tensor dequantized_linear_cuda(
    const at::Tensor& activation_values,
    const at::Tensor& activation_scales,
    const at::Tensor& weight_values,
    const at::Tensor& weight_scales) {
  const at::ScalarType payload_type = kFP8 ? at::kByte : at::kChar;
  check_linear_tensor(activation_values, "activation_values", payload_type, 2);
  check_linear_tensor(weight_values, "weight_values", payload_type, 2);
  check_linear_tensor(activation_scales, "activation_scales", at::kFloat, 1);
  check_linear_tensor(weight_scales, "weight_scales", at::kFloat, 1);
  TORCH_CHECK(
      activation_values.device() == activation_scales.device() &&
          activation_values.device() == weight_values.device() &&
          activation_values.device() == weight_scales.device(),
      "quantized values and scales must share one CUDA device");
  TORCH_CHECK(
      activation_values.size(0) > 0 && activation_values.size(1) > 0 &&
          weight_values.size(0) > 0,
      "quantized matrix dimensions must be positive");
  TORCH_CHECK(
      activation_values.size(1) == weight_values.size(1),
      "activation and weight inner dimensions must match");
  TORCH_CHECK(
      activation_scales.numel() == activation_values.size(0),
      "activation_scales must contain one value per activation row");
  TORCH_CHECK(
      weight_scales.numel() == weight_values.size(0),
      "weight_scales must contain one value per output channel");

  const c10::cuda::CUDAGuard device_guard(activation_values.device());
  const int64_t rows = activation_values.size(0);
  const int64_t output_channels = weight_values.size(0);
  const int64_t reduction = activation_values.size(1);
  const int64_t grid_x = (output_channels + kLinearTile - 1) / kLinearTile;
  const int64_t grid_y = (rows + kLinearTile - 1) / kLinearTile;
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(activation_values.get_device());
  TORCH_CHECK(
      grid_x <= static_cast<int64_t>(properties->maxGridSize[0]) &&
          grid_y <= static_cast<int64_t>(properties->maxGridSize[1]),
      "dequantized linear output exceeds the two-dimensional CUDA grid limit");

  auto output = at::empty(
      {rows, output_channels}, activation_scales.options().dtype(at::kFloat));
  const dim3 threads(kLinearTile, kLinearTile);
  const dim3 blocks(
      static_cast<unsigned int>(grid_x),
      static_cast<unsigned int>(grid_y));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(activation_values.get_device());
  dequantized_linear_kernel<Payload, kFP8><<<blocks, threads, 0, stream>>>(
      activation_values.const_data_ptr<Payload>(),
      activation_scales.const_data_ptr<float>(),
      weight_values.const_data_ptr<Payload>(),
      weight_scales.const_data_ptr<float>(),
      output.mutable_data_ptr<float>(),
      rows,
      output_channels,
      reduction);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<at::Tensor, at::Tensor> quantize_int8_per_row_cuda(
    const at::Tensor& input) {
  return quantize_per_row_cuda<int8_t, false>(input);
}

std::tuple<at::Tensor, at::Tensor> quantize_fp8_e4m3fn_per_row_cuda(
    const at::Tensor& input) {
  return quantize_per_row_cuda<uint8_t, true>(input);
}

at::Tensor dequantized_linear_int8_cuda(
    const at::Tensor& activation_values,
    const at::Tensor& activation_scales,
    const at::Tensor& weight_values,
    const at::Tensor& weight_scales) {
  return dequantized_linear_cuda<int8_t, false>(
      activation_values, activation_scales, weight_values, weight_scales);
}

at::Tensor dequantized_linear_fp8_e4m3fn_cuda(
    const at::Tensor& activation_values,
    const at::Tensor& activation_scales,
    const at::Tensor& weight_values,
    const at::Tensor& weight_scales) {
  return dequantized_linear_cuda<uint8_t, true>(
      activation_values, activation_scales, weight_values, weight_scales);
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("quantize_int8_per_row", TORCH_FN(quantize_int8_per_row_cuda));
  m.impl("quantize_fp8_e4m3fn_per_row", TORCH_FN(quantize_fp8_e4m3fn_per_row_cuda));
  m.impl("dequantized_linear_int8", TORCH_FN(dequantized_linear_int8_cuda));
  m.impl(
      "dequantized_linear_fp8_e4m3fn",
      TORCH_FN(dequantized_linear_fp8_e4m3fn_cuda));
}
