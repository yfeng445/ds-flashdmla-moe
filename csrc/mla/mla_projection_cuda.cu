#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kTile = 16;
constexpr int kElementwiseThreads = 256;

__global__ void linear_weight_float_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t sequence,
    int64_t rows,
    int64_t output_features,
    int64_t input_features,
    int64_t input_stride_batch,
    int64_t input_stride_sequence,
    int64_t input_stride_feature) {
  __shared__ float input_tile[kTile][kTile];
  __shared__ float weight_tile[kTile][kTile];

  const int64_t row = static_cast<int64_t>(blockIdx.y) * kTile + threadIdx.y;
  const int64_t output_feature =
      static_cast<int64_t>(blockIdx.x) * kTile + threadIdx.x;
  float accumulator = 0.0F;

  for (int64_t reduction_start = 0; reduction_start < input_features;
       reduction_start += kTile) {
    const int64_t reduction_feature = reduction_start + threadIdx.x;
    if (row < rows && reduction_feature < input_features) {
      const int64_t batch = row / sequence;
      const int64_t sequence_index = row - batch * sequence;
      input_tile[threadIdx.y][threadIdx.x] =
          input[batch * input_stride_batch +
                sequence_index * input_stride_sequence +
                reduction_feature * input_stride_feature];
    } else {
      input_tile[threadIdx.y][threadIdx.x] = 0.0F;
    }

    // During the load phase threadIdx.y names an output row in the row-major
    // [out_features, in_features] weight. During the compute phase
    // threadIdx.x names the output column consumed by this thread.
    const int64_t weight_output =
        static_cast<int64_t>(blockIdx.x) * kTile + threadIdx.y;
    weight_tile[threadIdx.y][threadIdx.x] =
        weight_output < output_features && reduction_feature < input_features
        ? weight[weight_output * input_features + reduction_feature]
        : 0.0F;
    __syncthreads();

#pragma unroll
    for (int reduction = 0; reduction < kTile; ++reduction) {
      accumulator = fmaf(
          input_tile[threadIdx.y][reduction],
          weight_tile[threadIdx.x][reduction],
          accumulator);
    }
    __syncthreads();
  }

  if (row < rows && output_feature < output_features) {
    output[row * output_features + output_feature] = accumulator;
  }
}

__global__ void rms_norm_prefix_float_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t sequence,
    int64_t rows,
    int64_t input_width,
    int64_t normalized_width,
    int64_t output_stride_batch,
    int64_t output_stride_sequence,
    int64_t output_stride_feature,
    int64_t output_sequence_start,
    float epsilon) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  __shared__ float reduction[kElementwiseThreads];
  float sum_squares = 0.0F;
  for (int64_t feature = threadIdx.x; feature < normalized_width;
       feature += blockDim.x) {
    const float value = input[row * input_width + feature];
    sum_squares = fmaf(value, value, sum_squares);
  }
  reduction[threadIdx.x] = sum_squares;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(reduction[0] / static_cast<float>(normalized_width) + epsilon);
  const int64_t batch = row / sequence;
  const int64_t sequence_index = row - batch * sequence;
  const int64_t output_offset = batch * output_stride_batch +
      (sequence_index + output_sequence_start) * output_stride_sequence;
  for (int64_t feature = threadIdx.x; feature < normalized_width;
       feature += blockDim.x) {
    output[output_offset + feature * output_stride_feature] =
        input[row * input_width + feature] * inverse_rms * weight[feature];
  }
}

__global__ void copy_query_nope_float_kernel(
    const float* __restrict__ projected,
    float* __restrict__ q_nope,
    int64_t total,
    int64_t heads,
    int64_t nope_dim,
    int64_t rope_dim) {
  const int64_t head_dim = nope_dim + rope_dim;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t nope_feature = index % nope_dim;
    const int64_t head_row = index / nope_dim;
    const int64_t head = head_row % heads;
    const int64_t row = head_row / heads;
    q_nope[index] =
        projected[row * heads * head_dim + head * head_dim + nope_feature];
  }
}

__global__ void query_rope_float_kernel(
    const float* __restrict__ projected,
    const int64_t* __restrict__ positions,
    float* __restrict__ q_pe,
    int64_t total,
    int64_t sequence,
    int64_t heads,
    int64_t nope_dim,
    int64_t rope_dim,
    int64_t position_stride,
    float theta) {
  const int64_t head_dim = nope_dim + rope_dim;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t rope_feature = index % rope_dim;
    const int64_t head_row = index / rope_dim;
    const int64_t head = head_row % heads;
    const int64_t row = head_row / heads;
    const int64_t sequence_index = row % sequence;
    const int64_t pair_feature = rope_feature & ~int64_t{1};
    const int64_t input_offset =
        row * heads * head_dim + head * head_dim + nope_dim + pair_feature;
    const float even = projected[input_offset];
    const float odd = projected[input_offset + 1];
    const float inverse_frequency =
        powf(theta, -static_cast<float>(pair_feature) / static_cast<float>(rope_dim));
    const float angle = static_cast<float>(positions[sequence_index * position_stride]) *
        inverse_frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    q_pe[index] = (rope_feature & 1) == 0
        ? even * cosine - odd * sine
        : even * sine + odd * cosine;
  }
}

__global__ void cache_rope_float_kernel(
    const float* __restrict__ projected,
    const int64_t* __restrict__ positions,
    float* __restrict__ pe,
    int64_t total,
    int64_t sequence,
    int64_t projected_width,
    int64_t latent_dim,
    int64_t rope_dim,
    int64_t position_stride,
    int64_t output_stride_batch,
    int64_t output_stride_sequence,
    int64_t output_stride_feature,
    int64_t output_sequence_start,
    float theta) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t rope_feature = index % rope_dim;
    const int64_t row = index / rope_dim;
    const int64_t batch = row / sequence;
    const int64_t sequence_index = row - batch * sequence;
    const int64_t pair_feature = rope_feature & ~int64_t{1};
    const int64_t input_offset =
        row * projected_width + latent_dim + pair_feature;
    const float even = projected[input_offset];
    const float odd = projected[input_offset + 1];
    const float inverse_frequency =
        powf(theta, -static_cast<float>(pair_feature) / static_cast<float>(rope_dim));
    const float angle = static_cast<float>(positions[sequence_index * position_stride]) *
        inverse_frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const int64_t output_offset = batch * output_stride_batch +
        (sequence_index + output_sequence_start) * output_stride_sequence +
        rope_feature * output_stride_feature;
    pe[output_offset] = (rope_feature & 1) == 0
        ? even * cosine - odd * sine
        : even * sine + odd * cosine;
  }
}

__global__ void copy_positions_kernel(
    const int64_t* __restrict__ positions,
    int64_t* __restrict__ position_storage,
    int64_t length,
    int64_t input_stride,
    int64_t start) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < length;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    position_storage[start + index] = positions[index * input_stride];
  }
}

void check_cuda_float_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must use float32");
}

void check_contiguous_cuda_float_tensor(const at::Tensor& tensor, const char* name) {
  check_cuda_float_tensor(tensor, name);
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_long_vector(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must use int64");
  TORCH_CHECK(tensor.dim() == 1, name, " must be a vector");
}

void check_rope_parameters(int64_t rope_dim, double theta) {
  TORCH_CHECK(rope_dim > 0 && rope_dim % 2 == 0, "RoPE dimension must be positive and even");
  TORCH_CHECK(std::isfinite(theta) && theta > 0.0, "rope_theta must be finite and positive");
}

void check_rms_epsilon(double epsilon) {
  TORCH_CHECK(
      std::isfinite(epsilon) && epsilon > 0.0,
      "rms_norm_eps must be finite and positive");
}

int elementwise_blocks(int64_t total, const cudaDeviceProp* properties) {
  const int64_t requested =
      total / kElementwiseThreads + (total % kElementwiseThreads != 0);
  return static_cast<int>(std::min<int64_t>(requested, properties->maxGridSize[0]));
}

void launch_linear_weight(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor& output,
    cudaStream_t stream) {
  const int64_t batch = input.size(0);
  const int64_t sequence = input.size(1);
  const int64_t rows = batch * sequence;
  const int64_t input_features = input.size(2);
  const int64_t output_features = weight.size(0);
  if (rows == 0 || output_features == 0) {
    return;
  }
  const int64_t grid_x = output_features / kTile + (output_features % kTile != 0);
  const int64_t grid_y = rows / kTile + (rows % kTile != 0);
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      grid_x <= static_cast<int64_t>(properties->maxGridSize[0]) &&
          grid_y <= static_cast<int64_t>(properties->maxGridSize[1]),
      "MLA projection grid exceeds the CUDA device limit");
  const dim3 threads(kTile, kTile);
  const dim3 blocks(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
  linear_weight_float_kernel<<<blocks, threads, 0, stream>>>(
      input.const_data_ptr<float>(),
      weight.const_data_ptr<float>(),
      output.mutable_data_ptr<float>(),
      sequence,
      rows,
      output_features,
      input_features,
      input.stride(0),
      input.stride(1),
      input.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_rms_norm_prefix(
    const at::Tensor& projected,
    const at::Tensor& weight,
    at::Tensor& output,
    int64_t batch,
    int64_t sequence,
    int64_t input_width,
    int64_t normalized_width,
    int64_t output_sequence_start,
    double epsilon,
    cudaStream_t stream) {
  const int64_t rows = batch * sequence;
  if (rows == 0) {
    return;
  }
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(projected.get_device());
  TORCH_CHECK(
      rows <= static_cast<int64_t>(properties->maxGridSize[0]),
      "too many MLA RMSNorm rows for a one-dimensional CUDA launch: ",
      rows);
  rms_norm_prefix_float_kernel<<<
      static_cast<unsigned int>(rows), kElementwiseThreads, 0, stream>>>(
      projected.const_data_ptr<float>(),
      weight.const_data_ptr<float>(),
      output.mutable_data_ptr<float>(),
      sequence,
      rows,
      input_width,
      normalized_width,
      output.stride(0),
      output.stride(1),
      output.stride(2),
      output_sequence_start,
      static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor> split_query_and_apply_rope(
    const at::Tensor& projected,
    const at::Tensor& positions,
    int64_t batch,
    int64_t sequence,
    int64_t heads,
    int64_t nope_dim,
    int64_t rope_dim,
    double theta,
    cudaStream_t stream) {
  auto q_nope = at::empty({batch, sequence, heads, nope_dim}, projected.options());
  auto q_pe = at::empty({batch, sequence, heads, rope_dim}, projected.options());
  const int64_t nope_total = batch * sequence * heads * nope_dim;
  const int64_t rope_total = batch * sequence * heads * rope_dim;
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(projected.get_device());
  if (nope_total > 0) {
    copy_query_nope_float_kernel<<<
        elementwise_blocks(nope_total, properties), kElementwiseThreads, 0, stream>>>(
        projected.const_data_ptr<float>(),
        q_nope.mutable_data_ptr<float>(),
        nope_total,
        heads,
        nope_dim,
        rope_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (rope_total > 0) {
    query_rope_float_kernel<<<
        elementwise_blocks(rope_total, properties), kElementwiseThreads, 0, stream>>>(
        projected.const_data_ptr<float>(),
        positions.const_data_ptr<int64_t>(),
        q_pe.mutable_data_ptr<float>(),
        rope_total,
        sequence,
        heads,
        nope_dim,
        rope_dim,
        positions.stride(0),
        static_cast<float>(theta));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {q_nope, q_pe};
}

void launch_cache_rope(
    const at::Tensor& projected,
    const at::Tensor& positions,
    at::Tensor& pe,
    int64_t batch,
    int64_t sequence,
    int64_t latent_dim,
    int64_t rope_dim,
    int64_t output_sequence_start,
    double theta,
    cudaStream_t stream) {
  const int64_t total = batch * sequence * rope_dim;
  if (total == 0) {
    return;
  }
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(projected.get_device());
  cache_rope_float_kernel<<<
      elementwise_blocks(total, properties), kElementwiseThreads, 0, stream>>>(
      projected.const_data_ptr<float>(),
      positions.const_data_ptr<int64_t>(),
      pe.mutable_data_ptr<float>(),
      total,
      sequence,
      latent_dim + rope_dim,
      latent_dim,
      rope_dim,
      positions.stride(0),
      pe.stride(0),
      pe.stride(1),
      pe.stride(2),
      output_sequence_start,
      static_cast<float>(theta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void validate_query_common(
    const at::Tensor& x,
    const at::Tensor& positions,
    int64_t heads,
    int64_t nope_dim,
    int64_t rope_dim,
    double theta) {
  check_cuda_float_tensor(x, "x");
  check_cuda_long_vector(positions, "positions");
  TORCH_CHECK(x.dim() == 3, "MLA query input must have shape [batch, sequence, model_dim]");
  TORCH_CHECK(x.device() == positions.device(), "query input and positions must share a device");
  TORCH_CHECK(positions.numel() == x.size(1), "positions must match query sequence length");
  TORCH_CHECK(heads > 0 && nope_dim > 0, "MLA head count and content dimension must be positive");
  check_rope_parameters(rope_dim, theta);
}

std::tuple<at::Tensor, at::Tensor> mla_query_projection_cuda(
    const at::Tensor& x,
    const at::Tensor& wq,
    const at::Tensor& positions,
    int64_t heads,
    int64_t nope_dim,
    int64_t rope_dim,
    double theta) {
  validate_query_common(x, positions, heads, nope_dim, rope_dim, theta);
  check_contiguous_cuda_float_tensor(wq, "wq");
  TORCH_CHECK(wq.device() == x.device(), "x and wq must share a CUDA device");
  TORCH_CHECK(wq.dim() == 2, "wq must be a matrix");
  TORCH_CHECK(wq.size(1) == x.size(2), "wq input dimension must match x");
  TORCH_CHECK(
      wq.size(0) == heads * (nope_dim + rope_dim),
      "wq output dimension must match the MLA head dimensions");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  auto projected = at::empty({x.size(0), x.size(1), wq.size(0)}, x.options());
  launch_linear_weight(x, wq, projected, stream);
  return split_query_and_apply_rope(
      projected,
      positions,
      x.size(0),
      x.size(1),
      heads,
      nope_dim,
      rope_dim,
      theta,
      stream);
}

std::tuple<at::Tensor, at::Tensor> mla_query_lora_projection_cuda(
    const at::Tensor& x,
    const at::Tensor& wq_a,
    const at::Tensor& q_norm_weight,
    const at::Tensor& wq_b,
    const at::Tensor& positions,
    int64_t heads,
    int64_t nope_dim,
    int64_t rope_dim,
    double theta,
    double epsilon) {
  validate_query_common(x, positions, heads, nope_dim, rope_dim, theta);
  check_contiguous_cuda_float_tensor(wq_a, "wq_a");
  check_contiguous_cuda_float_tensor(q_norm_weight, "q_norm_weight");
  check_contiguous_cuda_float_tensor(wq_b, "wq_b");
  check_rms_epsilon(epsilon);
  TORCH_CHECK(
      x.device() == wq_a.device() && x.device() == q_norm_weight.device() &&
          x.device() == wq_b.device(),
      "all MLA query projection tensors must share a CUDA device");
  TORCH_CHECK(wq_a.dim() == 2 && wq_b.dim() == 2, "wq_a and wq_b must be matrices");
  TORCH_CHECK(q_norm_weight.dim() == 1, "q_norm_weight must be a vector");
  const int64_t rank = wq_a.size(0);
  TORCH_CHECK(rank > 0 && wq_a.size(1) == x.size(2), "wq_a has an invalid shape");
  TORCH_CHECK(q_norm_weight.numel() == rank, "q_norm_weight must match q_lora_rank");
  TORCH_CHECK(
      wq_b.size(0) == heads * (nope_dim + rope_dim) && wq_b.size(1) == rank,
      "wq_b has an invalid shape");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  auto latent = at::empty({x.size(0), x.size(1), rank}, x.options());
  auto normalized = at::empty_like(latent);
  auto projected = at::empty({x.size(0), x.size(1), wq_b.size(0)}, x.options());
  launch_linear_weight(x, wq_a, latent, stream);
  launch_rms_norm_prefix(
      latent,
      q_norm_weight,
      normalized,
      x.size(0),
      x.size(1),
      rank,
      rank,
      0,
      epsilon,
      stream);
  launch_linear_weight(normalized, wq_b, projected, stream);
  return split_query_and_apply_rope(
      projected,
      positions,
      x.size(0),
      x.size(1),
      heads,
      nope_dim,
      rope_dim,
      theta,
      stream);
}

void validate_cache_projection_common(
    const at::Tensor& x,
    const at::Tensor& wkv_a,
    const at::Tensor& kv_norm_weight,
    const at::Tensor& positions,
    double theta,
    double epsilon) {
  check_cuda_float_tensor(x, "x");
  check_contiguous_cuda_float_tensor(wkv_a, "wkv_a");
  check_contiguous_cuda_float_tensor(kv_norm_weight, "kv_norm_weight");
  check_cuda_long_vector(positions, "positions");
  TORCH_CHECK(x.dim() == 3, "MLA cache input must have shape [batch, sequence, model_dim]");
  TORCH_CHECK(wkv_a.dim() == 2, "wkv_a must be a matrix");
  TORCH_CHECK(kv_norm_weight.dim() == 1, "kv_norm_weight must be a vector");
  TORCH_CHECK(
      x.device() == wkv_a.device() && x.device() == kv_norm_weight.device() &&
          x.device() == positions.device(),
      "all MLA cache projection tensors must share a CUDA device");
  TORCH_CHECK(wkv_a.size(1) == x.size(2), "wkv_a input dimension must match x");
  TORCH_CHECK(positions.numel() == x.size(1), "positions must match cache sequence length");
  const int64_t latent_dim = kv_norm_weight.numel();
  const int64_t rope_dim = wkv_a.size(0) - latent_dim;
  TORCH_CHECK(latent_dim > 0, "kv_lora_rank must be positive");
  check_rope_parameters(rope_dim, theta);
  check_rms_epsilon(epsilon);
}

std::tuple<at::Tensor, at::Tensor> mla_cache_projection_cuda(
    const at::Tensor& x,
    const at::Tensor& wkv_a,
    const at::Tensor& kv_norm_weight,
    const at::Tensor& positions,
    int64_t latent_dim,
    double theta,
    double epsilon) {
  validate_cache_projection_common(x, wkv_a, kv_norm_weight, positions, theta, epsilon);
  TORCH_CHECK(latent_dim == kv_norm_weight.numel(), "kv_lora_rank must match kv_norm_weight");
  const int64_t rope_dim = wkv_a.size(0) - latent_dim;
  const c10::cuda::CUDAGuard device_guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  auto projected = at::empty({x.size(0), x.size(1), wkv_a.size(0)}, x.options());
  auto kv = at::empty({x.size(0), x.size(1), latent_dim}, x.options());
  auto pe = at::empty({x.size(0), x.size(1), rope_dim}, x.options());
  launch_linear_weight(x, wkv_a, projected, stream);
  launch_rms_norm_prefix(
      projected,
      kv_norm_weight,
      kv,
      x.size(0),
      x.size(1),
      wkv_a.size(0),
      latent_dim,
      0,
      epsilon,
      stream);
  launch_cache_rope(
      projected,
      positions,
      pe,
      x.size(0),
      x.size(1),
      latent_dim,
      rope_dim,
      0,
      theta,
      stream);
  return {kv, pe};
}

void mla_cache_projection_write_cuda(
    const at::Tensor& x,
    const at::Tensor& wkv_a,
    const at::Tensor& kv_norm_weight,
    const at::Tensor& positions,
    at::Tensor& kv_storage,
    at::Tensor& pe_storage,
    at::Tensor& position_storage,
    int64_t start,
    double theta,
    double epsilon) {
  validate_cache_projection_common(x, wkv_a, kv_norm_weight, positions, theta, epsilon);
  check_contiguous_cuda_float_tensor(kv_storage, "kv_storage");
  check_contiguous_cuda_float_tensor(pe_storage, "pe_storage");
  TORCH_CHECK(position_storage.is_cuda(), "position_storage must be a CUDA tensor");
  TORCH_CHECK(position_storage.scalar_type() == at::kLong, "position_storage must use int64");
  TORCH_CHECK(position_storage.is_contiguous(), "position_storage must be contiguous");
  TORCH_CHECK(kv_storage.dim() == 3 && pe_storage.dim() == 3, "cache storage must be rank 3");
  TORCH_CHECK(position_storage.dim() == 1, "position_storage must be a vector");
  TORCH_CHECK(
      x.device() == kv_storage.device() && x.device() == pe_storage.device() &&
          x.device() == position_storage.device(),
      "all static cache tensors must share a CUDA device");
  const int64_t latent_dim = kv_norm_weight.numel();
  const int64_t rope_dim = wkv_a.size(0) - latent_dim;
  TORCH_CHECK(
      kv_storage.size(0) == x.size(0) && pe_storage.size(0) == x.size(0),
      "static cache batch dimension must match x");
  TORCH_CHECK(
      kv_storage.size(1) == pe_storage.size(1) &&
          kv_storage.size(1) == position_storage.numel(),
      "static cache capacities must match");
  TORCH_CHECK(
      kv_storage.size(2) == latent_dim && pe_storage.size(2) == rope_dim,
      "static cache feature dimensions do not match the projection weights");
  TORCH_CHECK(start >= 0, "static cache write start must be non-negative");
  TORCH_CHECK(start + x.size(1) <= kv_storage.size(1), "static cache write exceeds capacity");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  auto projected = at::empty({x.size(0), x.size(1), wkv_a.size(0)}, x.options());
  launch_linear_weight(x, wkv_a, projected, stream);
  launch_rms_norm_prefix(
      projected,
      kv_norm_weight,
      kv_storage,
      x.size(0),
      x.size(1),
      wkv_a.size(0),
      latent_dim,
      start,
      epsilon,
      stream);
  launch_cache_rope(
      projected,
      positions,
      pe_storage,
      x.size(0),
      x.size(1),
      latent_dim,
      rope_dim,
      start,
      theta,
      stream);
  if (x.size(1) > 0) {
    const cudaDeviceProp* properties = at::cuda::getDeviceProperties(x.get_device());
    copy_positions_kernel<<<
        elementwise_blocks(x.size(1), properties), kElementwiseThreads, 0, stream>>>(
        positions.const_data_ptr<int64_t>(),
        position_storage.mutable_data_ptr<int64_t>(),
        x.size(1),
        positions.stride(0),
        start);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

at::Tensor mla_output_projection_cuda(const at::Tensor& heads, const at::Tensor& wo) {
  check_contiguous_cuda_float_tensor(heads, "heads");
  check_contiguous_cuda_float_tensor(wo, "wo");
  TORCH_CHECK(heads.dim() == 4, "heads must have shape [batch, sequence, heads, value_dim]");
  TORCH_CHECK(wo.dim() == 2, "wo must be a matrix");
  TORCH_CHECK(heads.device() == wo.device(), "heads and wo must share a CUDA device");
  const int64_t inner = heads.size(2) * heads.size(3);
  TORCH_CHECK(inner > 0 && wo.size(0) > 0, "MLA output dimensions must be positive");
  TORCH_CHECK(wo.size(1) == inner, "wo input dimension must match flattened attention heads");

  const c10::cuda::CUDAGuard device_guard(heads.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(heads.get_device());
  auto flattened = heads.reshape({heads.size(0), heads.size(1), inner});
  auto output = at::empty({heads.size(0), heads.size(1), wo.size(0)}, heads.options());
  launch_linear_weight(flattened, wo, output, stream);
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("mla_query_projection", TORCH_FN(mla_query_projection_cuda));
  m.impl("mla_query_lora_projection", TORCH_FN(mla_query_lora_projection_cuda));
  m.impl("mla_cache_projection", TORCH_FN(mla_cache_projection_cuda));
  m.impl("mla_cache_projection_write", TORCH_FN(mla_cache_projection_write_cuda));
  m.impl("mla_output_projection", TORCH_FN(mla_output_projection_cuda));
}
