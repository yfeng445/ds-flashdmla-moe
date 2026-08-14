#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 128;

template <typename scalar_t>
__global__ void attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ output,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t value_dim,
    float scale,
    bool causal) {
  const int64_t query_row = static_cast<int64_t>(blockIdx.x);
  const int64_t query_position = query_row % query_length;
  const int64_t batch_head = query_row / query_length;

  const int64_t query_offset = (batch_head * query_length + query_position) * head_dim;
  const int64_t output_offset = (batch_head * query_length + query_position) * value_dim;

  extern __shared__ float shared[];
  float* query = shared;
  float* numerator = query + head_dim;
  float* reduction = numerator + value_dim;
  float* statistics = reduction + blockDim.x;

  for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
    query[column] = static_cast<float>(q[query_offset + column]);
  }
  for (int64_t column = threadIdx.x; column < value_dim; column += blockDim.x) {
    numerator[column] = 0.0F;
  }
  if (threadIdx.x == 0) {
    statistics[0] = -CUDART_INF_F;
    statistics[1] = 0.0F;
  }
  __syncthreads();

  int64_t visible_keys = key_length;
  if (causal) {
    const int64_t absolute_query_position = query_position + key_length - query_length;
    visible_keys = absolute_query_position + 1 < key_length
        ? absolute_query_position + 1
        : key_length;
  }

  for (int64_t key_position = 0; key_position < visible_keys; ++key_position) {
    const int64_t key_offset = (batch_head * key_length + key_position) * head_dim;
    float partial = 0.0F;
    for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
      partial += query[column] * static_cast<float>(k[key_offset + column]);
    }
    reduction[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const float score = reduction[0] * scale;
      const float previous_max = statistics[0];
      const float next_max = fmaxf(previous_max, score);
      const float previous_scale =
          isinf(previous_max) && previous_max < 0.0F ? 0.0F : expf(previous_max - next_max);
      const float current_scale = expf(score - next_max);
      statistics[0] = next_max;
      statistics[1] = statistics[1] * previous_scale + current_scale;
      reduction[0] = previous_scale;
      reduction[1] = current_scale;
    }
    __syncthreads();

    const float previous_scale = reduction[0];
    const float current_scale = reduction[1];
    const int64_t value_offset = (batch_head * key_length + key_position) * value_dim;
    for (int64_t column = threadIdx.x; column < value_dim; column += blockDim.x) {
      numerator[column] =
          numerator[column] * previous_scale +
          current_scale * static_cast<float>(v[value_offset + column]);
    }
    __syncthreads();
  }

  const float denominator = statistics[1];
  for (int64_t column = threadIdx.x; column < value_dim; column += blockDim.x) {
    output[output_offset + column] = static_cast<scalar_t>(
        denominator > 0.0F ? numerator[column] / denominator : 0.0F);
  }
}

at::Tensor attention_forward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && k.device() == v.device(),
              "q, k, and v must be on the same CUDA device");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q, k, and v must have shape [batch, heads, sequence, dimension]");
  const auto scalar_type = q.scalar_type();
  TORCH_CHECK(scalar_type == at::kFloat || scalar_type == at::kHalf ||
                  scalar_type == at::kBFloat16,
              "the CUDA attention forward kernel supports float32, float16, and bfloat16");
  TORCH_CHECK(k.scalar_type() == scalar_type && v.scalar_type() == scalar_type,
              "q, k, and v must have the same dtype");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "q, k, and v must be contiguous");
  TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0) &&
                  q.size(1) == k.size(1) && k.size(1) == v.size(1),
              "q, k, and v must have identical batch and head dimensions");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k must have the same head dimension");
  TORCH_CHECK(k.size(2) == v.size(2), "k and v must have the same sequence length");
  TORCH_CHECK(q.size(3) > 0, "attention head dimension must be positive");
  TORCH_CHECK(k.size(2) > 0, "key sequence length must be positive");
  TORCH_CHECK(!causal || q.size(2) <= k.size(2),
              "right-aligned causal attention requires query_length <= key_length");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");

  const auto batch = q.size(0);
  const auto heads = q.size(1);
  const auto query_length = q.size(2);
  const auto key_length = k.size(2);
  const auto head_dim = q.size(3);
  const auto value_dim = v.size(3);
  auto output = at::empty({batch, heads, query_length, value_dim}, v.options());

  if (batch == 0 || heads == 0 || query_length == 0 || value_dim == 0) {
    return output;
  }

  const int64_t query_rows = batch * heads * query_length;
  const c10::cuda::CUDAGuard device_guard(q.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(query_rows <= static_cast<int64_t>(properties->maxGridSize[0]),
              "too many query rows for a one-dimensional CUDA launch: ", query_rows);

  const size_t shared_bytes = static_cast<size_t>(head_dim + value_dim + kThreads + 2) *
      sizeof(float);
  TORCH_CHECK(shared_bytes <= static_cast<size_t>(properties->sharedMemPerBlock),
              "attention dimensions require ", shared_bytes,
              " bytes of shared memory, but the device limit is ", properties->sharedMemPerBlock);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "attention_forward_cuda",
      [&] {
        attention_forward_kernel<scalar_t><<<
            static_cast<unsigned int>(query_rows), kThreads, shared_bytes, stream>>>(
            q.const_data_ptr<scalar_t>(),
            k.const_data_ptr<scalar_t>(),
            v.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            query_length,
            key_length,
            head_dim,
            value_dim,
            static_cast<float>(scale),
            causal);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("attention_forward", TORCH_FN(attention_forward_cuda));
}
