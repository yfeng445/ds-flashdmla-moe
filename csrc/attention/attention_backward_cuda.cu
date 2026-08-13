#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kThreads = 128;
constexpr int kStatisticCount = 5;

__device__ float reduce_dot(
    const float* __restrict__ left,
    const float* __restrict__ right,
    int64_t dimension,
    float* __restrict__ reduction) {
  float partial = 0.0F;
  for (int64_t column = threadIdx.x; column < dimension; column += blockDim.x) {
    partial += left[column] * right[column];
  }
  reduction[threadIdx.x] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return reduction[0];
}

__global__ void attention_backward_float_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t value_dim,
    float scale,
    bool causal) {
  const int64_t query_row = static_cast<int64_t>(blockIdx.x);
  const int64_t query_position = query_row % query_length;
  const int64_t batch_head = query_row / query_length;
  const int64_t query_offset = query_row * head_dim;
  const int64_t grad_output_offset = query_row * value_dim;

  extern __shared__ float shared[];
  float* query = shared;
  float* upstream = query + head_dim;
  float* query_gradient = upstream + value_dim;
  float* reduction = query_gradient + head_dim;
  float* statistics = reduction + blockDim.x;

  for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
    query[column] = q[query_offset + column];
    query_gradient[column] = 0.0F;
  }
  for (int64_t column = threadIdx.x; column < value_dim; column += blockDim.x) {
    upstream[column] = grad_output[grad_output_offset + column];
  }
  if (threadIdx.x == 0) {
    statistics[0] = -CUDART_INF_F;  // row maximum
    statistics[1] = 0.0F;           // softmax denominator
    statistics[2] = 0.0F;           // row correction sum_j(P_j * dP_j)
  }
  __syncthreads();

  int64_t visible_keys = key_length;
  if (causal) {
    const int64_t absolute_query_position = query_position + key_length - query_length;
    visible_keys = absolute_query_position + 1 < key_length
        ? absolute_query_position + 1
        : key_length;
  }

  // Pass 1: stable online maximum and denominator for this query row.
  for (int64_t key_position = 0; key_position < visible_keys; ++key_position) {
    const int64_t key_offset = (batch_head * key_length + key_position) * head_dim;
    const float dot = reduce_dot(query, k + key_offset, head_dim, reduction);
    if (threadIdx.x == 0) {
      const float score = dot * scale;
      const float previous_max = statistics[0];
      const float next_max = fmaxf(previous_max, score);
      const float previous_scale =
          isinf(previous_max) && previous_max < 0.0F ? 0.0F : expf(previous_max - next_max);
      statistics[0] = next_max;
      statistics[1] = statistics[1] * previous_scale + expf(score - next_max);
    }
    __syncthreads();
  }

  // Pass 2: D_i = sum_j P_ij * (dO_i dot V_j).
  for (int64_t key_position = 0; key_position < visible_keys; ++key_position) {
    const int64_t key_offset = (batch_head * key_length + key_position) * head_dim;
    const int64_t value_offset = (batch_head * key_length + key_position) * value_dim;
    const float dot = reduce_dot(query, k + key_offset, head_dim, reduction);
    if (threadIdx.x == 0) {
      statistics[3] = expf(dot * scale - statistics[0]) / statistics[1];
    }
    __syncthreads();
    const float grad_probability = reduce_dot(upstream, v + value_offset, value_dim, reduction);
    if (threadIdx.x == 0) {
      statistics[2] += statistics[3] * grad_probability;
    }
    __syncthreads();
  }

  // Pass 3: form dS, then accumulate dQ locally and dK/dV atomically.
  for (int64_t key_position = 0; key_position < visible_keys; ++key_position) {
    const int64_t key_offset = (batch_head * key_length + key_position) * head_dim;
    const int64_t value_offset = (batch_head * key_length + key_position) * value_dim;
    const float dot = reduce_dot(query, k + key_offset, head_dim, reduction);
    if (threadIdx.x == 0) {
      statistics[3] = expf(dot * scale - statistics[0]) / statistics[1];  // probability
    }
    __syncthreads();
    const float grad_probability = reduce_dot(upstream, v + value_offset, value_dim, reduction);
    if (threadIdx.x == 0) {
      statistics[4] = statistics[3] * (grad_probability - statistics[2]) * scale;
    }
    __syncthreads();

    const float probability = statistics[3];
    const float grad_dot = statistics[4];
    for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
      query_gradient[column] += grad_dot * k[key_offset + column];
      atomicAdd(grad_k + key_offset + column, grad_dot * query[column]);
    }
    for (int64_t column = threadIdx.x; column < value_dim; column += blockDim.x) {
      atomicAdd(grad_v + value_offset + column, probability * upstream[column]);
    }
    __syncthreads();
  }

  for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
    grad_q[query_offset + column] = query_gradient[column];
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> attention_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    double scale) {
  TORCH_CHECK(!at::globalContext().deterministicAlgorithms(),
              "native attention backward uses atomic accumulation; disable deterministic "
              "algorithms or use the reference backward");
  TORCH_CHECK(grad_output.is_cuda() && q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "grad_output, q, k, and v must be CUDA tensors");
  TORCH_CHECK(grad_output.device() == q.device() && q.device() == k.device() &&
                  k.device() == v.device(),
              "grad_output, q, k, and v must be on the same CUDA device");
  TORCH_CHECK(grad_output.dim() == 4 && q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "grad_output, q, k, and v must be rank-four tensors");
  TORCH_CHECK(grad_output.scalar_type() == at::kFloat && q.scalar_type() == at::kFloat &&
                  k.scalar_type() == at::kFloat && v.scalar_type() == at::kFloat,
              "the CUDA attention backward kernel currently supports float32 only");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "q, k, and v must be contiguous");
  TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0) &&
                  q.size(1) == k.size(1) && k.size(1) == v.size(1),
              "q, k, and v must have identical batch and head dimensions");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k must have the same head dimension");
  TORCH_CHECK(k.size(2) == v.size(2), "k and v must have the same sequence length");
  TORCH_CHECK(grad_output.size(0) == q.size(0) && grad_output.size(1) == q.size(1) &&
                  grad_output.size(2) == q.size(2) && grad_output.size(3) == v.size(3),
              "grad_output must have shape [batch, heads, query_length, value_dim]");
  TORCH_CHECK(q.size(3) > 0, "attention head dimension must be positive");
  TORCH_CHECK(k.size(2) > 0, "key sequence length must be positive");
  TORCH_CHECK(!causal || q.size(2) <= k.size(2),
              "right-aligned causal attention requires query_length <= key_length");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");

  const auto grad_output_contiguous = grad_output.contiguous();
  auto grad_q = at::empty_like(q);
  auto grad_k = at::zeros_like(k);
  auto grad_v = at::zeros_like(v);
  const auto batch = q.size(0);
  const auto heads = q.size(1);
  const auto query_length = q.size(2);
  const auto key_length = k.size(2);
  const auto head_dim = q.size(3);
  const auto value_dim = v.size(3);

  if (batch == 0 || heads == 0 || query_length == 0) {
    return {grad_q, grad_k, grad_v};
  }
  if (value_dim == 0) {
    grad_q.zero_();
    grad_k.zero_();
    return {grad_q, grad_k, grad_v};
  }

  const int64_t query_rows = batch * heads * query_length;
  const c10::cuda::CUDAGuard device_guard(q.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(query_rows <= static_cast<int64_t>(properties->maxGridSize[0]),
              "too many query rows for a one-dimensional CUDA launch: ", query_rows);
  const size_t shared_bytes =
      static_cast<size_t>(2 * head_dim + value_dim + kThreads + kStatisticCount) * sizeof(float);
  TORCH_CHECK(shared_bytes <= static_cast<size_t>(properties->sharedMemPerBlock),
              "attention backward dimensions require ", shared_bytes,
              " bytes of shared memory, but the device limit is ", properties->sharedMemPerBlock);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  attention_backward_float_kernel<<<
      static_cast<unsigned int>(query_rows), kThreads, shared_bytes, stream>>>(
      grad_output_contiguous.const_data_ptr<float>(),
      q.const_data_ptr<float>(),
      k.const_data_ptr<float>(),
      v.const_data_ptr<float>(),
      grad_q.mutable_data_ptr<float>(),
      grad_k.mutable_data_ptr<float>(),
      grad_v.mutable_data_ptr<float>(),
      query_length,
      key_length,
      head_dim,
      value_dim,
      static_cast<float>(scale),
      causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_q, grad_k, grad_v};
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("attention_backward", TORCH_FN(attention_backward_cuda));
}
