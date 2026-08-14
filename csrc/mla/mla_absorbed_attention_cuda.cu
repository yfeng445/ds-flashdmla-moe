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
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

// Small DeepSeek-style head dimensions leave most of a 128-thread reduction
// block idle. Split the key sequence across four warps instead: each warp keeps
// an independent online-softmax state, then warp zero performs one stable merge
// and the latent-to-value projection. This removes block-wide barriers from the
// per-key loop while preserving the generic strided-tensor contract.
template <typename scalar_t>
__global__ void mla_absorbed_attention_warp_partition_kernel(
    const scalar_t* __restrict__ q_nope,
    const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ kv,
    const scalar_t* __restrict__ pe,
    const scalar_t* __restrict__ key_up,
    const scalar_t* __restrict__ value_up,
    const int64_t* __restrict__ query_positions,
    const int64_t* __restrict__ key_positions,
    scalar_t* __restrict__ output,
    int64_t heads,
    int64_t query_length,
    int64_t key_length,
    int64_t nope_dim,
    int64_t rope_dim,
    int64_t latent_dim,
    int64_t value_dim,
    int64_t q_nope_stride_batch,
    int64_t q_nope_stride_query,
    int64_t q_nope_stride_head,
    int64_t q_nope_stride_dim,
    int64_t q_pe_stride_batch,
    int64_t q_pe_stride_query,
    int64_t q_pe_stride_head,
    int64_t q_pe_stride_dim,
    int64_t kv_stride_batch,
    int64_t kv_stride_key,
    int64_t kv_stride_dim,
    int64_t pe_stride_batch,
    int64_t pe_stride_key,
    int64_t pe_stride_dim,
    int64_t key_up_stride_head,
    int64_t key_up_stride_nope,
    int64_t key_up_stride_latent,
    int64_t value_up_stride_head,
    int64_t value_up_stride_value,
    int64_t value_up_stride_latent,
    int64_t query_position_stride,
    int64_t key_position_stride,
    float scale,
    bool causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t query_index = row % query_length;
  const int64_t batch_head = row / query_length;
  const int64_t head = batch_head % heads;
  const int64_t batch = batch_head / heads;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  const int64_t q_nope_offset = batch * q_nope_stride_batch +
      query_index * q_nope_stride_query + head * q_nope_stride_head;
  const int64_t q_pe_offset = batch * q_pe_stride_batch +
      query_index * q_pe_stride_query + head * q_pe_stride_head;
  const int64_t key_up_offset = head * key_up_stride_head;
  const int64_t value_up_offset = head * value_up_stride_head;
  const int64_t output_offset =
      ((batch * query_length + query_index) * heads + head) * value_dim;

  __shared__ float shared_q_latent[kWarpSize];
  __shared__ float partition_max[kWarpsPerBlock];
  __shared__ float partition_denominator[kWarpsPerBlock];
  __shared__ float partition_numerator[kWarpsPerBlock][kWarpSize];
  __shared__ float partition_scale[kWarpsPerBlock];
  __shared__ float global_denominator;

  if (warp == 0 && lane < latent_dim) {
    float accumulator = 0.0F;
    for (int64_t column = 0; column < nope_dim; ++column) {
      accumulator = fmaf(
          static_cast<float>(q_nope[q_nope_offset + column * q_nope_stride_dim]),
          static_cast<float>(key_up[key_up_offset + column * key_up_stride_nope +
                                    lane * key_up_stride_latent]),
          accumulator);
    }
    shared_q_latent[lane] = accumulator;
  }
  __syncthreads();

  const float q_latent = lane < latent_dim ? shared_q_latent[lane] : 0.0F;
  float numerator = 0.0F;
  float running_max = -CUDART_INF_F;
  float denominator = 0.0F;

  for (int64_t key_index = warp; key_index < key_length;
       key_index += kWarpsPerBlock) {
    const bool visible = !causal ||
        key_positions[key_index * key_position_stride] <=
            query_positions[query_index * query_position_stride];
    if (!visible) {
      continue;
    }
    const int64_t kv_offset = batch * kv_stride_batch + key_index * kv_stride_key;
    const int64_t pe_offset = batch * pe_stride_batch + key_index * pe_stride_key;
    float partial = 0.0F;
    if (lane < latent_dim) {
      partial = q_latent * static_cast<float>(kv[kv_offset + lane * kv_stride_dim]);
    }
    if (lane < rope_dim) {
      partial = fmaf(
          static_cast<float>(q_pe[q_pe_offset + lane * q_pe_stride_dim]),
          static_cast<float>(pe[pe_offset + lane * pe_stride_dim]),
          partial);
    }
    partial = warp_sum(partial);

    float previous_scale = 0.0F;
    float current_scale = 0.0F;
    if (lane == 0) {
      const float score = partial * scale;
      const float next_max = fmaxf(running_max, score);
      previous_scale = isinf(running_max) && running_max < 0.0F
          ? 0.0F
          : expf(running_max - next_max);
      current_scale = expf(score - next_max);
      running_max = next_max;
      denominator = denominator * previous_scale + current_scale;
    }
    previous_scale = __shfl_sync(kFullWarpMask, previous_scale, 0);
    current_scale = __shfl_sync(kFullWarpMask, current_scale, 0);
    if (lane < latent_dim) {
      numerator = numerator * previous_scale +
          current_scale * static_cast<float>(kv[kv_offset + lane * kv_stride_dim]);
    }
  }

  if (lane == 0) {
    partition_max[warp] = running_max;
    partition_denominator[warp] = denominator;
  }
  if (lane < latent_dim) {
    partition_numerator[warp][lane] = numerator;
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }
  if (lane == 0) {
    float combined_max = -CUDART_INF_F;
#pragma unroll
    for (int partition = 0; partition < kWarpsPerBlock; ++partition) {
      if (partition_denominator[partition] > 0.0F) {
        combined_max = fmaxf(combined_max, partition_max[partition]);
      }
    }
    float combined_denominator = 0.0F;
#pragma unroll
    for (int partition = 0; partition < kWarpsPerBlock; ++partition) {
      const float partition_weight = partition_denominator[partition] > 0.0F
          ? expf(partition_max[partition] - combined_max)
          : 0.0F;
      partition_scale[partition] = partition_weight;
      combined_denominator +=
          partition_denominator[partition] * partition_weight;
    }
    global_denominator = combined_denominator;
  }
  __syncwarp();

  if (lane < value_dim) {
    float accumulator = 0.0F;
    if (global_denominator > 0.0F) {
      for (int64_t latent = 0; latent < latent_dim; ++latent) {
        float combined_numerator = 0.0F;
#pragma unroll
        for (int partition = 0; partition < kWarpsPerBlock; ++partition) {
          combined_numerator = fmaf(
              partition_numerator[partition][latent],
              partition_scale[partition],
              combined_numerator);
        }
        accumulator = fmaf(
            combined_numerator / global_denominator,
            static_cast<float>(value_up[value_up_offset + lane * value_up_stride_value +
                                        latent * value_up_stride_latent]),
            accumulator);
      }
    }
    output[output_offset + lane] = static_cast<scalar_t>(accumulator);
  }
}

template <typename scalar_t>
__global__ void mla_absorbed_attention_generic_kernel(
    const scalar_t* __restrict__ q_nope,
    const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ kv,
    const scalar_t* __restrict__ pe,
    const scalar_t* __restrict__ key_up,
    const scalar_t* __restrict__ value_up,
    const int64_t* __restrict__ query_positions,
    const int64_t* __restrict__ key_positions,
    scalar_t* __restrict__ output,
    int64_t heads,
    int64_t query_length,
    int64_t key_length,
    int64_t nope_dim,
    int64_t rope_dim,
    int64_t latent_dim,
    int64_t value_dim,
    int64_t q_nope_stride_batch,
    int64_t q_nope_stride_query,
    int64_t q_nope_stride_head,
    int64_t q_nope_stride_dim,
    int64_t q_pe_stride_batch,
    int64_t q_pe_stride_query,
    int64_t q_pe_stride_head,
    int64_t q_pe_stride_dim,
    int64_t kv_stride_batch,
    int64_t kv_stride_key,
    int64_t kv_stride_dim,
    int64_t pe_stride_batch,
    int64_t pe_stride_key,
    int64_t pe_stride_dim,
    int64_t key_up_stride_head,
    int64_t key_up_stride_nope,
    int64_t key_up_stride_latent,
    int64_t value_up_stride_head,
    int64_t value_up_stride_value,
    int64_t value_up_stride_latent,
    int64_t query_position_stride,
    int64_t key_position_stride,
    float scale,
    bool causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t query_index = row % query_length;
  const int64_t batch_head = row / query_length;
  const int64_t head = batch_head % heads;
  const int64_t batch = batch_head / heads;

  const int64_t q_nope_offset = batch * q_nope_stride_batch +
      query_index * q_nope_stride_query + head * q_nope_stride_head;
  const int64_t q_pe_offset = batch * q_pe_stride_batch +
      query_index * q_pe_stride_query + head * q_pe_stride_head;
  const int64_t key_up_offset = head * key_up_stride_head;
  const int64_t value_up_offset = head * value_up_stride_head;
  const int64_t output_offset = ((batch * query_length + query_index) * heads + head) * value_dim;

  extern __shared__ float shared[];
  float* q_latent = shared;
  float* numerator = q_latent + latent_dim;
  float* reduction = numerator + latent_dim;
  float* statistics = reduction + blockDim.x;

  for (int64_t latent = threadIdx.x; latent < latent_dim; latent += blockDim.x) {
    float accumulator = 0.0F;
    for (int64_t column = 0; column < nope_dim; ++column) {
      accumulator = fmaf(
          static_cast<float>(q_nope[q_nope_offset + column * q_nope_stride_dim]),
          static_cast<float>(key_up[key_up_offset + column * key_up_stride_nope +
                                    latent * key_up_stride_latent]),
          accumulator);
    }
    q_latent[latent] = accumulator;
    numerator[latent] = 0.0F;
  }
  if (threadIdx.x == 0) {
    statistics[0] = -CUDART_INF_F;
    statistics[1] = 0.0F;
  }
  __syncthreads();

  for (int64_t key_index = 0; key_index < key_length; ++key_index) {
    const bool visible = !causal ||
        key_positions[key_index * key_position_stride] <=
            query_positions[query_index * query_position_stride];
    if (!visible) {
      continue;
    }
    const int64_t kv_offset = batch * kv_stride_batch + key_index * kv_stride_key;
    const int64_t pe_offset = batch * pe_stride_batch + key_index * pe_stride_key;
    float partial = 0.0F;
    for (int64_t latent = threadIdx.x; latent < latent_dim; latent += blockDim.x) {
      partial = fmaf(
          q_latent[latent],
          static_cast<float>(kv[kv_offset + latent * kv_stride_dim]),
          partial);
    }
    for (int64_t column = threadIdx.x; column < rope_dim; column += blockDim.x) {
      partial = fmaf(
          static_cast<float>(q_pe[q_pe_offset + column * q_pe_stride_dim]),
          static_cast<float>(pe[pe_offset + column * pe_stride_dim]),
          partial);
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
    for (int64_t latent = threadIdx.x; latent < latent_dim; latent += blockDim.x) {
      numerator[latent] =
          numerator[latent] * previous_scale +
          current_scale * static_cast<float>(kv[kv_offset + latent * kv_stride_dim]);
    }
    __syncthreads();
  }

  const float denominator = statistics[1];
  for (int64_t value = threadIdx.x; value < value_dim; value += blockDim.x) {
    float accumulator = 0.0F;
    if (denominator > 0.0F) {
      for (int64_t latent = 0; latent < latent_dim; ++latent) {
        accumulator = fmaf(
            numerator[latent] / denominator,
            static_cast<float>(value_up[value_up_offset + value * value_up_stride_value +
                                        latent * value_up_stride_latent]),
            accumulator);
      }
    }
    output[output_offset + value] = static_cast<scalar_t>(accumulator);
  }
}

void check_cuda_mla_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      name,
      " must use float16, bfloat16, or float32");
}

void check_cuda_long_vector(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must use int64");
  TORCH_CHECK(tensor.dim() == 1, name, " must be a vector");
}

at::Tensor mla_absorbed_attention_cuda(
    const at::Tensor& q_nope,
    const at::Tensor& q_pe,
    const at::Tensor& kv,
    const at::Tensor& pe,
    const at::Tensor& key_up,
    const at::Tensor& value_up,
    const at::Tensor& query_positions,
    const at::Tensor& key_positions,
    bool causal,
    double scale) {
  check_cuda_mla_tensor(q_nope, "q_nope");
  check_cuda_mla_tensor(q_pe, "q_pe");
  check_cuda_mla_tensor(kv, "kv");
  check_cuda_mla_tensor(pe, "pe");
  check_cuda_mla_tensor(key_up, "key_up");
  check_cuda_mla_tensor(value_up, "value_up");
  const auto scalar_type = q_nope.scalar_type();
  TORCH_CHECK(
      q_pe.scalar_type() == scalar_type && kv.scalar_type() == scalar_type &&
          pe.scalar_type() == scalar_type && key_up.scalar_type() == scalar_type &&
          value_up.scalar_type() == scalar_type,
      "all floating-point MLA tensors must have the same dtype");
  check_cuda_long_vector(query_positions, "query_positions");
  check_cuda_long_vector(key_positions, "key_positions");
  TORCH_CHECK(
      q_nope.device() == q_pe.device() && q_nope.device() == kv.device() &&
          q_nope.device() == pe.device() && q_nope.device() == key_up.device() &&
          q_nope.device() == value_up.device() &&
          q_nope.device() == query_positions.device() &&
          q_nope.device() == key_positions.device(),
      "all MLA tensors must be on the same CUDA device");
  TORCH_CHECK(q_nope.dim() == 4 && q_pe.dim() == 4,
              "q_nope and q_pe must have shape [batch, query, heads, dimension]");
  TORCH_CHECK(kv.dim() == 3 && pe.dim() == 3,
              "kv and pe must have shape [batch, key, dimension]");
  TORCH_CHECK(key_up.dim() == 3 && value_up.dim() == 3,
              "key_up and value_up must have shape [heads, output, latent]");
  TORCH_CHECK(q_nope.size(0) == q_pe.size(0) && q_nope.size(0) == kv.size(0) &&
                  q_nope.size(0) == pe.size(0),
              "query and cache batch dimensions must match");
  TORCH_CHECK(q_nope.size(1) == q_pe.size(1) && q_nope.size(2) == q_pe.size(2),
              "q_nope and q_pe query/head dimensions must match");
  TORCH_CHECK(kv.size(0) == pe.size(0) && kv.size(1) == pe.size(1),
              "kv and pe cache dimensions must match");
  TORCH_CHECK(q_nope.size(2) == key_up.size(0) && key_up.size(0) == value_up.size(0),
              "query and absorbed weights must have the same head count");
  TORCH_CHECK(q_nope.size(3) == key_up.size(1),
              "q_nope dimension must match key_up");
  TORCH_CHECK(q_pe.size(3) == pe.size(2),
              "q_pe dimension must match cached pe");
  TORCH_CHECK(kv.size(2) == key_up.size(2) && kv.size(2) == value_up.size(2),
              "cache latent dimension must match absorbed weights");
  TORCH_CHECK(query_positions.numel() == q_nope.size(1),
              "query_positions must match query length");
  TORCH_CHECK(key_positions.numel() == kv.size(1),
              "key_positions must match cache length");
  TORCH_CHECK(q_nope.size(2) > 0 && q_nope.size(3) > 0 && q_pe.size(3) > 0 &&
                  kv.size(2) > 0 && value_up.size(1) > 0,
              "MLA head, content, positional, latent, and value dimensions must be positive");
  TORCH_CHECK(kv.size(1) > 0, "MLA cache length must be positive");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");

  const int64_t batch = q_nope.size(0);
  const int64_t query_length = q_nope.size(1);
  const int64_t heads = q_nope.size(2);
  const int64_t nope_dim = q_nope.size(3);
  const int64_t rope_dim = q_pe.size(3);
  const int64_t key_length = kv.size(1);
  const int64_t latent_dim = kv.size(2);
  const int64_t value_dim = value_up.size(1);
  auto output = at::empty({batch, query_length, heads, value_dim}, q_nope.options());
  if (batch == 0 || query_length == 0) {
    return output;
  }

  const int64_t rows = batch * heads * query_length;
  const c10::cuda::CUDAGuard device_guard(q_nope.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(q_nope.get_device());
  TORCH_CHECK(rows <= static_cast<int64_t>(properties->maxGridSize[0]),
              "too many MLA query rows for a one-dimensional CUDA launch: ", rows);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_nope.get_device());
  // One lane owns each latent, RoPE, or value component in the specialized
  // path. Larger dimensions retain the original block-wide implementation.
  const bool use_warp_partition =
      latent_dim <= kWarpSize && rope_dim <= kWarpSize && value_dim <= kWarpSize;
  if (use_warp_partition) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "mla_absorbed_attention_warp_partition_cuda",
        [&] {
          mla_absorbed_attention_warp_partition_kernel<scalar_t><<<
              static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
              q_nope.const_data_ptr<scalar_t>(),
              q_pe.const_data_ptr<scalar_t>(),
              kv.const_data_ptr<scalar_t>(),
              pe.const_data_ptr<scalar_t>(),
              key_up.const_data_ptr<scalar_t>(),
              value_up.const_data_ptr<scalar_t>(),
              query_positions.const_data_ptr<int64_t>(),
              key_positions.const_data_ptr<int64_t>(),
              output.mutable_data_ptr<scalar_t>(),
              heads,
              query_length,
              key_length,
              nope_dim,
              rope_dim,
              latent_dim,
              value_dim,
              q_nope.stride(0),
              q_nope.stride(1),
              q_nope.stride(2),
              q_nope.stride(3),
              q_pe.stride(0),
              q_pe.stride(1),
              q_pe.stride(2),
              q_pe.stride(3),
              kv.stride(0),
              kv.stride(1),
              kv.stride(2),
              pe.stride(0),
              pe.stride(1),
              pe.stride(2),
              key_up.stride(0),
              key_up.stride(1),
              key_up.stride(2),
              value_up.stride(0),
              value_up.stride(1),
              value_up.stride(2),
              query_positions.stride(0),
              key_positions.stride(0),
              static_cast<float>(scale),
              causal);
        });
  } else {
    const size_t shared_bytes =
        static_cast<size_t>(2 * latent_dim + kThreads + 2) * sizeof(float);
    TORCH_CHECK(
        shared_bytes <= static_cast<size_t>(properties->sharedMemPerBlock),
        "MLA latent dimension requires ", shared_bytes,
        " bytes of shared memory, but the device limit is ",
        properties->sharedMemPerBlock);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "mla_absorbed_attention_generic_cuda",
        [&] {
          mla_absorbed_attention_generic_kernel<scalar_t><<<
              static_cast<unsigned int>(rows), kThreads, shared_bytes, stream>>>(
              q_nope.const_data_ptr<scalar_t>(),
              q_pe.const_data_ptr<scalar_t>(),
              kv.const_data_ptr<scalar_t>(),
              pe.const_data_ptr<scalar_t>(),
              key_up.const_data_ptr<scalar_t>(),
              value_up.const_data_ptr<scalar_t>(),
              query_positions.const_data_ptr<int64_t>(),
              key_positions.const_data_ptr<int64_t>(),
              output.mutable_data_ptr<scalar_t>(),
              heads,
              query_length,
              key_length,
              nope_dim,
              rope_dim,
              latent_dim,
              value_dim,
              q_nope.stride(0),
              q_nope.stride(1),
              q_nope.stride(2),
              q_nope.stride(3),
              q_pe.stride(0),
              q_pe.stride(1),
              q_pe.stride(2),
              q_pe.stride(3),
              kv.stride(0),
              kv.stride(1),
              kv.stride(2),
              pe.stride(0),
              pe.stride(1),
              pe.stride(2),
              key_up.stride(0),
              key_up.stride(1),
              key_up.stride(2),
              value_up.stride(0),
              value_up.stride(1),
              value_up.stride(2),
              query_positions.stride(0),
              key_positions.stride(0),
              static_cast<float>(scale),
              causal);
        });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("mla_absorbed_attention", TORCH_FN(mla_absorbed_attention_cuda));
}
