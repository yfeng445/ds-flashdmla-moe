// Formal FlashAttention-1 forward operator following https://arxiv.org/abs/2205.14135.

#include "attention_common.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using ds_flash_mla_moe::attention::kKeyTile;
using ds_flash_mla_moe::attention::kQueryTile;
using ds_flash_mla_moe::attention::kThreads;
using ds_flash_mla_moe::attention::kWarpSize;
using ds_flash_mla_moe::attention::kWarps;
using ds_flash_mla_moe::attention::key_is_visible;
using ds_flash_mla_moe::attention::validate_formal_attention_forward_inputs;
using ds_flash_mla_moe::attention::warp_sum;

__device__ __forceinline__ float stable_rescale(float source_max, float target_max) {
  return source_max == -CUDART_INF_F ? 0.0F : expf(source_max - target_max);
}

__global__ void fa1_forward_kernel(
    const at::Half* q,
    const at::Half* k,
    const at::Half* v,
    float* normalized_output,
    float* row_max,
    float* row_sum,
    at::Half* output,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t value_dim,
    float scale,
    bool causal) {
  const int64_t batch_head = static_cast<int64_t>(blockIdx.x);
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  extern __shared__ float shared[];
  float* query_tile = shared;
  float* key_tile = query_tile + kQueryTile * head_dim;
  float* value_tile = key_tile + kKeyTile * head_dim;
  float* partial_output = value_tile + kKeyTile * value_dim;
  float* previous_output = partial_output + kQueryTile * kWarps * value_dim;
  float* local_max = previous_output + kQueryTile * value_dim;
  float* local_sum = local_max + kQueryTile * kWarps;
  float* merge_max = local_sum + kQueryTile * kWarps;
  float* merge_sum = merge_max + kQueryTile;

  for (int64_t key_block = 0; key_block < key_length; key_block += kKeyTile) {
    const int valid_keys =
        static_cast<int>(min(static_cast<int64_t>(kKeyTile), key_length - key_block));

    // Cooperative FP32 K/V load.
    for (int64_t index = threadIdx.x; index < valid_keys * head_dim;
         index += blockDim.x) {
      const int64_t key_in_tile = index / head_dim;
      const int64_t column = index % head_dim;
      const int64_t global_offset =
          ((batch_head * key_length + key_block + key_in_tile) * head_dim) + column;
      key_tile[index] = static_cast<float>(k[global_offset]);
    }
    for (int64_t index = threadIdx.x; index < valid_keys * value_dim;
         index += blockDim.x) {
      const int64_t key_in_tile = index / value_dim;
      const int64_t column = index % value_dim;
      const int64_t global_offset =
          ((batch_head * key_length + key_block + key_in_tile) * value_dim) + column;
      value_tile[index] = static_cast<float>(v[global_offset]);
    }
    __syncthreads();

    for (int64_t query_block = 0; query_block < query_length;
         query_block += kQueryTile) {
      const int valid_queries = static_cast<int>(
          min(static_cast<int64_t>(kQueryTile), query_length - query_block));

      // Skip this pair when the whole K/V tile lies above the causal boundary.
      const int64_t final_query_position = query_block + valid_queries - 1;
      if (causal &&
          key_block > final_query_position + key_length - query_length) {
        continue;
      }

      // Cooperative FP32 Q and normalized O/m/l reload.
      for (int64_t index = threadIdx.x; index < valid_queries * head_dim;
           index += blockDim.x) {
        const int64_t query_in_tile = index / head_dim;
        const int64_t column = index % head_dim;
        const int64_t global_offset =
            ((batch_head * query_length + query_block + query_in_tile) * head_dim) + column;
        query_tile[index] = static_cast<float>(q[global_offset]);
      }
      for (int64_t index = threadIdx.x; index < valid_queries * value_dim;
           index += blockDim.x) {
        const int64_t query_in_tile = index / value_dim;
        const int64_t column = index % value_dim;
        const int64_t workspace_offset =
            ((batch_head * query_length + query_block + query_in_tile) * value_dim) + column;
        previous_output[index] = normalized_output[workspace_offset];
      }
      for (int query_in_tile = threadIdx.x; query_in_tile < valid_queries;
           query_in_tile += blockDim.x) {
        const int64_t row = batch_head * query_length + query_block + query_in_tile;
        merge_max[query_in_tile] = row_max[row];
        merge_sum[query_in_tile] = row_sum[row];
      }
      for (int64_t index = threadIdx.x;
           index < static_cast<int64_t>(valid_queries) * kWarps * value_dim;
           index += blockDim.x) {
        partial_output[index] = 0.0F;
      }
      __syncthreads();

      for (int query_in_tile = 0; query_in_tile < valid_queries; ++query_in_tile) {
        float warp_max = -CUDART_INF_F;
        float warp_denominator = 0.0F;
        float* warp_numerator =
            partial_output + (query_in_tile * kWarps + warp_id) * value_dim;

        // Warps split key positions; lanes reduce every FP32 QK dot product.
        for (int key_in_tile = warp_id; key_in_tile < valid_keys;
             key_in_tile += kWarps) {
          const int64_t query_position = query_block + query_in_tile;
          const int64_t key_position = key_block + key_in_tile;
          if (!key_is_visible(
                  query_position, key_position, query_length, key_length, causal)) {
            continue;
          }

          float dot = 0.0F;
          for (int64_t column = lane_id; column < head_dim; column += kWarpSize) {
            dot += query_tile[query_in_tile * head_dim + column] *
                key_tile[key_in_tile * head_dim + column];
          }
          dot = warp_sum(dot);
          const float score = __shfl_sync(0xffffffff, dot * scale, 0);
          const float next_max = fmaxf(warp_max, score);
          const float previous_scale = stable_rescale(warp_max, next_max);
          const float current_scale = expf(score - next_max);

          for (int64_t column = lane_id; column < value_dim; column += kWarpSize) {
            warp_numerator[column] =
                warp_numerator[column] * previous_scale +
                current_scale * value_tile[key_in_tile * value_dim + column];
          }
          warp_denominator = warp_denominator * previous_scale + current_scale;
          warp_max = next_max;
        }

        if (lane_id == 0) {
          const int state_index = query_in_tile * kWarps + warp_id;
          local_max[state_index] = warp_max;
          local_sum[state_index] = warp_denominator;
        }
      }
      __syncthreads();

      // Merge the four warp-local m/l/numerator states with the global state.
      for (int query_in_tile = threadIdx.x; query_in_tile < valid_queries;
           query_in_tile += blockDim.x) {
        const float old_m = merge_max[query_in_tile];
        float next_m = old_m;
        for (int warp = 0; warp < kWarps; ++warp) {
          next_m = fmaxf(next_m, local_max[query_in_tile * kWarps + warp]);
        }

        float next_l = merge_sum[query_in_tile] * stable_rescale(old_m, next_m);
        for (int warp = 0; warp < kWarps; ++warp) {
          const int state_index = query_in_tile * kWarps + warp;
          next_l += local_sum[state_index] *
              stable_rescale(local_max[state_index], next_m);
        }
        merge_max[query_in_tile] = next_m;
        merge_sum[query_in_tile] = next_l;
      }
      __syncthreads();

      for (int64_t index = threadIdx.x; index < valid_queries * value_dim;
           index += blockDim.x) {
        const int query_in_tile = static_cast<int>(index / value_dim);
        const int64_t column = index % value_dim;
        const float old_m = row_max[batch_head * query_length + query_block + query_in_tile];
        const float old_l = row_sum[batch_head * query_length + query_block + query_in_tile];
        const float next_m = merge_max[query_in_tile];
        const float next_l = merge_sum[query_in_tile];

        float next_numerator =
            previous_output[index] * old_l * stable_rescale(old_m, next_m);
        for (int warp = 0; warp < kWarps; ++warp) {
          const int state_index = query_in_tile * kWarps + warp;
          next_numerator +=
              partial_output[(state_index * value_dim) + column] *
              stable_rescale(local_max[state_index], next_m);
        }
        const int64_t workspace_offset =
            ((batch_head * query_length + query_block + query_in_tile) * value_dim) + column;
        normalized_output[workspace_offset] =
            next_l > 0.0F ? next_numerator / next_l : 0.0F;
      }
      __syncthreads();
      for (int query_in_tile = threadIdx.x; query_in_tile < valid_queries;
           query_in_tile += blockDim.x) {
        const int64_t row = batch_head * query_length + query_block + query_in_tile;
        row_max[row] = merge_max[query_in_tile];
        row_sum[row] = merge_sum[query_in_tile];
      }
      __syncthreads();
    }
    __syncthreads();
  }

  // Cast the final normalized FP32 workspace to FP16 output once.
  const int64_t batch_head_elements = query_length * value_dim;
  for (int64_t index = threadIdx.x; index < batch_head_elements; index += blockDim.x) {
    const int64_t global_offset = batch_head * batch_head_elements + index;
    output[global_offset] = static_cast<at::Half>(normalized_output[global_offset]);
  }
}

at::Tensor attention_fa1_forward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    double scale) {
  validate_formal_attention_forward_inputs(q, k, v, causal, scale);
  const c10::cuda::CUDAGuard device_guard(q.device());

  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t query_length = q.size(2);
  const int64_t key_length = k.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  auto output = at::empty({batch, heads, query_length, value_dim}, v.options());
  if (batch == 0 || heads == 0 || query_length == 0 || value_dim == 0) {
    return output;
  }

  const int64_t batch_heads = batch * heads;
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(batch_heads <= static_cast<int64_t>(properties->maxGridSize[0]),
              "too many batch-head rows for a one-dimensional CUDA launch: ", batch_heads);

  const size_t shared_float_count =
      static_cast<size_t>(kQueryTile) * head_dim +
      static_cast<size_t>(kKeyTile) * head_dim +
      static_cast<size_t>(kKeyTile) * value_dim +
      static_cast<size_t>(kQueryTile) * kWarps * value_dim +
      static_cast<size_t>(kQueryTile) * value_dim +
      static_cast<size_t>(2 * kQueryTile * kWarps + 2 * kQueryTile);
  const size_t shared_bytes = shared_float_count * sizeof(float);
  TORCH_CHECK(shared_bytes <= static_cast<size_t>(properties->sharedMemPerBlock),
              "formal FA1 dimensions require ", shared_bytes,
              " bytes of shared memory, but the device limit is ",
              properties->sharedMemPerBlock);

  auto workspace_options = q.options().dtype(at::kFloat);
  auto normalized_output =
      at::zeros({batch, heads, query_length, value_dim}, workspace_options);
  auto row_max = at::full(
      {batch, heads, query_length}, -std::numeric_limits<float>::infinity(), workspace_options);
  auto row_sum = at::zeros({batch, heads, query_length}, workspace_options);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  fa1_forward_kernel<<<
      static_cast<unsigned int>(batch_heads), kThreads, shared_bytes, stream>>>(
      q.const_data_ptr<at::Half>(),
      k.const_data_ptr<at::Half>(),
      v.const_data_ptr<at::Half>(),
      normalized_output.mutable_data_ptr<float>(),
      row_max.mutable_data_ptr<float>(),
      row_sum.mutable_data_ptr<float>(),
      output.mutable_data_ptr<at::Half>(),
      query_length,
      key_length,
      head_dim,
      value_dim,
      static_cast<float>(scale),
      causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("attention_fa1_forward", TORCH_FN(attention_fa1_forward_cuda));
}
