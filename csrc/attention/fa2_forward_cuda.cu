// Formal FlashAttention-2 forward operator following https://arxiv.org/abs/2307.08691.

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

namespace {

using ds_flash_mla_moe::attention::kKeyTile;
using ds_flash_mla_moe::attention::kMaxHeadDim;
using ds_flash_mla_moe::attention::kMaxValueDim;
using ds_flash_mla_moe::attention::kQueryTile;
using ds_flash_mla_moe::attention::kThreads;
using ds_flash_mla_moe::attention::kWarpSize;
using ds_flash_mla_moe::attention::key_is_visible;
using ds_flash_mla_moe::attention::validate_formal_attention_forward_inputs;
using ds_flash_mla_moe::attention::warp_sum;

constexpr int kHeadValuesPerLane =
    (kMaxHeadDim + kWarpSize - 1) / kWarpSize;
constexpr int kOutputValuesPerLane =
    (kMaxValueDim + kWarpSize - 1) / kWarpSize;

__device__ __forceinline__ float stable_rescale(float source_max, float target_max) {
  return source_max == -CUDART_INF_F ? 0.0F : expf(source_max - target_max);
}

__global__ void fa2_forward_kernel(
    const at::Half* q,
    const at::Half* k,
    const at::Half* v,
    at::Half* output,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t value_dim,
    int64_t query_blocks,
    float scale,
    bool causal) {
  const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
  const int64_t query_block_index = flat_block % query_blocks;
  const int64_t batch_head = flat_block / query_blocks;
  const int64_t query_block = query_block_index * kQueryTile;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int64_t query_position = query_block + warp_id;
  const bool valid_query = query_position < query_length;

  extern __shared__ float shared[];
  float* key_tile = shared;
  float* value_tile = key_tile + kKeyTile * head_dim;

  // Load this CTA's four Q rows once and initialize FP32 m/l/O on chip.
  float query_values[kHeadValuesPerLane];
#pragma unroll
  for (int slot = 0; slot < kHeadValuesPerLane; ++slot) {
    const int64_t column = lane_id + slot * kWarpSize;
    query_values[slot] =
        valid_query && column < head_dim
        ? static_cast<float>(
              q[(batch_head * query_length + query_position) * head_dim + column])
        : 0.0F;
  }

  float output_values[kOutputValuesPerLane];
#pragma unroll
  for (int slot = 0; slot < kOutputValuesPerLane; ++slot) {
    output_values[slot] = 0.0F;
  }
  float row_max = -CUDART_INF_F;
  float row_sum = 0.0F;

  for (int64_t key_block = 0; key_block < key_length; key_block += kKeyTile) {
    const int valid_keys = static_cast<int>(
        min(static_cast<int64_t>(kKeyTile), key_length - key_block));

    // Skip the tile when it is fully masked for all valid Q rows in this CTA.
    const int64_t final_query_position =
        min(query_block + kQueryTile - 1, query_length - 1);
    if (causal &&
        key_block > final_query_position + key_length - query_length) {
      continue;
    }

    // Cooperatively load FP32 K/V tile.
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

    // warp_id exclusively updates query row query_block + warp_id.
    if (valid_query) {
      for (int key_in_tile = 0; key_in_tile < valid_keys; ++key_in_tile) {
        const int64_t key_position = key_block + key_in_tile;
        // FP32 dot reduction, right-aligned causal check, unnormalized recurrence.
        if (!key_is_visible(
                query_position, key_position, query_length, key_length, causal)) {
          continue;
        }

        float dot = 0.0F;
#pragma unroll
        for (int slot = 0; slot < kHeadValuesPerLane; ++slot) {
          const int64_t column = lane_id + slot * kWarpSize;
          if (column < head_dim) {
            dot += query_values[slot] * key_tile[key_in_tile * head_dim + column];
          }
        }
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * scale;
        const float next_max = fmaxf(row_max, score);
        const float alpha = stable_rescale(row_max, next_max);
        const float beta = expf(score - next_max);

#pragma unroll
        for (int slot = 0; slot < kOutputValuesPerLane; ++slot) {
          const int64_t column = lane_id + slot * kWarpSize;
          if (column < value_dim) {
            output_values[slot] = alpha * output_values[slot] +
                beta * value_tile[key_in_tile * value_dim + column];
          }
        }
        row_sum = alpha * row_sum + beta;
        row_max = next_max;
      }
    }
    __syncthreads();
  }

  // Divide each owned O row by l once and store FP16 output once.
  if (valid_query) {
#pragma unroll
    for (int slot = 0; slot < kOutputValuesPerLane; ++slot) {
      const int64_t column = lane_id + slot * kWarpSize;
      if (column < value_dim) {
        const int64_t output_offset =
            (batch_head * query_length + query_position) * value_dim + column;
        output[output_offset] = static_cast<at::Half>(
            row_sum > 0.0F ? output_values[slot] / row_sum : 0.0F);
      }
    }
  }
}

at::Tensor attention_fa2_forward_cuda(
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
  const int64_t query_blocks =
      (query_length + kQueryTile - 1) / kQueryTile;
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(
      batch_heads <=
          static_cast<int64_t>(properties->maxGridSize[0]) / query_blocks,
      "too many batch-head query blocks for a one-dimensional CUDA launch: ",
      batch_heads * query_blocks);
  const int64_t grid_blocks = batch_heads * query_blocks;

  const size_t shared_float_count =
      static_cast<size_t>(kKeyTile) * (head_dim + value_dim);
  const size_t shared_bytes = shared_float_count * sizeof(float);
  TORCH_CHECK(shared_bytes <= static_cast<size_t>(properties->sharedMemPerBlock),
              "formal FA2 dimensions require ", shared_bytes,
              " bytes of shared memory, but the device limit is ",
              properties->sharedMemPerBlock);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  fa2_forward_kernel<<<
      static_cast<unsigned int>(grid_blocks), kThreads, shared_bytes, stream>>>(
      q.const_data_ptr<at::Half>(),
      k.const_data_ptr<at::Half>(),
      v.const_data_ptr<at::Half>(),
      output.mutable_data_ptr<at::Half>(),
      query_length,
      key_length,
      head_dim,
      value_dim,
      query_blocks,
      static_cast<float>(scale),
      causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("attention_fa2_forward", TORCH_FN(attention_fa2_forward_cuda));
}
