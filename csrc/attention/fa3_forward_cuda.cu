// Teaching FlashAttention-3 direction: asynchronous double-buffered K/V staging.
//
// This forward-only kernel intentionally demonstrates one FA3 pipeline idea. It
// is not the Hopper/Blackwell production algorithm: there is no TMA descriptor,
// warp-specialized producer/consumer schedule, WGMMA, or FP8 path.

#include "attention_common.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

namespace cg = cooperative_groups;

using ds_flash_mla_moe::attention::kKeyTile;
using ds_flash_mla_moe::attention::kMaxHeadDim;
using ds_flash_mla_moe::attention::kMaxValueDim;
using ds_flash_mla_moe::attention::kQueryTile;
using ds_flash_mla_moe::attention::kThreads;
using ds_flash_mla_moe::attention::kWarpSize;
using ds_flash_mla_moe::attention::key_is_visible;
using ds_flash_mla_moe::attention::validate_formal_attention_forward_inputs;
using ds_flash_mla_moe::attention::warp_sum;

constexpr int kPipelineStages = 2;
constexpr int kHeadValuesPerLane =
    (kMaxHeadDim + kWarpSize - 1) / kWarpSize;
constexpr int kOutputValuesPerLane =
    (kMaxValueDim + kWarpSize - 1) / kWarpSize;

__device__ __forceinline__ float stable_rescale(float source_max, float target_max) {
  return source_max == -CUDART_INF_F ? 0.0F : expf(source_max - target_max);
}

__device__ __forceinline__ void enqueue_kv_tile(
    const cg::thread_block& cta,
    const at::Half* k,
    const at::Half* v,
    at::Half* key_stage,
    at::Half* value_stage,
    int64_t batch_head,
    int64_t key_block,
    int valid_keys,
    int64_t key_length,
    int64_t head_dim,
    int64_t value_dim) {
  const at::Half* key_source =
      k + (batch_head * key_length + key_block) * head_dim;
  const at::Half* value_source =
      v + (batch_head * key_length + key_block) * value_dim;
  cg::memcpy_async(
      cta,
      key_stage,
      key_source,
      static_cast<size_t>(valid_keys) * head_dim * sizeof(at::Half));
  cg::memcpy_async(
      cta,
      value_stage,
      value_source,
      static_cast<size_t>(valid_keys) * value_dim * sizeof(at::Half));
}

__global__ void fa3_teaching_forward_kernel(
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
  const cg::thread_block cta = cg::this_thread_block();

  extern __shared__ at::Half shared_half[];
  at::Half* key_stages[kPipelineStages] = {
      shared_half,
      shared_half + kKeyTile * head_dim,
  };
  at::Half* value_base =
      shared_half + kPipelineStages * kKeyTile * head_dim;
  at::Half* value_stages[kPipelineStages] = {
      value_base,
      value_base + kKeyTile * value_dim,
  };

  // One CTA owns a Q tile. Each warp keeps one Q row and its FP32 online
  // softmax/output state in registers for the full K/V traversal.
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

  const int first_valid_keys =
      static_cast<int>(min(static_cast<int64_t>(kKeyTile), key_length));
  enqueue_kv_tile(
      cta,
      k,
      v,
      key_stages[0],
      value_stages[0],
      batch_head,
      0,
      first_valid_keys,
      key_length,
      head_dim,
      value_dim);
  cg::wait(cta);
  cta.sync();

  int stage = 0;
  for (int64_t key_block = 0; key_block < key_length; key_block += kKeyTile) {
    const int valid_keys = static_cast<int>(
        min(static_cast<int64_t>(kKeyTile), key_length - key_block));
    const int64_t next_key_block = key_block + kKeyTile;
    const bool has_next_tile = next_key_block < key_length;

    // Producer phase: issue the next global-to-shared copy into the other
    // stage before consumers start calculating from the current stage.
    if (has_next_tile) {
      const int next_stage = 1 - stage;
      const int next_valid_keys = static_cast<int>(
          min(static_cast<int64_t>(kKeyTile), key_length - next_key_block));
      enqueue_kv_tile(
          cta,
          k,
          v,
          key_stages[next_stage],
          value_stages[next_stage],
          batch_head,
          next_key_block,
          next_valid_keys,
          key_length,
          head_dim,
          value_dim);
    }

    const int64_t final_query_position =
        min(query_block + kQueryTile - 1, query_length - 1);
    const bool tile_is_visible =
        !causal || key_block <= final_query_position + key_length - query_length;

    // Consumer phase: each warp exclusively updates one query row. The copy
    // for the next tile may progress while this arithmetic uses the current
    // half-precision shared-memory stage and FP32 register accumulators.
    if (valid_query && tile_is_visible) {
      for (int key_in_tile = 0; key_in_tile < valid_keys; ++key_in_tile) {
        const int64_t key_position = key_block + key_in_tile;
        if (!key_is_visible(
                query_position, key_position, query_length, key_length, causal)) {
          continue;
        }

        float dot = 0.0F;
#pragma unroll
        for (int slot = 0; slot < kHeadValuesPerLane; ++slot) {
          const int64_t column = lane_id + slot * kWarpSize;
          if (column < head_dim) {
            dot += query_values[slot] * static_cast<float>(
                key_stages[stage][key_in_tile * head_dim + column]);
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
                beta * static_cast<float>(
                    value_stages[stage][key_in_tile * value_dim + column]);
          }
        }
        row_sum = alpha * row_sum + beta;
        row_max = next_max;
      }
    }

    if (has_next_tile) {
      cg::wait(cta);
    }
    // No thread may recycle the just-consumed stage until every warp has
    // finished reading it. The next loop iteration then flips the buffers.
    cta.sync();
    stage = 1 - stage;
  }

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

at::Tensor attention_fa3_forward_cuda(
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

  const size_t shared_half_count =
      static_cast<size_t>(kPipelineStages) * kKeyTile * (head_dim + value_dim);
  const size_t shared_bytes = shared_half_count * sizeof(at::Half);
  TORCH_CHECK(
      shared_bytes <= static_cast<size_t>(properties->sharedMemPerBlock),
      "teaching FA3 dimensions require ",
      shared_bytes,
      " bytes of shared memory, but the device limit is ",
      properties->sharedMemPerBlock);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  fa3_teaching_forward_kernel<<<
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
  m.impl("attention_fa3_forward", TORCH_FN(attention_fa3_forward_cuda));
}
