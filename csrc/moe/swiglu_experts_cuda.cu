#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Half.h>
#include <torch/library.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cassert>
#include <climits>
#include <cstdint>

namespace {

constexpr int kTile = 16;
constexpr int kOffsetThreads = 128;
constexpr int kWarpThreads = 32;

__host__ __device__ int64_t ceil_div_positive(int64_t value, int64_t divisor) {
  return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

int64_t grouped_row_tile_upper_bound(int64_t rows, int64_t experts) {
  if (rows == 0) {
    return 0;
  }
  const int64_t nonempty_expert_upper_bound = rows < experts ? rows : experts;
  return nonempty_expert_upper_bound +
      (rows - nonempty_expert_upper_bound) / kTile;
}

__global__ void validate_offsets_kernel(
    const int64_t* __restrict__ offsets,
    int64_t experts,
    int64_t rows) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index <= experts;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t value = offsets[index];
    assert(value >= 0 && value <= rows);
    if (index == 0) {
      assert(value == 0);
    }
    if (index == experts) {
      assert(value == rows);
    }
    if (index < experts) {
      assert(value <= offsets[index + 1]);
    }
  }
}

__global__ void build_grouped_tile_offsets_kernel(
    const int64_t* __restrict__ expert_offsets,
    int64_t* __restrict__ task_offsets,
    int64_t experts,
    int64_t output_tiles,
    int64_t task_upper_bound) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t prefix = 0;
  task_offsets[0] = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    const int64_t count = expert_offsets[expert + 1] - expert_offsets[expert];
    assert(count >= 0);
    const int64_t row_tiles = count == 0 ? 0 : ceil_div_positive(count, kTile);
    assert(row_tiles == 0 || output_tiles <= INT64_MAX / row_tiles);
    const int64_t tasks = row_tiles * output_tiles;
    assert(tasks <= task_upper_bound - prefix);
    prefix += tasks;
    task_offsets[expert + 1] = prefix;
  }
  assert(prefix <= task_upper_bound);
}

__device__ int64_t task_expert(
    int64_t task,
    const int64_t* __restrict__ task_offsets,
    int64_t experts) {
  int64_t lower = 0;
  int64_t upper = experts;
  while (lower < upper) {
    const int64_t middle = lower + (upper - lower) / 2;
    if (task_offsets[middle + 1] <= task) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  assert(lower >= 0 && lower < experts);
  assert(task_offsets[lower] <= task && task < task_offsets[lower + 1]);
  return lower;
}

__device__ __forceinline__ float sigmoid_stable(float value) {
  if (value >= 0.0F) {
    const float exponential = expf(-value);
    return 1.0F / (1.0F + exponential);
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

// Each task is one active [row, hidden] tile for one expert. W1 and W3 share
// the activation tile but retain independent weight tiles and accumulators.
__global__ void swiglu_hidden_grouped_tiled_float_kernel(
    const float* __restrict__ activations,
    const int64_t* __restrict__ expert_offsets,
    const int64_t* __restrict__ task_offsets,
    const float* __restrict__ w1,
    const float* __restrict__ w3,
    float* __restrict__ hidden_state,
    int64_t experts,
    int64_t model_dim,
    int64_t hidden_dim,
    int64_t hidden_tiles,
    int64_t task_upper_bound) {
  __shared__ float activation_tile[kTile][kTile + 1];
  __shared__ float w1_tile[kTile][kTile + 1];
  __shared__ float w3_tile[kTile][kTile + 1];

  const int tile_row = threadIdx.y;
  const int tile_column = threadIdx.x;
  const int64_t task_count = task_offsets[experts];
  assert(task_count >= 0 && task_count <= task_upper_bound);
  for (int64_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int64_t expert = task_expert(task, task_offsets, experts);
    const int64_t local_task = task - task_offsets[expert];
    const int64_t local_row_tile = local_task / hidden_tiles;
    const int64_t hidden_tile = local_task % hidden_tiles;
    const int64_t row = expert_offsets[expert] + local_row_tile * kTile + tile_row;
    const int64_t expert_end = expert_offsets[expert + 1];
    const int64_t hidden = hidden_tile * kTile + tile_column;

    float gate_accumulator = 0.0F;
    float up_accumulator = 0.0F;
    for (int64_t inner_base = 0; inner_base < model_dim; inner_base += kTile) {
      const int64_t activation_column = inner_base + tile_column;
      activation_tile[tile_row][tile_column] =
          row < expert_end && activation_column < model_dim
          ? activations[row * model_dim + activation_column]
          : 0.0F;

      // W1/W3 are [hidden, model]. Adjacent x lanes load adjacent model
      // columns, then transpose into the shared [K, N] tile.
      const int64_t weight_hidden = hidden_tile * kTile + tile_row;
      const int64_t weight_column = inner_base + tile_column;
      if (weight_hidden < hidden_dim && weight_column < model_dim) {
        const int64_t weight_offset =
            (expert * hidden_dim + weight_hidden) * model_dim + weight_column;
        w1_tile[tile_column][tile_row] = w1[weight_offset];
        w3_tile[tile_column][tile_row] = w3[weight_offset];
      } else {
        w1_tile[tile_column][tile_row] = 0.0F;
        w3_tile[tile_column][tile_row] = 0.0F;
      }
      __syncthreads();

#pragma unroll
      for (int inner = 0; inner < kTile; ++inner) {
        const float activation = activation_tile[tile_row][inner];
        gate_accumulator =
            fmaf(activation, w1_tile[inner][tile_column], gate_accumulator);
        up_accumulator =
            fmaf(activation, w3_tile[inner][tile_column], up_accumulator);
      }
      __syncthreads();
    }
    if (row < expert_end && hidden < hidden_dim) {
      hidden_state[row * hidden_dim + hidden] =
          gate_accumulator * sigmoid_stable(gate_accumulator) * up_accumulator;
    }
  }
}

// Each task is one active [row, model] output tile for one expert. The hidden
// state is materialized between kernels so the SwiGLU nonlinearity remains an
// explicit stage and the down projection can use the same tiled GEMM mapping.
__global__ void swiglu_down_grouped_tiled_float_kernel(
    const float* __restrict__ hidden_state,
    const int64_t* __restrict__ expert_offsets,
    const int64_t* __restrict__ task_offsets,
    const float* __restrict__ w2,
    float* __restrict__ output,
    int64_t experts,
    int64_t model_dim,
    int64_t hidden_dim,
    int64_t model_tiles,
    int64_t task_upper_bound) {
  __shared__ float hidden_tile[kTile][kTile + 1];
  __shared__ float weight_tile[kTile][kTile + 1];

  const int tile_row = threadIdx.y;
  const int tile_column = threadIdx.x;
  const int64_t task_count = task_offsets[experts];
  assert(task_count >= 0 && task_count <= task_upper_bound);
  for (int64_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int64_t expert = task_expert(task, task_offsets, experts);
    const int64_t local_task = task - task_offsets[expert];
    const int64_t local_row_tile = local_task / model_tiles;
    const int64_t model_tile = local_task % model_tiles;
    const int64_t row = expert_offsets[expert] + local_row_tile * kTile + tile_row;
    const int64_t expert_end = expert_offsets[expert + 1];
    const int64_t model = model_tile * kTile + tile_column;

    float accumulator = 0.0F;
    for (int64_t inner_base = 0; inner_base < hidden_dim; inner_base += kTile) {
      const int64_t hidden = inner_base + tile_column;
      hidden_tile[tile_row][tile_column] =
          row < expert_end && hidden < hidden_dim
          ? hidden_state[row * hidden_dim + hidden]
          : 0.0F;

      // W2 is [model, hidden]. Coalesced hidden-column loads are transposed
      // into the shared [K, N] tile used by the output threads.
      const int64_t weight_model = model_tile * kTile + tile_row;
      const int64_t weight_hidden = inner_base + tile_column;
      weight_tile[tile_column][tile_row] =
          weight_model < model_dim && weight_hidden < hidden_dim
          ? w2[(expert * model_dim + weight_model) * hidden_dim + weight_hidden]
          : 0.0F;
      __syncthreads();

#pragma unroll
      for (int inner = 0; inner < kTile; ++inner) {
        accumulator =
            fmaf(hidden_tile[tile_row][inner], weight_tile[inner][tile_column], accumulator);
      }
      __syncthreads();
    }
    if (row < expert_end && model < model_dim) {
      output[row * model_dim + model] = accumulator;
    }
  }
}

// One converged warp owns one 16x16 output tile. Multiplicands are FP16 while
// WMMA accumulators and the temporary output tiles are FP32. Global inputs are
// staged through aligned shared tiles so arbitrary tensor strides in the
// reduction dimension do not violate WMMA's alignment/leading-dimension rules.
__global__ void swiglu_hidden_grouped_wmma_half_kernel(
    const half* __restrict__ activations,
    const int64_t* __restrict__ expert_offsets,
    const int64_t* __restrict__ task_offsets,
    const half* __restrict__ w1,
    const half* __restrict__ w3,
    half* __restrict__ hidden_state,
    int64_t experts,
    int64_t model_dim,
    int64_t hidden_dim,
    int64_t hidden_tiles,
    int64_t task_upper_bound) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __shared__ __align__(32) half activation_tile[kTile * kTile];
  __shared__ __align__(32) half w1_tile[kTile * kTile];
  __shared__ __align__(32) half w3_tile[kTile * kTile];
  __shared__ __align__(32) float gate_output[kTile * kTile];
  __shared__ __align__(32) float up_output[kTile * kTile];

  const int lane = threadIdx.x;
  const int64_t task_count = task_offsets[experts];
  assert(blockDim.x == warpSize);
  assert(task_count >= 0 && task_count <= task_upper_bound);
  for (int64_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int64_t expert = task_expert(task, task_offsets, experts);
    const int64_t local_task = task - task_offsets[expert];
    const int64_t local_row_tile = local_task / hidden_tiles;
    const int64_t hidden_tile = local_task % hidden_tiles;
    const int64_t row_start = expert_offsets[expert] + local_row_tile * kTile;
    const int64_t expert_end = expert_offsets[expert + 1];

    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, kTile, kTile, kTile, float>
        gate_accumulator;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, kTile, kTile, kTile, float>
        up_accumulator;
    nvcuda::wmma::fill_fragment(gate_accumulator, 0.0F);
    nvcuda::wmma::fill_fragment(up_accumulator, 0.0F);

    for (int64_t inner_base = 0; inner_base < model_dim; inner_base += kTile) {
      for (int index = lane; index < kTile * kTile; index += warpSize) {
        const int tile_row = index / kTile;
        const int tile_column = index % kTile;
        const int64_t row = row_start + tile_row;
        const int64_t activation_column = inner_base + tile_column;
        activation_tile[index] =
            row < expert_end && activation_column < model_dim
            ? activations[row * model_dim + activation_column]
            : __float2half_rn(0.0F);

        const int64_t weight_hidden = hidden_tile * kTile + tile_row;
        const int64_t weight_column = inner_base + tile_column;
        const int transposed_index = tile_column * kTile + tile_row;
        if (weight_hidden < hidden_dim && weight_column < model_dim) {
          const int64_t weight_offset =
              (expert * hidden_dim + weight_hidden) * model_dim + weight_column;
          w1_tile[transposed_index] = w1[weight_offset];
          w3_tile[transposed_index] = w3[weight_offset];
        } else {
          w1_tile[transposed_index] = __float2half_rn(0.0F);
          w3_tile[transposed_index] = __float2half_rn(0.0F);
        }
      }
      __syncwarp();

      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          kTile,
          kTile,
          kTile,
          half,
          nvcuda::wmma::row_major>
          activation_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          kTile,
          kTile,
          kTile,
          half,
          nvcuda::wmma::row_major>
          w1_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          kTile,
          kTile,
          kTile,
          half,
          nvcuda::wmma::row_major>
          w3_fragment;
      nvcuda::wmma::load_matrix_sync(activation_fragment, activation_tile, kTile);
      nvcuda::wmma::load_matrix_sync(w1_fragment, w1_tile, kTile);
      nvcuda::wmma::load_matrix_sync(w3_fragment, w3_tile, kTile);
      nvcuda::wmma::mma_sync(
          gate_accumulator,
          activation_fragment,
          w1_fragment,
          gate_accumulator);
      nvcuda::wmma::mma_sync(
          up_accumulator,
          activation_fragment,
          w3_fragment,
          up_accumulator);
      __syncwarp();
    }

    nvcuda::wmma::store_matrix_sync(
        gate_output,
        gate_accumulator,
        kTile,
        nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(
        up_output,
        up_accumulator,
        kTile,
        nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < kTile * kTile; index += warpSize) {
      const int tile_row = index / kTile;
      const int tile_column = index % kTile;
      const int64_t row = row_start + tile_row;
      const int64_t hidden = hidden_tile * kTile + tile_column;
      if (row < expert_end && hidden < hidden_dim) {
        const float gate = gate_output[index];
        const float value = gate * sigmoid_stable(gate) * up_output[index];
        hidden_state[row * hidden_dim + hidden] = __float2half_rn(value);
      }
    }
    __syncwarp();
  }
#else
  assert(false && "FP16 grouped WMMA experts require compute capability 7.0+");
#endif
}

__global__ void swiglu_down_grouped_wmma_half_kernel(
    const half* __restrict__ hidden_state,
    const int64_t* __restrict__ expert_offsets,
    const int64_t* __restrict__ task_offsets,
    const half* __restrict__ w2,
    half* __restrict__ output,
    int64_t experts,
    int64_t model_dim,
    int64_t hidden_dim,
    int64_t model_tiles,
    int64_t task_upper_bound) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __shared__ __align__(32) half hidden_tile[kTile * kTile];
  __shared__ __align__(32) half weight_tile[kTile * kTile];
  __shared__ __align__(32) float output_tile[kTile * kTile];

  const int lane = threadIdx.x;
  const int64_t task_count = task_offsets[experts];
  assert(blockDim.x == warpSize);
  assert(task_count >= 0 && task_count <= task_upper_bound);
  for (int64_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int64_t expert = task_expert(task, task_offsets, experts);
    const int64_t local_task = task - task_offsets[expert];
    const int64_t local_row_tile = local_task / model_tiles;
    const int64_t model_tile = local_task % model_tiles;
    const int64_t row_start = expert_offsets[expert] + local_row_tile * kTile;
    const int64_t expert_end = expert_offsets[expert + 1];

    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, kTile, kTile, kTile, float>
        accumulator;
    nvcuda::wmma::fill_fragment(accumulator, 0.0F);
    for (int64_t inner_base = 0; inner_base < hidden_dim; inner_base += kTile) {
      for (int index = lane; index < kTile * kTile; index += warpSize) {
        const int tile_row = index / kTile;
        const int tile_column = index % kTile;
        const int64_t row = row_start + tile_row;
        const int64_t hidden = inner_base + tile_column;
        hidden_tile[index] =
            row < expert_end && hidden < hidden_dim
            ? hidden_state[row * hidden_dim + hidden]
            : __float2half_rn(0.0F);

        const int64_t weight_model = model_tile * kTile + tile_row;
        const int64_t weight_hidden = inner_base + tile_column;
        const int transposed_index = tile_column * kTile + tile_row;
        weight_tile[transposed_index] =
            weight_model < model_dim && weight_hidden < hidden_dim
            ? w2[(expert * model_dim + weight_model) * hidden_dim + weight_hidden]
            : __float2half_rn(0.0F);
      }
      __syncwarp();

      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          kTile,
          kTile,
          kTile,
          half,
          nvcuda::wmma::row_major>
          hidden_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          kTile,
          kTile,
          kTile,
          half,
          nvcuda::wmma::row_major>
          weight_fragment;
      nvcuda::wmma::load_matrix_sync(hidden_fragment, hidden_tile, kTile);
      nvcuda::wmma::load_matrix_sync(weight_fragment, weight_tile, kTile);
      nvcuda::wmma::mma_sync(
          accumulator,
          hidden_fragment,
          weight_fragment,
          accumulator);
      __syncwarp();
    }

    nvcuda::wmma::store_matrix_sync(
        output_tile,
        accumulator,
        kTile,
        nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < kTile * kTile; index += warpSize) {
      const int tile_row = index / kTile;
      const int tile_column = index % kTile;
      const int64_t row = row_start + tile_row;
      const int64_t model = model_tile * kTile + tile_column;
      if (row < expert_end && model < model_dim) {
        output[row * model_dim + model] = __float2half_rn(output_tile[index]);
      }
    }
    __syncwarp();
  }
#else
  assert(false && "FP16 grouped WMMA experts require compute capability 7.0+");
#endif
}

void check_cuda_expert_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf,
      name,
      " must use float16 or float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t launch_blocks(
    int64_t task_upper_bound,
    const cudaDeviceProp* properties) {
  constexpr int64_t kBlocksPerMultiprocessor = 8;
  const int64_t occupancy_target =
      static_cast<int64_t>(properties->multiProcessorCount) * kBlocksPerMultiprocessor;
  const int64_t bounded_target = occupancy_target < properties->maxGridSize[0]
      ? occupancy_target
      : properties->maxGridSize[0];
  return task_upper_bound < bounded_target ? task_upper_bound : bounded_target;
}

at::Tensor swiglu_experts_cuda(
    const at::Tensor& activations,
    const at::Tensor& expert_offsets,
    const at::Tensor& expert_w1,
    const at::Tensor& expert_w2,
    const at::Tensor& expert_w3) {
  check_cuda_expert_tensor(activations, "activations");
  check_cuda_expert_tensor(expert_w1, "expert_w1");
  check_cuda_expert_tensor(expert_w2, "expert_w2");
  check_cuda_expert_tensor(expert_w3, "expert_w3");
  TORCH_CHECK(expert_offsets.is_cuda(), "expert_offsets must be a CUDA tensor");
  TORCH_CHECK(expert_offsets.scalar_type() == at::kLong,
              "expert_offsets must use int64");
  TORCH_CHECK(expert_offsets.is_contiguous(), "expert_offsets must be contiguous");
  TORCH_CHECK(
      activations.device() == expert_offsets.device() &&
          activations.device() == expert_w1.device() &&
          activations.device() == expert_w2.device() &&
          activations.device() == expert_w3.device(),
      "activations, offsets, and weights must be on the same CUDA device");
  TORCH_CHECK(
      activations.scalar_type() == expert_w1.scalar_type() &&
          activations.scalar_type() == expert_w2.scalar_type() &&
          activations.scalar_type() == expert_w3.scalar_type(),
      "activations and expert weights must share a dtype");
  TORCH_CHECK(activations.dim() == 2,
              "activations must have shape [rows, model_dim]");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a vector");
  TORCH_CHECK(expert_w1.dim() == 3 && expert_w2.dim() == 3 && expert_w3.dim() == 3,
              "expert weights must be rank-3 tensors");

  const int64_t experts = expert_w1.size(0);
  const int64_t hidden_dim = expert_w1.size(1);
  const int64_t model_dim = expert_w1.size(2);
  const int64_t rows = activations.size(0);
  TORCH_CHECK(model_dim > 0 && hidden_dim > 0,
              "model_dim and hidden_dim must be positive");
  TORCH_CHECK(activations.size(1) == model_dim,
              "activation model dimension does not match expert weights");
  TORCH_CHECK(expert_w3.sizes() == expert_w1.sizes(),
              "expert_w3 shape must match expert_w1");
  TORCH_CHECK(
      expert_w2.size(0) == experts && expert_w2.size(1) == model_dim &&
          expert_w2.size(2) == hidden_dim,
      "expert_w2 must have shape [experts, model_dim, hidden_dim]");
  TORCH_CHECK(expert_offsets.numel() == experts + 1,
              "expert_offsets must have shape [experts + 1]");
  TORCH_CHECK(experts > 0 || rows == 0,
              "non-empty activations require at least one expert");

  const int64_t hidden_tiles = ceil_div_positive(hidden_dim, kTile);
  const int64_t model_tiles = ceil_div_positive(model_dim, kTile);
  const int64_t row_tile_upper_bound =
      grouped_row_tile_upper_bound(rows, experts);
  TORCH_CHECK(
      row_tile_upper_bound == 0 ||
          hidden_tiles <= INT64_MAX / row_tile_upper_bound,
              "grouped hidden tile count overflows int64");
  TORCH_CHECK(
      row_tile_upper_bound == 0 ||
          model_tiles <= INT64_MAX / row_tile_upper_bound,
              "grouped down tile count overflows int64");
  const int64_t hidden_task_upper_bound = row_tile_upper_bound * hidden_tiles;
  const int64_t down_task_upper_bound = row_tile_upper_bound * model_tiles;

  const c10::cuda::CUDAGuard device_guard(activations.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(activations.get_device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(activations.get_device());
  if (activations.scalar_type() == at::kHalf) {
    TORCH_CHECK(
        properties->major >= 7,
        "FP16 grouped WMMA experts require compute capability 7.0 or newer");
  }
  auto output = at::empty({rows, model_dim}, activations.options());

  const int64_t offset_items = experts + 1;
  const int64_t offset_blocks_unbounded =
      ceil_div_positive(offset_items, kOffsetThreads);
  const int64_t offset_blocks = offset_blocks_unbounded < properties->maxGridSize[0]
      ? offset_blocks_unbounded
      : properties->maxGridSize[0];
  validate_offsets_kernel<<<
      static_cast<unsigned int>(offset_blocks), kOffsetThreads, 0, stream>>>(
      expert_offsets.const_data_ptr<int64_t>(), experts, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (rows == 0) {
    return output;
  }

  auto hidden_task_offsets = at::empty({experts + 1}, expert_offsets.options());
  auto down_task_offsets = at::empty({experts + 1}, expert_offsets.options());
  build_grouped_tile_offsets_kernel<<<1, 1, 0, stream>>>(
      expert_offsets.const_data_ptr<int64_t>(),
      hidden_task_offsets.mutable_data_ptr<int64_t>(),
      experts,
      hidden_tiles,
      hidden_task_upper_bound);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  build_grouped_tile_offsets_kernel<<<1, 1, 0, stream>>>(
      expert_offsets.const_data_ptr<int64_t>(),
      down_task_offsets.mutable_data_ptr<int64_t>(),
      experts,
      model_tiles,
      down_task_upper_bound);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto hidden_state = at::empty({rows, hidden_dim}, activations.options());
  const unsigned int hidden_blocks =
      static_cast<unsigned int>(launch_blocks(hidden_task_upper_bound, properties));
  const unsigned int down_blocks =
      static_cast<unsigned int>(launch_blocks(down_task_upper_bound, properties));
  if (activations.scalar_type() == at::kFloat) {
    const dim3 threads(kTile, kTile);
    swiglu_hidden_grouped_tiled_float_kernel<<<hidden_blocks, threads, 0, stream>>>(
        activations.const_data_ptr<float>(),
        expert_offsets.const_data_ptr<int64_t>(),
        hidden_task_offsets.const_data_ptr<int64_t>(),
        expert_w1.const_data_ptr<float>(),
        expert_w3.const_data_ptr<float>(),
        hidden_state.mutable_data_ptr<float>(),
        experts,
        model_dim,
        hidden_dim,
        hidden_tiles,
        hidden_task_upper_bound);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    swiglu_down_grouped_tiled_float_kernel<<<down_blocks, threads, 0, stream>>>(
        hidden_state.const_data_ptr<float>(),
        expert_offsets.const_data_ptr<int64_t>(),
        down_task_offsets.const_data_ptr<int64_t>(),
        expert_w2.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        experts,
        model_dim,
        hidden_dim,
        model_tiles,
        down_task_upper_bound);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const auto* activation_pointer = reinterpret_cast<const half*>(
        activations.const_data_ptr<at::Half>());
    const auto* w1_pointer = reinterpret_cast<const half*>(
        expert_w1.const_data_ptr<at::Half>());
    const auto* w2_pointer = reinterpret_cast<const half*>(
        expert_w2.const_data_ptr<at::Half>());
    const auto* w3_pointer = reinterpret_cast<const half*>(
        expert_w3.const_data_ptr<at::Half>());
    auto* hidden_pointer = reinterpret_cast<half*>(
        hidden_state.mutable_data_ptr<at::Half>());
    auto* output_pointer = reinterpret_cast<half*>(
        output.mutable_data_ptr<at::Half>());
    swiglu_hidden_grouped_wmma_half_kernel<<<hidden_blocks, kWarpThreads, 0, stream>>>(
        activation_pointer,
        expert_offsets.const_data_ptr<int64_t>(),
        hidden_task_offsets.const_data_ptr<int64_t>(),
        w1_pointer,
        w3_pointer,
        hidden_pointer,
        experts,
        model_dim,
        hidden_dim,
        hidden_tiles,
        hidden_task_upper_bound);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    swiglu_down_grouped_wmma_half_kernel<<<down_blocks, kWarpThreads, 0, stream>>>(
        hidden_pointer,
        expert_offsets.const_data_ptr<int64_t>(),
        down_task_offsets.const_data_ptr<int64_t>(),
        w2_pointer,
        output_pointer,
        experts,
        model_dim,
        hidden_dim,
        model_tiles,
        down_task_upper_bound);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("swiglu_experts", TORCH_FN(swiglu_experts_cuda));
}
