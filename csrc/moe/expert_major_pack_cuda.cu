#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Half.h>
#include <torch/library.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <climits>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kThreads = 256;

__device__ int64_t find_local_expert(
    int64_t global_expert,
    const int64_t* __restrict__ local_expert_ids,
    int64_t local_experts) {
  int64_t local_index = -1;
  for (int64_t index = 0; index < local_experts; ++index) {
    if (local_expert_ids[index] == global_expert) {
      assert(local_index == -1);
      local_index = index;
    }
  }
  assert(local_index >= 0);
  return local_index;
}

__global__ void count_local_experts_kernel(
    const int64_t* __restrict__ expert_indices,
    const int64_t* __restrict__ local_expert_ids,
    int64_t* __restrict__ counts,
    int64_t rows,
    int64_t local_experts) {
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows;
       row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t local_index =
        find_local_expert(expert_indices[row], local_expert_ids, local_experts);
    if (local_index < 0) {
      assert(false);
      continue;
    }
    atomicAdd(
        reinterpret_cast<unsigned long long*>(counts + local_index),
        static_cast<unsigned long long>(1));
  }
}

__global__ void exclusive_scan_kernel(
    const int64_t* __restrict__ counts,
    int64_t* __restrict__ offsets,
    int64_t local_experts) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t prefix = 0;
    offsets[0] = 0;
    for (int64_t expert = 0; expert < local_experts; ++expert) {
      prefix += counts[expert];
      offsets[expert + 1] = prefix;
    }
  }
}

template <typename scalar_t>
__global__ void pack_expert_major_kernel(
    const scalar_t* __restrict__ activations,
    const int64_t* __restrict__ expert_indices,
    const int64_t* __restrict__ local_expert_ids,
    const int64_t* __restrict__ offsets,
    int64_t* __restrict__ cursors,
    scalar_t* __restrict__ packed_activations,
    int64_t* __restrict__ inverse_permutation,
    int64_t rows,
    int64_t model_dim,
    int64_t local_experts) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t local_index =
      find_local_expert(expert_indices[row], local_expert_ids, local_experts);
  if (local_index < 0) {
    assert(false);
    return;
  }
  __shared__ int64_t packed_row;
  if (threadIdx.x == 0) {
    const auto relative = atomicAdd(
        reinterpret_cast<unsigned long long*>(cursors + local_index),
        static_cast<unsigned long long>(1));
    packed_row = offsets[local_index] + static_cast<int64_t>(relative);
    inverse_permutation[row] = packed_row;
  }
  __syncthreads();
  for (int64_t column = threadIdx.x; column < model_dim; column += blockDim.x) {
    packed_activations[packed_row * model_dim + column] =
        activations[row * model_dim + column];
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> expert_major_pack_cuda(
    const at::Tensor& activations,
    const at::Tensor& expert_indices,
    const at::Tensor& local_expert_ids) {
  TORCH_CHECK(
      !at::globalContext().deterministicAlgorithms(),
      "native expert-major pack uses atomic row assignment; disable deterministic algorithms "
      "or use the reference pack");
  TORCH_CHECK(activations.is_cuda(), "activations must be a CUDA tensor");
  TORCH_CHECK(
      activations.scalar_type() == at::kFloat ||
          activations.scalar_type() == at::kHalf,
      "activations must use float16 or float32");
  TORCH_CHECK(activations.is_contiguous(), "activations must be contiguous");
  TORCH_CHECK(expert_indices.is_cuda() && local_expert_ids.is_cuda(),
              "expert indices must be CUDA tensors");
  TORCH_CHECK(
      expert_indices.scalar_type() == at::kLong &&
          local_expert_ids.scalar_type() == at::kLong,
      "expert indices must use int64");
  TORCH_CHECK(expert_indices.is_contiguous() && local_expert_ids.is_contiguous(),
              "expert indices must be contiguous");
  TORCH_CHECK(
      activations.device() == expert_indices.device() &&
          activations.device() == local_expert_ids.device(),
      "activations and expert indices must be on the same CUDA device");
  TORCH_CHECK(activations.dim() == 2,
              "activations must have shape [rows, model_dim]");
  TORCH_CHECK(expert_indices.dim() == 1 && local_expert_ids.dim() == 1,
              "expert_indices and local_expert_ids must be vectors");
  TORCH_CHECK(activations.size(0) == expert_indices.numel(),
              "expert_indices must contain one id per activation row");
  TORCH_CHECK(activations.size(1) > 0, "model_dim must be positive");

  const int64_t rows = activations.size(0);
  const int64_t model_dim = activations.size(1);
  const int64_t local_experts = local_expert_ids.numel();
  TORCH_CHECK(local_experts > 0 || rows == 0,
              "non-empty activations require at least one local expert");

  const c10::cuda::CUDAGuard device_guard(activations.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(activations.get_device());
  auto long_options = expert_indices.options().dtype(at::kLong);
  auto counts = at::zeros({local_experts}, long_options);
  auto offsets = at::empty({local_experts + 1}, long_options);
  auto cursors = at::zeros({local_experts}, long_options);
  auto packed_activations = at::empty_like(activations);
  auto inverse_permutation = at::empty({rows}, long_options);

  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(activations.get_device());
  if (rows > 0) {
    const int64_t count_blocks_unbounded = (rows + kThreads - 1) / kThreads;
    const int64_t count_blocks = count_blocks_unbounded < properties->maxGridSize[0]
        ? count_blocks_unbounded
        : properties->maxGridSize[0];
    count_local_experts_kernel<<<
        static_cast<unsigned int>(count_blocks), kThreads, 0, stream>>>(
        expert_indices.const_data_ptr<int64_t>(),
        local_expert_ids.const_data_ptr<int64_t>(),
        counts.mutable_data_ptr<int64_t>(),
        rows,
        local_experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  exclusive_scan_kernel<<<1, 1, 0, stream>>>(
      counts.const_data_ptr<int64_t>(),
      offsets.mutable_data_ptr<int64_t>(),
      local_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (rows > 0) {
    TORCH_CHECK(
        rows <= static_cast<int64_t>(properties->maxGridSize[0]),
        "too many rows for the one-block-per-row expert-major pack kernel: ",
        rows);
    if (activations.scalar_type() == at::kFloat) {
      pack_expert_major_kernel<<<
          static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
          activations.const_data_ptr<float>(),
          expert_indices.const_data_ptr<int64_t>(),
          local_expert_ids.const_data_ptr<int64_t>(),
          offsets.const_data_ptr<int64_t>(),
          cursors.mutable_data_ptr<int64_t>(),
          packed_activations.mutable_data_ptr<float>(),
          inverse_permutation.mutable_data_ptr<int64_t>(),
          rows,
          model_dim,
          local_experts);
    } else {
      const auto* activation_pointer = reinterpret_cast<const half*>(
          activations.const_data_ptr<at::Half>());
      auto* packed_pointer = reinterpret_cast<half*>(
          packed_activations.mutable_data_ptr<at::Half>());
      pack_expert_major_kernel<<<
          static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
          activation_pointer,
          expert_indices.const_data_ptr<int64_t>(),
          local_expert_ids.const_data_ptr<int64_t>(),
          offsets.const_data_ptr<int64_t>(),
          cursors.mutable_data_ptr<int64_t>(),
          packed_pointer,
          inverse_permutation.mutable_data_ptr<int64_t>(),
          rows,
          model_dim,
          local_experts);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed_activations, offsets, inverse_permutation};
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("expert_major_pack", TORCH_FN(expert_major_pack_cuda));
}
