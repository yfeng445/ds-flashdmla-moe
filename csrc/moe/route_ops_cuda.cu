#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <climits>
#include <cassert>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kThreads = 256;

__global__ void count_route_keys_kernel(
    const int64_t* __restrict__ expert_indices,
    const int64_t* __restrict__ expert_owner,
    int64_t* __restrict__ key_counts,
    int64_t route_count,
    int64_t experts,
    int64_t world_size) {
  for (int64_t route = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       route < route_count;
       route += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t expert = expert_indices[route];
    assert(expert >= 0 && expert < experts);
    const int64_t destination = expert_owner[expert];
    assert(destination >= 0 && destination < world_size);
    const int64_t key = destination * experts + expert;
    atomicAdd(
        reinterpret_cast<unsigned long long*>(key_counts + key),
        static_cast<unsigned long long>(1));
  }
}

__global__ void exclusive_scan_kernel(
    const int64_t* __restrict__ counts,
    int64_t* __restrict__ offsets,
    int64_t count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t prefix = 0;
    offsets[0] = 0;
    for (int64_t index = 0; index < count; ++index) {
      prefix += counts[index];
      offsets[index + 1] = prefix;
    }
  }
}

__global__ void summarize_counts_kernel(
    const int64_t* __restrict__ key_counts,
    const int64_t* __restrict__ expert_owner,
    int64_t* __restrict__ counts_per_expert,
    int64_t* __restrict__ rank_counts,
    int64_t experts,
    int64_t world_size) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < experts + world_size;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    if (index < experts) {
      const int64_t destination = expert_owner[index];
      assert(destination >= 0 && destination < world_size);
      counts_per_expert[index] = key_counts[destination * experts + index];
      continue;
    }

    const int64_t rank = index - experts;
    int64_t total = 0;
    const int64_t base = rank * experts;
    for (int64_t expert = 0; expert < experts; ++expert) {
      total += key_counts[base + expert];
    }
    rank_counts[rank] = total;
  }
}

__global__ void pack_routes_float_kernel(
    const float* __restrict__ x,
    const float* __restrict__ route_weights,
    const int64_t* __restrict__ expert_indices,
    const int64_t* __restrict__ expert_owner,
    const int64_t* __restrict__ key_offsets,
    int64_t* __restrict__ key_cursors,
    float* __restrict__ packed_activations,
    float* __restrict__ packed_weights,
    int64_t* __restrict__ packed_route_indices,
    int64_t* __restrict__ packed_expert_indices,
    int64_t route_count,
    int64_t topk,
    int64_t model_dim,
    int64_t experts,
    int64_t world_size) {
  const int64_t route = static_cast<int64_t>(blockIdx.x);
  if (route >= route_count) {
    return;
  }

  const int64_t expert = expert_indices[route];
  assert(expert >= 0 && expert < experts);
  const int64_t destination = expert_owner[expert];
  assert(destination >= 0 && destination < world_size);
  const int64_t key = destination * experts + expert;
  __shared__ int64_t packed_row;
  if (threadIdx.x == 0) {
    const auto offset = atomicAdd(
        reinterpret_cast<unsigned long long*>(key_cursors + key),
        static_cast<unsigned long long>(1));
    packed_row = key_offsets[key] + static_cast<int64_t>(offset);
    packed_weights[packed_row] = route_weights[route];
    packed_route_indices[packed_row] = route;
    packed_expert_indices[packed_row] = expert;
  }
  __syncthreads();

  const int64_t token = route / topk;
  for (int64_t column = threadIdx.x; column < model_dim; column += blockDim.x) {
    packed_activations[packed_row * model_dim + column] = x[token * model_dim + column];
  }
}

__global__ void combine_routes_float_kernel(
    const float* __restrict__ contributions,
    const float* __restrict__ route_weights,
    const int64_t* __restrict__ token_indices,
    float* __restrict__ output,
    int64_t rows,
    int64_t model_dim,
    int64_t token_count) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t token = token_indices[row];
  assert(token >= 0 && token < token_count);
  const float weight = route_weights[row];
  for (int64_t column = threadIdx.x; column < model_dim; column += blockDim.x) {
    atomicAdd(
        output + token * model_dim + column,
        contributions[row * model_dim + column] * weight);
  }
}

void check_cuda_float_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must use float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_long_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must use int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
route_pack_cuda(
    const at::Tensor& x,
    const at::Tensor& route_weights,
    const at::Tensor& expert_indices,
    const at::Tensor& expert_owner,
    int64_t world_size) {
  TORCH_CHECK(
      !at::globalContext().deterministicAlgorithms(),
      "native route pack uses atomic row assignment; disable deterministic algorithms or use "
      "the reference pack");
  check_cuda_float_tensor(x, "x");
  check_cuda_float_tensor(route_weights, "route_weights");
  check_cuda_long_tensor(expert_indices, "expert_indices");
  check_cuda_long_tensor(expert_owner, "expert_owner");
  TORCH_CHECK(
      x.device() == route_weights.device() && x.device() == expert_indices.device() &&
          x.device() == expert_owner.device(),
      "all route-pack tensors must be on the same CUDA device");
  TORCH_CHECK(x.dim() == 2, "x must have shape [tokens, model_dim]");
  TORCH_CHECK(
      route_weights.dim() == 2 && expert_indices.dim() == 2,
      "route weights and expert indices must have shape [tokens, topk]");
  TORCH_CHECK(
      route_weights.sizes() == expert_indices.sizes(),
      "route weights and expert indices must have identical shapes");
  TORCH_CHECK(
      x.size(0) == route_weights.size(0),
      "x and routing tensors must have the same token count");
  TORCH_CHECK(x.size(1) > 0, "model_dim must be positive");
  TORCH_CHECK(route_weights.size(1) > 0, "topk must be positive");
  TORCH_CHECK(expert_owner.dim() == 1 && expert_owner.numel() > 0,
              "expert_owner must be a non-empty vector");
  TORCH_CHECK(world_size > 0, "world_size must be positive");

  const int64_t experts = expert_owner.numel();
  const int64_t tokens = x.size(0);
  const int64_t topk = route_weights.size(1);
  const int64_t model_dim = x.size(1);
  TORCH_CHECK(tokens <= INT64_MAX / topk, "tokens * topk overflows int64");
  const int64_t route_count = tokens * topk;
  TORCH_CHECK(
      experts <= INT64_MAX / world_size,
      "world_size * experts overflows int64");
  const int64_t key_count = world_size * experts;

  const c10::cuda::CUDAGuard device_guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(
      route_count <= static_cast<int64_t>(properties->maxGridSize[0]),
      "too many routes for the one-block-per-route CUDA pack kernel: ",
      route_count);

  auto long_options = expert_indices.options().dtype(at::kLong);
  auto key_counts = at::zeros({key_count}, long_options);
  auto key_offsets = at::empty({key_count + 1}, long_options);
  auto key_cursors = at::zeros({key_count}, long_options);
  auto counts_per_expert = at::empty({experts}, long_options);
  auto rank_counts = at::empty({world_size}, long_options);
  auto packed_activations = at::empty({route_count, model_dim}, x.options());
  auto packed_weights = at::empty({route_count}, route_weights.options());
  auto packed_route_indices = at::empty({route_count}, long_options);
  auto packed_expert_indices = at::empty({route_count}, long_options);

  if (route_count > 0) {
    const int64_t count_blocks =
        (route_count + kThreads - 1) / kThreads < properties->maxGridSize[0]
        ? (route_count + kThreads - 1) / kThreads
        : properties->maxGridSize[0];
    count_route_keys_kernel<<<
        static_cast<unsigned int>(count_blocks), kThreads, 0, stream>>>(
        expert_indices.const_data_ptr<int64_t>(),
        expert_owner.const_data_ptr<int64_t>(),
        key_counts.mutable_data_ptr<int64_t>(),
        route_count,
        experts,
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  exclusive_scan_kernel<<<1, 1, 0, stream>>>(
      key_counts.const_data_ptr<int64_t>(),
      key_offsets.mutable_data_ptr<int64_t>(),
      key_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  TORCH_CHECK(experts <= INT64_MAX - world_size, "experts + world_size overflows int64");
  const int64_t summary_items = experts + world_size;
  const int64_t summary_blocks = (summary_items + kThreads - 1) / kThreads;
  summarize_counts_kernel<<<
      static_cast<unsigned int>(summary_blocks), kThreads, 0, stream>>>(
      key_counts.const_data_ptr<int64_t>(),
      expert_owner.const_data_ptr<int64_t>(),
      counts_per_expert.mutable_data_ptr<int64_t>(),
      rank_counts.mutable_data_ptr<int64_t>(),
      experts,
      world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (route_count > 0) {
    pack_routes_float_kernel<<<
        static_cast<unsigned int>(route_count), kThreads, 0, stream>>>(
        x.const_data_ptr<float>(),
        route_weights.const_data_ptr<float>(),
        expert_indices.const_data_ptr<int64_t>(),
        expert_owner.const_data_ptr<int64_t>(),
        key_offsets.const_data_ptr<int64_t>(),
        key_cursors.mutable_data_ptr<int64_t>(),
        packed_activations.mutable_data_ptr<float>(),
        packed_weights.mutable_data_ptr<float>(),
        packed_route_indices.mutable_data_ptr<int64_t>(),
        packed_expert_indices.mutable_data_ptr<int64_t>(),
        route_count,
        topk,
        model_dim,
        experts,
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {
      packed_activations,
      packed_weights,
      packed_route_indices,
      packed_expert_indices,
      counts_per_expert,
      rank_counts};
}

at::Tensor route_combine_cuda(
    const at::Tensor& contributions,
    const at::Tensor& route_weights,
    const at::Tensor& token_indices,
    int64_t token_count) {
  TORCH_CHECK(
      !at::globalContext().deterministicAlgorithms(),
      "native route combine uses atomic accumulation; disable deterministic algorithms or use "
      "the reference combine");
  check_cuda_float_tensor(contributions, "contributions");
  check_cuda_float_tensor(route_weights, "route_weights");
  check_cuda_long_tensor(token_indices, "token_indices");
  TORCH_CHECK(
      contributions.device() == route_weights.device() &&
          contributions.device() == token_indices.device(),
      "all route-combine tensors must be on the same CUDA device");
  TORCH_CHECK(contributions.dim() == 2, "contributions must have shape [rows, model_dim]");
  TORCH_CHECK(route_weights.dim() == 1 && token_indices.dim() == 1,
              "route_weights and token_indices must be vectors");
  TORCH_CHECK(
      contributions.size(0) == route_weights.numel() &&
          contributions.size(0) == token_indices.numel(),
      "route-combine row counts must match");
  TORCH_CHECK(contributions.size(1) > 0, "model_dim must be positive");
  TORCH_CHECK(token_count >= 0, "token_count must be non-negative");
  const int64_t rows = contributions.size(0);
  const int64_t model_dim = contributions.size(1);
  auto output = at::zeros({token_count, model_dim}, contributions.options());
  if (rows == 0 || token_count == 0) {
    return output;
  }

  const c10::cuda::CUDAGuard device_guard(contributions.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(contributions.get_device());
  TORCH_CHECK(
      rows <= static_cast<int64_t>(properties->maxGridSize[0]),
      "too many rows for the one-block-per-route CUDA combine kernel: ",
      rows);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(contributions.get_device());
  combine_routes_float_kernel<<<
      static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
      contributions.const_data_ptr<float>(),
      route_weights.const_data_ptr<float>(),
      token_indices.const_data_ptr<int64_t>(),
      output.mutable_data_ptr<float>(),
      rows,
      model_dim,
      token_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("route_pack", TORCH_FN(route_pack_cuda));
  m.impl("route_combine", TORCH_FN(route_combine_cuda));
}
