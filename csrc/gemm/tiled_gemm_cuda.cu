#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <optional>

namespace {

constexpr int kTile = 16;

__global__ void tiled_gemm_float_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ output,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    float beta) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];

  const int64_t row = static_cast<int64_t>(blockIdx.y) * kTile + threadIdx.y;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * kTile + threadIdx.x;
  float accumulator = 0.0F;

  for (int64_t reduction_start = 0; reduction_start < k; reduction_start += kTile) {
    const int64_t a_column = reduction_start + threadIdx.x;
    const int64_t b_row = reduction_start + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        row < m && a_column < k ? a[row * k + a_column] : 0.0F;
    b_tile[threadIdx.y][threadIdx.x] =
        b_row < k && column < n ? b[b_row * n + column] : 0.0F;
    __syncthreads();

#pragma unroll
    for (int reduction = 0; reduction < kTile; ++reduction) {
      accumulator += a_tile[threadIdx.y][reduction] * b_tile[reduction][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && column < n) {
    const int64_t output_index = row * n + column;
    const float epilogue = c == nullptr ? 0.0F : beta * c[output_index];
    output[output_index] = alpha * accumulator + epilogue;
  }
}

void check_cuda_float_matrix(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must use float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a rank-2 matrix");
}

at::Tensor tiled_gemm_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const std::optional<at::Tensor>& c,
    double alpha,
    double beta) {
  check_cuda_float_matrix(a, "a");
  check_cuda_float_matrix(b, "b");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(a.size(1) == b.size(0), "GEMM inner dimensions must match");
  TORCH_CHECK(std::isfinite(alpha) && std::isfinite(beta), "alpha and beta must be finite");

  if (c.has_value()) {
    check_cuda_float_matrix(*c, "c");
    TORCH_CHECK(c->device() == a.device(), "c must be on the same CUDA device as a and b");
    TORCH_CHECK(
        c->size(0) == a.size(0) && c->size(1) == b.size(1),
        "GEMM epilogue matrix c must have shape [m, n]");
  } else {
    TORCH_CHECK(beta == 0.0, "a nonzero beta requires an epilogue matrix c");
  }

  const int64_t m = a.size(0);
  const int64_t n = b.size(1);
  const int64_t k = a.size(1);
  const c10::cuda::CUDAGuard device_guard(a.device());
  auto output = at::empty({m, n}, a.options());
  if (m == 0 || n == 0) {
    return output;
  }

  const int64_t grid_x = n / kTile + (n % kTile != 0);
  const int64_t grid_y = m / kTile + (m % kTile != 0);
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(
      grid_x <= static_cast<int64_t>(properties->maxGridSize[0]) &&
          grid_y <= static_cast<int64_t>(properties->maxGridSize[1]),
      "GEMM output shape requires grid (",
      grid_x,
      ", ",
      grid_y,
      "), exceeding the device limit");

  const dim3 threads(kTile, kTile);
  const dim3 blocks(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());
  const float* c_pointer = c.has_value() ? c->const_data_ptr<float>() : nullptr;
  tiled_gemm_float_kernel<<<blocks, threads, 0, stream>>>(
      a.const_data_ptr<float>(),
      b.const_data_ptr<float>(),
      c_pointer,
      output.mutable_data_ptr<float>(),
      m,
      n,
      k,
      static_cast<float>(alpha),
      static_cast<float>(beta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("tiled_gemm", TORCH_FN(tiled_gemm_cuda));
}
