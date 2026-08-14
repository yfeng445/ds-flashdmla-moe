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
#include <unordered_set>

namespace {

constexpr int kThreads = 128;

template <typename scalar_t>
__global__ void mla_paged_absorbed_attention_kernel(
    const scalar_t* __restrict__ q_nope,
    const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ kv_storage,
    const scalar_t* __restrict__ pe_storage,
    const int64_t* __restrict__ position_storage,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ sequence_lengths,
    const scalar_t* __restrict__ key_up,
    const scalar_t* __restrict__ value_up,
    const int64_t* __restrict__ query_positions,
    scalar_t* __restrict__ output,
    int64_t num_pages,
    int64_t page_size,
    int64_t heads,
    int64_t query_length,
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
    int64_t kv_stride_page,
    int64_t kv_stride_token,
    int64_t kv_stride_dim,
    int64_t pe_stride_page,
    int64_t pe_stride_token,
    int64_t pe_stride_dim,
    int64_t position_stride_page,
    int64_t position_stride_token,
    int64_t block_table_stride_batch,
    int64_t block_table_stride_page,
    int64_t sequence_length_stride,
    int64_t key_up_stride_head,
    int64_t key_up_stride_nope,
    int64_t key_up_stride_latent,
    int64_t value_up_stride_head,
    int64_t value_up_stride_value,
    int64_t value_up_stride_latent,
    int64_t query_position_stride_batch,
    int64_t query_position_stride_query,
    float scale,
    bool causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t query_index = row % query_length;
  const int64_t batch_head = row / query_length;
  const int64_t head = batch_head % heads;
  const int64_t batch = batch_head / heads;
  const int64_t key_length = sequence_lengths[batch * sequence_length_stride];

  const int64_t q_nope_offset = batch * q_nope_stride_batch +
      query_index * q_nope_stride_query + head * q_nope_stride_head;
  const int64_t q_pe_offset = batch * q_pe_stride_batch +
      query_index * q_pe_stride_query + head * q_pe_stride_head;
  const int64_t key_up_offset = head * key_up_stride_head;
  const int64_t value_up_offset = head * value_up_stride_head;
  const int64_t output_offset =
      ((batch * query_length + query_index) * heads + head) * value_dim;
  const int64_t query_position =
      query_positions[batch * query_position_stride_batch +
                      query_index * query_position_stride_query];

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
    const int64_t logical_page = key_index / page_size;
    const int64_t page_offset = key_index - logical_page * page_size;
    const int64_t physical_page =
        block_table[batch * block_table_stride_batch +
                    logical_page * block_table_stride_page];
    if (physical_page < 0 || physical_page >= num_pages) {
      continue;
    }
    const int64_t key_position =
        position_storage[physical_page * position_stride_page +
                         page_offset * position_stride_token];
    if (causal && key_position > query_position) {
      continue;
    }
    const int64_t kv_offset =
        physical_page * kv_stride_page + page_offset * kv_stride_token;
    const int64_t pe_offset =
        physical_page * pe_stride_page + page_offset * pe_stride_token;

    float partial = 0.0F;
    for (int64_t latent = threadIdx.x; latent < latent_dim; latent += blockDim.x) {
      partial = fmaf(
          q_latent[latent],
          static_cast<float>(kv_storage[kv_offset + latent * kv_stride_dim]),
          partial);
    }
    for (int64_t column = threadIdx.x; column < rope_dim; column += blockDim.x) {
      partial = fmaf(
          static_cast<float>(q_pe[q_pe_offset + column * q_pe_stride_dim]),
          static_cast<float>(pe_storage[pe_offset + column * pe_stride_dim]),
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
      numerator[latent] = numerator[latent] * previous_scale +
          current_scale *
              static_cast<float>(kv_storage[kv_offset + latent * kv_stride_dim]);
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

void check_cuda_float_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      name,
      " must use float16, bfloat16, or float32");
}

void check_cuda_long_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must use int64");
}

at::Tensor mla_paged_absorbed_attention_cuda(
    const at::Tensor& q_nope,
    const at::Tensor& q_pe,
    const at::Tensor& kv_storage,
    const at::Tensor& pe_storage,
    const at::Tensor& position_storage,
    const at::Tensor& block_table,
    const at::Tensor& sequence_lengths,
    const at::Tensor& key_up,
    const at::Tensor& value_up,
    const at::Tensor& query_positions,
    bool metadata_validated,
    bool causal,
    double scale) {
  check_cuda_float_tensor(q_nope, "q_nope");
  check_cuda_float_tensor(q_pe, "q_pe");
  check_cuda_float_tensor(kv_storage, "kv_storage");
  check_cuda_float_tensor(pe_storage, "pe_storage");
  check_cuda_float_tensor(key_up, "key_up");
  check_cuda_float_tensor(value_up, "value_up");
  check_cuda_long_tensor(position_storage, "position_storage");
  check_cuda_long_tensor(block_table, "block_table");
  check_cuda_long_tensor(sequence_lengths, "sequence_lengths");
  check_cuda_long_tensor(query_positions, "query_positions");

  const auto scalar_type = q_nope.scalar_type();
  TORCH_CHECK(
      q_pe.scalar_type() == scalar_type && kv_storage.scalar_type() == scalar_type &&
          pe_storage.scalar_type() == scalar_type && key_up.scalar_type() == scalar_type &&
          value_up.scalar_type() == scalar_type,
      "all floating-point paged MLA tensors must have the same dtype");
  TORCH_CHECK(
      q_nope.device() == q_pe.device() && q_nope.device() == kv_storage.device() &&
          q_nope.device() == pe_storage.device() &&
          q_nope.device() == position_storage.device() &&
          q_nope.device() == block_table.device() &&
          q_nope.device() == sequence_lengths.device() &&
          q_nope.device() == key_up.device() && q_nope.device() == value_up.device() &&
          q_nope.device() == query_positions.device(),
      "all paged MLA tensors must share a CUDA device");
  TORCH_CHECK(q_nope.dim() == 4 && q_pe.dim() == 4,
              "q_nope and q_pe must have shape [batch, query, heads, dimension]");
  TORCH_CHECK(kv_storage.dim() == 3 && pe_storage.dim() == 3,
              "paged cache payload storage must have shape [pages, page_size, dimension]");
  TORCH_CHECK(position_storage.dim() == 2,
              "paged position storage must have shape [pages, page_size]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [batch, logical_pages]");
  TORCH_CHECK(sequence_lengths.dim() == 1,
              "sequence_lengths must contain one entry per batch row");
  TORCH_CHECK(query_positions.dim() == 2,
              "query_positions must have shape [batch, query]");
  TORCH_CHECK(key_up.dim() == 3 && value_up.dim() == 3,
              "absorbed weights must have shape [heads, output, latent]");

  const int64_t batch = q_nope.size(0);
  const int64_t query_length = q_nope.size(1);
  const int64_t heads = q_nope.size(2);
  const int64_t nope_dim = q_nope.size(3);
  const int64_t num_pages = kv_storage.size(0);
  const int64_t page_size = kv_storage.size(1);
  const int64_t latent_dim = kv_storage.size(2);
  const int64_t rope_dim = pe_storage.size(2);
  const int64_t value_dim = value_up.size(1);
  TORCH_CHECK(num_pages > 0 && page_size > 0, "paged cache dimensions must be positive");
  TORCH_CHECK(q_nope.sizes().slice(0, 3) == q_pe.sizes().slice(0, 3),
              "q_nope and q_pe batch/query/head dimensions must match");
  TORCH_CHECK(
      kv_storage.sizes().slice(0, 2) == pe_storage.sizes().slice(0, 2) &&
          kv_storage.sizes().slice(0, 2) == position_storage.sizes(),
      "paged cache page dimensions must match");
  TORCH_CHECK(block_table.size(0) == batch && sequence_lengths.numel() == batch,
              "paged metadata batch dimension must match the query");
  TORCH_CHECK(
      query_positions.size(0) == batch && query_positions.size(1) == query_length,
      "query_positions must match query batch and sequence dimensions");
  TORCH_CHECK(heads == key_up.size(0) && heads == value_up.size(0),
              "query and absorbed weights must have the same head count");
  TORCH_CHECK(nope_dim == key_up.size(1), "q_nope dimension must match key_up");
  TORCH_CHECK(q_pe.size(3) == rope_dim, "q_pe dimension must match paged pe storage");
  TORCH_CHECK(latent_dim == key_up.size(2) && latent_dim == value_up.size(2),
              "paged cache latent dimension must match absorbed weights");
  TORCH_CHECK(
      heads > 0 && nope_dim > 0 && rope_dim > 0 && latent_dim > 0 && value_dim > 0,
      "paged MLA head and feature dimensions must be positive");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");

  if (!metadata_validated) {
    auto lengths_cpu = sequence_lengths.cpu().contiguous();
    auto table_cpu = block_table.cpu().contiguous();
    auto positions_cpu = position_storage.cpu().contiguous();
    auto query_positions_cpu = query_positions.cpu().contiguous();
    const int64_t* lengths_ptr = lengths_cpu.const_data_ptr<int64_t>();
    const int64_t* table_ptr = table_cpu.const_data_ptr<int64_t>();
    const int64_t* positions_ptr = positions_cpu.const_data_ptr<int64_t>();
    const int64_t* query_positions_ptr = query_positions_cpu.const_data_ptr<int64_t>();
    const int64_t logical_capacity = block_table.size(1) * page_size;
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      const int64_t length = lengths_ptr[batch_index];
      TORCH_CHECK(
          length >= 0 && length <= logical_capacity,
          "sequence_lengths[",
          batch_index,
          "] exceeds the block-table capacity");
      const int64_t required_pages = (length + page_size - 1) / page_size;
      std::unordered_set<int64_t> used_pages;
      for (int64_t logical_page = 0; logical_page < block_table.size(1); ++logical_page) {
        const int64_t physical_page =
            table_ptr[batch_index * block_table.size(1) + logical_page];
        if (logical_page < required_pages) {
          TORCH_CHECK(
              physical_page >= 0 && physical_page < num_pages,
              "block_table row ",
              batch_index,
              " contains an out-of-range page");
          TORCH_CHECK(
              used_pages.insert(physical_page).second,
              "block_table row ",
              batch_index,
              " repeats a physical page");
        } else {
          TORCH_CHECK(
              physical_page == -1,
              "unused block_table entries must be -1");
        }
      }
      int64_t previous_position = -1;
      for (int64_t key_index = 0; key_index < length; ++key_index) {
        const int64_t logical_page = key_index / page_size;
        const int64_t page_offset = key_index % page_size;
        const int64_t physical_page =
            table_ptr[batch_index * block_table.size(1) + logical_page];
        const int64_t position = positions_ptr[physical_page * page_size + page_offset];
        TORCH_CHECK(position >= 0, "block_table references an unwritten paged-cache slot");
        TORCH_CHECK(
            key_index == 0 || position > previous_position,
            "paged cache positions must increase within each logical sequence");
        previous_position = position;
      }
      int64_t previous_query_position = -1;
      for (int64_t query_index = 0; query_index < query_length; ++query_index) {
        const int64_t position =
            query_positions_ptr[batch_index * query_length + query_index];
        TORCH_CHECK(position >= 0, "query_positions must be non-negative");
        TORCH_CHECK(
            query_index == 0 || position > previous_query_position,
            "query_positions must increase within each batch row");
        previous_query_position = position;
      }
    }
  }

  auto output = at::empty({batch, query_length, heads, value_dim}, q_nope.options());
  if (batch == 0 || query_length == 0) {
    return output;
  }
  const int64_t rows = batch * query_length * heads;
  const c10::cuda::CUDAGuard device_guard(q_nope.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(q_nope.get_device());
  TORCH_CHECK(
      rows <= static_cast<int64_t>(properties->maxGridSize[0]),
      "too many paged MLA query rows for a one-dimensional CUDA launch: ",
      rows);
  const size_t shared_bytes =
      static_cast<size_t>(2 * latent_dim + kThreads + 2) * sizeof(float);
  TORCH_CHECK(
      shared_bytes <= static_cast<size_t>(properties->sharedMemPerBlock),
      "paged MLA latent dimension requires ",
      shared_bytes,
      " bytes of shared memory, but the device limit is ",
      properties->sharedMemPerBlock);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_nope.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "mla_paged_absorbed_attention_cuda",
      [&] {
        mla_paged_absorbed_attention_kernel<scalar_t><<<
            static_cast<unsigned int>(rows), kThreads, shared_bytes, stream>>>(
            q_nope.const_data_ptr<scalar_t>(),
            q_pe.const_data_ptr<scalar_t>(),
            kv_storage.const_data_ptr<scalar_t>(),
            pe_storage.const_data_ptr<scalar_t>(),
            position_storage.const_data_ptr<int64_t>(),
            block_table.const_data_ptr<int64_t>(),
            sequence_lengths.const_data_ptr<int64_t>(),
            key_up.const_data_ptr<scalar_t>(),
            value_up.const_data_ptr<scalar_t>(),
            query_positions.const_data_ptr<int64_t>(),
            output.mutable_data_ptr<scalar_t>(),
            num_pages,
            page_size,
            heads,
            query_length,
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
            kv_storage.stride(0),
            kv_storage.stride(1),
            kv_storage.stride(2),
            pe_storage.stride(0),
            pe_storage.stride(1),
            pe_storage.stride(2),
            position_storage.stride(0),
            position_storage.stride(1),
            block_table.stride(0),
            block_table.stride(1),
            sequence_lengths.stride(0),
            key_up.stride(0),
            key_up.stride(1),
            key_up.stride(2),
            value_up.stride(0),
            value_up.stride(1),
            value_up.stride(2),
            query_positions.stride(0),
            query_positions.stride(1),
            static_cast<float>(scale),
            causal);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("mla_paged_absorbed_attention", TORCH_FN(mla_paged_absorbed_attention_cuda));
}
