#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>
#include <cmath>

using namespace nvcuda;

__global__ void fa2_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int Nq, int Nk, int d, int dv, float scale)
{
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    const int warps_in_block = (blockDim.x / 32) > 0 ? (blockDim.x / 32) : 1;
    const int rows_per_warp  = 16;

    const int row_tile_start = blockIdx.y * (rows_per_warp * warps_in_block) + warp_id * rows_per_warp;
    int remain_rows = Nq - row_tile_start;
    const int valid_rows = (remain_rows > 0) ? (remain_rows < rows_per_warp ? remain_rows : rows_per_warp) : 0;
    if (valid_rows <= 0) return;

    extern __shared__ unsigned char smem_raw[];
    const size_t per_warp_bytes =
          16*16*sizeof(__half)
        + 16*16*sizeof(__half)
        + 16*16*sizeof(float);
    unsigned char* base = smem_raw + warp_id * per_warp_bytes;
    __half* Ash = reinterpret_cast<__half*>(base);
    __half* Bsh = Ash + 16*16;
    float*  Ssh = reinterpret_cast<float*>(Bsh + 16*16);

    float m_row[16];
    float l_row[16];
    if (lane < 16){
        for (int r = 0; r < valid_rows; ++r){ m_row[r] = -CUDART_INF_F; l_row[r] = 0.f; }
    }

    for (int j0 = 0; j0 < Nk; j0 += 16){
        int remain_cols = Nk - j0;
        const int tile_cols = (remain_cols < 16 ? remain_cols : 16);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> C;
        wmma::fill_fragment(C, 0.0f);

        for (int k0 = 0; k0 < d; k0 += 16){
            for (int t = lane; t < 16*16; t += 32){
                int r = t / 16;
                int kk = t % 16;
                __half hv = __float2half(0.f);
                if (r < valid_rows){
                    int gr = row_tile_start + r;
                    int gk = k0 + kk;
                    if (gk < d){
                        hv = __float2half(Q[gr * d + gk]);
                    }
                }
                Ash[r*16 + kk] = hv;
            }
            for (int t = lane; t < 16*16; t += 32){
                int c  = t / 16;
                int kk = t % 16;
                __half hv = __float2half(0.f);
                if (c < tile_cols){
                    int gc = j0 + c;
                    int gk = k0 + kk;
                    if (gk < d){
                        hv = __float2half(K[gc * d + gk]);
                    }
                }
                Bsh[kk + c*16] = hv;
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> A;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> B;
            wmma::load_matrix_sync(A, Ash, 16);
            wmma::load_matrix_sync(B, Bsh, 16);
            wmma::mma_sync(C, A, B, C);
        }

        wmma::store_matrix_sync(Ssh, C, 16, wmma::mem_row_major);
        __syncwarp();

        if (lane < 16){
            for (int r = 0; r < valid_rows; ++r){
                float row_vals[16];
                float m_tile = -CUDART_INF_F;
                for (int c = 0; c < tile_cols; ++c){
                    float v = Ssh[r*16 + c] * scale;
                    row_vals[c] = v;
                    m_tile = v > m_tile ? v : m_tile;
                }
                float m_prev = m_row[r];
                float m_new  = (m_prev > m_tile ? m_prev : m_tile);
                float alpha  = (m_prev == -CUDART_INF_F) ? 0.0f : expf(m_prev - m_new);

                float e_sum = 0.f;
                float e_val[16];
                for (int c = 0; c < tile_cols; ++c){
                    float e = expf(row_vals[c] - m_new);
                    e_val[c] = e;
                    e_sum += e;
                }
                float l_prev = l_row[r];
                float l_new  = l_prev * alpha + e_sum;
                float beta   = (l_new > 0.f) ? (l_prev * alpha / l_new) : 0.f;
                float gamma  = (l_new > 0.f) ? (1.f / l_new) : 0.f;

                if (j0 == 0){
                    for (int v = 0; v < dv; ++v){
                        O[(row_tile_start + r) * dv + v] = 0.f;
                    }
                }
                for (int v = 0; v < dv; ++v){
                    float acc = 0.f;
                    for (int c = 0; c < tile_cols; ++c){
                        acc += e_val[c] * V[(j0 + c) * dv + v];
                    }
                    float prev = O[(row_tile_start + r) * dv + v];
                    O[(row_tile_start + r) * dv + v] = prev * beta + gamma * acc;
                }

                m_row[r] = m_new;
                l_row[r] = l_new;
            }
        }
        __syncwarp();
    }
}

static inline size_t fa2_forward_smem_bytes(int warps_per_block){
    return (size_t)warps_per_block * (16*16*sizeof(__half) + 16*16*sizeof(__half) + 16*16*sizeof(float));
}

void fa2_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int Nq, int Nk, int d, int dv,
    float scale,
    int warps_per_block,
    cudaStream_t stream)
{
    if (d % 16 != 0) return;
    if (warps_per_block <= 0) warps_per_block = 4;
    dim3 block(32 * warps_per_block, 1, 1);
    int rows_per_block = 16 * warps_per_block;
    dim3 grid(1, (Nq + rows_per_block - 1) / rows_per_block, 1);
    size_t smem = fa2_forward_smem_bytes(warps_per_block);
    fa2_fwd_kernel<<<grid, block, smem, stream>>>(Q, K, V, O, Nq, Nk, d, dv, scale);
}
