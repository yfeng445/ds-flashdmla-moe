#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cassert>
using namespace nvcuda;

static inline __host__ __device__ size_t align16(size_t x){ return (x + 15) & ~size_t(15); }

__device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v){
    #pragma unroll
    for (int off=16; off>0; off>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}

__global__ void flashAttn2_bwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int N, int d, int dv, float scale, int Br, int Bc)
{
    const int W       = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    const int i_tile  = blockIdx.y;
    const int i0      = i_tile * Br;

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    float* Kds = reinterpret_cast<float*>(ptr);
    ptr += align16((size_t)Bc * d * sizeof(float));
    float* m_row = reinterpret_cast<float*>(ptr);
    ptr += align16((size_t)Br * sizeof(float));
    float* l_row = reinterpret_cast<float*>(ptr);
    ptr += align16((size_t)Br * sizeof(float));
    float* delta_row = reinterpret_cast<float*>(ptr);
    ptr += align16((size_t)Br * sizeof(float));
    __half* Vds = reinterpret_cast<__half*>(ptr);
    ptr += align16((size_t)Bc * dv * sizeof(__half));

    __half* Ash  = reinterpret_cast<__half*>(ptr);     (void)Ash;
    ptr += (size_t)W * (2 * 16 * 16 * sizeof(__half));
    float*  Sblk = reinterpret_cast<float*>(ptr);      (void)Sblk;

    __shared__ float sum_warp[32];

    for (int r = threadIdx.x; r < Br; r += blockDim.x){
        m_row[r]     = -INFINITY;
        l_row[r]     = 0.f;
        delta_row[r] = 0.f;
    }
    __syncthreads();

    for (;;){
        const int i_start = i0 + warp_id * 16;
        if (i_start >= N) break;
        const int mr = min(16, N - i_start);

        for (int k0 = 0; k0 < d; k0 += 16){
            __half* As = Ash + warp_id * (2 * 16 * 16);
            __half* Bs = As + 16 * 16;

            for (int t = lane; t < 16 * 16; t += 32){
                int r = t / 16, c = t % 16;
                __half a = (r < mr && (k0 + c) < d) ? __float2half(Q[(size_t)(i_start + r) * d + (k0 + c)]) : __float2half(0.f);
                As[r * 16 + c] = a;
            }
            __syncwarp();

            float m_loc = -INFINITY;
            for (int j0 = 0; j0 < N; j0 += Bc){
                const int cols = min(Bc, N - j0);

                for (int kk = lane; kk < cols * d; kk += 32){
                    int cc = kk / d, dd = kk % d;
                    Kds[(size_t)cc * d + dd] = K[(size_t)(j0 + cc) * d + dd];
                }
                __syncwarp();

                for (int c_off = 0; c_off < cols; c_off += 16){
                    const int cc = min(16, cols - c_off);

                    __half* Bs0 = Bs;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> C;
                    wmma::fill_fragment(C, 0.0f);

                    for (int kk2 = 0; kk2 < 16; kk2 += 16){
                        for (int t = lane; t < 16 * 16; t += 32){
                            int r = t / 16, c = t % 16;
                            int gj = c_off + c;
                            __half b = (c < cc && (k0 + r) < d) ? __float2half(Kds[(size_t)gj * d + (k0 + r)]) : __float2half(0.0f);
                            Bs0[r + c * 16] = b;
                        }
                        __syncwarp();

                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> A;
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> B;
                        wmma::load_matrix_sync(A, As, 16);
                        wmma::load_matrix_sync(B, Bs0, 16);
                        wmma::mma_sync(C, A, B, C);
                        __syncwarp();
                    }

                    wmma::store_matrix_sync(Sblk + warp_id * 16 * 16, C, 16, wmma::mem_row_major);
                    __syncwarp();

                    if (lane < 16){
                        float* S = Sblk + warp_id * 16 * 16;
                        for (int r = 0; r < mr; ++r){
                            float m_prev = m_row[warp_id * 16 + r];
                            float m_tile = -INFINITY;
                            for (int c = 0; c < cc; ++c){
                                float v = S[r * 16 + c] * scale;
                                m_tile = fmaxf(m_tile, v);
                            }
                            float m_new = fmaxf(m_prev, m_tile);
                            float alpha = (m_prev == -INFINITY) ? 0.f : __expf(m_prev - m_new);

                            float e_sum = 0.f;
                            for (int c = 0; c < cc; ++c){
                                e_sum += __expf(S[r * 16 + c] * scale - m_new);
                            }
                            float l_prev = l_row[warp_id * 16 + r];
                            float l_new  = l_prev * alpha + e_sum;

                            m_row[warp_id * 16 + r] = m_new;
                            l_row[warp_id * 16 + r] = l_new;
                        }
                    }
                    __syncwarp();
                }
                __syncwarp();
            }
        }
        __syncwarp();
    }

    __syncthreads();

    for (int c0 = 0; c0 < N; c0 += Bc){
        const int cols = min(Bc, N - c0);

        for (int kk = threadIdx.x; kk < cols * dv; kk += blockDim.x){
            int c = kk / dv, v = kk % dv;
            Vds[(size_t)c * dv + v] = __float2half(V[(size_t)(c0 + c) * dv + v]);
        }
        __syncthreads();

        for (int i_off = 0; i_off < Br; i_off += 16){
            const int i_start = i0 + i_off;
            if (i_start >= N) break;
            const int mr_tail = min(16, N - i_start);

            for (int c_off = 0; c_off < cols; c_off += 16){
                const int cc = min(16, cols - c_off);

                float* Sw = Sblk;
                if (warp_id == 0){
                    __half* As = Ash;
                    __half* Bs = As + 16 * 16;

                    for (int t = lane; t < 16 * 16; t += 32){
                        int r = t / 16, c = t % 16;
                        __half a = (r < mr_tail) ? __float2half(Q[(size_t)(i_start + r) * d + c]) : __float2half(0.f);
                        As[r * 16 + c] = a;
                    }
                    for (int t = lane; t < 16 * 16; t += 32){
                        int r = t / 16, c = t % 16;
                        int gj = c0 + c_off + c;
                        __half b = (c < cc) ? __float2half(Kds[(size_t)gj * d + r]) : __float2half(0.0f);
                        Bs[r + c * 16] = b;
                    }
                    __syncwarp();

                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> A;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> B;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> C;
                    wmma::fill_fragment(C, 0.0f);
                    wmma::load_matrix_sync(A, As, 16);
                    wmma::load_matrix_sync(B, Bs, 16);
                    wmma::mma_sync(C, A, B, C);
                    wmma::store_matrix_sync(Sw, C, 16, wmma::mem_row_major);
                }
                __syncthreads();

                for (int r = threadIdx.x; r < mr_tail; r += blockDim.x){
                    int gi = i_start + r;
                    float m = m_row[(i_off + r)];
                    float l = l_row[(i_off + r)];

                    float tmp = 0.f;
                    for (int c = 0; c < cc; ++c){
                        float e = __expf(Sw[r * 16 + c] * scale - m);
                        float p = e / l;
                        int gj = c0 + c_off + c;

                        float acc = 0.f;
                        for (int v = 0; v < dv; ++v){
                            acc += p * (float)Vds[(size_t)(c_off + c) * dv + v] * dO[(size_t)gi * dv + v];
                        }
                        tmp += acc;
                    }
                    atomicAdd(delta_row + (i_off + r), tmp);
                }
                __syncthreads();

                for (int c = threadIdx.x; c < cc; c += blockDim.x){
                    int gj = c0 + c_off + c;

                    float acc = 0.f;
                    for (int r = 0; r < mr_tail; ++r){
                        int gi = i_start + r;
                        float m = m_row[(i_off + r)];
                        float l = l_row[(i_off + r)];
                        float e = __expf(Sw[r * 16 + c] * scale - m);
                        float p = e / l;

                        float tval = 0.f;
                        for (int v = 0; v < dv; ++v){
                            tval += (float)Vds[(size_t)(c_off + c) * dv + v] * dO[(size_t)gi * dv + v];
                        }
                        float dot = p * (tval - delta_row[(i_off + r)]);
                        acc += dot;
                    }
                    atomicAdd(&dK[(size_t)gj * d + 0], 0.f);
                    for (int k0 = 0; k0 < d; ++k0){
                        float grad = 0.f;
                        for (int r = 0; r < mr_tail; ++r){
                            int gi = i_start + r;
                            float m = m_row[(i_off + r)];
                            float l = l_row[(i_off + r)];
                            float e = __expf(Sw[r * 16 + c] * scale - m);
                            float p = e / l;

                            float tval = 0.f;
                            for (int v = 0; v < dv; ++v){
                                tval += (float)Vds[(size_t)(c_off + c) * dv + v] * dO[(size_t)gi * dv + v];
                            }
                            float dot = p * (tval - delta_row[(i_off + r)]);
                            grad += dot * Q[(size_t)gi * d + k0];
                        }
                        atomicAdd(&dK[(size_t)gj * d + k0], grad);
                    }
                }
                __syncthreads();

                for (int v0 = threadIdx.x; v0 < dv; v0 += blockDim.x){
                    for (int c = 0; c < cc; ++c){
                        int gj = c0 + c_off + c;
                        float acc = 0.f;
                        for (int r = 0; r < mr_tail; ++r){
                            int gi = i_start + r;
                            float m = m_row[(i_off + r)];
                            float l = l_row[(i_off + r)];
                            float e = __expf(Sw[r * 16 + c] * scale - m);
                            float p = e / l;
                            acc += p * dO[(size_t)gi * dv + v0];
                        }
                        atomicAdd(&dV[(size_t)gj * dv + v0], acc);
                    }
                }
                __syncthreads();

                for (int r = threadIdx.x; r < mr_tail; r += blockDim.x){
                    int gi = i_start + r;
                    float acc = 0.f;
                    for (int c = 0; c < cc; ++c){
                        int gj = c0 + c_off + c;
                        float m = m_row[(i_off + r)];
                        float l = l_row[(i_off + r)];
                        float e = __expf(Sw[r * 16 + c] * scale - m);
                        float p = e / l;

                        float tval = 0.f;
                        for (int v = 0; v < dv; ++v){
                            tval += (float)Vds[(size_t)(c_off + c) * dv + v] * dO[(size_t)gi * dv + v];
                        }
                        float dot = p * (tval - delta_row[(i_off + r)]);
                        acc += dot * K[(size_t)gj * d + 0];
                    }
                    for (int k0 = 0; k0 < d; ++k0){
                        float acc_k = 0.f;
                        for (int c = 0; c < cc; ++c){
                            int gj = c0 + c_off + c;
                            float m = m_row[(i_off + r)];
                            float l = l_row[(i_off + r)];
                            float e = __expf(Sw[r * 16 + c] * scale - m);
                            float p = e / l;

                            float tval = 0.f;
                            for (int v = 0; v < dv; ++v){
                                tval += (float)Vds[(size_t)(c_off + c) * dv + v] * dO[(size_t)gi * dv + v];
                            }
                            float dot = p * (tval - delta_row[(i_off + r)]);
                            acc_k += dot * K[(size_t)gj * d + k0];
                        }
                        atomicAdd(&dQ[(size_t)gi * d + k0], acc_k);
                    }
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

static inline size_t fa2_backward_smem_bytes(int Br, int Bc, int d, int dv, int block_x){
    const int W = block_x / 32;
    size_t bytes = 0;
    bytes += (size_t)Bc * d  * sizeof(float);
    bytes += (size_t)Br      * sizeof(float);
    bytes += (size_t)Br      * sizeof(float);
    bytes += (size_t)Br      * sizeof(float);
    bytes += (size_t)Bc * dv * sizeof(__half);
    bytes += (size_t)W * (2 * 16 * 16 * sizeof(__half));
    bytes += (size_t)W * (16 * 16 * sizeof(float));
    return bytes;
}

extern "C" void fa2_backward(
    const float* Q,
    const float* K,
    const float* V,
    const float* dO,
    float* dQ,
    float* dK,
    float* dV,
    int N,
    int d,
    int dv,
    float scale,
    int warps_per_block,
    cudaStream_t stream)
{
    if ((d % 16) != 0) return;
    if (warps_per_block <= 0) warps_per_block = 4;
    const int Br = 16 * warps_per_block;
    const int Bc = 16 * warps_per_block;
    dim3 block(32 * warps_per_block, 1, 1);
    dim3 grid(1, (N + Br - 1) / Br, 1);
    size_t sharedBytes = fa2_backward_smem_bytes(Br, Bc, d, dv, block.x);
    flashAttn2_bwd_kernel<<<grid, block, sharedBytes, stream>>>(
        Q, K, V, dO, dQ, dK, dV, N, d, dv, scale, Br, Bc);
}
