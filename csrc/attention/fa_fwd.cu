#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

static __device__ __forceinline__ float warp_reduce_max(float v){
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
static __device__ __forceinline__ float warp_reduce_sum(float v){
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__global__ void fa1_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, int dv, float scale,
    int Br, int Bc)
{
    extern __shared__ float smem[];
    float* Kds   = smem;                         // [Bc, d]
    float* Vds   = Kds + (size_t)Bc * d;         // [Bc, dv]
    float* Etile = Vds + (size_t)Bc * dv;        // [Br, Bc] (only mr x cols used)
    float* m_row = Etile + (size_t)Br * Bc;      // [Br]
    float* l_row = m_row + Br;                   // [Br]
    float* tmpA  = l_row + Br;                   // [Br]  m_new or prev_scale
    float* tmpB  = tmpA  + Br;                   // [Br]  alpha or inv_l_new

    __shared__ float red[32];

    const int tid = threadIdx.x;
    const int i0  = blockIdx.y * Br;
    const int mr  = min(Br, N - i0);
    if (mr <= 0) return;

    for (int r = tid; r < mr; r += blockDim.x) { m_row[r] = -CUDART_INF_F; l_row[r] = 0.f; }
    __syncthreads();

    for (int idx = tid; idx < mr * dv; idx += blockDim.x){
        int r = idx / dv, v = idx % dv;
        O[(size_t)(i0 + r) * dv + v] = 0.f;
    }
    __syncthreads();

    for (int c0 = 0; c0 < N; c0 += Bc){
        const int cols = min(Bc, N - c0);

        for (int t = tid; t < cols * d; t += blockDim.x){
            int c = t / d, kx = t % d;
            Kds[(size_t)c * d + kx] = K[(size_t)(c0 + c) * d + kx];
        }
        for (int t = tid; t < cols * dv; t += blockDim.x){
            int c = t / dv, vx = t % dv;
            Vds[(size_t)c * dv + vx] = V[(size_t)(c0 + c) * dv + vx];
        }
        __syncthreads();

        for (int r = 0; r < mr; ++r){
            const float* __restrict__ Qi = Q + (size_t)(i0 + r) * d;

            float local_max = -CUDART_INF_F;
            for (int c = tid; c < cols; c += blockDim.x){
                const float* __restrict__ Kj = Kds + (size_t)c * d;
                float dot = 0.f;
                #pragma unroll 1
                for (int kx = 0; kx < d; ++kx) dot += Qi[kx] * Kj[kx];
                local_max = fmaxf(local_max, dot * scale);
            }
            float wmax = warp_reduce_max(local_max);
            if ((tid & 31) == 0) red[tid >> 5] = wmax;
            __syncthreads();
            float m_tile = -CUDART_INF_F;
            if ((tid >> 5) == 0){
                float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : -CUDART_INF_F;
                v = warp_reduce_max(v);
                if (tid == 0) m_tile = v;
            }
            __syncthreads();
            if (tid == 0){
                float m_prev = m_row[r];
                float m_new  = fmaxf(m_prev, m_tile);
                float alpha  = (m_prev == -CUDART_INF_F) ? 0.f : __expf(m_prev - m_new);
                tmpA[r] = m_new;     // stash m_new
                tmpB[r] = alpha;     // stash alpha
            }
            __syncthreads();

            float m_new = tmpA[r];
            float e_sum_local = 0.f;
            for (int c = tid; c < cols; c += blockDim.x){
                const float* __restrict__ Kj = Kds + (size_t)c * d;
                float dot = 0.f;
                #pragma unroll 1
                for (int kx = 0; kx < d; ++kx) dot += Qi[kx] * Kj[kx];
                float e = __expf(dot * scale - m_new);
                Etile[(size_t)r * Bc + c] = e;
                e_sum_local += e;
            }
            float wsum = warp_reduce_sum(e_sum_local);
            if ((tid & 31) == 0) red[tid >> 5] = wsum;
            __syncthreads();
            float e_sum = 0.f;
            if ((tid >> 5) == 0){
                float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.f;
                v = warp_reduce_sum(v);
                if (tid == 0) e_sum = v;
            }
            __syncthreads();

            if (tid == 0){
                float m_prev = m_row[r];
                float l_prev = l_row[r];
                float alpha  = tmpB[r];
                float l_new  = l_prev * alpha + e_sum;
                float inv_l_new = (l_new > 0.f) ? (1.f / l_new) : 0.f;
                float prev_scale = (l_new > 0.f) ? (l_prev * alpha * inv_l_new) : 0.f;
                tmpA[r] = prev_scale;
                tmpB[r] = inv_l_new;
                m_row[r] = m_new;
                l_row[r] = l_new;
            }
            __syncthreads();

            const float prev_scale = tmpA[r];
            const float inv_l_new  = tmpB[r];

            for (int v = tid; v < dv; v += blockDim.x){
                float acc = 0.f;
                for (int c = 0; c < cols; ++c){
                    float e = Etile[(size_t)r * Bc + c];
                    acc += e * Vds[(size_t)c * dv + v];
                }
                float* __restrict__ Orow = O + (size_t)(i0 + r) * dv;
                Orow[v] = Orow[v] * prev_scale + inv_l_new * acc;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

static inline size_t fa1_forward_smem_bytes(int Br, int Bc, int d, int dv){
    size_t bytes = 0;
    bytes += (size_t)Bc * d  * sizeof(float);    // Kds
    bytes += (size_t)Bc * dv * sizeof(float);    // Vds
    bytes += (size_t)Br * Bc * sizeof(float);    // Etile
    bytes += (size_t)(4 * Br) * sizeof(float);   // m_row, l_row, tmpA, tmpB
    return bytes;
}

extern "C" void fa1_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N, int Nk, int d, int dv,
    float scale,
    int warps_per_block,
    cudaStream_t stream)
{
    (void)Nk; // FA1 assumes Nk == N (full attention). If not, pass Nk for K/V loops accordingly.
    if (N <= 0 || d <= 0 || dv <= 0) return;
    if (warps_per_block <= 0) warps_per_block = 4;

    const int Br = 16 * warps_per_block;
    const int Bc = 16 * warps_per_block;
    const int threads = 256;

    dim3 block(threads, 1, 1);
    dim3 grid(1, (N + Br - 1) / Br, 1);

    size_t smem = fa1_forward_smem_bytes(Br, Bc, d, dv);
    fa1_forward_kernel<<<grid, block, smem, stream>>>(Q, K, V, O, N, d, dv, scale, Br, Bc);
}
