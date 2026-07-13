/*
 * cuda_search.cu — CUDA 单卡搜索（阶段 5）。C 风格 + extern "C" 边界。
 *
 * 两 kernel:
 *   slime_map_kernel : 一维，每线程判定一格，写 (w+16)×(h+16) 的 0/1 图（四周外扩 8）。
 *   search_kernel    : block(16,16)，__shared__ tile[32][32]，逐格数甜甜圈，atomicAdd 写结果。
 *
 * 设备端判定与 CPU 侧 slime.h 逐位一致：精确 nextInt(10) 去偏（禁用偏置版），
 * z*z 先 32 位回绕再 64 位乘 4392871。
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" {
#include "slimerander.h"
}

#define SR_WINDOW   17
#define SR_OFFSET   8
#define SR_LCG_MUL  0x5DEECE66DULL
#define SR_LCG_ADD  0xBULL
#define SR_LCG_MASK ((1ULL << 48) - 1ULL)

/* ---- 设备端判定（与 slime.h 逐位一致） ---- */
__device__ __forceinline__ int32_t d_mul32(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a * (uint32_t)b);
}
__device__ __forceinline__ int64_t d_chunk_seed(int64_t world_seed, int32_t x, int32_t z) {
    int32_t t1 = d_mul32(d_mul32(x, x), 4987142);
    int32_t t2 = d_mul32(x, 5947611);
    int64_t t3 = (int64_t)d_mul32(z, z) * 4392871LL;
    int32_t t4 = d_mul32(z, 389711);
    uint64_t u = (uint64_t)world_seed;
    u += (uint64_t)(int64_t)t1;
    u += (uint64_t)(int64_t)t2;
    u += (uint64_t)t3;
    u += (uint64_t)(int64_t)t4;
    u ^= (uint64_t)987234911LL;
    return (int64_t)u;
}
__device__ __forceinline__ int d_is_slime(int64_t world_seed, int32_t x, int32_t z) {
    uint64_t s = ((uint64_t)d_chunk_seed(world_seed, x, z) ^ SR_LCG_MUL) & SR_LCG_MASK;
    for (;;) {
        s = (s * SR_LCG_MUL + SR_LCG_ADD) & SR_LCG_MASK;
        int32_t bits = (int32_t)(s >> (48 - 31));
        int32_t val  = bits % 10;
        if (((uint32_t)bits - (uint32_t)val + 9u) < 0x80000000u)
            return val == 0;
    }
}

/* ---- kernel 1: 填 slime map ---- */
__global__ void slime_map_kernel(int64_t seed, int32_t map_x0, int32_t map_z0,
                                 uint32_t map_w, uint32_t map_h, uint8_t *map) {
    uint64_t idx   = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)map_w * map_h;
    if (idx >= total) return;
    int32_t rel_x = (int32_t)(idx % map_w);
    int32_t rel_z = (int32_t)(idx / map_w);
    map[idx] = (uint8_t)d_is_slime(seed, map_x0 + rel_x, map_z0 + rel_z);
}

/* ---- kernel 2: 每中心数甜甜圈 ---- */
__global__ void search_kernel(const uint8_t *map, uint32_t map_w,
                              uint8_t thr, int32_t x0, int32_t z0,
                              uint32_t width, uint32_t height,
                              SrResult *out, uint32_t cap,
                              uint32_t *out_count, int *overflow) {
    const uint32_t bx = blockIdx.x * blockDim.x;
    const uint32_t bz = blockIdx.y * blockDim.y;
    const uint32_t tx = threadIdx.x;
    const uint32_t tz = threadIdx.y;
    const uint32_t rel_x = bx + tx;
    const uint32_t rel_z = bz + tz;

    __shared__ uint8_t tile[32][32];
    for (uint32_t lz = tz; lz < 32; lz += blockDim.y) {
        for (uint32_t lx = tx; lx < 32; lx += blockDim.x) {
            uint32_t map_x = bx + lx;
            uint32_t map_z = bz + lz;
            tile[lz][lx] = (map_x < map_w && map_z < height + 16)
                         ? map[(uint64_t)map_z * map_w + map_x] : 0;
        }
    }
    __syncthreads();

    if (rel_x >= width || rel_z >= height) return;

    uint32_t count = 0;
    for (int32_t dx = -SR_OFFSET; dx <= SR_OFFSET; ++dx) {
        for (int32_t dz = -SR_OFFSET; dz <= SR_OFFSET; ++dz) {
            int32_t d2 = dx * dx + dz * dz;
            if (!(d2 > 1 && d2 <= 64)) continue;
            count += tile[(int32_t)tz + dz + 8][(int32_t)tx + dx + 8];
        }
    }

    if (count >= thr) {
        uint32_t oi = atomicAdd(out_count, 1u);
        if (oi < cap) { out[oi].x = x0 + (int32_t)rel_x; out[oi].z = z0 + (int32_t)rel_z; out[oi].count = count; }
        else atomicExch(overflow, 1);
    }
}

/* ---- 上下文 ---- */
struct SrCudaCtx {
    int       device;
    uint32_t  max_w, max_h, max_cap;
    uint8_t  *d_map;
    SrResult *d_results;
    uint32_t *d_count;
    int      *d_overflow;
    SrResult *h_results;   /* pinned */
    uint32_t *h_count;     /* pinned */
    int      *h_overflow;  /* pinned */
    cudaStream_t stream;
};

extern "C" {

int sr_cuda_ctx_deinit(SrCudaCtx *c);  /* 前向声明：ctx_init 失败时调用 */

int sr_cuda_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return -1;
    return n;
}

int sr_cuda_device_score(int idx, float *score, uint64_t *free_mem, uint64_t *total_mem) {
    if (cudaSetDevice(idx) != cudaSuccess) return 1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, idx) != cudaSuccess) return 1;
    size_t fb = 0, tb = 0;
    if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) return 1;
    if (free_mem)  *free_mem  = (uint64_t)fb;
    if (total_mem) *total_mem = (uint64_t)tb;
    /* CUDA 13 移除了 cudaDeviceProp::clockRate，改用属性查询（kHz）。 */
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, idx);
    float perf = (float)prop.multiProcessorCount * (float)clock_khz;
    float cap  = (float)(prop.major * 10 + prop.minor);
    float mem  = (float)((double)fb / (double)(1ull << 30));
    if (score) *score = perf * (1.0f + cap * 0.02f) * (1.0f + mem * 0.02f);
    return 0;
}

int sr_cuda_ctx_init(int device, uint32_t max_w, uint32_t max_h, uint32_t max_cap, SrCudaCtx **out) {
    if (!out) return 1;
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    SrCudaCtx *c = (SrCudaCtx *)calloc(1, sizeof(SrCudaCtx));
    if (!c) return 1;
    c->device = device; c->max_w = max_w; c->max_h = max_h; c->max_cap = max_cap;

    uint64_t map_cells = (uint64_t)(max_w + 16) * (uint64_t)(max_h + 16);
    int ok =
        cudaMalloc(&c->d_map, map_cells) == cudaSuccess &&
        cudaMalloc(&c->d_results, (size_t)max_cap * sizeof(SrResult)) == cudaSuccess &&
        cudaMalloc(&c->d_count, sizeof(uint32_t)) == cudaSuccess &&
        cudaMalloc(&c->d_overflow, sizeof(int)) == cudaSuccess &&
        cudaMallocHost(&c->h_results, (size_t)max_cap * sizeof(SrResult)) == cudaSuccess &&
        cudaMallocHost(&c->h_count, sizeof(uint32_t)) == cudaSuccess &&
        cudaMallocHost(&c->h_overflow, sizeof(int)) == cudaSuccess &&
        cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking) == cudaSuccess;
    if (!ok) { sr_cuda_ctx_deinit(c); return 1; }
    *out = c;
    return 0;
}

int sr_cuda_ctx_deinit(SrCudaCtx *c) {
    if (!c) return 0;
    if (c->d_map) cudaFree(c->d_map);
    if (c->d_results) cudaFree(c->d_results);
    if (c->d_count) cudaFree(c->d_count);
    if (c->d_overflow) cudaFree(c->d_overflow);
    if (c->h_results) cudaFreeHost(c->h_results);
    if (c->h_count) cudaFreeHost(c->h_count);
    if (c->h_overflow) cudaFreeHost(c->h_overflow);
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
    return 0;
}

int sr_cuda_search_batch(SrCudaCtx *c, int64_t seed, uint8_t thr,
                         int32_t x0, int32_t z0, uint32_t w, uint32_t h,
                         SrResult *out, uint32_t cap, uint32_t *out_count) {
    if (!c || w == 0 || h == 0) { if (out_count) *out_count = 0; return 0; }
    if (w > c->max_w || h > c->max_h || cap > c->max_cap) return 1;

    cudaSetDevice(c->device);
    *c->h_count = 0; *c->h_overflow = 0;
    cudaMemcpyAsync(c->d_count, c->h_count, sizeof(uint32_t), cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->d_overflow, c->h_overflow, sizeof(int), cudaMemcpyHostToDevice, c->stream);

    /* kernel 1: 填图 */
    uint32_t map_w = w + 16, map_h = h + 16;
    int32_t  map_x0 = x0 - SR_OFFSET, map_z0 = z0 - SR_OFFSET;
    uint64_t map_total = (uint64_t)map_w * map_h;
    uint32_t mt = 256;
    uint32_t mb = (uint32_t)((map_total + mt - 1) / mt);
    slime_map_kernel<<<mb, mt, 0, c->stream>>>(seed, map_x0, map_z0, map_w, map_h, c->d_map);

    /* kernel 2: 搜索 */
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    search_kernel<<<grid, block, 0, c->stream>>>(c->d_map, map_w, thr, x0, z0, w, h,
                                                 c->d_results, cap, c->d_count, c->d_overflow);

    cudaMemcpyAsync(c->h_count, c->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, c->stream);
    cudaMemcpyAsync(c->h_overflow, c->d_overflow, sizeof(int), cudaMemcpyDeviceToHost, c->stream);
    cudaStreamSynchronize(c->stream);

    if (cudaGetLastError() != cudaSuccess) return 1;

    uint32_t found = *c->h_count;
    uint32_t copy_n = found < cap ? found : cap;
    if (copy_n > 0) {
        cudaMemcpyAsync(c->h_results, c->d_results, (size_t)copy_n * sizeof(SrResult),
                        cudaMemcpyDeviceToHost, c->stream);
        cudaStreamSynchronize(c->stream);
        memcpy(out, c->h_results, (size_t)copy_n * sizeof(SrResult));
    }
    if (out_count) *out_count = found;
    return (*c->h_overflow) ? 2 : 0;
}

} /* extern "C" */
