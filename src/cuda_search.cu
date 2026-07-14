/*
 * cuda_search.cu — CUDA 单卡搜索（阶段 5，热路径优化版）。
 *
 * 优化要点（均与 slime.h 逐位一致，regress_cuda.sh 三种子对拍红线）:
 *   1. 种子代数拆分：chunk_seed = (zbase(z) + xterm(x)) ^ 987234911，其中 xterm 只依赖 x、
 *      zbase 只依赖 z（含 seed）。用两个 1D kernel 预算 d_xterm[map_w] / d_zbase[map_h]，
 *      map kernel 每格只做一次加 + 一次异或 + 一步 LCG，消掉每线程 4×32 位乘 + 1×64 位乘。
 *      与 slime.h 的 sr_xterm / sr_zbase 同源（u64 加法结合/交换 + 末位异或，故逐位一致）。
 *   2. nextInt(10) 快路径：魔数除法 q=(u*0xCCCCCCCD)>>35 代替 % 10；u<REJECT(2147483640)
 *      时去偏必过、直接返回，避免无条件 for(;;)。仅 ~4e-9 的尾部落入精确 rejection 循环，
 *      循环体与原实现逐位一致。
 *   3. map kernel 2D launch，消掉每线程 idx%map_w / idx/map_w 的整数除模。
 *   4. search kernel 用 20 段甜甜圈 run（常量内存）代替 289 次逐格 d² 分支。
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
#define SR_XOR      987234911ULL
/* nextInt(10) 去偏阈值：bits ∈ [0,2^31) 中，bits>=REJECT 的 8 个值去偏必失败，需重抽。
 * 等价于 slime.h 的 (bits - bits%10 + 9) < 2^31 判定。 */
#define SR_NEXTINT_REJECT 2147483640u

/* ---- 设备端 32 位回绕乘（与 slime.h sr_mul32 一致） ---- */
__device__ __forceinline__ int32_t d_mul32(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a * (uint32_t)b);
}

/* xterm(x) = t1(x)+t2(x)，与 slime.h sr_xterm 逐位一致 */
__device__ __forceinline__ uint64_t d_xterm(int32_t x) {
    int32_t t1 = d_mul32(d_mul32(x, x), 4987142);
    int32_t t2 = d_mul32(x, 5947611);
    return (uint64_t)(int64_t)t1 + (uint64_t)(int64_t)t2;
}
/* zbase(seed,z) = seed + t3(z)+t4(z)，与 slime.h sr_zbase 逐位一致 */
__device__ __forceinline__ uint64_t d_zbase(int64_t world_seed, int32_t z) {
    int64_t t3 = (int64_t)d_mul32(z, z) * 4392871LL;
    int32_t t4 = d_mul32(z, 389711);
    return (uint64_t)world_seed + (uint64_t)t3 + (uint64_t)(int64_t)t4;
}

/* 由拆分后的 chunk_seed 判定 slime。快路径消 % 10 与无条件循环。
 * 与 slime.h sr_is_slime_from_cs 逐位一致（含精确去偏 rejection 尾部）。 */
__device__ __forceinline__ int d_is_slime_from_cs(uint64_t chunk_seed) {
    uint64_t s = (chunk_seed ^ SR_LCG_MUL) & SR_LCG_MASK;
    s = (s * SR_LCG_MUL + SR_LCG_ADD) & SR_LCG_MASK;      /* 首抽 */
    uint32_t u = (uint32_t)(s >> 17);                     /* bits ∈ [0,2^31) */
    if (u < SR_NEXTINT_REJECT) {                          /* 去偏必过：魔数除法判 %10==0 */
        uint32_t q = (uint32_t)(((uint64_t)u * 0xCCCCCCCDULL) >> 35);
        return (u - q * 10u) == 0u;
    }
    /* 罕见尾部（~4e-9）：精确去偏 rejection，逐位复刻 slime.h 循环 */
    for (;;) {
        int32_t bits = (int32_t)(s >> 17);
        int32_t val  = bits % 10;
        if (((uint32_t)bits - (uint32_t)val + 9u) < 0x80000000u)
            return val == 0;
        s = (s * SR_LCG_MUL + SR_LCG_ADD) & SR_LCG_MASK;
    }
}

/* ---- 甜甜圈 20 段 run（常量内存）：与 slime.h sr_build_donut_runs 同源 ----
 * 每段 (drow, c1, c2)：窗口行 drow∈[0,17)，列闭区间 [c1,c2]∈[0,17)。
 * 原逐格实现映射 tile[tz+col][tx+drow]，即固定 map_x=bx+tx+drow、变 map_z=bz+tz+col。
 * 故 run 是 map 空间里的一条竖直段（沿 z）。用沿 z 的列前缀和把每段降到 2 次查表。 */
struct DRun { int32_t drow, c1, c2; };
__constant__ DRun c_runs[20];
__constant__ int  c_nruns;

/* ---- kernel A: 预算 x 项表（每列一次） ---- */
__global__ void xterm_kernel(int32_t map_x0, uint32_t map_w, uint64_t *xt) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < map_w) xt[i] = d_xterm(map_x0 + (int32_t)i);
}
/* ---- kernel B: 预算 z 项表（每行一次，含 seed） ---- */
__global__ void zbase_kernel(int64_t seed, int32_t map_z0, uint32_t map_h, uint64_t *zb) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < map_h) zb[j] = d_zbase(seed, map_z0 + (int32_t)j);
}

/* ---- kernel 1: 融合 (slime 判定 + 沿 z 列前缀和)，消掉中间 map 写回+读取 ----
 * 每线程负责一列 map_x，串行走 map_z 累加。
 *   ps[(z+1)*map_w + x] = Σ_{k<=z} slime(x,k)   （独占前缀，ps 行数 = map_h+1）
 * 相邻线程 = 相邻 x = 相邻内存，读 zb[z]（全 warp 同 z，广播）、写 ps 均合并访存。
 * 判定 cs 与 slime.h 逐位一致。 */
__global__ void colprefix_kernel(const uint64_t * __restrict__ xt,
                                 const uint64_t * __restrict__ zb,
                                 uint32_t map_w, uint32_t map_h,
                                 uint16_t * __restrict__ ps) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= map_w) return;
    uint64_t xterm = xt[x];
    uint32_t acc = 0;                            /* map_h<=16400，列前缀和 <65535，uint16 足够 */
    ps[x] = 0;                                   /* ps[0][x] = 0 */
    for (uint32_t z = 0; z < map_h; ++z) {
        uint64_t cs = (zb[z] + xterm) ^ SR_XOR;
        acc += (uint32_t)d_is_slime_from_cs(cs);
        ps[(uint64_t)(z + 1) * map_w + x] = (uint16_t)acc; /* ps[z+1][x] */
    }
}

/* ---- kernel 2: 每中心数甜甜圈（列前缀和差分，20 段 → 40 次查表） ---- */
__global__ void search_kernel(const uint16_t * __restrict__ ps, uint32_t map_w,
                              uint8_t thr, int32_t x0, int32_t z0,
                              uint32_t width, uint32_t height,
                              SrResult *out, uint32_t cap,
                              uint32_t *out_count, int *overflow) {
    const uint32_t rel_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rel_z = blockIdx.y * blockDim.y + threadIdx.y;
    if (rel_x >= width || rel_z >= height) return;

    /* run 段沿 z：固定 map_x = rel_x + drow，z ∈ [rel_z+c1, rel_z+c2]。
     * Σ = ps[(rel_z+c2+1)][mapx] - ps[(rel_z+c1)][mapx]  （独占前缀差分）。 */
    uint32_t count = 0;
    const int nr = c_nruns;
    #pragma unroll
    for (int r = 0; r < 20; ++r) {
        if (r >= nr) break;
        uint32_t mapx = rel_x + (uint32_t)c_runs[r].drow;
        uint32_t lo   = rel_z + (uint32_t)c_runs[r].c1;
        uint32_t hi1  = rel_z + (uint32_t)c_runs[r].c2 + 1u;
        count += (uint32_t)(ps[(uint64_t)hi1 * map_w + mapx]
                          - ps[(uint64_t)lo  * map_w + mapx]);
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
    uint16_t *d_ps;        /* 列前缀和 [(max_h+16+1) * (max_w+16)] uint16（列和<65535） */
    uint64_t *d_xterm;     /* [max_w + 16] */
    uint64_t *d_zbase;     /* [max_h + 16] */
    SrResult *d_results;
    uint32_t *d_count;
    int      *d_overflow;
    SrResult *h_results;   /* pinned */
    uint32_t *h_count;     /* pinned */
    int      *h_overflow;  /* pinned */
    cudaStream_t stream;
    int       runs_uploaded;
};

/* 主机侧构造 20 段甜甜圈 run（与 slime.h sr_build_donut_runs 逻辑一致），上传常量内存。 */
static int upload_donut_runs(void) {
    DRun runs[20];
    int n = 0;
    for (int row = 0; row < SR_WINDOW; ++row) {
        int dx = row - SR_OFFSET;
        int in_run = 0, start = 0;
        for (int col = 0; col < SR_WINDOW; ++col) {
            int dz = col - SR_OFFSET;
            int d2 = dx * dx + dz * dz;
            int inside = (d2 > 1 && d2 <= 64);
            if (inside && !in_run) { start = col; in_run = 1; }
            else if (!inside && in_run) {
                runs[n].drow = row; runs[n].c1 = start; runs[n].c2 = col - 1; ++n;
                in_run = 0;
            }
        }
        if (in_run) { runs[n].drow = row; runs[n].c1 = start; runs[n].c2 = SR_WINDOW - 1; ++n; }
    }
    if (n != 20) return 1;
    if (cudaMemcpyToSymbol(c_runs, runs, sizeof(runs)) != cudaSuccess) return 1;
    if (cudaMemcpyToSymbol(c_nruns, &n, sizeof(n)) != cudaSuccess) return 1;
    return 0;
}

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

    /* 前缀和多一行（独占前缀），列宽 = max_w+16。 */
    uint64_t ps_cells = (uint64_t)(max_w + 16) * (uint64_t)(max_h + 16 + 1);
    int ok =
        cudaMalloc(&c->d_ps, ps_cells * sizeof(uint16_t)) == cudaSuccess &&
        cudaMalloc(&c->d_xterm, (size_t)(max_w + 16) * sizeof(uint64_t)) == cudaSuccess &&
        cudaMalloc(&c->d_zbase, (size_t)(max_h + 16) * sizeof(uint64_t)) == cudaSuccess &&
        cudaMalloc(&c->d_results, (size_t)max_cap * sizeof(SrResult)) == cudaSuccess &&
        cudaMalloc(&c->d_count, sizeof(uint32_t)) == cudaSuccess &&
        cudaMalloc(&c->d_overflow, sizeof(int)) == cudaSuccess &&
        cudaMallocHost(&c->h_results, (size_t)max_cap * sizeof(SrResult)) == cudaSuccess &&
        cudaMallocHost(&c->h_count, sizeof(uint32_t)) == cudaSuccess &&
        cudaMallocHost(&c->h_overflow, sizeof(int)) == cudaSuccess &&
        cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking) == cudaSuccess;
    if (!ok) { sr_cuda_ctx_deinit(c); return 1; }
    if (upload_donut_runs() != 0) { sr_cuda_ctx_deinit(c); return 1; }
    c->runs_uploaded = 1;
    *out = c;
    return 0;
}

int sr_cuda_ctx_deinit(SrCudaCtx *c) {
    if (!c) return 0;
    if (c->d_ps) cudaFree(c->d_ps);
    if (c->d_xterm) cudaFree(c->d_xterm);
    if (c->d_zbase) cudaFree(c->d_zbase);
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

    uint32_t map_w = w + 16, map_h = h + 16;
    int32_t  map_x0 = x0 - SR_OFFSET, map_z0 = z0 - SR_OFFSET;

    /* 预算 x/z 项表（O(map_w+map_h)，相对 map_w*map_h 可忽略） */
    xterm_kernel<<<(map_w + 255) / 256, 256, 0, c->stream>>>(map_x0, map_w, c->d_xterm);
    zbase_kernel<<<(map_h + 255) / 256, 256, 0, c->stream>>>(seed, map_z0, map_h, c->d_zbase);

    /* kernel 1: 融合判定 + 沿 z 列前缀和（每线程一列 map_x） */
    colprefix_kernel<<<(map_w + 255) / 256, 256, 0, c->stream>>>(
        c->d_xterm, c->d_zbase, map_w, map_h, c->d_ps);

    /* kernel 2: 搜索（列前缀和差分数甜甜圈） */
    dim3 block(32, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    search_kernel<<<grid, block, 0, c->stream>>>(c->d_ps, map_w, thr, x0, z0, w, h,
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
