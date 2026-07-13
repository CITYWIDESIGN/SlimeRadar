/*
 * bench_opt.c — 验证 "x 项预计算表" 优化：
 *   chunk_seed = (seed + t1(x)+t2(x) + t3(z)+t4(z)) ^ 987234911
 *   t1/t2 只依赖 x → 每 block 预计算 512 个 xterm，跨 511 行复用。
 *   t3/t4 只依赖 z → 每行预计算一次 zbase。
 * 对拍逐格 == 基线 sr_is_slime，并对比建 SAT 耗时。
 *
 * 编译: gcc -std=c11 -O3 -march=native -funroll-loops bench_opt.c -o bench_opt.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>
#include "src/slime.h"

#define SIZE   512
#define SATDIM (SIZE + 1)
#define SAT_AT(sat, r, c) ((sat)[(size_t)(r) * SATDIM + (size_t)(c)])

static double now_ms(LARGE_INTEGER freq) {
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return 1000.0 * (double)t.QuadPart / (double)freq.QuadPart;
}

/* 由预计算的 chunk_seed 直接判定（跳过 sr_chunk_seed 的 6 次乘法）。 */
static inline int is_slime_from_cs(int64_t cs) {
    uint64_t s = ((uint64_t)cs ^ SR_LCG_MUL) & SR_LCG_MASK;
    for (;;) {
        s = (s * SR_LCG_MUL + SR_LCG_ADD) & SR_LCG_MASK;
        int32_t bits = (int32_t)(s >> (48 - 31));
        int32_t val  = bits % 10;
        if (((uint32_t)bits - (uint32_t)val + 9u) < 0x80000000u)
            return val == 0;
    }
}

/* x 项：t1(x)+t2(x)，符号扩展成 uint64。 */
static inline uint64_t xterm_of(int32_t x) {
    int32_t t1 = sr_mul32(sr_mul32(x, x), 4987142);
    int32_t t2 = sr_mul32(x, 5947611);
    return (uint64_t)(int64_t)t1 + (uint64_t)(int64_t)t2;
}
/* z 项：seed + t3(z)+t4(z)。 */
static inline uint64_t zbase_of(int64_t seed, int32_t z) {
    int64_t t3 = (int64_t)sr_mul32(z, z) * 4392871LL;
    int32_t t4 = sr_mul32(z, 389711);
    return (uint64_t)seed + (uint64_t)t3 + (uint64_t)(int64_t)t4;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s SEED [NBLOCKS]\n", argv[0]); return 1; }
    int64_t seed = (int64_t)strtoll(argv[1], NULL, 10);
    int nblocks  = (argc >= 3) ? atoi(argv[2]) : 256;

    uint16_t *sat_a = (uint16_t *)malloc((size_t)SATDIM * SATDIM * sizeof(uint16_t));
    uint16_t *sat_b = (uint16_t *)malloc((size_t)SATDIM * SATDIM * sizeof(uint16_t));
    uint64_t *xterm = (uint64_t *)malloc((size_t)SIZE * sizeof(uint64_t));
    if (!sat_a || !sat_b || !xterm) return 2;

    LARGE_INTEGER freq; QueryPerformanceFrequency(&freq);
    double t_base = 0, t_opt = 0;
    volatile uint64_t sink = 0;
    uint64_t mismatch = 0;

    for (int b = 0; b < nblocks; ++b) {
        int32_t bx0 = (int32_t)((int64_t)b * (SIZE-16) - 100000);
        int32_t bz0 = (int32_t)((int64_t)(b * 7) * (SIZE-16) + 50000);
        int32_t sx0 = bx0 - SR_OFFSET, sz0 = bz0 - SR_OFFSET;

        /* --- 基线：逐格 sr_is_slime --- */
        double a0 = now_ms(freq);
        for (int c = 0; c < SATDIM; ++c) SAT_AT(sat_a, 0, c) = 0;
        for (int r = 1; r < SATDIM; ++r) SAT_AT(sat_a, r, 0) = 0;
        for (int j = 0; j < SIZE; ++j) {
            int32_t z = sz0 + j;
            uint16_t run = 0;
            for (int i = 0; i < SIZE; ++i) {
                run = (uint16_t)(run + (uint16_t)sr_is_slime(seed, sx0 + i, z));
                SAT_AT(sat_a, j + 1, i + 1) = (uint16_t)(SAT_AT(sat_a, j, i + 1) + run);
            }
        }
        double a1 = now_ms(freq);

        /* --- 优化：x 项预表 + 每行 zbase --- */
        for (int c = 0; c < SATDIM; ++c) SAT_AT(sat_b, 0, c) = 0;
        for (int r = 1; r < SATDIM; ++r) SAT_AT(sat_b, r, 0) = 0;
        for (int i = 0; i < SIZE; ++i) xterm[i] = xterm_of(sx0 + i);
        for (int j = 0; j < SIZE; ++j) {
            uint64_t zbase = zbase_of(seed, sz0 + j);
            uint16_t run = 0;
            for (int i = 0; i < SIZE; ++i) {
                int64_t cs = (int64_t)((zbase + xterm[i]) ^ (uint64_t)987234911LL);
                run = (uint16_t)(run + (uint16_t)is_slime_from_cs(cs));
                SAT_AT(sat_b, j + 1, i + 1) = (uint16_t)(SAT_AT(sat_b, j, i + 1) + run);
            }
        }
        double a2 = now_ms(freq);

        /* 逐格对拍整张 SAT */
        for (int r = 0; r < SATDIM; ++r)
            for (int c = 0; c < SATDIM; ++c)
                if (SAT_AT(sat_a, r, c) != SAT_AT(sat_b, r, c)) ++mismatch;

        sink += SAT_AT(sat_a, SIZE, SIZE);
        t_base += (a1 - a0);
        t_opt  += (a2 - a1);
    }

    printf("blocks=%d seed=%lld (sink=%llu)\n", nblocks, (long long)seed, (unsigned long long)sink);
    printf("SAT mismatch cells : %llu  %s\n", (unsigned long long)mismatch,
           mismatch ? "*** FAIL ***" : "OK (bit-identical)");
    printf("build SAT baseline : %8.2f ms\n", t_base);
    printf("build SAT optimized: %8.2f ms  (%.2fx)\n", t_opt, t_base / t_opt);
    free(sat_a); free(sat_b); free(xterm);
    return 0;
}
