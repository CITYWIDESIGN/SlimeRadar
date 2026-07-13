/*
 * bench_phases.c — 量化 CPU 单 block 内 "建 SAT" vs "扫描候选点" 的时间占比，
 * 用于判断 SIMD（只加速判定+建 SAT 那趟）值不值得做。
 *
 * 复用 slime.h 的判定与甜甜圈段，逻辑与 cpu_search.c 的 search_one_block 一致。
 * 对多个不同位置的 block 各跑一次，分别累计两段耗时。单线程、标量。
 *
 * 编译: gcc -std=c11 -O3 -funroll-loops bench_phases.c -o bench_phases.exe
 * 用法: bench_phases SEED [NBLOCKS]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>
#include "src/slime.h"

#define SIZE   512
#define TESTED (SIZE - SR_WINDOW + 1)
#define SATDIM (SIZE + 1)
#define SAT_AT(sat, r, c) ((sat)[(size_t)(r) * SATDIM + (size_t)(c)])

static double now_ms(LARGE_INTEGER freq) {
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return 1000.0 * (double)t.QuadPart / (double)freq.QuadPart;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s SEED [NBLOCKS]\n", argv[0]); return 1; }
    int64_t seed = (int64_t)strtoll(argv[1], NULL, 10);
    int nblocks  = (argc >= 3) ? atoi(argv[2]) : 64;
    unsigned thr = 45;

    SrDonutRun runs[SR_DONUT_RUNS];
    sr_build_donut_runs(runs);

    uint16_t *sat = (uint16_t *)malloc((size_t)SATDIM * SATDIM * sizeof(uint16_t));
    if (!sat) return 2;

    LARGE_INTEGER freq; QueryPerformanceFrequency(&freq);
    double t_sat = 0, t_scan = 0;
    volatile uint64_t sink = 0;   /* 防止扫描被优化掉 */

    for (int b = 0; b < nblocks; ++b) {
        /* 让各 block 落在不同区域，模拟真实分布 */
        int32_t bx0 = (int32_t)((int64_t)b * TESTED - 100000);
        int32_t bz0 = (int32_t)((int64_t)(b * 7) * TESTED + 50000);
        int32_t sx0 = bx0 - SR_OFFSET, sz0 = bz0 - SR_OFFSET;

        /* --- 段 1: 建 SAT --- */
        double a0 = now_ms(freq);
        for (int c = 0; c < SATDIM; ++c) SAT_AT(sat, 0, c) = 0;
        for (int r = 1; r < SATDIM; ++r) SAT_AT(sat, r, 0) = 0;
        for (int j = 0; j < SIZE; ++j) {
            int32_t z = sz0 + j;
            uint16_t run = 0;
            for (int i = 0; i < SIZE; ++i) {
                int32_t x = sx0 + i;
                run = (uint16_t)(run + (uint16_t)sr_is_slime(seed, x, z));
                SAT_AT(sat, j + 1, i + 1) = (uint16_t)(SAT_AT(sat, j, i + 1) + run);
            }
        }
        double a1 = now_ms(freq);

        /* --- 段 2: 扫描候选点（预筛 + 精确） --- */
        uint64_t hits = 0;
        for (int32_t kz = 0; kz < TESTED; ++kz) {
            for (int32_t kx = 0; kx < TESTED; ++kx) {
                uint16_t box = (uint16_t)(
                      SAT_AT(sat, kz + SR_WINDOW, kx + SR_WINDOW)
                    - SAT_AT(sat, kz,             kx + SR_WINDOW)
                    - SAT_AT(sat, kz + SR_WINDOW, kx)
                    + SAT_AT(sat, kz,             kx));
                if (box < thr) continue;
                uint16_t exact = 0;
                for (int s = 0; s < SR_DONUT_RUNS; ++s) {
                    int rr = kz + runs[s].dx;
                    int c1 = kx + runs[s].c1;
                    int c2 = kx + runs[s].c2 + 1;
                    exact = (uint16_t)(exact
                        + SAT_AT(sat, rr + 1, c2) - SAT_AT(sat, rr, c2)
                        - SAT_AT(sat, rr + 1, c1) + SAT_AT(sat, rr, c1));
                }
                if (exact >= thr) ++hits;
            }
        }
        double a2 = now_ms(freq);
        sink += hits;

        t_sat  += (a1 - a0);
        t_scan += (a2 - a1);
    }

    double total = t_sat + t_scan;
    printf("blocks=%d  seed=%lld  (sink=%llu)\n", nblocks, (long long)seed, (unsigned long long)sink);
    printf("build SAT : %8.2f ms  (%5.1f%%)  <- SIMD 只能加速这段\n", t_sat,  100.0 * t_sat  / total);
    printf("scan pts  : %8.2f ms  (%5.1f%%)\n", t_scan, 100.0 * t_scan / total);
    printf("total     : %8.2f ms\n", total);
    printf("\nSIMD 上限估算: 若判定占建SAT的大头且 8x 向量化，最好情况整体约省 %.1f%%\n",
           100.0 * t_sat / total * (7.0/8.0));
    free(sat);
    return 0;
}
