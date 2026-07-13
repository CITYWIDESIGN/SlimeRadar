/*
 * main.c — 正式 CLI（阶段 4）
 *
 * 用法: slimerander [OPTIONS] SEED RANGE [THRESHOLD]
 *   -m cpu|cuda   模式（默认 cpu）
 *   -j THREADS     CPU 线程数（0/缺省=自动）
 *   -k CARDS       最多显卡数（0=全部）
 *   -f human|csv   输出格式（默认 human）
 *   -u             关闭排序
 *   -q             关闭进度
 *   -h / -v        帮助 / 版本
 *
 * 区域 [-RANGE,RANGE)²（区块坐标半开区间）；x0>x1 / z0>z1 自动交换。
 * 结果收集: 回调只 append 进缓冲；搜完统一 qsort（count↓ → d²↑ → x↑ → z↑）。
 * human: "(%5d, %5d)   %d"；csv: "x,z,count"。进度打到 stderr。
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
  #define _POSIX_C_SOURCE 199309L   /* clock_gettime / CLOCK_MONOTONIC (Linux) */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "slimerander.h"

/* ---- 单调墙钟毫秒（进度条已用/剩余时间），跨平台。 ---- */
#if defined(_WIN32)
  #include <windows.h>
  static uint64_t sr_now_ms(void) { return (uint64_t)GetTickCount64(); }
#else
  #include <time.h>
  static uint64_t sr_now_ms(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
  }
#endif

#define SR_VERSION "slimerander 0.4 (stage 4)"

/* ---- 结果收集缓冲 ---- */
typedef struct {
    SrResult *buf;
    size_t    len, cap;
} ResultVec;

static void vec_push(ResultVec *v, SrResult r) {
    if (v->len == v->cap) {
        size_t ncap = v->cap ? v->cap * 2 : 4096;
        SrResult *nb = (SrResult *)realloc(v->buf, ncap * sizeof(SrResult));
        if (!nb) { fprintf(stderr, "OOM collecting results\n"); exit(2); }
        v->buf = nb; v->cap = ncap;
    }
    v->buf[v->len++] = r;
}

/* ---- 进度（stderr） ---- */
typedef struct {
    int enabled;
    uint64_t start_ms;  /* 首次回调时的墙钟起点（sr_now_ms） */
    int started;
} ProgressCtx;

/* 秒数格式化为 mm:ss（<1h）或 h:mm:ss。写入 buf。 */
static void fmt_hms(double sec, char *buf, size_t n) {
    if (sec < 0) sec = 0;
    unsigned long t = (unsigned long)(sec + 0.5);
    unsigned h = (unsigned)(t / 3600), m = (unsigned)((t % 3600) / 60), s = (unsigned)(t % 60);
    if (h) snprintf(buf, n, "%u:%02u:%02u", h, m, s);
    else   snprintf(buf, n, "%02u:%02u", m, s);
}

static void on_progress(void *ctx, uint64_t done, uint64_t total) {
    ProgressCtx *pc = (ProgressCtx *)ctx;
    if (!pc->enabled || total == 0) return;
    if (!pc->started) { pc->start_ms = sr_now_ms(); pc->started = 1; }

    double elapsed = (double)(sr_now_ms() - pc->start_ms) / 1000.0;
    double frac = (double)done / (double)total;

    char eb[24], rb[24];
    fmt_hms(elapsed, eb, sizeof eb);
    if (done > 0 && done < total) {
        /* 按当前速率外推：剩余 = 已用 * (剩余量 / 已完成量)。 */
        double remain = elapsed * (double)(total - done) / (double)done;
        fmt_hms(remain, rb, sizeof rb);
    } else {
        snprintf(rb, sizeof rb, "--:--");
    }

    /* 标签用英文避免 UTF-8 源码在 GBK 控制台下乱码；中文汇总留给 run.bat。 */
    fprintf(stderr, "\r[%5.1f%%] %llu/%llu  elapsed %s  ETA %s   ",
            100.0 * frac,
            (unsigned long long)done, (unsigned long long)total,
            eb, rb);
    if (done == total) fprintf(stderr, "\n");
    fflush(stderr);
}

/* 搜索期间同时用作 result + progress 的 ctx */
typedef struct { ResultVec vec; ProgressCtx prog; } SearchCtx;
static void cb_result(void *ctx, SrResult r)   { vec_push(&((SearchCtx*)ctx)->vec, r); }
static void cb_progress(void *ctx, uint64_t d, uint64_t t) {
    on_progress(&((SearchCtx*)ctx)->prog, d, t);
}

/* ---- 四级排序键: count↓ → d²↑ → x↑ → z↑ ---- */
static int cmp_result(const void *pa, const void *pb) {
    const SrResult *a = (const SrResult *)pa;
    const SrResult *b = (const SrResult *)pb;
    if (a->count != b->count) return (a->count > b->count) ? -1 : 1;  /* 降序 */
    int64_t da = (int64_t)a->x * a->x + (int64_t)a->z * a->z;
    int64_t db = (int64_t)b->x * b->x + (int64_t)b->z * b->z;
    if (da != db) return (da < db) ? -1 : 1;                          /* 升序 */
    if (a->x != b->x) return (a->x < b->x) ? -1 : 1;
    if (a->z != b->z) return (a->z < b->z) ? -1 : 1;
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [OPTIONS] SEED RANGE [THRESHOLD]\n"
        "  -m cpu|cuda    mode (default cpu)\n"
        "  -j THREADS     CPU threads (0=auto)\n"
        "  -k CARDS       max GPUs (0=all)\n"
        "  -f human|csv   output format (default human)\n"
        "  -u             disable sort\n"
        "  -q             disable progress\n"
        "  -b             benchmark mode (measure hardware/scan throughput, no result list)\n"
        "  -h / -v        help / version\n", prog);
}

int main(int argc, char **argv) {
    const char *mode = "cpu";
    const char *fmt  = "human";
    unsigned threads = 0, cards = 0;
    int sort = 1, progress = 1, bench = 0;

    /* 手写选项解析。带值选项 -x 支持两种写法: "-x VALUE" 与紧贴 "-xVALUE"。 */
    const char *pos[3]; int npos = 0;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] && !(a[1] >= '0' && a[1] <= '9')) {
            /* 取带值选项的值：优先紧贴 a[2..]，否则下一个 argv。 */
            #define SR_OPTVAL() (a[2] ? &a[2] : (++i < argc ? argv[i] : NULL))
            const char *val;
            switch (a[1]) {
                case 'm': val = SR_OPTVAL(); if (!val) { usage(argv[0]); return 1; } mode = val; break;
                case 'j': val = SR_OPTVAL(); if (!val) { usage(argv[0]); return 1; } threads = (unsigned)strtoul(val, NULL, 10); break;
                case 'k': val = SR_OPTVAL(); if (!val) { usage(argv[0]); return 1; } cards = (unsigned)strtoul(val, NULL, 10); break;
                case 'f': val = SR_OPTVAL(); if (!val) { usage(argv[0]); return 1; } fmt = val; break;
                case 'u': sort = 0; break;
                case 'q': progress = 0; break;
                case 'b': bench = 1; break;
                case 'h': usage(argv[0]); return 0;
                case 'v': printf("%s\n", SR_VERSION); return 0;
                default: fprintf(stderr, "unknown option: %s\n", a); usage(argv[0]); return 1;
            }
        } else {
            if (npos < 3) pos[npos++] = a;
            else { fprintf(stderr, "too many arguments\n"); usage(argv[0]); return 1; }
        }
    }
    if (npos < 2) { usage(argv[0]); return 1; }

    int64_t seed  = (int64_t)strtoll(pos[0], NULL, 10);
    int64_t range = (int64_t)strtoll(pos[1], NULL, 10);
    unsigned thr  = (npos >= 3) ? (unsigned)strtoul(pos[2], NULL, 10) : 45u;
    if (range <= 0) { fprintf(stderr, "RANGE must be > 0\n"); return 1; }
    if (thr > 255)  { fprintf(stderr, "THRESHOLD must be 0..255\n"); return 1; }

    SrParams p;
    p.world_seed = seed;
    p.threshold  = (uint8_t)thr;
    p.x0 = (int32_t)(-range); p.x1 = (int32_t)range;
    p.z0 = (int32_t)(-range); p.z1 = (int32_t)range;
    /* x0>x1/z0>z1 自动交换（此处 -range<range 恒成立，保留通用逻辑） */
    if (p.x0 > p.x1) { int32_t t = p.x0; p.x0 = p.x1; p.x1 = t; }
    if (p.z0 > p.z1) { int32_t t = p.z0; p.z0 = p.z1; p.z1 = t; }

    SearchCtx sc;
    sc.vec.buf = NULL; sc.vec.len = 0; sc.vec.cap = 0;
    sc.prog.enabled = progress;
    sc.prog.started = 0;
    sc.prog.start_ms = 0;

    /* bench 模式：关进度、静默计时，测扫描吞吐。 */
    if (bench) { sc.prog.enabled = 0; progress = 0; }

    uint64_t t_start = sr_now_ms();
    int rc;
    if (strcmp(mode, "cpu") == 0) {
        rc = sr_search_cpu(&p, threads, &sc, cb_result, cb_progress);
    } else if (strcmp(mode, "cuda") == 0) {
        rc = sr_search_cuda(&p, cards, &sc, cb_result, cb_progress);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode); return 1;
    }
    uint64_t t_end = sr_now_ms();
    if (rc != SR_OK) {
        fprintf(stderr, "search error %d\n", rc);
        free(sc.vec.buf);
        return rc;
    }

    if (bench) {
        /* 工作量换算：TESTED=496 个候选中心/块边，块内 512×512 个 slime 判定。 */
        const int64_t TESTED = 512 - 17 + 1;         /* 496，与 cpu_search 一致 */
        int64_t width  = (int64_t)p.x1 - p.x0;
        int64_t height = (int64_t)p.z1 - p.z0;
        int64_t bx = (width  + TESTED - 1) / TESTED;
        int64_t bz = (height + TESTED - 1) / TESTED;
        int64_t blocks = bx * bz;
        int64_t cand   = width * height;            /* 候选中心点数（=搜索区域格子数） */
        int64_t judged = blocks * 512LL * 512LL;    /* slime 判定总次数（含块间 16 格重叠） */
        double  sec    = (double)(t_end - t_start) / 1000.0;
        if (sec <= 0) sec = 1e-9;

        const char *thr_desc;
        char thr_buf[32];
        if (strcmp(mode, "cuda") == 0) { thr_desc = "GPU"; }
        else if (threads == 0) { snprintf(thr_buf, sizeof thr_buf, "CPU auto-threads"); thr_desc = thr_buf; }
        else { snprintf(thr_buf, sizeof thr_buf, "CPU %u threads", threads); thr_desc = thr_buf; }

        printf("=== SlimeRadar benchmark ===\n");
        printf("mode          : %s (%s)\n", mode, thr_desc);
        printf("seed / range  : %lld / %lld  (threshold %u)\n",
               (long long)seed, (long long)range, thr);
        printf("area          : %lld x %lld chunks  (%lld candidate centers)\n",
               (long long)width, (long long)height, (long long)cand);
        printf("blocks        : %lld  (%lld slime tests, incl. 16-chunk overlap)\n",
               (long long)blocks, (long long)judged);
        printf("hits          : %zu\n", sc.vec.len);
        printf("elapsed       : %.3f s\n", sec);
        printf("candidates/s  : %.2f M/s\n", (double)cand   / sec / 1e6);
        printf("slime tests/s : %.2f M/s\n", (double)judged / sec / 1e6);
        fflush(stdout);
        free(sc.vec.buf);
        return 0;
    }

    if (sort) qsort(sc.vec.buf, sc.vec.len, sizeof(SrResult), cmp_result);

    int csv = (strcmp(fmt, "csv") == 0);
    for (size_t i = 0; i < sc.vec.len; ++i) {
        SrResult r = sc.vec.buf[i];
        if (csv) printf("%d,%d,%u\n", r.x, r.z, r.count);
        else     printf("(%5d, %5d)   %u\n", r.x, r.z, r.count);
    }

    fflush(stdout);
    free(sc.vec.buf);
    return 0;
}
