/*
 * cuda_host.c — CUDA 模式 host 调度（纯 C）。阶段 5/6: 多卡按算力分行 + 按行分批。
 * 替换 cuda_stub.c（仅 SLIMY_ENABLE_CUDA 构建时链接本文件）。
 *
 * 多卡: 发现设备并按 score 降序，按 score 占比把 z 方向的行切给各卡；
 *       每卡一个 host 线程独立跑批（上下文复用）。结果先各卡收进独立缓冲，
 *       主线程 join 后串行回调（避免并发回调的线程安全问题）。
 * 分批: 每卡任务内沿 z 按行切，chunk_rows = min(剩余, 512MB/bytes_per_row, 16384)。
 *       每批中心区域 [x0,x1) × [z0+cursor, z0+cursor+rows)，kernel 输出即绝对坐标。
 * 单卡时自然退化为单卡跑全部行。
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "slimerander.h"
#include "cuda_search.h"

/* ---- 平台线程抽象：Windows 走 Win32，其余(Linux) 走 pthreads。 ---- */
#if defined(_WIN32)
  #include <windows.h>
  typedef HANDLE          sr_thread_t;
  #define SR_THREAD_RET   DWORD WINAPI
  #define SR_THREAD_PARAM LPVOID
#else
  #include <pthread.h>
  typedef pthread_t       sr_thread_t;
  #define SR_THREAD_RET   void *
  #define SR_THREAD_PARAM void *
#endif

#define SR_CUDA_MAX_ROWS   16384u
#define SR_CUDA_BYTES_CAP  (512u * 1024u * 1024u)
#define SR_CUDA_MIN_CAP    (1u << 16)
#define SR_CUDA_MAX_DEV    64

typedef struct {
    int      device;
    int64_t  seed;
    uint8_t  thr;
    int32_t  x0;
    uint32_t width;
    int32_t  z_base;      /* 该卡起始行对应的绝对 z0 */
    int64_t  rows;        /* 该卡负责的行数 */
    /* 输出（线程内收集，主线程 join 后回调） */
    SrResult *buf;
    size_t    len, cap;
    int       err;        /* 0 ok，非 0 错误码 */
    int       overflow;
} DevTask;

static void task_push(DevTask *t, SrResult r) {
    if (t->len == t->cap) {
        size_t nc = t->cap ? t->cap * 2 : 65536;
        SrResult *nb = (SrResult *)realloc(t->buf, nc * sizeof(SrResult));
        if (!nb) { t->err = SR_ERR_ALLOC; return; }
        t->buf = nb; t->cap = nc;
    }
    t->buf[t->len++] = r;
}

static SR_THREAD_RET dev_worker(SR_THREAD_PARAM param) {
    DevTask *t = (DevTask *)param;
    if (t->rows <= 0) return 0;

    uint32_t bytes_per_row = t->width * (uint32_t)sizeof(SrResult);
    if (bytes_per_row == 0) bytes_per_row = 1;
    uint32_t chunk_rows = SR_CUDA_BYTES_CAP / bytes_per_row;
    if (chunk_rows > SR_CUDA_MAX_ROWS) chunk_rows = SR_CUDA_MAX_ROWS;
    if (chunk_rows == 0) chunk_rows = 1;
    if ((int64_t)chunk_rows > t->rows) chunk_rows = (uint32_t)t->rows;

    uint32_t cap = t->width * chunk_rows;
    if (cap < SR_CUDA_MIN_CAP) cap = SR_CUDA_MIN_CAP;

    SrCudaCtx *cc = NULL;
    if (sr_cuda_ctx_init(t->device, t->width, chunk_rows, cap, &cc) != 0) {
        t->err = SR_ERR_ALLOC; return 0;   /* 错误经 t->err 传递，返回值不被检查 */
    }
    SrResult *out = (SrResult *)malloc((size_t)cap * sizeof(SrResult));
    if (!out) { sr_cuda_ctx_deinit(cc); t->err = SR_ERR_ALLOC; return 0; }

    for (int64_t cursor = 0; cursor < t->rows; cursor += chunk_rows) {
        uint32_t rows = ((t->rows - cursor) < chunk_rows) ? (uint32_t)(t->rows - cursor) : chunk_rows;
        int32_t bz0 = (int32_t)(t->z_base + cursor);
        uint32_t got = 0;
        int br = sr_cuda_search_batch(cc, t->seed, t->thr, t->x0, bz0,
                                      t->width, rows, out, cap, &got);
        if (br == 2) t->overflow = 1;
        else if (br != 0) { t->err = SR_ERR_ALLOC; break; }
        uint32_t emit = got < cap ? got : cap;
        for (uint32_t i = 0; i < emit; ++i) {
            task_push(t, out[i]);
            if (t->err) break;
        }
        if (t->err) break;
    }

    free(out);
    sr_cuda_ctx_deinit(cc);
    return 0;
}

int sr_search_cuda(const SrParams *p, unsigned max_cards,
                   void *ctx, sr_result_fn on_result, sr_progress_fn on_progress) {
    if (!p || !on_result) return SR_ERR_PARAM;
    if (p->x1 <= p->x0 || p->z1 <= p->z0) return SR_OK;

    int ndev = sr_cuda_device_count();
    if (ndev <= 0) return SR_ERR_CUDA_UNSUPPORTED;
    if (ndev > SR_CUDA_MAX_DEV) ndev = SR_CUDA_MAX_DEV;

    /* 收集各卡 score，降序排序（简单选择排序，卡数少）。 */
    int   dev_idx[SR_CUDA_MAX_DEV];
    float dev_score[SR_CUDA_MAX_DEV];
    int   nvalid = 0;
    for (int i = 0; i < ndev; ++i) {
        float sc; uint64_t fm, tm;
        if (sr_cuda_device_score(i, &sc, &fm, &tm) == 0) {
            dev_idx[nvalid] = i; dev_score[nvalid] = sc; ++nvalid;
        }
    }
    if (nvalid == 0) return SR_ERR_CUDA_UNSUPPORTED;
    for (int a = 0; a < nvalid; ++a)
        for (int b = a + 1; b < nvalid; ++b)
            if (dev_score[b] > dev_score[a]) {
                float ts = dev_score[a]; dev_score[a] = dev_score[b]; dev_score[b] = ts;
                int ti = dev_idx[a]; dev_idx[a] = dev_idx[b]; dev_idx[b] = ti;
            }

    /* max_cards=0 用全部；否则截断。 */
    int ncards = nvalid;
    if (max_cards > 0 && (int)max_cards < ncards) ncards = (int)max_cards;

    const uint32_t width  = (uint32_t)((int64_t)p->x1 - p->x0);
    const int64_t  height = (int64_t)p->z1 - p->z0;

    /* 按 score 占比分行（前缀比例，末卡兜底到 height）。 */
    double total_score = 0;
    for (int i = 0; i < ncards; ++i) total_score += dev_score[i];
    if (total_score < 1.0) total_score = 1.0;

    DevTask tasks[SR_CUDA_MAX_DEV];
    sr_thread_t handles[SR_CUDA_MAX_DEV];
    int64_t assigned = 0;
    double  prefix = 0;
    for (int i = 0; i < ncards; ++i) {
        int64_t start = assigned;
        int64_t end;
        if (i + 1 == ncards) {
            end = height;
        } else {
            prefix += dev_score[i];
            int64_t target = (int64_t)((double)height * (prefix / total_score));
            if (target < assigned) target = assigned;
            if (target > height) target = height;
            end = target;
        }
        assigned = end;

        tasks[i].device = dev_idx[i];
        tasks[i].seed   = p->world_seed;
        tasks[i].thr    = p->threshold;
        tasks[i].x0     = p->x0;
        tasks[i].width  = width;
        tasks[i].z_base = (int32_t)(p->z0 + start);
        tasks[i].rows   = end - start;
        tasks[i].buf = NULL; tasks[i].len = 0; tasks[i].cap = 0;
        tasks[i].err = 0; tasks[i].overflow = 0;
    }

    /* 每卡一个线程 */
    unsigned char spawned[SR_CUDA_MAX_DEV] = {0};
    for (int i = 0; i < ncards; ++i) {
#if defined(_WIN32)
        handles[i] = CreateThread(NULL, 0, dev_worker, &tasks[i], 0, NULL);
        spawned[i] = (handles[i] != NULL);
        if (!spawned[i]) dev_worker(&tasks[i]);  /* 回退：同步跑 */
#else
        spawned[i] = (pthread_create(&handles[i], NULL, dev_worker, &tasks[i]) == 0);
        if (!spawned[i]) dev_worker(&tasks[i]);  /* 回退：同步跑 */
#endif
    }
    for (int i = 0; i < ncards; ++i) {
        if (!spawned[i]) continue;
#if defined(_WIN32)
        WaitForSingleObject(handles[i], INFINITE);
        CloseHandle(handles[i]);
#else
        pthread_join(handles[i], NULL);
#endif
    }

    /* join 后串行回调（线程安全）。 */
    int rc = SR_OK, any_overflow = 0;
    for (int i = 0; i < ncards; ++i) {
        if (tasks[i].err) rc = tasks[i].err;
        if (tasks[i].overflow) any_overflow = 1;
        for (size_t k = 0; k < tasks[i].len; ++k)
            on_result(ctx, tasks[i].buf[k]);
        free(tasks[i].buf);
    }
    if (on_progress) on_progress(ctx, (uint64_t)height, (uint64_t)height);

    if (any_overflow)
        fprintf(stderr, "[cuda] WARNING: result overflow; output may be truncated. "
                        "Lower RANGE or raise THRESHOLD.\n");
    return rc;
}
