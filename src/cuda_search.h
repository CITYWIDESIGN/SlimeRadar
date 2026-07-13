/*
 * cuda_search.h — CUDA host API 的纯 C 声明（.cu 内 extern "C" 实现）。
 * 参见 代码需求.md §3.3。所有函数返回 0 成功，非 0 错误码（2=结果溢出）。
 */
#ifndef SLIMERANDER_CUDA_SEARCH_H
#define SLIMERANDER_CUDA_SEARCH_H

#include <stdint.h>
#include "slimerander.h"   /* SrResult */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SrCudaCtx SrCudaCtx;  /* 不透明句柄 */

int sr_cuda_device_count(void);
int sr_cuda_device_score(int idx, float *score, uint64_t *free_mem, uint64_t *total_mem);

/* 预分配 device/pinned 内存 + 非阻塞 stream；max_w/max_h 为单批最大中心区域，max_cap 为结果上限。 */
int sr_cuda_ctx_init(int device, uint32_t max_w, uint32_t max_h, uint32_t max_cap, SrCudaCtx **out);
int sr_cuda_ctx_deinit(SrCudaCtx *ctx);

/* 单批搜索：中心区域 [x0,x0+w) × [z0,z0+h)。命中写 out[]，数量写 *out_count。
 * 返回 0 成功，2 溢出（*out_count 会 > cap，out[] 只含前 cap 条）。 */
int sr_cuda_search_batch(SrCudaCtx *ctx, int64_t seed, uint8_t thr,
                         int32_t x0, int32_t z0, uint32_t w, uint32_t h,
                         SrResult *out, uint32_t cap, uint32_t *out_count);

#ifdef __cplusplus
}
#endif

#endif /* SLIMERANDER_CUDA_SEARCH_H */
