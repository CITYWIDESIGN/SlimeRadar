/*
 * cuda_stub.c — 无 CUDA 构建时的 sr_search_cuda 占位实现。
 * 启用 CUDA 时用真实实现替换本文件（不链接此桩）。
 */
#include "slimerander.h"

int sr_search_cuda(const SrParams *p, unsigned max_cards,
                   void *ctx, sr_result_fn on_result, sr_progress_fn on_progress) {
    (void)p; (void)max_cards; (void)ctx; (void)on_result; (void)on_progress;
    return SR_ERR_CUDA_UNSUPPORTED;
}
