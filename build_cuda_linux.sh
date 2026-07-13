#!/usr/bin/env bash
# Linux CUDA 全量构建 (nvcc + gcc host)。在 WSL 或原生 Linux (装了 CUDA Toolkit) 上运行。
# 产物 slimerander_cuda 同时支持 -m cpu 和 -m cuda。
# 用法: bash build_cuda_linux.sh [输出路径]
set -e

# 定位 nvcc（优先 PATH，其次常见安装位置）
if command -v nvcc >/dev/null 2>&1; then
    NVCC=nvcc
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC=/usr/local/cuda/bin/nvcc
else
    NVCC=$(ls /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1)
fi
[ -z "$NVCC" ] && { echo "找不到 nvcc，请先安装 CUDA Toolkit"; exit 1; }
echo "using nvcc: $NVCC"

OUT="${1:-dist/SlimeRadar-cuda-linux/slimerander_cuda}"
mkdir -p "$(dirname "$OUT")"

# CUDA 构建下所有 .c 也交给 nvcc（host 编译器 gcc），保证单一工具链。
# sm_86 = RTX 3060 Ampere；含 cpu_simd_avx2.c（-Xcompiler -mavx2 给 host 编译器开 AVX2）。
"$NVCC" -O3 -arch=sm_86 -DSLIMY_ENABLE_CUDA=1 \
    -Xcompiler -mavx2 \
    src/cuda_search.cu src/cuda_host.c src/main.c src/cpu_search.c src/cpu_simd_avx2.c \
    -lpthread -o "$OUT"

echo "CUDA BUILD OK -> $OUT"
file "$OUT"
