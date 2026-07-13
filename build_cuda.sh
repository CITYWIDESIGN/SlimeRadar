#!/usr/bin/env bash
# 已验证可用的 CUDA 全量构建（nvcc 13.3 + VS2019 MSVC host）。
# 产物 slimerander_cuda.exe 同时支持 -m cpu 和 -m cuda。
# 用法: bash build_cuda.sh
set -e
CUDA_BIN="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin"
CLBIN='C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64'
export PATH="$PATH:$CUDA_BIN"

# 注意: CUDA 构建下所有 .c 也由 nvcc 转 MSVC 编译，保证全套 MSVC ABI（勿混入 GCC 目标）。
# C4819（头文件 GBK 编码警告）无害，已过滤。
nvcc -O3 -arch=sm_86 -ccbin "$CLBIN" -DSLIMY_ENABLE_CUDA=1 \
     src/cuda_search.cu src/cuda_host.c src/main.c src/cpu_search.c src/cpu_simd_avx2.c \
     -o slimerander_cuda.exe 2>&1 | grep -viE "C4819" || true

[ -f slimerander_cuda.exe ] && echo "CUDA BUILD OK -> slimerander_cuda.exe" || { echo "CUDA BUILD FAILED"; exit 1; }
