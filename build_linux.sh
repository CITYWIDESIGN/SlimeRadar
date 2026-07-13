#!/usr/bin/env bash
# Linux (Debian/Ubuntu) CPU-only 构建。在 WSL 或原生 Linux 上运行。
# AVX2 文件单独带 -mavx2（运行时 CPUID 分派，默认走标量）；其余通用 x86-64。
set -e
OUT="${1:-dist/SlimeRadar-linux/slimerander}"
mkdir -p "$(dirname "$OUT")"

gcc -std=c11 -O3 -mavx2 -c src/cpu_simd_avx2.c -o /tmp/sr_simd.o
gcc -std=c11 -O3 -funroll-loops \
    src/main.c src/cpu_search.c src/cuda_stub.c /tmp/sr_simd.o \
    -lpthread -o "$OUT"

echo "BUILD OK -> $OUT"
file "$OUT"
