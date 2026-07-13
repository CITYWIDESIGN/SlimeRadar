#!/usr/bin/env bash
# CUDA 模式回归: 三种子 -m cuda 对拍参考数据。用法: bash regress_cuda.sh [可执行文件]
set -u
BIN="${1:-./slimerander_cuda}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
seeds="0 5023147298867078368 -184718958561915"
fail=0
for seed in $seeds; do
  ref="data/result_${seed}_10000.csv"
  "$BIN" -q -m cuda -f csv "$seed" 10000 45 2>/dev/null | tr -d '\r' > /tmp/cuda_mine.csv
  tr -d '\r' < "$ref" > /tmp/cuda_ref.csv
  if diff -q /tmp/cuda_mine.csv /tmp/cuda_ref.csv >/dev/null; then
    echo "seed $seed CUDA OK ($(wc -l < /tmp/cuda_mine.csv) rows)"
  else
    echo "seed $seed CUDA MISMATCH"; diff /tmp/cuda_mine.csv /tmp/cuda_ref.csv | head; fail=1
  fi
done
[ $fail -eq 0 ] && echo "ALL PASS" || echo "CUDA REGRESSION FAILED"
exit $fail
