#!/usr/bin/env bash
# 回归测试: 三种子 -f csv 逐行 diff 参考文件（连排序顺序一致）。
# 用法: ./regress.sh [可执行文件，默认 ./slimerander.exe]
set -u
BIN="${1:-./slimerander.exe}"
seeds="0 5023147298867078368 -184718958561915"
fail=0
for seed in $seeds; do
  ref="data/result_${seed}_10000.csv"
  "$BIN" -q -f csv "$seed" 10000 45 2>/dev/null | tr -d '\r' > /tmp/mine.csv
  tr -d '\r' < "$ref" > /tmp/ref.csv
  if diff -q /tmp/mine.csv /tmp/ref.csv >/dev/null; then
    echo "seed $seed OK ($(wc -l < /tmp/mine.csv) rows)"
  else
    echo "seed $seed MISMATCH"; diff /tmp/mine.csv /tmp/ref.csv | head -20; fail=1
  fi
done
[ $fail -eq 0 ] && echo "ALL PASS" || echo "REGRESSION FAILED"
exit $fail
