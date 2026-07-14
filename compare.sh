#!/usr/bin/env bash
# 对拍脚本: 忽略顺序比较集合是否相等。
# 用法: ./compare.sh SEED
set -u
seed="$1"
ref="data/result_${seed}_10000.csv"
mine="mine_${seed}.csv"

if [ ! -f "$ref" ]; then echo "no ref file: $ref"; exit 2; fi

# 参考 CSV 是 Windows CRLF 行尾，剥掉 \r 再比，否则每行都因末尾 \r 假性不等。
./slime_ref.exe "$seed" 10000 45 2>/dev/null | tr -d '\r' | sort > "$mine"
tr -d '\r' < "$ref" | sort > "ref_${seed}.sorted"

if diff -q "$mine" "ref_${seed}.sorted" >/dev/null; then
    echo "seed $seed OK ($(wc -l < "$mine") rows)"
else
    echo "seed $seed MISMATCH"
    echo "--- only in MINE ---";  comm -23 "$mine" "ref_${seed}.sorted"
    echo "--- only in REF  ---";  comm -13 "$mine" "ref_${seed}.sorted"
fi
