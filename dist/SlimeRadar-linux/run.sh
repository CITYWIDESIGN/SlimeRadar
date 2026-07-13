#!/usr/bin/env bash
# SlimeRadar 交互式启动脚本 (Linux)
# 双击或在终端运行: ./run.sh
cd "$(dirname "$0")"
EXE="./slimerander"

if [ ! -x "$EXE" ]; then
    echo "错误: 找不到可执行文件 slimerander，或没有执行权限。"
    echo "请先运行: chmod +x slimerander run.sh"
    exit 1
fi

while true; do
    clear
    echo "============================================"
    echo "   SlimeRadar - 史莱姆区块搜索器"
    echo "============================================"
    echo

    read -rp "请输入世界种子 (整数, 可为负): " seed
    if [ -z "$seed" ]; then
        echo "[错误] 种子不能为空。"
        read -rp "按回车继续..." _
        continue
    fi

    read -rp "请输入搜索范围 (区块, 默认 10000): " range
    [ -z "$range" ] && range=10000

    read -rp "请输入数量阈值 (默认 45): " thr
    [ -z "$thr" ] && thr=45

    read -rp "保存为 CSV 文件吗? (y=是, 回车=屏幕显示): " fmt

    echo
    echo "--------------------------------------------"
    echo " 种子=$seed   范围=±$range   阈值=$thr"
    echo "--------------------------------------------"
    echo

    t0=$(date +%s.%N)
    if [ "$fmt" = "y" ] || [ "$fmt" = "Y" ]; then
        outfile="result_${seed}_${range}.csv"
        "$EXE" -f csv "$seed" "$range" "$thr" > "$outfile"
        echo "已保存到: $outfile"
    else
        "$EXE" "$seed" "$range" "$thr"
    fi
    t1=$(date +%s.%N)

    elapsed=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
    echo
    echo "--------------------------------------------"
    echo " 搜索完成，用时 ${elapsed} 秒"
    echo "--------------------------------------------"
    read -rp "按回车返回菜单，或 Ctrl+C 退出。" _
done
