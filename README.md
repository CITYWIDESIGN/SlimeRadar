# SlimeRadar

Minecraft 史莱姆区块聚集搜索器。给定世界种子,在区块坐标区域内暴力搜索 17×17 甜甜圈窗口内史莱姆区块数量最多的中心点。纯 C 实现,含标量 / AVX2(备用)/ 多线程 / CUDA 多卡四条计算路径,输出与参考实现逐行 diff 一致。

## 状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | 精确判定 + 甜甜圈 + 20 段 runs / 朴素黄金参照 | ✅ 完成 |
| 1 | 标量 SAT + 包围盒预筛 + 20 段精确 | ✅ 完成 |
| 2 | SIMD (AVX2 8-lane 判定) | ✅ 实现并三路对拍;**默认不启用**(见下) |
| 3 | Win32 多线程 + 无锁收集 | ✅ 完成,1→8 线程 4.46× |
| 4 | CLI + 四级排序 + human/csv 输出 | ✅ 完成,逐行 diff 一致 |
| 5 | CUDA 单卡(双 kernel + host API) | ✅ 完成,单卡三种子 diff 一致 |
| 6 | CUDA 多卡(按 score 分行 + 每卡线程) | ⚠️ 单卡退化验证通过 / **真多卡待硬件实测** |

## 构建

两条构建路径,**切勿混用**(GCC 与 MSVC 的 ABI 不兼容)。

### CPU-only (MSYS2 UCRT64 GCC)

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
gcc -std=c11 -O3 -march=native -funroll-loops \
    src/main.c src/cpu_search.c src/cpu_simd_avx2.c src/cuda_stub.c \
    -o slimerander.exe
```

或用 CMake:

```bash
cmake -B build && cmake --build build
```

### CUDA 全量 (nvcc 13.3 + VS2019 MSVC)

CUDA 构建下**所有 `.c` 也交给 nvcc→MSVC 编译**(保证单一 ABI),`-DSLIMY_ENABLE_CUDA=ON` 是开关。产物同时支持 `-m cpu` 和 `-m cuda`。

```bash
bash build_cuda.sh          # 已验证脚本 → slimerander_cuda.exe
# 或:
cmake -B build-cuda -DSLIMY_ENABLE_CUDA=ON && cmake --build build-cuda
```

目标架构 sm_86 (RTX 3060 Ampere)。

## 用法

```
slimerander [OPTIONS] SEED RANGE [THRESHOLD]
  -m cpu|cuda    模式(默认 cpu)
  -j THREADS     CPU 线程数(0=自动)
  -k CARDS       最多显卡数(0=全部)
  -f human|csv   输出格式(默认 human)
  -u             关闭排序
  -q             关闭进度
  -b             基准模式(测硬件/扫描吞吐,不输出结果列表)
  -h / -v        帮助 / 版本
```

- 搜索区域为 `[-RANGE, RANGE)²` 区块坐标(半开区间)。
- `THRESHOLD` 默认 45(甜甜圈内史莱姆区块数下限)。
- 输出排序键: **count↓ → 到原点距离²↑ → x↑ → z↑**(与参考 CSV 完全一致)。

示例:

```bash
slimerander -q -f csv 0 10000 45          # 种子 0,±10000 区块,阈值 45,CSV
slimerander -m cuda -k 0 12345 20000 50   # CUDA 全部显卡
slimerander -b 0 20000 45                 # 基准:测扫描吞吐(候选点/s、判定/s)
```

## 性能

基准模式 `-b` 实测吞吐(种子 0,阈值 45)。判定次数含 16 格块间重叠。

### CPU(Intel i9-13900K)

标量热路径(默认,非 AVX2):端到端峰值 **~117.17 亿区块判定/秒(11.717 G tests/s)**。

### CUDA 单卡(NVIDIA RTX 3060,sm_86)

CUDA 热路径优化后(种子代数拆分 + `nextInt(10)` 魔数除法快路径 + 融合判定/列前缀和 + uint16 前缀和差分数甜甜圈):

| RANGE | 区域(区块) | 候选中心 | elapsed | slime tests/s |
|-------|-----------|----------|---------|---------------|
| 20000 | 40000² | 16.0 亿 | 0.390 s | 4.41 G/s |
| 40000 | 80000² | 64.0 亿 | 0.891 s | 7.72 G/s |
| 80000 | 160000² | 256 亿 | 2.921 s | **9.36 G/s** |

优化前后端到端(种子 0,`-f csv`,取三次最优):

| RANGE | 优化前 | 优化后 | 加速 |
|-------|--------|--------|------|
| 10000 | 342 ms | 257 ms | 1.33× |
| 20000 | 675 ms | 400 ms | 1.69× |
| 40000 | 2039 ms | 895 ms | 2.28× |
| 80000 | 7517 ms | 2968 ms | 2.53× |

> 加速比随区域增大而升高:消除的是甜甜圈计数中 O(窗口面积) 的重复累加。GPU 侧瓶颈现集中在甜甜圈计数 kernel(约 89%),判定/列前缀和 kernel 已压至约 11%。

## 验证

一键回归:三种子 `-f csv` 输出与参考文件逐行 diff。

```bash
bash regress.sh ./slimerander.exe       # 或 ./slimerander_cuda.exe
```

## 已知限制

- **真多卡并行未实测**:阶段 6 逻辑就位,本机单卡只做了单卡退化验证。真正的双卡分行并行待第二张卡验证。
- **AVX2 备而不用**:AVX2 判定已实现且三路对拍 = golden,但**不快于标量**——LCG 的 64 位乘在 AVX2 上无原生 `_mm256_mullo_epi64`,拼装开销抵消 8 路并行,而 GCC16 对标量已优化到位。默认走标量;`SR_USE_AVX2=1` 环境变量可显式启用(供不同 CPU/编译器重评估)。详见 `NOTES.md`。
- **AVX-512 未实现**:本机无 AVX-512 硬件,无法对拍验证。刻意不写默认可达的未验证向量路径(CPUID 命中 AVX-512 也降级走 AVX2)。
