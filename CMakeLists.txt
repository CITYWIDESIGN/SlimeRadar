cmake_minimum_required(VERSION 3.20)

option(SLIMY_ENABLE_CUDA "Build CUDA mode" OFF)

if(SLIMY_ENABLE_CUDA)
    project(SlimeRadar C CUDA)
else()
    project(SlimeRadar C)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(SR_SOURCES src/main.c src/cpu_search.c src/cpu_simd_avx2.c)

if(SLIMY_ENABLE_CUDA)
    # CUDA 构建: 所有源都走 nvcc + MSVC host（保证单一 ABI，勿混 GCC）。
    # 目标架构 sm_86 (RTX 3060 Ampere)；CUDA 13 已移除 cudaDeviceProp::clockRate。
    set(CMAKE_CUDA_ARCHITECTURES 86)
    list(APPEND SR_SOURCES src/cuda_search.cu src/cuda_host.c)
    add_definitions(-DSLIMY_ENABLE_CUDA=1)
    # 让 .c 也用 CUDA 语言编译，统一交给 nvcc/MSVC，避免 GCC/MSVC 混链。
    set_source_files_properties(src/main.c src/cpu_search.c src/cpu_simd_avx2.c src/cuda_host.c
                                PROPERTIES LANGUAGE CUDA)
else()
    list(APPEND SR_SOURCES src/cuda_stub.c)
endif()

add_executable(slimerander ${SR_SOURCES})

# 阶段 0 朴素黄金参照（对拍用，纯 C）
add_executable(slime_ref src/slime_ref.c)

if(NOT MSVC AND NOT SLIMY_ENABLE_CUDA)
    target_compile_options(slimerander PRIVATE -O3 -funroll-loops)
    # AVX2 intrinsics 文件需 -mavx2 才能编（GCC 默认不开 AVX2）。
    # 仅此文件带该标志，不污染其余源；运行时仍靠 CPUID 分派，默认走标量。
    # CUDA 构建下该文件走 MSVC（默认支持 AVX2 intrinsics，无需此标志），故只在 GCC 路径加。
    set_source_files_properties(src/cpu_simd_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

# 构建备注:
#   CPU-only (GCC):  cmake -B build && cmake --build build
#   CUDA (需在 VS Developer 环境或指定 host):
#       cmake -B build-cuda -DSLIMY_ENABLE_CUDA=ON
#       cmake --build build-cuda
#   或直接用已验证脚本: bash build_cuda.sh
