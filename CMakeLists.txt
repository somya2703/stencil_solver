cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# ── Project declaration ───────────────────────────────────────────────────────
project(stencil_solver
  VERSION 0.1.0
  DESCRIPTION "GPU-accelerated finite-difference stencil solver (heat / wave)"
  LANGUAGES CXX CUDA
)

# ── Global standards ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CUDA_STANDARD          17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type: Debug | Release | RelWithDebInfo | MinSizeRel" FORCE)
endif()

# ── Options ───────────────────────────────────────────────────────────────────
option(STENCIL_BUILD_TESTS       "Build unit + integration tests"    ON)
option(STENCIL_BUILD_BENCHMARKS  "Build performance benchmarks"      ON)
option(STENCIL_ENABLE_MPI        "Enable MPI multi-node path"        OFF)
option(STENCIL_ENABLE_NCCL       "Enable NCCL multi-GPU path"        OFF)
option(STENCIL_ENABLE_NVTX       "Enable NVTX range markers"         ON)
option(STENCIL_CPU_FALLBACK      "Replace CUDA kernels with OpenMP"  OFF)
option(STENCIL_ENABLE_COVERAGE   "Enable gcov/lcov coverage flags"   OFF)
option(STENCIL_FP64              "Use double precision (default: fp32)" OFF)

# ── CUDA architecture ─────────────────────────────────────────────────────────
# Override: cmake -DCMAKE_CUDA_ARCHITECTURES="80;86"
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90"
      CACHE STRING "CUDA SM architectures to compile for")
endif()

# ── Module path ───────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ── Conan toolchain (optional — only injected when conan install was run) ─────
if(EXISTS "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
  include("${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
endif()

# ── Core dependencies ─────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

if(STENCIL_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  message(STATUS "MPI: ${MPI_CXX_COMPILER}")
endif()

if(STENCIL_ENABLE_NCCL)
  find_package(NCCL REQUIRED)          # provided by cmake/FindNCCL.cmake
  message(STATUS "NCCL: ${NCCL_LIBRARIES}")
endif()

if(STENCIL_CPU_FALLBACK)
  find_package(OpenMP REQUIRED)
endif()

# ── GoogleTest (fetched if not found via Conan) ───────────────────────────────
if(STENCIL_BUILD_TESTS OR STENCIL_BUILD_BENCHMARKS)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
      GIT_SHALLOW    TRUE
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

# ── Interface library — shared settings for all targets ───────────────────────
add_library(stencil_common INTERFACE)

target_include_directories(stencil_common INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(stencil_common INTERFACE
  # Host flags
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wno-unused-parameter>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3;-march=native>
  # CUDA device flags
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>   # Nsight source correlation
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
)

target_compile_definitions(stencil_common INTERFACE
  $<$<BOOL:${STENCIL_FP64}>:STENCIL_FP64>
  $<$<BOOL:${STENCIL_CPU_FALLBACK}>:STENCIL_CPU_FALLBACK>
  $<$<BOOL:${STENCIL_ENABLE_NVTX}>:ENABLE_NVTX>
  $<$<BOOL:${STENCIL_ENABLE_MPI}>:ENABLE_MPI>
  $<$<BOOL:${STENCIL_ENABLE_NCCL}>:ENABLE_NCCL>
)

target_link_libraries(stencil_common INTERFACE
  CUDA::cudart
  CUDA::cuda_driver
  $<$<BOOL:${STENCIL_ENABLE_NVTX}>:CUDA::nvtx3>
  $<$<BOOL:${STENCIL_ENABLE_MPI}>:MPI::MPI_CXX>
  $<$<BOOL:${STENCIL_CPU_FALLBACK}>:OpenMP::OpenMP_CXX>
)

if(STENCIL_ENABLE_NCCL AND NCCL_FOUND)
  target_include_directories(stencil_common INTERFACE ${NCCL_INCLUDE_DIRS})
  target_link_libraries(stencil_common INTERFACE ${NCCL_LIBRARIES})
endif()

if(STENCIL_ENABLE_COVERAGE)
  target_compile_options(stencil_common INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:--coverage;-fprofile-arcs;-ftest-coverage>)
  target_link_options(stencil_common INTERFACE --coverage)
endif()

# ── Subdirectories ────────────────────────────────────────────────────────────
add_subdirectory(src)

if(STENCIL_BUILD_TESTS OR STENCIL_BUILD_BENCHMARKS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(DIRECTORY include/ DESTINATION include)
install(TARGETS stencil_common EXPORT stencil_solverTargets)

# ── Configure summary ─────────────────────────────────────────────────────────
if(STENCIL_FP64)
  set(_precision_str "FP64")
else()
  set(_precision_str "FP32")
endif()
message(STATUS "")
message(STATUS "╔══════════════════════════════════════╗")
message(STATUS "║  stencil-solver ${PROJECT_VERSION}              ║")
message(STATUS "╠══════════════════════════════════════╣")
message(STATUS "║  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "║  CUDA archs   : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "║  Precision    : ${_precision_str}")
message(STATUS "║  CPU fallback : ${STENCIL_CPU_FALLBACK}")
message(STATUS "║  MPI          : ${STENCIL_ENABLE_MPI}")
message(STATUS "║  NCCL         : ${STENCIL_ENABLE_NCCL}")
message(STATUS "║  NVTX         : ${STENCIL_ENABLE_NVTX}")
message(STATUS "║  Tests        : ${STENCIL_BUILD_TESTS}")
message(STATUS "║  Benchmarks   : ${STENCIL_BUILD_BENCHMARKS}")
message(STATUS "║  Coverage     : ${STENCIL_ENABLE_COVERAGE}")
message(STATUS "╚══════════════════════════════════════╝")
message(STATUS "")
