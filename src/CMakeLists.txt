# src/CMakeLists.txt — builds all solver targets

# ── Utility + solver static library ──────────────────────────────────────────
# CPU-only sources compiled as a plain static lib.
# Kernels are compiled separately so nvcc only touches .cu files.
add_library(stencil_utils STATIC
  utils/timer.cpp
  utils/io.cpp
  utils/grid.cpp
  solver/solver_base.cpp
  solver/heat_diffusion.cpp
  solver/wave_prop.cpp
)
target_link_libraries(stencil_utils PUBLIC stencil_common)
set_target_properties(stencil_utils PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# ── Kernel object library (CUDA) ──────────────────────────────────────────────
# Compiled with nvcc; linked into each executable separately so we can profile
# naive vs tiled in isolation.
add_library(kernels_naive OBJECT
  kernels/stencil_naive.cu
)
target_link_libraries(kernels_naive PUBLIC stencil_common)
set_target_properties(kernels_naive PROPERTIES
  CUDA_SEPARABLE_COMPILATION $<NOT:$<BOOL:${STENCIL_CPU_FALLBACK}>>
)

add_library(kernels_tiled OBJECT
  kernels/stencil_tiled.cu
)
target_link_libraries(kernels_tiled PUBLIC stencil_common)
set_target_properties(kernels_tiled PROPERTIES
  CUDA_SEPARABLE_COMPILATION $<NOT:$<BOOL:${STENCIL_CPU_FALLBACK}>>
)

# ── Naive executable ──────────────────────────────────────────────────────────
add_executable(stencil_naive
  main_naive.cpp
  $<TARGET_OBJECTS:kernels_naive>
  $<TARGET_OBJECTS:kernels_tiled>
)
target_link_libraries(stencil_naive PRIVATE stencil_utils stencil_common)
set_target_properties(stencil_naive PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── Optimised (tiled) executable ─────────────────────────────────────────────
add_executable(stencil_tiled
  main_tiled.cpp
  $<TARGET_OBJECTS:kernels_naive>    # tiled builds on naive for correctness checks
  $<TARGET_OBJECTS:kernels_tiled>
)
target_link_libraries(stencil_tiled PRIVATE stencil_utils stencil_common)
set_target_properties(stencil_tiled PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── Multi-GPU kernel (always built — wraps launch_tiled) ─────────────────────
if(NOT STENCIL_CPU_FALLBACK)
add_library(kernels_multigpu OBJECT
  kernels/stencil_multi_gpu.cu
)
target_link_libraries(kernels_multigpu PUBLIC stencil_common)
set_target_properties(kernels_multigpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ── Multi-GPU executable (requires MPI or NCCL) ───────────────────────────────
if(STENCIL_ENABLE_MPI OR STENCIL_ENABLE_NCCL)
  add_library(stencil_comm STATIC
    comm/nccl_exchange.cpp
    comm/mpi_halo.cpp
  )
  target_link_libraries(stencil_comm PUBLIC stencil_common)

  add_executable(stencil_multigpu
    main_multigpu.cpp
    $<TARGET_OBJECTS:kernels_naive>
    $<TARGET_OBJECTS:kernels_tiled>
    $<TARGET_OBJECTS:kernels_multigpu>
  )
  target_link_libraries(stencil_multigpu PRIVATE
    stencil_utils
    stencil_comm
    stencil_common
  )
  set_target_properties(stencil_multigpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  install(TARGETS stencil_multigpu DESTINATION bin)
endif()
endif()  # NOT STENCIL_CPU_FALLBACK

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS stencil_naive stencil_tiled DESTINATION bin)
