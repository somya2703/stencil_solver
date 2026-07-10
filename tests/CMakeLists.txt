# tests/CMakeLists.txt

include(GoogleTest)

# ── Helper: unit test (CPU-only, always runs in CI) ───────────────────────────
function(add_unit_test target)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE
    stencil_utils
    stencil_common
    GTest::gtest
    GTest::gtest_main
    ${ARG_LIBS}
  )
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests"
    CUDA_SEPARABLE_COMPILATION ON
  )
  gtest_discover_tests(${target}
    PROPERTIES LABELS "unit"   # filtered in by CI: --label-regex unit
    DISCOVERY_TIMEOUT 60
  )
endfunction()

# ── Helper: GPU integration test (requires a CUDA device) ────────────────────
# These are excluded from the standard CI ctest run (no GPU on ubuntu-latest).
# They run on self-hosted GPU runners and in the benchmark workflow.
function(add_gpu_test target)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE
    stencil_utils
    stencil_common
    GTest::gtest
    GTest::gtest_main
    ${ARG_LIBS}
  )
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests"
    CUDA_SEPARABLE_COMPILATION ON
  )
  gtest_discover_tests(${target}
    PROPERTIES LABELS "gpu"   # CI runs these only on self-hosted GPU runners
    DISCOVERY_TIMEOUT 120
  )
endfunction()

# ── Unit tests ────────────────────────────────────────────────────────────────
if(STENCIL_BUILD_TESTS)
  add_unit_test(test_fd_coeffs   SOURCES unit/test_fd_coeffs.cpp)
  add_unit_test(test_grid        SOURCES unit/test_grid.cpp)
  add_unit_test(test_timer       SOURCES unit/test_timer.cpp)
  add_unit_test(test_solver_cpu  SOURCES unit/test_solver_cpu.cpp)
  add_unit_test(test_solver      SOURCES unit/test_solver.cpp
    LIBS    $<TARGET_OBJECTS:kernels_naive>
            $<TARGET_OBJECTS:kernels_tiled>)
set_target_properties(test_solver PROPERTIES LINKER_LANGUAGE CUDA)

  # test_kernels: stub populated in Step 5
  add_unit_test(test_kernels     SOURCES unit/test_kernels.cpp
    LIBS    $<TARGET_OBJECTS:kernels_naive>
            $<TARGET_OBJECTS:kernels_tiled>)
set_target_properties(test_kernels PROPERTIES LINKER_LANGUAGE CUDA)

  # ── GPU integration tests ─────────────────────────────────────────────────
  add_gpu_test(test_wave_3d
    SOURCES integration/test_wave_3d.cpp
    LIBS    $<TARGET_OBJECTS:kernels_naive>
            $<TARGET_OBJECTS:kernels_tiled>
  )
set_target_properties(test_wave_3d PROPERTIES LINKER_LANGUAGE CUDA)
  add_gpu_test(test_heat_3d
    SOURCES integration/test_heat_3d.cpp
    LIBS    $<TARGET_OBJECTS:kernels_naive>
            $<TARGET_OBJECTS:kernels_tiled>
  )
set_target_properties(test_heat_3d PROPERTIES LINKER_LANGUAGE CUDA)
  add_gpu_test(test_kernel_correctness
    SOURCES integration/test_kernel_correctness.cpp
    LIBS    $<TARGET_OBJECTS:kernels_naive>
            $<TARGET_OBJECTS:kernels_tiled>
  )
set_target_properties(test_kernel_correctness PROPERTIES LINKER_LANGUAGE CUDA)

  # ── Custom CTest targets for convenience ──────────────────────────────────
  # make test_unit   → runs only unit tests
  # make test_gpu    → runs only GPU integration tests
  add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND}
      --test-dir "${CMAKE_BINARY_DIR}"
      --label-regex "unit"
      --output-on-failure
      --parallel "${CMAKE_BUILD_PARALLEL_LEVEL}"
    COMMENT "Running unit tests"
  )
  add_custom_target(test_gpu
    COMMAND ${CMAKE_CTEST_COMMAND}
      --test-dir "${CMAKE_BINARY_DIR}"
      --label-regex "gpu"
      --output-on-failure
    COMMENT "Running GPU integration tests"
  )
endif()

# ── Benchmarks ────────────────────────────────────────────────────────────────
if(STENCIL_BUILD_BENCHMARKS)
  function(add_bench target)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${target} ${ARG_SOURCES})
    target_link_libraries(${target} PRIVATE
      stencil_utils stencil_common ${ARG_LIBS})
    set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/bench"
      CUDA_SEPARABLE_COMPILATION ON
    )
  endfunction()

  add_bench(bench_naive
    SOURCES perf/bench_naive.cpp
    LIBS    $<TARGET_OBJECTS:kernels_naive>
            $<TARGET_OBJECTS:kernels_tiled>
  )
  add_bench(bench_tiled
    SOURCES perf/bench_tiled.cpp
    LIBS    $<TARGET_OBJECTS:kernels_naive>
            $<TARGET_OBJECTS:kernels_tiled>
  )
  add_bench(bench_scaling
    SOURCES perf/bench_scaling.cpp
    LIBS    $<TARGET_OBJECTS:kernels_naive>
            $<TARGET_OBJECTS:kernels_tiled>
            $<$<NOT:$<BOOL:${STENCIL_CPU_FALLBACK}>>:$<TARGET_OBJECTS:kernels_multigpu>>
  )
  set_target_properties(bench_scaling PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
  )
endif()
