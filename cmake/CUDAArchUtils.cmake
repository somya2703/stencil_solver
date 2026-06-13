# cmake/CUDAArchUtils.cmake
# Helpers for CUDA architecture detection and validation.

# auto_detect_cuda_arch()
# Sets CMAKE_CUDA_ARCHITECTURES to the SM of every GPU present on this machine.
# Falls back to 70;75;80;86;90 if CUDA is unavailable or no device is found.
function(auto_detect_cuda_arch)
  set(_detect_src [=[
    #include <cstdio>
    int main() {
      int n = 0;
      cudaGetDeviceCount(&n);
      for (int i = 0; i < n; ++i) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("%d%d\n", p.major, p.minor);
      }
      return 0;
    }
  ]=])

  set(_detect_file "${CMAKE_BINARY_DIR}/detect_cuda_arch.cu")
  set(_detect_bin  "${CMAKE_BINARY_DIR}/detect_cuda_arch")
  file(WRITE "${_detect_file}" "${_detect_src}")

  execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} "${_detect_file}" -o "${_detect_bin}"
            -lcudart
    RESULT_VARIABLE _compile_result
    OUTPUT_QUIET ERROR_QUIET
  )

  if(_compile_result EQUAL 0)
    execute_process(
      COMMAND "${_detect_bin}"
      OUTPUT_VARIABLE _arch_output
      RESULT_VARIABLE _run_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_run_result EQUAL 0 AND _arch_output)
      string(REPLACE "\n" ";" _arch_list "${_arch_output}")
      list(REMOVE_DUPLICATES _arch_list)
      set(CMAKE_CUDA_ARCHITECTURES "${_arch_list}" PARENT_SCOPE)
      message(STATUS "Auto-detected CUDA architectures: ${_arch_list}")
      return()
    endif()
  endif()

  # Fallback
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90" PARENT_SCOPE)
  message(STATUS "CUDA arch detection failed — using default: 70;75;80;86;90")
endfunction()
