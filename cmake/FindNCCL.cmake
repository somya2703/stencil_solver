# cmake/FindNCCL.cmake
# Finds the NCCL library.
#
# Imported targets:
#   NCCL::NCCL
#
# Result variables:
#   NCCL_FOUND
#   NCCL_INCLUDE_DIRS
#   NCCL_LIBRARIES
#   NCCL_VERSION

# Search hints: env var NCCL_HOME, common install prefixes
set(_nccl_hints
  "$ENV{NCCL_HOME}"
  "$ENV{CUDA_HOME}"
  /usr/local/nccl
  /usr/local/cuda
  /opt/nccl
)

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${_nccl_hints}
  PATH_SUFFIXES include
  DOC "NCCL include directory"
)

find_library(NCCL_LIBRARIES
  NAMES nccl nccl_static
  HINTS ${_nccl_hints}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "NCCL library"
)

# Extract version from nccl.h
if(NCCL_INCLUDE_DIRS)
  file(STRINGS "${NCCL_INCLUDE_DIRS}/nccl.h" _nccl_ver_line
       REGEX "^#define NCCL_VERSION_CODE")
  if(_nccl_ver_line)
    string(REGEX REPLACE ".*NCCL_VERSION_CODE ([0-9]+).*" "\\1"
           _nccl_ver_code "${_nccl_ver_line}")
    math(EXPR NCCL_VERSION_MAJOR "${_nccl_ver_code} / 10000")
    math(EXPR NCCL_VERSION_MINOR "(${_nccl_ver_code} % 10000) / 100")
    math(EXPR NCCL_VERSION_PATCH "${_nccl_ver_code} % 100")
    set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_LIBRARIES NCCL_INCLUDE_DIRS
  VERSION_VAR   NCCL_VERSION
)

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
  add_library(NCCL::NCCL UNKNOWN IMPORTED)
  set_target_properties(NCCL::NCCL PROPERTIES
    IMPORTED_LOCATION             "${NCCL_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}"
  )
endif()

mark_as_advanced(NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
