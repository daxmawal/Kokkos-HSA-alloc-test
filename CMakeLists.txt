cmake_minimum_required(VERSION 3.18)
project(kokkos_hsa_alloc_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)

set(HSA_ROOT "" CACHE PATH "Root of ROCm/HSA installation")

find_path(HSA_INCLUDE_DIR
  NAMES hsa/hsa.h
  HINTS ${HSA_ROOT}
  PATH_SUFFIXES include
)

find_library(HSA_RUNTIME_LIBRARY
  NAMES hsa-runtime64
  HINTS ${HSA_ROOT}
  PATH_SUFFIXES lib lib64
)

if(NOT HSA_INCLUDE_DIR)
  message(FATAL_ERROR "HSA headers not found. Set HSA_ROOT or HSA_INCLUDE_DIR.")
endif()
if(NOT HSA_RUNTIME_LIBRARY)
  message(FATAL_ERROR "HSA runtime library not found. Set HSA_ROOT or HSA_RUNTIME_LIBRARY.")
endif()

add_executable(kokkos_hsa_alloc_test kokkos_hsa_alloc_test.cpp)
target_include_directories(kokkos_hsa_alloc_test PRIVATE ${HSA_INCLUDE_DIR})
target_link_libraries(kokkos_hsa_alloc_test PRIVATE Kokkos::kokkos ${HSA_RUNTIME_LIBRARY})
