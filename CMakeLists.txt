cmake_minimum_required(VERSION 3.18)
project(kokkos_hsa_alloc_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kokkos/CMakeLists.txt")
  message(
    FATAL_ERROR
      "Kokkos submodule not found. Run `git submodule update --init --recursive`."
  )
endif()

add_subdirectory(kokkos)

add_executable(kokkos_hsa_alloc_test kokkos_hsa_alloc_test.cpp)
target_link_libraries(kokkos_hsa_alloc_test PRIVATE Kokkos::kokkos)

add_executable(kokkos_mat_add kokkos_mat_add.cpp)
target_link_libraries(kokkos_mat_add PRIVATE Kokkos::kokkos)

add_executable(hip_mat_add hip_mat_add.cpp)
add_executable(hip_managed_mat_add hip_managed_mat_add.cpp)

set(HIP_ARCH
    ""
    CACHE STRING
          "HIP offload architecture for HIP examples (e.g., gfx90a, gfx942)")
set(_HIP_OFFLOAD_ARCH "")
if(HIP_ARCH)
  set(_HIP_OFFLOAD_ARCH "${HIP_ARCH}")
elseif(Kokkos_ARCH_AMD_GFX942_APU OR Kokkos_ARCH_AMD_GFX942)
  set(_HIP_OFFLOAD_ARCH "gfx942")
elseif(Kokkos_ARCH_AMD_GFX90A)
  set(_HIP_OFFLOAD_ARCH "gfx90a")
endif()

if(_HIP_OFFLOAD_ARCH)
  foreach(hip_target hip_mat_add hip_managed_mat_add)
    if(TARGET ${hip_target})
      target_compile_options(${hip_target}
                             PRIVATE "--offload-arch=${_HIP_OFFLOAD_ARCH}")
      target_link_options(${hip_target} PRIVATE
                          "--offload-arch=${_HIP_OFFLOAD_ARCH}")
    endif()
  endforeach()
endif()

option(ENABLE_HSA_TEST "Build HSA-only allocation test" ON)
if(ENABLE_HSA_TEST)
  set(HSA_ROOT
      ""
      CACHE PATH "Root of ROCm/HSA installation")

  find_path(
    HSA_INCLUDE_DIR
    NAMES hsa/hsa.h
    HINTS ${HSA_ROOT}
    PATH_SUFFIXES include)

  find_library(
    HSA_RUNTIME_LIBRARY
    NAMES hsa-runtime64
    HINTS ${HSA_ROOT}
    PATH_SUFFIXES lib lib64)

  if(NOT HSA_INCLUDE_DIR OR NOT HSA_RUNTIME_LIBRARY)
    message(
      FATAL_ERROR
        "HSA runtime not found. Set HSA_ROOT or disable with -DENABLE_HSA_TEST=OFF."
    )
  endif()

  add_executable(hsa_alloc_test hsa_alloc_test.cpp)
  target_include_directories(hsa_alloc_test PRIVATE ${HSA_INCLUDE_DIR})
  target_link_libraries(hsa_alloc_test PRIVATE ${HSA_RUNTIME_LIBRARY})
endif()
