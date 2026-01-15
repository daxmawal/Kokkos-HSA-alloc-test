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
