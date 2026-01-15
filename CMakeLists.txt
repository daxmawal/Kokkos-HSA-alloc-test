cmake_minimum_required(VERSION 3.18)
project(kokkos_hsa_alloc_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)

add_executable(kokkos_hsa_alloc_test kokkos_hsa_alloc_test.cpp)
target_link_libraries(kokkos_hsa_alloc_test PRIVATE Kokkos::kokkos)
