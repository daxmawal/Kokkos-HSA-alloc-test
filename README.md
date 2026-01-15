# Kokkos HSA Alloc Test

Minimal Kokkos test that allocates a number of buffers each time step using
Kokkos memory spaces.

## Clone with submodule

```bash
git clone --recurse-submodules git@github.com:daxmawal/Kokkos-HSA-alloc-test.git
```

If already cloned:

```bash
git submodule update --init --recursive
```

## Build on Adastra (MI300A)

From the project root:

```bash
module purge
module load PrgEnv-amd amd/6.2.1 cray-mpich cray-hdf5-parallel cmake/3.27.9

rm -rf build
mkdir build
cd build

CC=hipcc CXX=hipcc cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_AMD_GFX942_APU=ON \
  -DKokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC=ON

cmake --build .
```

## Build for MI250

Use `GFX90A` for MI250 / MI250X:

```bash
CC=hipcc CXX=hipcc cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_AMD_GFX90A=ON \
  -DKokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC=ON
```

## Run

```bash
./kokkos_hsa_alloc_test [steps] [allocs_per_step] [bytes]
```

Example:

```bash
./kokkos_hsa_alloc_test 10 10 $((64*1024*1024))
```
