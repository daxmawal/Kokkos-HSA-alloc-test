#include <hip/hip_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

void hip_check(hipError_t status, const char *context) {
  if (status == hipSuccess) {
    return;
  }
  std::fprintf(stderr, "HIP error (%s) at %s\n", hipGetErrorString(status),
               context);
  std::exit(EXIT_FAILURE);
}

__global__ void init_arrays(float *a, float *b, uint64_t n) {
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }
}

__global__ void add_arrays(const float *a, const float *b, float *c,
                           uint64_t n) {
  uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

int main(int argc, char **argv) {
  int steps = 10;
  uint64_t dim = 10000;

  if (argc > 1) {
    steps = std::atoi(argv[1]);
  }
  if (argc > 2) {
    dim = static_cast<uint64_t>(std::atoll(argv[2]));
  }

  constexpr int threads_per_block = 256;

  for (int step = 0; step < steps; ++step) {
    const uint64_t elements = dim * dim;
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    const size_t total_bytes = bytes * 3;

    std::printf(
        "step %d: dim=%" PRIu64 " elements=%" PRIu64
        " bytes/array=%zu total=%zu\n",
        step, dim, elements, bytes, total_bytes);

    float *a = nullptr;
    float *b = nullptr;
    float *c = nullptr;
    hip_check(hipMalloc(&a, bytes), "hipMalloc(a)");
    hip_check(hipMalloc(&b, bytes), "hipMalloc(b)");
    hip_check(hipMalloc(&c, bytes), "hipMalloc(c)");

    const uint64_t blocks =
        (elements + threads_per_block - 1) / threads_per_block;
    const unsigned int grid_x = static_cast<unsigned int>(blocks);
    hipLaunchKernelGGL(init_arrays, dim3(grid_x), dim3(threads_per_block), 0, 0,
                       a, b, elements);
    hip_check(hipGetLastError(), "init_arrays launch");
    hip_check(hipDeviceSynchronize(), "init_arrays sync");

    hipLaunchKernelGGL(add_arrays, dim3(grid_x), dim3(threads_per_block), 0, 0,
                       a, b, c, elements);
    hip_check(hipGetLastError(), "add_arrays launch");
    hip_check(hipDeviceSynchronize(), "add_arrays sync");

    hip_check(hipFree(a), "hipFree(a)");
    hip_check(hipFree(b), "hipFree(b)");
    hip_check(hipFree(c), "hipFree(c)");

    dim = dim + dim / 5;
  }

  return 0;
}
