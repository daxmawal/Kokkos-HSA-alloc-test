#include <Kokkos_Core.hpp>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  int steps = 10;
  size_t dim = 10000;

  if (argc > 1) {
    steps = std::atoi(argv[1]);
  }
  if (argc > 2) {
    dim = static_cast<size_t>(std::atoll(argv[2]));
  }

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace = ExecSpace::memory_space;
  using View = Kokkos::View<float *, MemSpace>;
  using Index = int64_t;
  using Range = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<Index>>;

  for (int step = 0; step < steps; ++step) {
    const Index elements =
        static_cast<Index>(dim) * static_cast<Index>(dim);
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    const size_t total_bytes = bytes * 3;

    std::printf(
        "step %d: dim=%zu elements=%" PRId64 " bytes/array=%zu total=%zu\n",
        step, dim, elements, bytes, total_bytes);

    View a("a", elements);
    View b("b", elements);
    View c("c", elements);

    Kokkos::parallel_for(
        "init_a_b", Range(0, elements), KOKKOS_LAMBDA(const Index i) {
          a(i) = 1.0f;
          b(i) = 2.0f;
        });
    Kokkos::parallel_for(
        "add", Range(0, elements), KOKKOS_LAMBDA(const Index i) {
          c(i) = a(i) + b(i);
        });
    Kokkos::fence();

    dim = dim + dim / 5;
  }

  return 0;
}
