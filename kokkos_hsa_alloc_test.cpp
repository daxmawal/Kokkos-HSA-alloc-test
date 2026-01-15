#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  int steps = 10;
  int allocs_per_step = 10;
  size_t bytes = 1 << 20;

  if (argc > 1) {
    steps = std::atoi(argv[1]);
  }
  if (argc > 2) {
    allocs_per_step = std::atoi(argv[2]);
  }
  if (argc > 3) {
    bytes = static_cast<size_t>(std::atoll(argv[3]));
  }

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace = ExecSpace::memory_space;
  using Buffer = Kokkos::View<unsigned char *, MemSpace>;

  for (int step = 0; step < steps; ++step) {
    std::vector<Buffer> buffers;
    buffers.reserve(static_cast<size_t>(allocs_per_step));

    for (int i = 0; i < allocs_per_step; ++i) {
      buffers.emplace_back(
          Kokkos::ViewAllocateWithoutInitializing("step_alloc"), bytes);
    }

    buffers.clear();
  }

  return 0;
}
