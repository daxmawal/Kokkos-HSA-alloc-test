#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace {

void check_hsa(hsa_status_t status, const char *context) {
  if (status == HSA_STATUS_SUCCESS) {
    return;
  }
  const char *msg = nullptr;
  hsa_status_string(status, &msg);
  std::fprintf(stderr, "HSA error %d (%s) at %s\n", status,
               msg ? msg : "unknown", context);
  std::exit(EXIT_FAILURE);
}

struct AgentSelector {
  hsa_agent_t agent{};
  bool found = false;
};

hsa_status_t find_gpu_agent_cb(hsa_agent_t agent, void *data) {
  auto *selector = static_cast<AgentSelector *>(data);
  hsa_device_type_t type;
  hsa_status_t status =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (type == HSA_DEVICE_TYPE_GPU) {
    selector->agent = agent;
    selector->found = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t find_first_agent_cb(hsa_agent_t agent, void *data) {
  auto *selector = static_cast<AgentSelector *>(data);
  if (!selector->found) {
    selector->agent = agent;
    selector->found = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

struct PoolSelector {
  hsa_amd_memory_pool_t pool{};
  bool found = false;
};

hsa_status_t find_pool_cb(hsa_amd_memory_pool_t pool, void *data) {
  auto *selector = static_cast<PoolSelector *>(data);
  hsa_amd_segment_t segment;
  hsa_status_t status =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                   &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  bool alloc_allowed = false;
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  uint32_t flags = 0;
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
#ifdef HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG) {
    return HSA_STATUS_SUCCESS;
  }
#else
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
    return HSA_STATUS_SUCCESS;
  }
#endif

  selector->pool = pool;
  selector->found = true;
  return HSA_STATUS_INFO_BREAK;
}

hsa_agent_t pick_agent() {
  AgentSelector selector;
  check_hsa(hsa_iterate_agents(find_gpu_agent_cb, &selector),
            "hsa_iterate_agents(gpu)");
  if (selector.found) {
    return selector.agent;
  }

  check_hsa(hsa_iterate_agents(find_first_agent_cb, &selector),
            "hsa_iterate_agents(first)");
  if (!selector.found) {
    std::fprintf(stderr, "No HSA agent found\n");
    std::exit(EXIT_FAILURE);
  }
  return selector.agent;
}

hsa_amd_memory_pool_t pick_pool(hsa_agent_t agent) {
  PoolSelector selector;
  check_hsa(hsa_amd_agent_iterate_memory_pools(agent, find_pool_cb, &selector),
            "hsa_amd_agent_iterate_memory_pools");
  if (!selector.found) {
    std::fprintf(stderr, "No suitable HSA memory pool found\n");
    std::exit(EXIT_FAILURE);
  }
  return selector.pool;
}

}  // namespace

int main(int argc, char **argv) {
  check_hsa(hsa_init(), "hsa_init");

  int steps = 10;
  int allocs_per_step = 10;
  size_t bytes = 64 << 20;

  if (argc > 1) {
    steps = std::atoi(argv[1]);
  }
  if (argc > 2) {
    allocs_per_step = std::atoi(argv[2]);
  }
  if (argc > 3) {
    bytes = static_cast<size_t>(std::atoll(argv[3]));
  }

  hsa_agent_t agent = pick_agent();
  hsa_amd_memory_pool_t pool = pick_pool(agent);

  for (int step = 0; step < steps; ++step) {
    std::vector<void *> ptrs;
    ptrs.reserve(static_cast<size_t>(allocs_per_step));

    for (int i = 0; i < allocs_per_step; ++i) {
      void *ptr = nullptr;
      check_hsa(hsa_amd_memory_pool_allocate(pool, bytes, 0, &ptr),
                "hsa_amd_memory_pool_allocate");
      ptrs.push_back(ptr);
      std::printf("step %d alloc %d: %zu bytes\n", step, i, bytes);
    }

    for (void *ptr : ptrs) {
      check_hsa(hsa_amd_memory_pool_free(ptr),
                "hsa_amd_memory_pool_free");
    }
  }

  check_hsa(hsa_shut_down(), "hsa_shut_down");
  return 0;
}
