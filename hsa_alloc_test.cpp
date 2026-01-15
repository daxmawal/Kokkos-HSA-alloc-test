#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

void check_hsa(hsa_status_t status, const char *context) {
  if (status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK) {
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

struct PoolInfo {
  hsa_amd_memory_pool_t pool{};
  hsa_amd_segment_t segment = HSA_AMD_SEGMENT_GLOBAL;
  bool alloc_allowed = false;
  uint32_t global_flags = 0;
  bool is_kernarg = false;
  bool is_fine_grain = false;
  size_t size_bytes = 0;
};

struct PoolCollector {
  std::vector<PoolInfo> pools;
};

const char *segment_name(hsa_amd_segment_t segment) {
  switch (segment) {
    case HSA_AMD_SEGMENT_GLOBAL:
      return "GLOBAL";
    case HSA_AMD_SEGMENT_READONLY:
      return "READONLY";
    case HSA_AMD_SEGMENT_GROUP:
      return "GROUP";
    case HSA_AMD_SEGMENT_PRIVATE:
      return "PRIVATE";
    case HSA_AMD_SEGMENT_KERNARG:
      return "KERNARG";
    default:
      return "UNKNOWN";
  }
}

bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value) {
    return false;
  }
  return std::strcmp(value, "0") != 0;
}

int parse_env_index(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value) {
    return -1;
  }
  char *end = nullptr;
  long idx = std::strtol(value, &end, 10);
  if (*end != '\0' || idx < 0 ||
      idx > static_cast<long>(std::numeric_limits<int>::max())) {
    return -1;
  }
  return static_cast<int>(idx);
}

hsa_status_t fill_pool_info(hsa_amd_memory_pool_t pool, PoolInfo *info) {
  info->pool = pool;
  hsa_status_t status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &info->segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (info->segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
      &info->alloc_allowed);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &info->global_flags);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &info->size_bytes);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

#ifdef HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED
  info->is_fine_grain =
      (info->global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) != 0;
#endif

#ifdef HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG
  info->is_kernarg =
      (info->global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG) != 0;
#else
  info->is_kernarg =
      (info->global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0;
#endif

  return HSA_STATUS_SUCCESS;
}

hsa_status_t collect_pool_cb(hsa_amd_memory_pool_t pool, void *data) {
  auto *collector = static_cast<PoolCollector *>(data);
  PoolInfo info;
  hsa_status_t status = fill_pool_info(pool, &info);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  collector->pools.push_back(info);
  return HSA_STATUS_SUCCESS;
}

bool pool_is_usable(const PoolInfo &info) {
  return info.segment == HSA_AMD_SEGMENT_GLOBAL && info.alloc_allowed &&
         !info.is_kernarg;
}

void print_usable_pools(const std::vector<const PoolInfo *> &pools) {
  std::printf("HSA usable pools (index within usable list):\n");
  for (size_t i = 0; i < pools.size(); ++i) {
    const PoolInfo *info = pools[i];
    std::printf(
        "  %zu: segment=%s alloc=%d fine=%d kernarg=%d size=%zuMB flags=0x%08x\n",
        i, segment_name(info->segment), info->alloc_allowed ? 1 : 0,
        info->is_fine_grain ? 1 : 0, info->is_kernarg ? 1 : 0,
        info->size_bytes / (1024 * 1024), info->global_flags);
  }
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
  PoolCollector collector;
  check_hsa(
      hsa_amd_agent_iterate_memory_pools(agent, collect_pool_cb, &collector),
            "hsa_amd_agent_iterate_memory_pools");
  std::vector<const PoolInfo *> usable;
  usable.reserve(collector.pools.size());
  for (const auto &info : collector.pools) {
    if (pool_is_usable(info)) {
      usable.push_back(&info);
    }
  }

  if (usable.empty()) {
    std::fprintf(stderr, "No suitable HSA memory pool found\n");
    std::exit(EXIT_FAILURE);
  }

  bool list_pools = env_flag_enabled("HSA_ALLOC_LIST_POOLS");
  if (list_pools) {
    print_usable_pools(usable);
  }

  int forced_index = parse_env_index("HSA_ALLOC_POOL_INDEX");
  if (forced_index >= 0) {
    if (forced_index >= static_cast<int>(usable.size())) {
      std::fprintf(stderr,
                   "Invalid HSA_ALLOC_POOL_INDEX=%d (max %zu)\n",
                   forced_index, usable.size() - 1);
      std::exit(EXIT_FAILURE);
    }
    if (list_pools) {
      std::printf("Using HSA pool index %d\n", forced_index);
    }
    return usable[static_cast<size_t>(forced_index)]->pool;
  }

  const PoolInfo *choice = nullptr;
  for (const PoolInfo *info : usable) {
    if (!info->is_fine_grain) {
      choice = info;
      break;
    }
  }
  if (!choice) {
    choice = usable[0];
  }
  if (list_pools) {
    std::printf("Selected pool: segment=%s fine=%d size=%zuMB flags=0x%08x\n",
                segment_name(choice->segment), choice->is_fine_grain ? 1 : 0,
                choice->size_bytes / (1024 * 1024), choice->global_flags);
  }
  return choice->pool;
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

    size_t alloc_bytes = bytes;
    for (int i = 0; i < allocs_per_step; ++i) {
      void *ptr = nullptr;
      check_hsa(hsa_amd_memory_pool_allocate(pool, alloc_bytes, 0, &ptr),
                "hsa_amd_memory_pool_allocate");
      ptrs.push_back(ptr);
      std::printf("step %d alloc %d: %zu bytes\n", step, i, alloc_bytes);
      alloc_bytes = static_cast<size_t>(alloc_bytes * 1.2);
    }

    for (void *ptr : ptrs) {
      check_hsa(hsa_amd_memory_pool_free(ptr),
                "hsa_amd_memory_pool_free");
    }
  }

  check_hsa(hsa_shut_down(), "hsa_shut_down");
  return 0;
}
