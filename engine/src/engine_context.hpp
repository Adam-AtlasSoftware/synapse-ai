#pragma once
#include <sycl/sycl.hpp>

// Internal (src-only) header — free to include SYCL. Provides the one process-wide
// queue every engine kernel uses. Default selector keeps the engine hardware-
// agnostic: it prefers an accelerator (your GPU) and falls back to the CPU/OpenMP
// backend when none is available.
namespace synapse::detail {

inline sycl::queue& queue() {
  static sycl::queue q{sycl::default_selector_v};
  return q;
}

}  // namespace synapse::detail
