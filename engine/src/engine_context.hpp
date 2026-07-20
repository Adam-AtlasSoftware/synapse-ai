#pragma once
#include <sycl/sycl.hpp>

// Internal (src-only) header — free to include SYCL. Provides the one process-wide
// queue every engine kernel uses. Default selector keeps the engine hardware-
// agnostic: it prefers an accelerator (your GPU) and falls back to the CPU/OpenMP
// backend when none is available.
namespace synapse::detail {

// In-order so a chain of dependent kernels (e.g. a batched training epoch) executes
// in submission order without an explicit host-device sync between each one — that
// per-kernel synchronization was the bottleneck for batched GPU training.
inline sycl::queue& queue() {
  static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
  return q;
}

}  // namespace synapse::detail
