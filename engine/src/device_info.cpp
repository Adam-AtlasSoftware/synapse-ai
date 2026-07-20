#include "synapse/device_info.hpp"

#include <sycl/sycl.hpp>  // Confined to the .cpp — never reaches a public header.

#include "engine_context.hpp"  // the shared process-wide queue

namespace synapse {

using detail::queue;

std::string active_device_name() {
  return queue().get_device().get_info<sycl::info::device::name>();
}

bool device_self_test() {
  auto& q = queue();
  constexpr int n = 1024;

  // Unified Shared Memory: one pointer the host and device both see.
  float* a = sycl::malloc_shared<float>(n, q);
  float* b = sycl::malloc_shared<float>(n, q);
  float* c = sycl::malloc_shared<float>(n, q);

  for (int i = 0; i < n; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
    c[i] = 0.0f;
  }

  q.parallel_for(sycl::range<1>(n),
                 [=](sycl::id<1> i) { c[i] = a[i] + b[i]; })
      .wait();

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    if (c[i] != 3.0f) {
      ok = false;
      break;
    }
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return ok;
}

}  // namespace synapse
