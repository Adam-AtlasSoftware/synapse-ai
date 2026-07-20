#pragma once
#include <utility>
#include <vector>

#include "engine_context.hpp"

// Internal Tensor: a 2D block of floats in Unified Shared Memory, so the host and
// device share one pointer. This is the grown-up version of the sandbox's
// GPUMatrix — float instead of double, move-only, RAII-clean. Kept out of the
// public headers on purpose so no SYCL type ever leaks to the Qt side.
namespace synapse::detail {

struct Tensor {
  int rows = 0;
  int cols = 0;
  float* data = nullptr;  // USM shared, row-major

  Tensor() = default;
  Tensor(int r, int c) { allocate(r, c); }

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& o) noexcept { *this = std::move(o); }
  Tensor& operator=(Tensor&& o) noexcept {
    if (this != &o) {
      release();
      rows = o.rows;
      cols = o.cols;
      data = o.data;
      o.data = nullptr;
      o.rows = o.cols = 0;
    }
    return *this;
  }
  ~Tensor() { release(); }

  int size() const { return rows * cols; }

  void allocate(int r, int c) {
    release();
    rows = r;
    cols = c;
    data = sycl::malloc_shared<float>(static_cast<size_t>(size()), queue());
  }

  void release() {
    if (data) {
      sycl::free(data, queue());
      data = nullptr;
    }
  }

  // Host-side helpers (valid because USM shared memory is host-accessible once
  // any in-flight kernels have completed).
  std::vector<float> download() const { return std::vector<float>(data, data + size()); }
  void upload(const std::vector<float>& v) {
    const int n = std::min(size(), static_cast<int>(v.size()));
    for (int i = 0; i < n; ++i) data[i] = v[i];
  }
};

}  // namespace synapse::detail
