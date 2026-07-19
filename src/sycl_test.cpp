#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  // 1. Create a queue targeting an available GPU
  sycl::queue q(sycl::gpu_selector_v);
  std::cout << "Running on: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  const int N = 1000;
  // 2. Allocate Unified Shared Memory (accessible by Host CPU and Device GPU)
  float *a = sycl::malloc_shared<float>(N, q);
  float *b = sycl::malloc_shared<float>(N, q);
  float *c = sycl::malloc_shared<float>(N, q);

  // Initialize data on CPU
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // 3. Submit parallel work to the GPU queue
  q.submit([&](sycl::handler &h) {
     // 4. Run 'N' instances of this lambda in parallel on the GPU
     h.parallel_for(sycl::range<1>(N),
                    [=](sycl::id<1> idx) { c[idx] = a[idx] + b[idx]; });
   }).wait(); // Wait for the GPU to finish

  std::cout << "Result index 500: " << c[500] << " (Expected: 3.0)\n";

  // Clean up USM
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return 0;
}