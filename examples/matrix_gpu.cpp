#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sycl/sycl.hpp>

// A SYCL-optimized Matrix struct using Unified Shared Memory (USM)
struct GPUMatrix {
  int rows;
  int cols;
  double *data;   // Raw pointer required for SYCL USM allocation
  sycl::queue &q; // Reference to our GPU pipeline

  GPUMatrix(int r, int c, sycl::queue &queue, bool init_random = false)
      : rows(r), cols(c), q(queue) {

    // Allocate memory that both the CPU and GPU can see seamlessly
    data = sycl::malloc_shared<double>(r * c, q);

    if (init_random) {
      for (int i = 0; i < r * c; ++i) {
        data[i] = ((double)rand() / (RAND_MAX)) * 2.0 - 1.0;
      }
    } else {
      for (int i = 0; i < r * c; ++i)
        data[i] = 0.0;
    }
  }

  // Clean up our GPU memory allocation when the matrix falls out of scope
  ~GPUMatrix() { sycl::free(data, q); }

  // Overloading parenthesis for easy element mapping on the Host (CPU) side
  double &operator()(int r, int c) { return data[r * cols + c]; }
  const double &operator()(int r, int c) const { return data[r * cols + c]; }

  // --- HARDCORE GPU PARALLEL MATRIX MULTIPLICATION ---
  static void multiply(const GPUMatrix &A, const GPUMatrix &B, GPUMatrix &C) {
    sycl::queue queue = A.q;

    // Capture local variables into the lambda function for the GPU
    int A_rows = A.rows;
    int A_cols = A.cols;
    int B_cols = B.cols;
    double *a_ptr = A.data;
    double *b_ptr = B.data;
    double *c_ptr = C.data;

    // Submit the work to the GPU queue
    queue
        .submit([&](sycl::handler &h) {
          // Define a 2D execution space matching the output matrix size
          sycl::range<2> num_threads(A_rows, B_cols);

          // Execute this lambda in parallel across thousands of GPU cores
          h.parallel_for(num_threads, [=](sycl::id<2> idx) {
            int r = idx[0]; // Global row ID assigned to this GPU core
            int c = idx[1]; // Global column ID assigned to this GPU core

            double sum = 0.0;
            for (int k = 0; k < A_cols; ++k) {
              sum += a_ptr[r * A_cols + k] * b_ptr[k * B_cols + c];
            }
            c_ptr[r * B_cols + c] = sum;
          });
        })
        .wait(); // Block the CPU until the GPU finishes calculating the matrix
  }

  // Element-wise GPU operations for activations
  void apply_sigmoid() {
    double *d_ptr = data;
    int size = rows * cols;

    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
       d_ptr[idx] = 1.0 / (1.0 + std::exp(-d_ptr[idx]));
     }).wait();
  }

  // Element-wise addition on the GPU
  void add(const GPUMatrix &B) {
    double *a_ptr = data;
    double *b_ptr = B.data;
    int size = rows * cols;

    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
       a_ptr[idx] += b_ptr[idx];
     }).wait();
  }

  void print() const {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::cout << (*this)(r, c) << " ";
      }
      std::cout << "\n";
    }
  }
};

int main() {
  // 1. Initialize our GPU selector queue
  sycl::queue q(sycl::gpu_selector_v);
  std::cout << "Training AI Matrix math layers on: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

  // 2. Build simple testing matrices using our USM allocator
  GPUMatrix A(2, 3, q);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 4;
  A(1, 1) = 5;
  A(1, 2) = 6;

  GPUMatrix B(3, 2, q);
  B(0, 0) = 7;
  B(0, 1) = 8;
  B(1, 0) = 9;
  B(1, 1) = 10;
  B(2, 0) = 11;
  B(2, 1) = 12;

  GPUMatrix C(2, 2, q); // The output placeholder matrix

  // 3. Execute the matrix multiplication natively on the GPU
  GPUMatrix::multiply(A, B, C);

  std::cout
      << "GPU Matrix Multiplication Output (Expected: 58 64 / 139 154):\n";
  C.print();

  // 4. Test our parallel activation execution
  C.apply_sigmoid();
  std::cout << "\nGPU Sigmoid Activation Output:\n";
  C.print();

  return 0;
}