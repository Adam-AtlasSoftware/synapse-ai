# The original SYCL sandbox files, preserved as reference and still buildable —
# now through CMake/AdaptiveCpp instead of a hand-written compile line. Each is a
# standalone executable. matrix_gpu.cpp in particular is the conceptual seed of
# the engine's real Tensor type.
foreach(sample sycl_test matrix_gpu)
  add_executable(${sample} ${sample}.cpp)
  target_compile_features(${sample} PRIVATE cxx_std_17)
  add_sycl_to_target(TARGET ${sample} SOURCES ${sample}.cpp)
endforeach()
