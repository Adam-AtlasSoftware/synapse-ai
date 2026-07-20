# Headless engine tests. Deliberately NOT routed through add_sycl_to_target: they
# compile with the plain Clang toolchain and link the engine, which is what proves
# the public headers are free of SYCL.
add_executable(forward_smoke forward_smoke.cpp)
target_link_libraries(forward_smoke PRIVATE synapse_engine)
add_test(NAME forward_smoke COMMAND forward_smoke ${CMAKE_SOURCE_DIR}/models/blueprints/xor.json)

add_executable(train_smoke train_smoke.cpp)
target_link_libraries(train_smoke PRIVATE synapse_engine)
# XOR exercises sigmoid + MSE; seven-segment exercises softmax + cross-entropy.
add_test(NAME train_xor COMMAND train_smoke ${CMAKE_SOURCE_DIR}/models/blueprints/xor.json 0.3 5000)
add_test(NAME train_classifier COMMAND train_smoke ${CMAKE_SOURCE_DIR}/models/blueprints/seven_segment.json 0.1 5000)
