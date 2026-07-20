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
# Full-batch GPU training path (batched matrix ops).
add_test(NAME train_gpu_batched COMMAND train_smoke ${CMAKE_SOURCE_DIR}/models/blueprints/seven_segment.json 0.5 8000 gpu)

# Tier-2: a custom activation (registered as C++) is gradient-correct and trains. Opts
# into the internal engine header, so it also proves the registry works end to end.
add_executable(custom_activation_smoke custom_activation_smoke.cpp)
target_link_libraries(custom_activation_smoke PRIVATE synapse_engine)
target_include_directories(custom_activation_smoke PRIVATE ${CMAKE_SOURCE_DIR}/engine/src)
add_test(NAME custom_activation COMMAND custom_activation_smoke)
