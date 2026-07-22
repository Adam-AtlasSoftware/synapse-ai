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

# Phase 7: pluggable optimizers (sgd/momentum/adam) + parameter save/restore.
add_executable(optimizer_smoke optimizer_smoke.cpp)
target_link_libraries(optimizer_smoke PRIVATE synapse_engine)
add_test(NAME optimizers COMMAND optimizer_smoke)

# Phase 8: the auto-training coach's policy, driven with synthetic loss curves. Plain C++,
# so it needs the engine only for the Metrics struct — no Qt, no GUI.
add_executable(auto_trainer_smoke auto_trainer_smoke.cpp ${CMAKE_SOURCE_DIR}/dashboard/src/AutoTrainer.cpp)
target_include_directories(auto_trainer_smoke PRIVATE ${CMAKE_SOURCE_DIR}/dashboard/src)
target_link_libraries(auto_trainer_smoke PRIVATE synapse_engine)
add_test(NAME auto_trainer COMMAND auto_trainer_smoke)

# Phase 6b: the engine running as a SEPARATE PROCESS. Drives synapse_engine_host over
# pipes and checks it streams the telemetry contract as JSON and trains XOR.
add_test(NAME engine_host_ipc COMMAND ${CMAKE_COMMAND}
  -DHOST_EXE=$<TARGET_FILE:synapse_engine_host>
  -DWORK_DIR=${CMAKE_SOURCE_DIR}
  -DCOMMANDS=${CMAKE_CURRENT_SOURCE_DIR}/host_ipc_commands.txt
  -P ${CMAKE_CURRENT_SOURCE_DIR}/host_ipc_test.cmake)
