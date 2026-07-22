# The engine running as its own process. Links the engine and talks the telemetry
# contract over stdin/stdout as newline-delimited JSON, so the dashboard can drive it
# without linking any SYCL. Compiled by the plain Clang toolchain (NOT routed through
# add_sycl_to_target) — more proof the public headers stay SYCL-free.
add_executable(synapse_engine_host
  main.cpp
  protocol.cpp
  protocol.hpp
)

target_link_libraries(synapse_engine_host PRIVATE synapse_engine)

# json.hpp is an internal engine header; the host reads commands with the same parser
# the model-spec loader uses.
target_include_directories(synapse_engine_host PRIVATE ${CMAKE_SOURCE_DIR}/engine/src)
