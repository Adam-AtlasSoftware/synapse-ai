# The neural-net engine. A static library so the dashboard can link it directly.
# AdaptiveCpp compiles the sources; consumers see only the pure-C++ public headers.

set(SYNAPSE_ENGINE_SOURCES
  src/device_info.cpp
  src/telemetry.cpp
  src/activation_registry.cpp
  src/custom_activations.cpp
  src/json.cpp
  src/model_spec.cpp
  src/network.cpp
)

add_library(synapse_engine STATIC ${SYNAPSE_ENGINE_SOURCES})

# PUBLIC include/ so consumers get the headers; PRIVATE src/ for internal headers
# (engine_context.hpp, tensor.hpp, json.hpp) that must never reach the GUI.
target_include_directories(synapse_engine
  PUBLIC  include
  PRIVATE src
)
target_compile_features(synapse_engine PUBLIC cxx_std_17)

# Route the whole engine through AdaptiveCpp and link its runtime. The launch rule
# is target-granular, which is exactly why SYCL is quarantined in this library.
add_sycl_to_target(TARGET synapse_engine SOURCES ${SYNAPSE_ENGINE_SOURCES})
