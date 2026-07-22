#include <cmath>

#include "activation_registry.hpp"

// Custom activation functions — real C++ the engine compiles. Edit, then Rebuild.
namespace synapse {

void register_custom_activations() {
      // "sine" — a periodic activation (SIREN); good at smooth, repeating signals.
  register_activation({
      "sine", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) act[i] = std::sin(pre[i]);
      },
      [](float pre, float post) { return std::cos(pre); }});

}

}  // namespace synapse
