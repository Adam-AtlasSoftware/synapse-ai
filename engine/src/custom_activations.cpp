#include <cmath>

#include "activation_registry.hpp"

// Custom activation functions — real C++ the engine compiles. Edit, then Rebuild.
namespace synapse {

void register_custom_activations() {
  // "swish" (a.k.a. SiLU) = x * sigmoid(x): a smooth, self-gated activation.
  register_activation({
      "swish", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) {
          float s = 1.0f / (1.0f + std::exp(-pre[i]));
          act[i] = pre[i] * s;                    // x · sigmoid(x)
        }
      },
      [](float pre, float post) {
        float s = 1.0f / (1.0f + std::exp(-pre));
        return s + pre * s * (1.0f - s);          // sigmoid(x) + x·sigmoid'(x)
      }});
}

}  // namespace synapse
