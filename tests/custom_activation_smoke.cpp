// Tier-2 guarantee: a CUSTOM activation — registered as real C++, exactly like the
// ones the Code Lab compiles into the engine — is differentiated correctly by backprop
// and lets the network learn. Registers "swish" at runtime, builds a net that uses it,
// then finite-difference gradient-checks and trains.
#include <cmath>
#include <iostream>
#include <vector>

#include "activation_registry.hpp"  // internal engine header — this test opts into src/
#include "synapse/network.hpp"
#include "synapse/telemetry.hpp"

using namespace synapse;

int main() {
  // swish(x) = x · sigmoid(x); swish'(x) = sigmoid(x) + x·sigmoid(x)(1−sigmoid(x)).
  register_activation({
      "swish", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) {
          float s = 1.0f / (1.0f + std::exp(-pre[i]));
          act[i] = pre[i] * s;
        }
      },
      [](float pre, float post) {
        float s = 1.0f / (1.0f + std::exp(-pre));
        return s + pre * s * (1.0f - s);
      }});

  // A small net that USES the custom activation in its hidden layer.
  Topology topo;
  topo.name = "swish-test";
  topo.input_dim = 3;
  topo.layers = {
      LayerInfo{"L0", "dense", 3, 5, "swish"},
      LayerInfo{"L1", "dense", 5, 2, "sigmoid"},
  };
  Network net;
  net.build(topo, 42);

  const std::vector<float> in{0.5f, -1.0f, 0.25f};
  const std::vector<float> tg{1.0f, 0.0f};

  // 1) The core correctness check: analytic backprop vs finite differences through swish.
  const double err = net.gradient_check(in, tg);
  std::cout << "custom-activation gradient check: max relative error = " << err << "\n";
  if (err > 0.02) {
    std::cerr << "FAIL: custom activation gradients disagree with finite differences\n";
    return 1;
  }

  // 2) The net actually trains with the custom activation.
  const float first = net.evaluate_loss(in, tg);
  for (int i = 0; i < 2000; ++i) net.train_step(in, tg, 0.1f);
  const float last = net.evaluate_loss(in, tg);
  std::cout << "loss " << first << " -> " << last << "\n";
  if (!(last < first * 0.5f)) {
    std::cerr << "FAIL: custom activation did not train\n";
    return 2;
  }

  std::cout << "OK\n";
  return 0;
}
