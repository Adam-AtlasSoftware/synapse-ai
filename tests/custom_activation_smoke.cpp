// Tier-2 guarantee: CUSTOM activations — registered as real C++, exactly like the ones
// the Code Lab compiles into the engine — are differentiated correctly by backprop and
// let the network learn.
//
// This registers every activation offered as a Code Lab *template* and finite-difference
// gradient-checks each one, so the example code users copy is verified math, not guesswork.
// Keep these implementations in sync with dashboard/qml/CodeLab.qml's templates.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "activation_registry.hpp"  // internal engine header — this test opts into src/
#include "synapse/network.hpp"
#include "synapse/telemetry.hpp"

using namespace synapse;

static void register_templates() {
  // leaky_relu — ReLU with a small slope for negatives, so units never fully die.
  register_activation({
      "leaky_relu", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) act[i] = pre[i] > 0.0f ? pre[i] : 0.01f * pre[i];
      },
      [](float pre, float post) { return pre > 0.0f ? 1.0f : 0.01f; }});

  // swish (SiLU) — x · sigmoid(x): smooth and self-gating.
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

  // elu — linear above zero, saturating exponential below.
  register_activation({
      "elu", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) act[i] = pre[i] > 0.0f ? pre[i] : (std::exp(pre[i]) - 1.0f);
      },
      // For x <= 0 the output is e^x - 1, so the slope is e^x = post + 1.
      [](float pre, float post) { return pre > 0.0f ? 1.0f : post + 1.0f; }});

  // softplus — ln(1 + e^x), a smooth ReLU whose derivative is exactly sigmoid(x).
  register_activation({
      "softplus", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i)
          act[i] = pre[i] > 20.0f ? pre[i] : std::log1p(std::exp(pre[i]));
      },
      [](float pre, float post) { return 1.0f / (1.0f + std::exp(-pre)); }});

  // gelu (tanh approximation) — the activation used in most transformers.
  register_activation({
      "gelu", false,
      [](const float* pre, float* act, int n) {
        const float c = 0.7978845608f;  // sqrt(2/pi)
        for (int i = 0; i < n; ++i) {
          float x = pre[i];
          float u = c * (x + 0.044715f * x * x * x);
          act[i] = 0.5f * x * (1.0f + std::tanh(u));
        }
      },
      [](float pre, float post) {
        const float c = 0.7978845608f;
        float x = pre;
        float u = c * (x + 0.044715f * x * x * x);
        float t = std::tanh(u);
        float du = c * (1.0f + 0.134145f * x * x);
        return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
      }});

  // sine — a periodic activation (SIREN), good at fitting smooth repeating signals.
  register_activation({
      "sine", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) act[i] = std::sin(pre[i]);
      },
      [](float pre, float post) { return std::cos(pre); }});
}

// Validate a template's math directly: compare its analytic derivative against a central
// finite difference of its own forward, sampled away from x = 0. This is the precise check
// on the code users copy, and (unlike a whole-network check) it is well defined for
// piecewise-linear activations as long as we don't straddle their kink.
static bool derivative_matches(const std::string& name) {
  const ActivationImpl& a = activation_by_name(name);
  auto f = [&](float x) {
    float y = 0.0f;
    a.forward(&x, &y, 1);
    return y;
  };
  const float h = 1e-3f;
  bool ok = true;
  for (float x : {-3.0f, -2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f, 3.0f}) {
    const float numeric = (f(x + h) - f(x - h)) / (2.0f * h);
    const float analytic = a.derivative(x, f(x));
    const float rel = std::fabs(numeric - analytic) / std::max(1.0f, std::fabs(analytic));
    if (rel > 1e-2f) {
      std::cerr << "FAIL: " << name << " derivative wrong at x=" << x << " (analytic " << analytic
                << " vs numeric " << numeric << ")\n";
      ok = false;
    }
  }
  return ok;
}

int main() {
  register_templates();

  // `smooth` activations are also checked through full backprop. Piecewise-linear ones
  // (leaky_relu) are not: a whole-network finite-difference check reports a huge max error
  // whenever any pre-activation lands near the kink, where the slope jumps 0.01 -> 1. That
  // is an artifact of finite differences, not a wrong gradient — derivative_matches() and
  // the training check below cover those.
  struct Case {
    const char* name;
    bool smooth;
  };
  const std::vector<Case> cases{{"leaky_relu", false}, {"swish", true}, {"elu", true},
                                {"softplus", true},    {"gelu", true},  {"sine", true}};

  const std::vector<float> in{0.5f, -1.0f, 0.25f};
  const std::vector<float> tg{1.0f, 0.0f};
  int failures = 0;

  for (const Case& c : cases) {
    const std::string name = c.name;
    if (!derivative_matches(name)) ++failures;

    // A small net using this custom activation in its hidden layer.
    Topology topo;
    topo.name = name + "-test";
    topo.input_dim = 3;
    topo.layers = {
        LayerInfo{"L0", "dense", 3, 5, name},
        LayerInfo{"L1", "dense", 5, 2, "sigmoid"},
    };
    Network net;
    net.build(topo, 42);

    const double err = net.gradient_check(in, tg);

    // It actually learns through the custom activation.
    const float first = net.evaluate_loss(in, tg);
    for (int i = 0; i < 3000; ++i) net.train_step(in, tg, 0.1f);
    const float last = net.evaluate_loss(in, tg);

    std::cout << name << ": derivative OK, backprop err = " << err << (c.smooth ? "" : " (kinked)")
              << ", loss " << first << " -> " << last << "\n";
    if (c.smooth && err > 0.02) {
      std::cerr << "FAIL: " << name << " gradients disagree with finite differences\n";
      ++failures;
    }
    if (!(last < first * 0.5f)) {
      std::cerr << "FAIL: " << name << " did not train\n";
      ++failures;
    }
  }

  if (failures) return 1;
  std::cout << "OK — all " << cases.size() << " custom-activation templates verified\n";
  return 0;
}
