// Phase 7 Stage A: optimizers are pluggable and correct.
//   1. every registered optimizer trains (loss falls a lot)
//   2. backprop is unaffected by the choice of optimizer (gradient check still passes)
//   3. parameters() / set_parameters() round-trip exactly — the basis of saving a trained
//      model and of surviving an engine hot-swap
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "synapse/model_spec.hpp"
#include "synapse/network.hpp"
#include "synapse/optimizer.hpp"

using namespace synapse;

namespace {

Topology tiny_topology() {
  Topology t;
  t.name = "opt-test";
  t.input_dim = 3;
  t.layers = {LayerInfo{"L0", "dense", 3, 6, "tanh"}, LayerInfo{"L1", "dense", 6, 2, "sigmoid"}};
  return t;
}

// A small fixed regression problem: four samples the net must fit.
struct Sample2 {
  std::vector<float> in, target;
};
const std::vector<Sample2>& samples() {
  static const std::vector<Sample2> s{{{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
                                      {{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                                      {{0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                                      {{1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}}};
  return s;
}

float mean_loss(Network& net) {
  float s = 0.0f;
  for (const Sample2& x : samples()) s += net.evaluate_loss(x.in, x.target);
  return s / static_cast<float>(samples().size());
}

// Train for a fixed number of epochs and report the final mean loss.
float train_with(const std::string& opt, float lr, int epochs) {
  Network net;
  net.build(tiny_topology(), 42);  // identical starting weights for every optimizer
  net.set_optimizer(opt);
  for (int e = 0; e < epochs; ++e)
    for (const Sample2& x : samples()) net.train_step(x.in, x.target, lr);
  return mean_loss(net);
}

}  // namespace

int main() {
  int failures = 0;

  // 1) Every registered optimizer must actually train, and leave backprop correct.
  const std::vector<std::string> names = optimizer_names();
  std::cout << "registered optimizers:";
  for (const std::string& n : names) std::cout << " " << n;
  std::cout << "\n";
  if (names.size() < 3) {
    std::cerr << "FAIL: expected at least sgd/momentum/adam\n";
    ++failures;
  }

  for (const std::string& name : names) {
    Network net;
    net.build(tiny_topology(), 42);
    net.set_optimizer(name);
    if (net.optimizer() != name) {
      std::cerr << "FAIL: optimizer did not stick: " << name << "\n";
      ++failures;
    }
    // Adam's per-parameter scaling wants a smaller step than plain SGD.
    const float lr = (name == "sgd" || name == "momentum") ? 0.2f : 0.02f;
    const float before = mean_loss(net);
    for (int e = 0; e < 400; ++e)
      for (const Sample2& x : samples()) net.train_step(x.in, x.target, lr);
    const float after = mean_loss(net);

    // Gradients must still match finite differences whatever the update rule is.
    const double gerr = net.gradient_check(samples()[0].in, samples()[0].target);

    std::cout << name << ": loss " << before << " -> " << after << ", gradient err " << gerr
              << "\n";
    if (!(after < before * 0.5f)) {
      std::cerr << "FAIL: " << name << " did not train\n";
      ++failures;
    }
    if (gerr > 0.02) {
      std::cerr << "FAIL: " << name << " broke the gradient check\n";
      ++failures;
    }
  }

  // 2) Momentum and Adam should beat plain SGD given the same budget — that is the whole
  //    reason they exist, and it is the lesson the dashboard will show side by side.
  const int epochs = 150;
  const float sgd_loss = train_with("sgd", 0.2f, epochs);
  const float mom_loss = train_with("momentum", 0.2f, epochs);
  const float adam_loss = train_with("adam", 0.05f, epochs);
  std::cout << "after " << epochs << " epochs — sgd " << sgd_loss << ", momentum " << mom_loss
            << ", adam " << adam_loss << "\n";
  if (!(mom_loss < sgd_loss)) {
    std::cerr << "FAIL: momentum did not converge faster than sgd\n";
    ++failures;
  }
  if (!(adam_loss < sgd_loss)) {
    std::cerr << "FAIL: adam did not converge faster than sgd\n";
    ++failures;
  }

  // 3) parameters() / set_parameters() round-trip.
  {
    Network net;
    net.build(tiny_topology(), 7);
    for (int e = 0; e < 100; ++e)
      for (const Sample2& x : samples()) net.train_step(x.in, x.target, 0.2f);

    const std::vector<float> saved = net.parameters();
    const std::vector<float> out_before = net.predict(samples()[0].in);
    if (static_cast<int>(saved.size()) != net.parameter_count()) {
      std::cerr << "FAIL: parameters() size != parameter_count()\n";
      ++failures;
    }

    // Wreck the weights, then restore them.
    for (int e = 0; e < 200; ++e)
      for (const Sample2& x : samples()) net.train_step(x.in, x.target, 0.9f);
    const std::vector<float> out_wrecked = net.predict(samples()[0].in);

    if (!net.set_parameters(saved)) {
      std::cerr << "FAIL: set_parameters rejected a correctly-sized vector\n";
      ++failures;
    }
    const std::vector<float> out_after = net.predict(samples()[0].in);

    bool same = out_before.size() == out_after.size();
    for (size_t i = 0; same && i < out_before.size(); ++i)
      same = std::fabs(out_before[i] - out_after[i]) < 1e-6f;
    bool changed = false;
    for (size_t i = 0; i < out_before.size() && i < out_wrecked.size(); ++i)
      if (std::fabs(out_before[i] - out_wrecked[i]) > 1e-4f) changed = true;

    std::cout << "params round-trip: restored=" << (same ? "exact" : "MISMATCH")
              << ", intermediate state differed=" << (changed ? "yes" : "no") << "\n";
    if (!same) {
      std::cerr << "FAIL: restored parameters do not reproduce the original output\n";
      ++failures;
    }
    if (!changed) {
      std::cerr << "FAIL: the wrecking step changed nothing, so the test proves nothing\n";
      ++failures;
    }

    // Wrong size must be rejected, not silently accepted.
    std::vector<float> wrong = saved;
    wrong.pop_back();
    if (net.set_parameters(wrong)) {
      std::cerr << "FAIL: set_parameters accepted a wrong-sized vector\n";
      ++failures;
    }
  }

  if (failures) return 1;
  std::cout << "OK — optimizers and parameter round-trip verified\n";
  return 0;
}
