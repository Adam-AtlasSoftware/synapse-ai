#include "optimizer_registry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "synapse/optimizer.hpp"

namespace synapse {
namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// ── built-in update rules ────────────────────────────────────────────────────

// Plain gradient descent: step straight downhill.
void upd_sgd(float* p, float grad, float*, float lr, int) { *p -= lr * grad; }

// Momentum: keep a running velocity so consistent directions accelerate and noisy
// ones cancel out — this is why it escapes the ravines plain SGD crawls along.
void upd_momentum(float* p, float grad, float* state, float lr, int) {
  constexpr float mu = 0.9f;
  state[0] = mu * state[0] - lr * grad;  // velocity
  *p += state[0];
}

// Adam: momentum on the gradient (m) AND on its square (v), so every parameter gets its
// own adaptive step size. The bias correction with `t` matters most in the first steps,
// when m and v are still warming up from zero.
void upd_adam(float* p, float grad, float* state, float lr, int t) {
  constexpr float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  state[0] = b1 * state[0] + (1.0f - b1) * grad;         // m
  state[1] = b2 * state[1] + (1.0f - b2) * grad * grad;  // v
  const float mhat = state[0] / (1.0f - std::pow(b1, static_cast<float>(t)));
  const float vhat = state[1] / (1.0f - std::pow(b2, static_cast<float>(t)));
  *p -= lr * mhat / (std::sqrt(vhat) + eps);
}

std::vector<OptimizerImpl>& registry() {
  static std::vector<OptimizerImpl> r;
  return r;
}

bool& initialized() {
  static bool v = false;
  return v;
}

void ensure_init() {
  if (initialized()) return;
  initialized() = true;
  register_optimizer({"sgd", 0, upd_sgd});
  register_optimizer({"momentum", 1, upd_momentum});
  register_optimizer({"adam", 2, upd_adam});
  register_custom_optimizers();
}

}  // namespace

void register_optimizer(OptimizerImpl impl) {
  impl.name = lower(impl.name);
  auto& r = registry();
  for (OptimizerImpl& e : r) {
    if (e.name == impl.name) {
      e = impl;
      return;
    }
  }
  r.push_back(impl);
}

const OptimizerImpl& optimizer_by_name(const std::string& name) {
  ensure_init();
  const std::string key = lower(name);
  for (const OptimizerImpl& e : registry())
    if (e.name == key) return e;
  for (const OptimizerImpl& e : registry())
    if (e.name == "sgd") return e;
  return registry().front();
}

std::vector<std::string> registered_optimizer_names() {
  ensure_init();
  std::vector<std::string> names;
  names.reserve(registry().size());
  for (const OptimizerImpl& e : registry()) names.push_back(e.name);
  return names;
}

bool is_registered_optimizer(const std::string& name) {
  ensure_init();
  const std::string key = lower(name);
  for (const OptimizerImpl& e : registry())
    if (e.name == key) return true;
  return false;
}

// ── public API (synapse/optimizer.hpp) — names only ──────────────────────────
std::vector<std::string> optimizer_names() { return registered_optimizer_names(); }
bool optimizer_exists(const std::string& name) { return is_registered_optimizer(name); }

}  // namespace synapse
