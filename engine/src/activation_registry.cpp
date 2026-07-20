#include "activation_registry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "synapse/activation.hpp"

namespace synapse {
namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// ── built-in activation math (host) ──────────────────────────────────────────
void fwd_linear(const float* pre, float* act, int n) {
  for (int i = 0; i < n; ++i) act[i] = pre[i];
}
void fwd_sigmoid(const float* pre, float* act, int n) {
  for (int i = 0; i < n; ++i) act[i] = 1.0f / (1.0f + std::exp(-pre[i]));
}
void fwd_relu(const float* pre, float* act, int n) {
  for (int i = 0; i < n; ++i) act[i] = pre[i] > 0.0f ? pre[i] : 0.0f;
}
void fwd_tanh(const float* pre, float* act, int n) {
  for (int i = 0; i < n; ++i) act[i] = std::tanh(pre[i]);
}
void fwd_softmax(const float* pre, float* act, int n) {
  float m = pre[0];
  for (int i = 1; i < n; ++i) m = pre[i] > m ? pre[i] : m;
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    float e = std::exp(pre[i] - m);
    act[i] = e;
    sum += e;
  }
  for (int i = 0; i < n; ++i) act[i] /= sum;
}

float dv_linear(float, float) { return 1.0f; }
float dv_sigmoid(float, float post) { return post * (1.0f - post); }
float dv_relu(float pre, float) { return pre > 0.0f ? 1.0f : 0.0f; }
float dv_tanh(float, float post) { return 1.0f - post * post; }
float dv_one(float, float) { return 1.0f; }  // softmax: handled by cross-entropy

// The registry itself. A function-local static so registration order is well-defined
// and initialization is lazy (no static-init-order surprises across translation units).
std::vector<ActivationImpl>& registry() {
  static std::vector<ActivationImpl> r;
  return r;
}

bool& initialized() {
  static bool v = false;
  return v;
}

void ensure_init() {
  if (initialized()) return;
  initialized() = true;  // set first so a custom registration during init doesn't recurse
  register_activation({"linear", false, fwd_linear, dv_linear});
  register_activation({"sigmoid", false, fwd_sigmoid, dv_sigmoid});
  register_activation({"relu", false, fwd_relu, dv_relu});
  register_activation({"tanh", false, fwd_tanh, dv_tanh});
  register_activation({"softmax", true, fwd_softmax, dv_one});
  register_custom_activations();  // user-added activations (custom_activations.cpp)
}

}  // namespace

void register_activation(ActivationImpl impl) {
  impl.name = lower(impl.name);
  auto& r = registry();
  for (ActivationImpl& e : r) {
    if (e.name == impl.name) {  // redefining a name replaces it
      e = impl;
      return;
    }
  }
  r.push_back(impl);
}

const ActivationImpl& activation_by_name(const std::string& name) {
  ensure_init();
  const std::string key = lower(name);
  for (const ActivationImpl& e : registry())
    if (e.name == key) return e;
  for (const ActivationImpl& e : registry())
    if (e.name == "sigmoid") return e;  // safe default
  return registry().front();
}

std::vector<std::string> registered_activation_names() {
  ensure_init();
  std::vector<std::string> names;
  names.reserve(registry().size());
  for (const ActivationImpl& e : registry()) names.push_back(e.name);
  return names;
}

bool is_registered_activation(const std::string& name) {
  ensure_init();
  const std::string key = lower(name);
  for (const ActivationImpl& e : registry())
    if (e.name == key) return true;
  return false;
}

// ── public API (synapse/activation.hpp) — names only, no function pointers leak ──
std::vector<std::string> activation_names() { return registered_activation_names(); }
bool activation_exists(const std::string& name) { return is_registered_activation(name); }

}  // namespace synapse
