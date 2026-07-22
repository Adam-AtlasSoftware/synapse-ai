#include <cmath>

#include "optimizer_registry.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  YOUR CUSTOM OPTIMIZERS
//
//  Real C++ the engine compiles — the same edit-recompile-run loop as custom
//  activations. An optimizer decides HOW a weight moves once backprop has told it
//  which way is downhill.
//
//  You provide:
//    name        what it's called in the Training panel's dropdown.
//    state_slots how many floats of private memory each parameter needs
//                (0 = stateless like plain SGD, 1 = a velocity, 2 = Adam's m and v).
//    update      the rule itself, run once per parameter:
//                  p     the weight to modify (in place)
//                  grad  d(loss)/d(p) from backprop
//                  state this parameter's private slice, zero-initialized
//                  lr    the learning rate from the slider
//                  t     1-based step number (Adam uses it for bias correction)
//
//  Register with:
//    register_optimizer({ "name", state_slots, update });
// ─────────────────────────────────────────────────────────────────────────────
namespace synapse {

void register_custom_optimizers() {
  // ── Example: "rmsprop" — divide the step by a running RMS of recent gradients,
  // so steep directions get damped and flat ones get amplified. Uncomment to enable.
  //
  // register_optimizer({
  //     "rmsprop", 1,
  //     [](float* p, float grad, float* state, float lr, int t) {
  //       constexpr float decay = 0.9f, eps = 1e-8f;
  //       state[0] = decay * state[0] + (1.0f - decay) * grad * grad;  // mean square
  //       *p -= lr * grad / (std::sqrt(state[0]) + eps);
  //     }});
}

}  // namespace synapse
