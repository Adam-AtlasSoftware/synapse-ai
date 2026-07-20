#pragma once
#include <string>
#include <vector>

// Internal activation registry (engine-private — never included by the GUI). Maps an
// activation NAME to its host implementation, so new activations can be added as plain
// C++ without touching the compute code. The built-ins register themselves; the user's
// register_custom_activations() (custom_activations.cpp) adds more.
//
// Activations run on the HOST here — the forward gets the whole layer vector (so softmax
// can normalize across the row) and the derivative is per-element w.r.t. the pre-activation.
// The batched GPU trainer keeps its own built-in-only fast path in network.cpp.
namespace synapse {

struct ActivationImpl {
  std::string name;
  bool is_softmax = false;                                  // cross-entropy handles the derivative
  void (*forward)(const float* pre, float* act, int n) = nullptr;   // whole-layer forward
  float (*derivative)(float pre, float post) = nullptr;             // d act / d pre
};

// Add (or replace, by name) an activation. Called by the built-ins and by user code.
void register_activation(ActivationImpl impl);

// Look up by name (case-insensitive); returns "sigmoid" if the name is unknown. Never null.
const ActivationImpl& activation_by_name(const std::string& name);

// All registered names, built-ins first then custom, in registration order.
std::vector<std::string> registered_activation_names();
bool is_registered_activation(const std::string& name);

// Defined in custom_activations.cpp — the user-editable file the GUI's Code Lab edits.
// Called once when the registry initializes.
void register_custom_activations();

}  // namespace synapse
