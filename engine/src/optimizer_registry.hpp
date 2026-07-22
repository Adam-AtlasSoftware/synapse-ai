#pragma once
#include <string>
#include <vector>

// Internal optimizer registry (engine-private). Mirrors activation_registry.* exactly, so
// an optimizer is just as pluggable as an activation: name it, say how much per-parameter
// state it needs, and write the update rule.
//
// The update runs per parameter on the host. `state` points at this parameter's private
// slice (state_slots floats, zero-initialized); `t` is the 1-based step count, which is
// what Adam needs for bias correction.
namespace synapse {

struct OptimizerImpl {
  std::string name;
  int state_slots = 0;  // floats of state per parameter (sgd 0, momentum 1, adam 2)
  void (*update)(float* p, float grad, float* state, float lr, int t) = nullptr;
};

void register_optimizer(OptimizerImpl impl);

// Look up by name (case-insensitive); returns "sgd" if unknown. Never null.
const OptimizerImpl& optimizer_by_name(const std::string& name);

std::vector<std::string> registered_optimizer_names();
bool is_registered_optimizer(const std::string& name);

// Defined in custom_optimizers.cpp — user-editable, like custom_activations.cpp.
void register_custom_optimizers();

}  // namespace synapse
