#pragma once
#include <string>
#include <vector>

// Public activation API — pure C++/STL, safe for the GUI to include. Exposes only the
// NAMES of the activations the engine knows (built-ins plus any custom ones compiled in),
// so the dashboard can populate its activation dropdowns. The actual implementations live
// behind the engine boundary (src/activation_registry.*), never reaching the GUI.
namespace synapse {

std::vector<std::string> activation_names();
bool activation_exists(const std::string& name);

}  // namespace synapse
