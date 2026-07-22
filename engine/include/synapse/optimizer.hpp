#pragma once
#include <string>
#include <vector>

// Public optimizer API — pure C++/STL, safe for the GUI to include. Like activation.hpp,
// it exposes only the NAMES of the optimizers the engine knows (built-ins plus any custom
// ones compiled in), so the dashboard can offer them in a dropdown. The update rules stay
// behind the engine boundary (src/optimizer_registry.*).
namespace synapse {

std::vector<std::string> optimizer_names();
bool optimizer_exists(const std::string& name);

}  // namespace synapse
