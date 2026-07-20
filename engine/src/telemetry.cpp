#include "synapse/telemetry.hpp"

// The telemetry contract is header-only data (Topology / StepSnapshot / Observer).
// Activation handling moved to the name-based registry (activation_registry.*), so
// this translation unit now carries no definitions — kept as a build anchor.
namespace synapse {}  // namespace synapse
