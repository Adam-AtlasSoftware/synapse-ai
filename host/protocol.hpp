#pragma once
#include <string>

#include "synapse/model_spec.hpp"
#include "synapse/telemetry.hpp"

// Wire format for the engine host: newline-delimited JSON, one message per line.
//
//   dashboard → engine (stdin) : {"id":7,"cmd":"forward","input":[...]}
//   engine → dashboard (stdout): {"ev":"topology","topology":{...}}
//                                {"ev":"step","step":{...}}
//                                {"ev":"result","id":7,...}
//                                {"ev":"error","id":7,"message":"..."}
//
// This is exactly the decoupling contract from telemetry.hpp put on a wire — the
// GUI still only ever sees a Topology and StepSnapshots.
namespace synapse::proto {

// Serialize the telemetry structs. Floats use %.6g: compact, and far more precision
// than the dashboard needs to draw a neuron.
std::string to_json(const Topology& t);
std::string to_json(const StepSnapshot& s);

// Small helpers for building event/result lines.
std::string escape(const std::string& s);
std::string event_topology(const Topology& t);
std::string event_step(const StepSnapshot& s);
std::string event_error(long id, const std::string& message);

}  // namespace synapse::proto
