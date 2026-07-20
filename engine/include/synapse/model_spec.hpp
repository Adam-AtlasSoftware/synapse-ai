#pragma once
#include <string>
#include <vector>

#include "synapse/telemetry.hpp"

// Reading/writing the JSON model spec — the "raw data" tier of customization.
// The GUI edits these files through nice controls; the engine builds a network
// from them with no recompile. Schema:
//
//   {
//     "name": "XOR",
//     "input_dim": 2,
//     "layers": [
//       { "type": "dense", "units": 4, "activation": "tanh" },
//       { "type": "dense", "units": 1, "activation": "sigmoid" }
//     ]
//   }
//
// Each layer's input_dim is inferred by chaining from the previous layer's units
// (or the network input_dim for the first layer).
namespace synapse {

Topology parse_topology_json(const std::string& text);   // throws on malformed input
Topology load_topology_file(const std::string& path);    // throws if unreadable
std::string topology_to_json(const Topology& t);
void save_topology_file(const Topology& t, const std::string& path);

// A labeled training example and a collection of them. The optional "dataset"
// array in a blueprint file supplies these — [{ "input": [...], "target": [...] }].
struct Sample {
  std::vector<float> input;
  std::vector<float> target;
};
struct Dataset {
  std::vector<Sample> samples;
  bool empty() const { return samples.empty(); }
  int size() const { return static_cast<int>(samples.size()); }
};

Dataset parse_dataset_json(const std::string& text);  // empty if no "dataset" key
Dataset load_dataset_file(const std::string& path);

}  // namespace synapse
