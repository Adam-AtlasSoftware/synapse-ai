// Headless proof that the engine's forward path and telemetry contract work —
// and, because this file links synapse_engine while including ONLY public headers
// (and is NOT compiled through AdaptiveCpp), it also proves no SYCL type leaks
// across the library boundary. If a SYCL header had crept into a public header,
// this translation unit would fail to compile.
#include <iostream>
#include <vector>

#include "synapse/network.hpp"
#include "synapse/telemetry.hpp"

using namespace synapse;

// A telemetry sink that prints the topology and each forward pass's activations —
// exactly the role the GUI's bridge will play, just to a terminal.
struct PrintObserver : Observer {
  void on_topology(const Topology& t) override {
    std::cout << "topology '" << t.name << "': in=" << t.input_dim;
    for (const auto& L : t.layers)
      std::cout << "  ->  [" << L.name << " " << L.input_dim << "x" << L.output_dim << " "
                << to_string(L.activation) << "]";
    std::cout << "\n";
  }
  void on_step(const StepSnapshot& s) override {
    std::cout << "    [phase=" << s.phase << " active_layer=" << s.active_layer << "]";
    for (const auto& tv : s.tensors) {
      if (tv.name.rfind("activations.", 0) != 0) continue;
      std::cout << "  " << tv.name << " = [";
      for (size_t i = 0; i < tv.data.size(); ++i)
        std::cout << (i ? ", " : "") << tv.data[i];
      std::cout << "]";
    }
    std::cout << "\n";
  }
};

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "models/xor.json";
  try {
    Network net = Network::from_json_file(path);
    if (!net.valid()) {
      std::cerr << "FAIL: network did not build\n";
      return 1;
    }

    PrintObserver obs;
    net.set_observer(&obs);

    const std::vector<std::vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for (const auto& in : inputs) {
      std::cout << "input (" << in[0] << ", " << in[1] << "):\n";
      std::vector<float> out = net.forward(in);
      std::cout << "  output = ";
      for (float v : out) std::cout << v << " ";
      std::cout << "\n";
    }

    // Shape sanity: output length must equal the topology's final width.
    const int got = static_cast<int>(net.forward({0, 0}).size());
    if (got != net.topology().output_dim()) {
      std::cerr << "FAIL: output dim " << got << " != " << net.topology().output_dim() << "\n";
      return 2;
    }

    // Stepped pass: begin, then advance one layer at a time. There must be exactly
    // one step per layer, and the pass must report "done" only at the end.
    std::cout << "stepped pass for input (1, 0):\n";
    net.begin_forward({1, 0});
    int steps = 0;
    while (net.step_forward()) ++steps;
    const int layerCount = static_cast<int>(net.topology().layers.size());
    if (steps != layerCount) {
      std::cerr << "FAIL: took " << steps << " steps for " << layerCount << " layers\n";
      return 4;
    }
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 3;
  }

  std::cout << "OK\n";
  return 0;
}
