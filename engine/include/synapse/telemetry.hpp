#pragma once
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// THE DECOUPLING CONTRACT (pure C++/STL — no SYCL, no Qt).
//
// The engine describes itself and streams what it computes through these plain
// structs. The dashboard renders whatever it is handed and knows nothing about
// any specific model. Swap a 2-4-1 XOR net for a 784-128-10 digit net and the
// visualization adapts with zero code changes — because all it ever sees is a
// `Topology` (how to lay out the graph) and `StepSnapshot`s (what the numbers are
// right now). These structs are also trivially serializable, so the engine can
// later move to a separate process streaming the same data over IPC.
// ─────────────────────────────────────────────────────────────────────────────
namespace synapse {

// The non-linearity applied at the output of a layer.
enum class Activation { Linear, Sigmoid, ReLU, Tanh, Softmax };

std::string to_string(Activation a);
Activation activation_from_string(const std::string& s);  // case-insensitive

// One layer's shape and role. `input_dim`/`output_dim` are what the GUI uses to
// draw the incoming edges and the column of neurons this layer produces.
struct LayerInfo {
  std::string name;          // "L0", "L1", ... (display + telemetry key)
  std::string type = "dense";
  int input_dim = 0;
  int output_dim = 0;
  Activation activation = Activation::Sigmoid;
};

// The whole network's structure. The first visible column has `input_dim`
// neurons; each layer then adds a column of `output_dim` neurons.
struct Topology {
  std::string name = "model";
  int input_dim = 0;
  std::vector<LayerInfo> layers;

  int output_dim() const {
    return layers.empty() ? input_dim : layers.back().output_dim;
  }
};

// A named, host-side snapshot of a block of numbers copied off the device.
// Row-major. Names follow a simple convention the GUI can pattern-match:
//   "input", "weights.L0", "biases.L0", "preact.L0", "activations.L0", ...
struct TensorView {
  std::string name;
  int rows = 0;
  int cols = 0;
  std::vector<float> data;
};

// Everything captured at one moment of computation.
struct StepSnapshot {
  long step = 0;
  int epoch = 0;
  float loss = 0.0f;
  // Phase of computation. For the forward pass: "forward" (instant, all at once),
  // or "begin"/"step"/"done" when stepping through one layer at a time.
  std::string phase;
  // Which layer was just computed this step (index into Topology::layers), or -1
  // when not applicable. Lets the GUI highlight the column that just "fired".
  int active_layer = -1;
  std::vector<TensorView> tensors;
};

// The engine pushes; the GUI (or a test) implements. Default no-op methods so an
// observer can listen to only what it cares about.
class Observer {
 public:
  virtual ~Observer() = default;
  virtual void on_topology(const Topology&) {}
  virtual void on_step(const StepSnapshot&) {}
};

}  // namespace synapse
