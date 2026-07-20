#pragma once
#include <memory>
#include <string>
#include <vector>

#include "synapse/telemetry.hpp"

// Public engine API. Pure C++ — the dashboard includes this and telemetry.hpp and
// nothing else. All SYCL/device detail lives behind the PIMPL in network.cpp.
namespace synapse {

class Network {
 public:
  Network();
  ~Network();
  Network(Network&&) noexcept;
  Network& operator=(Network&&) noexcept;
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;

  // Build (or rebuild) from a topology, re-initializing weights (Xavier/Glorot).
  // This is the Tier-1 "edit the model, no recompile" entry point.
  void build(const Topology& topo, unsigned seed = 42);

  // Convenience: load a JSON model spec, then build.
  static Network from_json_file(const std::string& path, unsigned seed = 42);

  const Topology& topology() const;
  bool valid() const;

  // Run a single-sample forward pass. `input.size()` must equal input_dim.
  // Returns the output activations, and (if an observer is attached) emits a
  // "forward" StepSnapshot carrying inputs, weights, biases, and activations.
  std::vector<float> forward(const std::vector<float>& input);

  // --- stepped forward pass (for the "watch it in motion" playback) ---------
  // begin_forward loads the input and emits a "begin" snapshot (only the input
  // column is populated; every layer is reset). Each step_forward computes ONE
  // more layer and emits a "step" (or "done") snapshot with active_layer set.
  // step_forward returns false once the whole pass is complete.
  void begin_forward(const std::vector<float>& input);
  bool step_forward();
  bool forward_done() const;

  // --- training (backprop + SGD, from scratch) ------------------------------
  // Loss at the current output given a target (run forward() first). Cross-entropy
  // when the output layer is softmax, otherwise mean-squared error.
  float loss(const std::vector<float>& target) const;

  // One stochastic-gradient-descent step on a single sample: forward, compute the
  // loss, backpropagate the gradients, and nudge every weight by -lr * gradient.
  // Returns the loss *before* the update. Emits no telemetry (kept fast for loops).
  float train_step(const std::vector<float>& input, const std::vector<float>& target,
                   float learning_rate);

  // Verify backprop against finite differences. Returns the largest relative error
  // between the analytic gradient and a numerical one across all parameters; a tiny
  // value (≈1e-3) means the backprop math is correct.
  double gradient_check(const std::vector<float>& input, const std::vector<float>& target);

  // Forward + loss without changing weights or emitting telemetry (for eval/plots).
  float evaluate_loss(const std::vector<float>& input, const std::vector<float>& target);

  // Stepped learn-step (gradient-flow animation): animate ONE SGD step in slow
  // motion — forward, then loss, then backprop layer by layer, then the weight
  // update. Emits snapshots with phases "begin"/"step"/"loss"/"backward"/"update"
  // and delta/grad tensors. learn_step_advance() returns false when the step ends.
  void begin_learn_step(const std::vector<float>& input, const std::vector<float>& target,
                        float lr);
  bool learn_step_advance();
  bool learn_active() const;

  // Batched (full-batch) gradient-descent step on the GPU. `inputs` is N*input_dim and
  // `targets` is N*output_dim (row-major). Returns the mean loss. Far faster for larger
  // models than per-sample SGD — real matrix ops instead of thousands of tiny kernels.
  float train_epoch_batched(const std::vector<float>& inputs, const std::vector<float>& targets,
                            int n, float lr);

  // Attach a telemetry sink (may be nullptr). Re-emits topology immediately so a
  // freshly-attached GUI can lay out the graph before the first forward pass.
  void set_observer(Observer* obs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace synapse
