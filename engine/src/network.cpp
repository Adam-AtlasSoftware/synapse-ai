#include "synapse/network.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "engine_context.hpp"
#include "synapse/model_spec.hpp"
#include "tensor.hpp"

namespace synapse {

using detail::queue;
using detail::Tensor;

namespace {

// Apply a layer's non-linearity element-wise, pre -> act (length n).
// The simple element-wise activations run as tiny device kernels. Softmax needs a
// reduction across the row, which is awkward on-device for a small output layer,
// so we normalize on the host reading USM shared memory directly.
void apply_activation(Activation a, const float* pre, float* act, int n) {
  auto& q = queue();
  switch (a) {
    case Activation::Linear:
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { act[i] = pre[i]; }).wait();
      break;
    case Activation::Sigmoid:
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
         act[i] = 1.0f / (1.0f + std::exp(-pre[i]));
       }).wait();
      break;
    case Activation::ReLU:
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
         act[i] = pre[i] > 0.0f ? pre[i] : 0.0f;
       }).wait();
      break;
    case Activation::Tanh:
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { act[i] = std::tanh(pre[i]); }).wait();
      break;
    case Activation::Softmax: {
      float m = pre[0];
      for (int i = 1; i < n; ++i) m = pre[i] > m ? pre[i] : m;
      float sum = 0.0f;
      for (int i = 0; i < n; ++i) {
        float e = std::exp(pre[i] - m);
        act[i] = e;
        sum += e;
      }
      for (int i = 0; i < n; ++i) act[i] /= sum;
      break;
    }
  }
}

}  // namespace

// The derivative of an activation w.r.t. its pre-activation, expressed in terms of
// the post-activation `a` (and `z` where needed). Softmax is handled specially by
// the cross-entropy output delta, so it returns 1 here.
float act_deriv(Activation a, float post, float pre) {
  switch (a) {
    case Activation::Linear: return 1.0f;
    case Activation::Sigmoid: return post * (1.0f - post);
    case Activation::Tanh: return 1.0f - post * post;
    case Activation::ReLU: return pre > 0.0f ? 1.0f : 0.0f;
    case Activation::Softmax: return 1.0f;
  }
  return 1.0f;
}

// One dense layer's parameters and its scratch buffers for forward + backprop.
struct Layer {
  LayerInfo info;
  Tensor W;      // output_dim x input_dim  (one row of incoming weights per neuron)
  Tensor b;      // 1 x output_dim
  Tensor pre;    // 1 x output_dim  (pre-activation:  W·x + b)
  Tensor act;    // 1 x output_dim  (post-activation)
  Tensor dW;     // output_dim x input_dim  (∂loss/∂W)
  Tensor db;     // 1 x output_dim          (∂loss/∂b)
  Tensor delta;  // 1 x output_dim          (∂loss/∂pre — the backprop signal)
};

struct Network::Impl {
  Topology topo;
  std::vector<Layer> layers;
  Tensor input;   // 1 x input_dim
  Observer* obs = nullptr;
  bool built = false;
  long step = 0;
  int cursor = 0;    // stepped-forward position: index of the next layer to compute
  float last_loss = 0.0f;

  // stepped learn-step (gradient-flow animation) state
  bool learn_on = false;
  int learn_fwd = 0;
  int learn_bwd = -1;
  bool learn_loss_done = false;
  bool learn_updated = false;
  std::vector<float> learn_target;
  float learn_lr = 0.1f;

  void build(const Topology& t, unsigned seed) {
    topo = t;
    layers.clear();
    std::mt19937 rng(seed);

    for (size_t k = 0; k < t.layers.size(); ++k) {
      Layer L;
      L.info = t.layers[k];
      if (L.info.name.empty()) L.info.name = "L" + std::to_string(k);
      const int in = L.info.input_dim;
      const int out = L.info.output_dim;

      L.W.allocate(out, in);
      L.b.allocate(1, out);
      L.pre.allocate(1, out);
      L.act.allocate(1, out);
      L.dW.allocate(out, in);
      L.db.allocate(1, out);
      L.delta.allocate(1, out);

      // Weight init that keeps signal variance stable across layers — He for ReLU
      // (which halves the variance), Xavier/Glorot otherwise. Far better than the
      // sandbox's uniform[-1,1], and it's what lets ReLU classifiers actually train.
      const float limit = (L.info.activation == Activation::ReLU)
                              ? std::sqrt(6.0f / static_cast<float>(in))
                              : std::sqrt(6.0f / static_cast<float>(in + out));
      std::uniform_real_distribution<float> dist(-limit, limit);
      for (int i = 0; i < L.W.size(); ++i) L.W.data[i] = dist(rng);
      for (int i = 0; i < L.b.size(); ++i) L.b.data[i] = 0.0f;

      layers.push_back(std::move(L));
    }

    input.allocate(1, t.input_dim);
    cursor = static_cast<int>(layers.size());  // "done" until a pass is started
    built = true;
    notify_topology();
  }

  void notify_topology() {
    if (obs && built) obs->on_topology(topo);
  }

  static TensorView view(const std::string& name, const Tensor& t) {
    TensorView v;
    v.name = name;
    v.rows = t.rows;
    v.cols = t.cols;
    v.data = t.download();
    return v;
  }

  void set_input(const std::vector<float>& in) {
    if (!built) throw std::runtime_error("network not built");
    if (static_cast<int>(in.size()) != topo.input_dim)
      throw std::runtime_error("input size mismatch");
    for (int i = 0; i < topo.input_dim; ++i) input.data[i] = in[i];
  }

  // Compute one layer: pre = W·x + b (one GPU thread per output neuron), then the
  // activation. `x` is the input column for k==0, else the previous layer's output.
  void compute_layer(int k) {
    Layer& L = layers[k];
    const int inN = L.info.input_dim;
    const int outN = L.info.output_dim;
    float* w = L.W.data;
    float* b = L.b.data;
    float* pre = L.pre.data;
    float* x = (k == 0) ? input.data : layers[k - 1].act.data;

    queue().parallel_for(sycl::range<1>(outN), [=](sycl::id<1> idx) {
             const int o = idx[0];
             float s = b[o];
             for (int i = 0; i < inN; ++i) s += w[o * inN + i] * x[i];
             pre[o] = s;
           }).wait();

    apply_activation(L.info.activation, L.pre.data, L.act.data, outN);
  }

  // Snapshot the whole current state (input + every layer's params/activations)
  // so the GUI can render nodes and edges. `active` marks the just-computed layer.
  void emit_snapshot(const std::string& phase, int active) {
    if (!obs) return;
    StepSnapshot snap;
    snap.step = step;
    snap.phase = phase;
    snap.active_layer = active;
    snap.loss = last_loss;
    snap.tensors.push_back(view("input", input));
    for (Layer& L : layers) {
      snap.tensors.push_back(view("weights." + L.info.name, L.W));
      snap.tensors.push_back(view("biases." + L.info.name, L.b));
      snap.tensors.push_back(view("preact." + L.info.name, L.pre));
      snap.tensors.push_back(view("activations." + L.info.name, L.act));
      snap.tensors.push_back(view("delta." + L.info.name, L.delta));   // ∂loss/∂pre
      snap.tensors.push_back(view("grad." + L.info.name, L.dW));        // ∂loss/∂W
    }
    obs->on_step(snap);
  }

  // Compute every layer without emitting telemetry (used by forward + training).
  void run_forward(const std::vector<float>& in) {
    set_input(in);
    for (int k = 0; k < static_cast<int>(layers.size()); ++k) compute_layer(k);
  }

  // Instant forward pass: compute every layer, emit one snapshot, return output.
  std::vector<float> forward(const std::vector<float>& in) {
    run_forward(in);
    cursor = static_cast<int>(layers.size());
    emit_snapshot("forward", -1);
    ++step;
    return layers.back().act.download();
  }

  // Loss at the current output. Cross-entropy for a softmax head, else MSE (½Σ(a−y)²).
  float loss(const std::vector<float>& target) const {
    const Layer& L = layers.back();
    const int n = L.info.output_dim;
    const float* a = L.act.data;
    float s = 0.0f;
    if (L.info.activation == Activation::Softmax) {
      for (int i = 0; i < n; ++i) s -= target[i] * std::log(std::max(a[i], 1e-9f));
    } else {
      for (int i = 0; i < n; ++i) {
        const float d = a[i] - target[i];
        s += d * d;
      }
      s *= 0.5f;
    }
    return s;
  }

  // Backprop, from scratch, split into pieces so the gradient-flow animation can
  // run them one layer at a time. Runs on the host reading USM memory — clearest
  // for learning; the forward pass already shows the GPU-kernel style.

  // Output layer delta: softmax+cross-entropy collapses to (a − y); else (a − y)·act'(z).
  void compute_output_delta(const std::vector<float>& target) {
    Layer& L = layers.back();
    const int out = L.info.output_dim;
    for (int o = 0; o < out; ++o) {
      const float a = L.act.data[o];
      if (L.info.activation == Activation::Softmax)
        L.delta.data[o] = a - target[o];
      else
        L.delta.data[o] = (a - target[o]) * act_deriv(L.info.activation, a, L.pre.data[o]);
    }
  }

  // Hidden-layer delta: (Wᵀ_next · delta_next) · activation'(z).
  void compute_hidden_delta(int k) {
    Layer& L = layers[k];
    Layer& next = layers[k + 1];
    const int units = L.info.output_dim;  // == next.info.input_dim
    const int nextOut = next.info.output_dim;
    const int nextIn = next.info.input_dim;
    for (int i = 0; i < units; ++i) {
      float s = 0.0f;
      for (int o = 0; o < nextOut; ++o) s += next.W.data[o * nextIn + i] * next.delta.data[o];
      L.delta.data[i] = s * act_deriv(L.info.activation, L.act.data[i], L.pre.data[i]);
    }
  }

  // Gradients from the deltas: dW[o,i] = delta[o] · a_prev[i];  db[o] = delta[o].
  void compute_gradients() {
    for (int k = 0; k < static_cast<int>(layers.size()); ++k) {
      Layer& L = layers[k];
      const int in = L.info.input_dim;
      const int out = L.info.output_dim;
      const float* aprev = (k == 0) ? input.data : layers[k - 1].act.data;
      for (int o = 0; o < out; ++o) {
        L.db.data[o] = L.delta.data[o];
        for (int i = 0; i < in; ++i) L.dW.data[o * in + i] = L.delta.data[o] * aprev[i];
      }
    }
  }

  void backward(const std::vector<float>& target) {
    const int n = static_cast<int>(layers.size());
    compute_output_delta(target);
    for (int k = n - 2; k >= 0; --k) compute_hidden_delta(k);
    compute_gradients();
  }

  void apply_gradients(float lr) {
    for (Layer& L : layers) {
      for (int i = 0; i < L.W.size(); ++i) L.W.data[i] -= lr * L.dW.data[i];
      for (int i = 0; i < L.b.size(); ++i) L.b.data[i] -= lr * L.db.data[i];
    }
  }

  float train_step(const std::vector<float>& in, const std::vector<float>& target, float lr) {
    run_forward(in);
    const float l = loss(target);  // loss of the current prediction, before the update
    backward(target);
    apply_gradients(lr);
    return l;
  }

  float evaluate_loss(const std::vector<float>& in, const std::vector<float>& target) {
    run_forward(in);
    return loss(target);
  }

  // ── stepped learn-step: one gradient-descent step in slow motion ──
  void begin_learn_step(const std::vector<float>& in, const std::vector<float>& target, float lr) {
    set_input(in);
    for (Layer& L : layers) {
      for (int i = 0; i < L.pre.size(); ++i) {
        L.pre.data[i] = 0.0f;
        L.act.data[i] = 0.0f;
        L.delta.data[i] = 0.0f;
        L.db.data[i] = 0.0f;
      }
      for (int i = 0; i < L.dW.size(); ++i) L.dW.data[i] = 0.0f;
    }
    learn_target = target;
    learn_lr = lr;
    learn_fwd = 0;
    learn_bwd = -1;
    learn_loss_done = false;
    learn_updated = false;
    learn_on = true;
    last_loss = 0.0f;
    emit_snapshot("begin", -1);
  }

  // Advance one sub-step: forward each layer, then the loss, then backward each
  // layer, then the weight update. Returns false once the whole step is complete.
  bool learn_step_advance() {
    const int n = static_cast<int>(layers.size());
    if (!learn_on) return false;
    if (learn_fwd < n) {
      compute_layer(learn_fwd);
      const int a = learn_fwd;
      ++learn_fwd;
      emit_snapshot("step", a);
      return true;
    }
    if (!learn_loss_done) {
      compute_output_delta(learn_target);
      last_loss = loss(learn_target);
      learn_loss_done = true;
      learn_bwd = n - 2;
      emit_snapshot("loss", n - 1);
      return true;
    }
    if (learn_bwd >= 0) {
      compute_hidden_delta(learn_bwd);
      const int a = learn_bwd;
      --learn_bwd;
      emit_snapshot("backward", a);
      return true;
    }
    if (!learn_updated) {
      compute_gradients();
      apply_gradients(learn_lr);
      learn_updated = true;
      ++step;
      emit_snapshot("update", -1);
      return true;
    }
    learn_on = false;
    return false;
  }

  bool learn_active() const { return learn_on; }

  // Compare analytic gradients (backprop) to numerical ones (finite differences).
  // Returns the largest relative error over all parameters — tiny means backprop is right.
  double gradient_check(const std::vector<float>& in, const std::vector<float>& target) {
    run_forward(in);
    backward(target);  // analytic gradients into dW/db
    const float eps = 1e-3f;
    double maxRel = 0.0;

    auto check = [&](float* param, float analytic) {
      const float orig = *param;
      *param = orig + eps; run_forward(in); const float lp = loss(target);
      *param = orig - eps; run_forward(in); const float lm = loss(target);
      *param = orig;
      const double numeric = (lp - lm) / (2.0 * static_cast<double>(eps));
      const double denom = std::abs(numeric) + std::abs(static_cast<double>(analytic)) + 1e-6;
      const double rel = std::abs(numeric - static_cast<double>(analytic)) / denom;
      if (rel > maxRel) maxRel = rel;
    };

    for (Layer& L : layers) {
      for (int i = 0; i < L.W.size(); ++i) check(&L.W.data[i], L.dW.data[i]);
      for (int i = 0; i < L.b.size(); ++i) check(&L.b.data[i], L.db.data[i]);
    }
    run_forward(in);  // restore the unperturbed forward state
    return maxRel;
  }

  // Stepped forward pass: load input, reset every layer's buffers to zero so the
  // uncomputed columns read as "pending", and show just the input column.
  void begin_forward(const std::vector<float>& in) {
    set_input(in);
    for (Layer& L : layers) {
      for (int i = 0; i < L.pre.size(); ++i) L.pre.data[i] = 0.0f;
      for (int i = 0; i < L.act.size(); ++i) L.act.data[i] = 0.0f;
    }
    cursor = 0;
    emit_snapshot("begin", -1);
  }

  // Compute the next single layer. Returns false once the pass is complete.
  bool step_forward() {
    if (cursor >= static_cast<int>(layers.size())) return false;
    compute_layer(cursor);
    const int active = cursor;
    ++cursor;
    const bool done = cursor >= static_cast<int>(layers.size());
    emit_snapshot(done ? "done" : "step", active);
    if (done) ++step;
    return true;
  }

  bool forward_done() const { return cursor >= static_cast<int>(layers.size()); }
};

Network::Network() : impl_(std::make_unique<Impl>()) {}
Network::~Network() = default;
Network::Network(Network&&) noexcept = default;
Network& Network::operator=(Network&&) noexcept = default;

void Network::build(const Topology& t, unsigned seed) { impl_->build(t, seed); }
const Topology& Network::topology() const { return impl_->topo; }
bool Network::valid() const { return impl_->built; }
std::vector<float> Network::forward(const std::vector<float>& in) { return impl_->forward(in); }
void Network::begin_forward(const std::vector<float>& in) { impl_->begin_forward(in); }
bool Network::step_forward() { return impl_->step_forward(); }
bool Network::forward_done() const { return impl_->forward_done(); }

float Network::loss(const std::vector<float>& target) const { return impl_->loss(target); }
float Network::train_step(const std::vector<float>& in, const std::vector<float>& target,
                          float lr) {
  return impl_->train_step(in, target, lr);
}
double Network::gradient_check(const std::vector<float>& in, const std::vector<float>& target) {
  return impl_->gradient_check(in, target);
}
float Network::evaluate_loss(const std::vector<float>& in, const std::vector<float>& target) {
  return impl_->evaluate_loss(in, target);
}
void Network::begin_learn_step(const std::vector<float>& in, const std::vector<float>& target,
                               float lr) {
  impl_->begin_learn_step(in, target, lr);
}
bool Network::learn_step_advance() { return impl_->learn_step_advance(); }
bool Network::learn_active() const { return impl_->learn_active(); }

void Network::set_observer(Observer* obs) {
  impl_->obs = obs;
  impl_->notify_topology();
}

Network Network::from_json_file(const std::string& path, unsigned seed) {
  Topology t = load_topology_file(path);
  Network n;
  n.build(t, seed);
  return n;
}

}  // namespace synapse
