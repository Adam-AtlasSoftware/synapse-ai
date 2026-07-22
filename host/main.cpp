// synapse_engine_host — the engine running as its own process.
//
// Speaks the telemetry contract over pipes: newline-delimited JSON commands on stdin,
// newline-delimited JSON events on stdout. The dashboard drives it and renders whatever
// Topology/StepSnapshot it is handed, exactly as it does in-process — which is the whole
// point of keeping telemetry.hpp free of SYCL and Qt.
//
// Granularity matters: the dashboard used to call train_step() once per sample (tens of
// thousands of calls per UI tick). That would be hopeless as RPC, so the training LOOP
// lives here — one "train" command runs as many epochs as fit in a time budget.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "json.hpp"
#include "protocol.hpp"
#include "synapse/activation.hpp"
#include "synapse/device_info.hpp"
#include "synapse/metrics.hpp"
#include "synapse/model_spec.hpp"
#include "synapse/network.hpp"
#include "synapse/optimizer.hpp"

using namespace synapse;

namespace {

void emit(const std::string& line) {
  std::cout << line << "\n";
  std::cout.flush();  // the dashboard is waiting on this pipe
}

std::vector<float> floats(const json::Value* v) {
  std::vector<float> out;
  if (!v || !v->is_array()) return out;
  out.reserve(v->arr_v.size());
  for (const json::Value& e : v->arr_v) out.push_back(static_cast<float>(e.num_v));
  return out;
}

// The four numbers the Training panel lives on, as JSON object fields.
std::string metrics_json(const Metrics& m) {
  return "\"train_loss\":" + std::to_string(m.train_loss) +
         ",\"val_loss\":" + std::to_string(m.val_loss) +
         ",\"train_acc\":" + std::to_string(m.train_acc) +
         ",\"val_acc\":" + std::to_string(m.val_acc) +
         ",\"train_n\":" + std::to_string(m.train_n) +
         ",\"val_n\":" + std::to_string(m.val_n) + ",\"epoch\":" + std::to_string(m.epoch);
}

double number_or(const json::Value& o, const std::string& key, double fallback) {
  const json::Value* v = o.find(key);
  return (v && v->type == json::Value::Type::Number) ? v->num_v : fallback;
}

// Streams every snapshot the engine produces straight out as an event.
struct StdoutObserver : Observer {
  void on_topology(const Topology& t) override { emit(proto::event_topology(t)); }
  void on_step(const StepSnapshot& s) override { emit(proto::event_step(s)); }
};

struct Host {
  Network net;
  Topology topo;
  Dataset dataset;
  DatasetSplit split;
  float val_fraction = 0.0f;  // 0 = train on everything (the default)
  StdoutObserver observer;
  std::mt19937 rng{1};
  int epoch = 0;
  std::vector<float> flat_in, flat_out;
  bool flat_dirty = true;

  void rebuild(unsigned seed) {
    net.build(topo, seed);
    net.set_observer(&observer);
    epoch = 0;
    flat_dirty = true;
    reslice();
  }

  void reslice() { split.rebuild(dataset.size(), val_fraction); }

  // Loss + accuracy over a set of sample indices, using the silent predict() so a full
  // sweep never floods the GUI with snapshots.
  void score(const std::vector<int>& idx, float* loss, float* acc) const {
    if (idx.empty()) {
      *loss = 0.0f;
      *acc = 0.0f;
      return;
    }
    float l = 0.0f;
    int correct = 0;
    for (int i : idx) {
      const Sample& s = dataset.samples[static_cast<size_t>(i)];
      const std::vector<float> out = const_cast<Network&>(net).predict(s.input);
      l += const_cast<Network&>(net).evaluate_loss(s.input, s.target);
      if (prediction_correct(out, s.target)) ++correct;
    }
    *loss = l / static_cast<float>(idx.size());
    *acc = static_cast<float>(correct) / static_cast<float>(idx.size());
  }

  Metrics measure() {
    Metrics m;
    score(split.train, &m.train_loss, &m.train_acc);
    score(split.val, &m.val_loss, &m.val_acc);
    m.train_n = static_cast<int>(split.train.size());
    m.val_n = static_cast<int>(split.val.size());
    m.epoch = epoch;
    return m;
  }

  // Flatten only the TRAINING samples — held-out data must never reach the optimizer.
  void ensure_flat() {
    if (!flat_dirty) return;
    flat_in.clear();
    flat_out.clear();
    for (int i : split.train) {
      const Sample& s = dataset.samples[static_cast<size_t>(i)];
      flat_in.insert(flat_in.end(), s.input.begin(), s.input.end());
      flat_out.insert(flat_out.end(), s.target.begin(), s.target.end());
    }
    flat_dirty = false;
  }

  // One shuffled pass of per-sample SGD over the TRAINING split; returns the mean loss.
  float run_one_epoch(float lr) {
    if (split.train.empty()) return 0.0f;
    std::vector<int> order = split.train;
    std::shuffle(order.begin(), order.end(), rng);
    float sum = 0.0f;
    for (int idx : order) {
      const Sample& s = dataset.samples[static_cast<size_t>(idx)];
      sum += net.train_step(s.input, s.target, lr);
    }
    ++epoch;
    return sum / static_cast<float>(order.size());
  }
};

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  Host host;
  std::string line;

  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    long id = 0;
    try {
      const json::Value msg = json::parse(line);
      id = static_cast<long>(number_or(msg, "id", 0));
      const std::string cmd = msg.string_or("cmd", "");

      if (cmd == "quit") break;

      if (cmd == "load") {
        const std::string path = msg.string_or("path", "");
        host.topo = load_topology_file(path);
        host.dataset = load_dataset_file(path);
        host.rebuild(static_cast<unsigned>(number_or(msg, "seed", 42)));
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"ok\":true,\"dataset_size\":" + std::to_string(host.dataset.size()) + "}");

      } else if (cmd == "build") {
        // The model-spec fields (name/input_dim/layers) ride at the TOP level of the
        // command, so the engine's own parser can read the line as-is; it ignores
        // the extra "id"/"cmd"/"seed" keys.
        host.topo = parse_topology_json(line);
        host.rebuild(static_cast<unsigned>(number_or(msg, "seed", 42)));
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) + ",\"ok\":true}");

      } else if (cmd == "set_dataset") {
        host.dataset.samples.clear();
        if (const json::Value* arr = msg.find("samples")) {
          for (const json::Value& s : arr->arr_v)
            host.dataset.samples.push_back(Sample{floats(s.find("input")), floats(s.find("target"))});
        }
        host.flat_dirty = true;
        host.reslice();
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"dataset_size\":" + std::to_string(host.dataset.size()) + "}");

      } else if (cmd == "set_split") {
        host.val_fraction = static_cast<float>(number_or(msg, "val_fraction", 0.0));
        host.flat_dirty = true;
        host.reslice();
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"train_n\":" + std::to_string(host.split.train.size()) +
             ",\"val_n\":" + std::to_string(host.split.val.size()) + "}");

      } else if (cmd == "metrics") {
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) + "," +
             metrics_json(host.measure()) + "}");

      } else if (cmd == "set_optimizer") {
        host.net.set_optimizer(msg.string_or("name", "sgd"));
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) + ",\"optimizer\":\"" +
             proto::escape(host.net.optimizer()) + "\"}");

      } else if (cmd == "optimizers") {
        std::string j = "{\"ev\":\"result\",\"id\":" + std::to_string(id) + ",\"names\":[";
        const std::vector<std::string> names = optimizer_names();
        for (size_t i = 0; i < names.size(); ++i) {
          if (i) j += ",";
          j += "\"" + proto::escape(names[i]) + "\"";
        }
        emit(j + "]}");

      } else if (cmd == "get_params") {
        const std::vector<float> p = host.net.parameters();
        std::string j = "{\"ev\":\"result\",\"id\":" + std::to_string(id) +
                        ",\"count\":" + std::to_string(p.size()) + ",\"params\":[";
        for (size_t i = 0; i < p.size(); ++i) {
          if (i) j += ",";
          j += std::to_string(p[i]);
        }
        emit(j + "]}");

      } else if (cmd == "set_params") {
        const bool ok = host.net.set_parameters(floats(msg.find("params")));
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"ok\":" + (ok ? "true" : "false") + "}");

      } else if (cmd == "forward") {
        const std::vector<float> out = host.net.forward(floats(msg.find("input")));
        std::string j = "{\"ev\":\"result\",\"id\":" + std::to_string(id) + ",\"output\":[";
        for (size_t i = 0; i < out.size(); ++i) {
          if (i) j += ",";
          j += std::to_string(out[i]);
        }
        emit(j + "]}");

      } else if (cmd == "begin_forward") {
        host.net.begin_forward(floats(msg.find("input")));
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) + ",\"ok\":true}");

      } else if (cmd == "step_forward") {
        const bool more = host.net.step_forward();
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"more\":" + (more ? "true" : "false") + "}");

      } else if (cmd == "forward_done") {
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"done\":" + (host.net.forward_done() ? "true" : "false") + "}");

      } else if (cmd == "learn_active") {
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"active\":" + (host.net.learn_active() ? "true" : "false") + "}");


      } else if (cmd == "begin_learn") {
        host.net.begin_learn_step(floats(msg.find("input")), floats(msg.find("target")),
                                  static_cast<float>(number_or(msg, "lr", 0.1)));
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) + ",\"ok\":true}");

      } else if (cmd == "learn_advance") {
        const bool more = host.net.learn_step_advance();
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"more\":" + (more ? "true" : "false") + "}");

      } else if (cmd == "train") {
        // The loop that used to live in the GUI: run as many epochs as fit the budget.
        const float lr = static_cast<float>(number_or(msg, "lr", 0.1));
        const int budget_ms = static_cast<int>(number_or(msg, "budget_ms", 20));
        const int max_epochs = static_cast<int>(number_or(msg, "max_epochs", 2000));
        const bool gpu = msg.string_or("mode", "cpu") == "gpu";
        std::vector<float> losses;
        const auto t0 = std::chrono::steady_clock::now();
        int did = 0;
        if (!host.dataset.empty()) {
          do {
            float l;
            if (gpu) {
              host.ensure_flat();
              l = host.net.train_epoch_batched(host.flat_in, host.flat_out, host.dataset.size(), lr);
              ++host.epoch;
            } else {
              l = host.run_one_epoch(lr);
            }
            losses.push_back(l);
            ++did;
          } while (std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count() < budget_ms &&
                   did < max_epochs);
        }
        // Re-run the caller's current input so the GUI sees the updated network.
        if (const json::Value* ri = msg.find("refresh_input"))
          if (ri->is_array() && !ri->arr_v.empty()) host.net.forward(floats(ri));

        std::string j = "{\"ev\":\"result\",\"id\":" + std::to_string(id) + "," +
                        metrics_json(host.measure()) + ",\"losses\":[";
        for (size_t i = 0; i < losses.size(); ++i) {
          if (i) j += ",";
          j += std::to_string(losses[i]);
        }
        emit(j + "]}");

      } else if (cmd == "eval_loss") {
        const float l = host.net.evaluate_loss(floats(msg.find("input")), floats(msg.find("target")));
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"loss\":" + std::to_string(l) + "}");

      } else if (cmd == "gradient_check") {
        const double e = host.net.gradient_check(floats(msg.find("input")), floats(msg.find("target")));
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) +
             ",\"error\":" + std::to_string(e) + "}");

      } else if (cmd == "activations") {
        std::string j = "{\"ev\":\"result\",\"id\":" + std::to_string(id) + ",\"names\":[";
        const std::vector<std::string> names = activation_names();
        for (size_t i = 0; i < names.size(); ++i) {
          if (i) j += ",";
          j += "\"" + proto::escape(names[i]) + "\"";
        }
        emit(j + "]}");

      } else if (cmd == "device") {
        emit("{\"ev\":\"result\",\"id\":" + std::to_string(id) + ",\"device\":\"" +
             proto::escape(active_device_name()) + "\"}");

      } else {
        emit(proto::event_error(id, "unknown command '" + cmd + "'"));
      }
    } catch (const std::exception& e) {
      emit(proto::event_error(id, e.what()));  // never let one bad command kill the engine
    }
  }
  return 0;
}
