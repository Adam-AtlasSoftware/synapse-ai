// Headless proof that the network LEARNS and that backprop is correct — for both
// output styles: sigmoid+MSE (e.g. XOR) and softmax+cross-entropy (classifiers).
//   1. gradient_check — analytic gradients match finite differences
//   2. training drives the loss down
//   3. the trained net predicts every training sample correctly
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "synapse/model_spec.hpp"
#include "synapse/network.hpp"

using namespace synapse;

static int argmax(const std::vector<float>& v) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(v.size()); ++i)
    if (v[i] > v[best]) best = i;
  return best;
}

// Single-output nets (XOR, adder bits) use a 0.5 threshold; multi-output nets are
// classifiers judged by argmax.
static bool correct(const std::vector<float>& y, const std::vector<float>& t) {
  if (t.size() == 1) return (y[0] > 0.5f) == (t[0] > 0.5f);
  return argmax(y) == argmax(t);
}

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "models/blueprints/xor.json";
  try {
    Network net = Network::from_json_file(path);
    Dataset ds = load_dataset_file(path);
    if (ds.empty()) {
      std::cerr << "FAIL: blueprint has no dataset\n";
      return 1;
    }

    // 1) Gradient check — the single most important correctness test.
    const double err = net.gradient_check(ds.samples[0].input, ds.samples[0].target);
    std::cout << "gradient check: max relative error = " << err << "\n";
    if (err > 0.02) {
      std::cerr << "FAIL: gradients disagree with finite differences\n";
      return 2;
    }

    // 2) Train with plain SGD and watch the loss fall.
    std::mt19937 rng(1);
    const float lr = argc > 2 ? std::stof(argv[2]) : 0.3f;
    const int epochs = argc > 3 ? std::stoi(argv[3]) : 5000;
    std::vector<int> order(ds.size());
    std::iota(order.begin(), order.end(), 0);
    float firstLoss = 0.0f, lastLoss = 0.0f;
    for (int e = 0; e < epochs; ++e) {
      std::shuffle(order.begin(), order.end(), rng);
      float epochLoss = 0.0f;
      for (int idx : order)
        epochLoss += net.train_step(ds.samples[idx].input, ds.samples[idx].target, lr);
      epochLoss /= ds.size();
      if (e == 0) firstLoss = epochLoss;
      lastLoss = epochLoss;
    }
    std::cout << "loss: " << firstLoss << "  ->  " << lastLoss << "\n";
    if (!(lastLoss < firstLoss * 0.5f)) {
      std::cerr << "FAIL: loss did not fall meaningfully\n";
      return 3;
    }

    // 3) Every training sample must now be classified correctly.
    int right = 0;
    for (const Sample& s : ds.samples)
      if (correct(net.forward(s.input), s.target)) ++right;
    std::cout << "accuracy: " << right << "/" << ds.size() << "\n";
    if (right != ds.size()) {
      std::cerr << "FAIL: trained net misclassifies its own training data\n";
      return 4;
    }
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 5;
  }

  std::cout << "OK\n";
  return 0;
}
