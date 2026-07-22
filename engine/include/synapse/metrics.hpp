#pragma once
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

// Shared, header-only definitions for "how well is it doing?" — used by BOTH engine
// backends (the in-process session and the out-of-process host) so they can never drift
// apart and report different numbers for the same model.
//
// Pure C++/STL: no SYCL, no Qt.
namespace synapse {

// Did the network get this example right?
//   single output  → a threshold at 0.5 (XOR, adder bits)
//   many outputs   → argmax matches the target's argmax (classifiers)
// Same rule tests/train_smoke.cpp has always used to judge a trained net.
inline bool prediction_correct(const std::vector<float>& out, const std::vector<float>& target) {
  if (out.empty() || target.empty()) return false;
  if (target.size() == 1) return (out[0] > 0.5f) == (target[0] > 0.5f);
  const size_t n = std::min(out.size(), target.size());
  size_t bo = 0, bt = 0;
  for (size_t i = 1; i < n; ++i) {
    if (out[i] > out[bo]) bo = i;
    if (target[i] > target[bt]) bt = i;
  }
  return bo == bt;
}

// What the Training panel shows. Accuracy is only meaningful for classifiers; the GUI
// decides whether to display it based on the blueprint's output kind.
struct Metrics {
  float train_loss = 0.0f;
  float val_loss = 0.0f;
  float train_acc = 0.0f;
  float val_acc = 0.0f;
  int train_n = 0;
  int val_n = 0;
  int epoch = 0;
};

// A deterministic train/validation partition of a dataset's indices. Held out data is
// never trained on, which is what makes overfitting visible: training loss keeps falling
// while validation loss turns back up.
struct DatasetSplit {
  std::vector<int> train;
  std::vector<int> val;

  bool has_validation() const { return !val.empty(); }

  // fraction <= 0 (or too small a dataset) means "train on everything".
  void rebuild(int n, float val_fraction, unsigned seed = 1234) {
    train.clear();
    val.clear();
    if (n <= 0) return;
    std::vector<int> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    int n_val = (val_fraction > 0.0f) ? static_cast<int>(static_cast<float>(n) * val_fraction) : 0;
    // Never hold out so much that training has nothing left.
    n_val = std::max(0, std::min(n_val, n - 1));
    if (n_val > 0) {
      std::mt19937 rng(seed);
      std::shuffle(idx.begin(), idx.end(), rng);
      val.assign(idx.begin(), idx.begin() + n_val);
      train.assign(idx.begin() + n_val, idx.end());
      std::sort(val.begin(), val.end());
      std::sort(train.begin(), train.end());
    } else {
      train = std::move(idx);
    }
  }
};

}  // namespace synapse
