// The auto-training coach's policy, tested against synthetic loss curves. Driving it here
// rather than through the GUI means we can assert it does the right thing in situations that
// are slow or fiddly to reproduce by hand — a plateau, a blow-up, a run that starts overfitting.
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "AutoTrainer.h"

using synapse::Metrics;

namespace {

Metrics mk(float trainLoss, float valLoss = 0.0f, float trainAcc = 0.0f, float valAcc = 0.0f) {
  Metrics m;
  m.train_loss = trainLoss;
  m.val_loss = valLoss;
  m.train_acc = trainAcc;
  m.val_acc = valAcc;
  return m;
}

const char* kindName(AutoTrainer::Action::Kind k) {
  switch (k) {
    case AutoTrainer::Action::Continue: return "continue";
    case AutoTrainer::Action::LowerLr:  return "lower-lr";
    case AutoTrainer::Action::Rewind:   return "rewind";
    case AutoTrainer::Action::Stop:     return "stop";
  }
  return "?";
}

int failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

}  // namespace

int main() {
  // 1) A plateau should make it lower the learning rate rather than spin forever.
  {
    AutoTrainer at;
    at.patience = 4;
    at.begin(0.10f, false, true);
    at.step(mk(1.0f));                       // first look
    at.step(mk(0.5f));                       // improving
    bool lowered = false;
    for (int i = 0; i < 10 && !lowered; ++i) {
      AutoTrainer::Action a = at.step(mk(0.5f));   // flat: no improvement at all
      if (a.kind == AutoTrainer::Action::LowerLr) {
        lowered = true;
        check(a.lr < 0.10f, "plateau lowered the learning rate");
        std::cout << "plateau  -> " << kindName(a.kind) << " lr=" << a.lr << "  \"" << a.message
                  << "\"\n";
      }
    }
    check(lowered, "a plateau eventually lowers the learning rate");
  }

  // 2) A blow-up should rewind to the best weights and back the rate off hard.
  {
    AutoTrainer at;
    at.begin(0.5f, false, true);
    at.step(mk(1.0f));
    at.step(mk(0.2f));                                   // best so far
    AutoTrainer::Action a = at.step(mk(50.0f));          // diverged
    check(a.kind == AutoTrainer::Action::Rewind, "divergence rewinds");
    check(a.lr < 0.5f, "divergence lowers the learning rate");
    std::cout << "diverge  -> " << kindName(a.kind) << " lr=" << a.lr << "  \"" << a.message
              << "\"\n";

    // NaN must be handled too, not propagated.
    AutoTrainer at2;
    at2.begin(0.5f, false, true);
    at2.step(mk(1.0f));
    AutoTrainer::Action b = at2.step(mk(std::numeric_limits<float>::quiet_NaN()));
    check(b.kind == AutoTrainer::Action::Rewind, "NaN loss rewinds");
  }

  // 3) The headline case: training loss keeps falling while held-out loss climbs.
  //    It must stop early rather than happily memorizing.
  {
    AutoTrainer at;
    at.overfitPatience = 3;
    at.begin(0.1f, true, true);
    at.step(mk(1.0f, 1.0f));
    float tr = 0.9f, va = 0.9f;
    bool stopped = false;
    for (int i = 0; i < 12 && !stopped; ++i) {
      tr *= 0.7f;                     // training keeps getting better
      va += 0.05f;                    // held-out keeps getting worse
      AutoTrainer::Action a = at.step(mk(tr, va, 1.0f, 0.4f));
      if (a.kind == AutoTrainer::Action::Stop) {
        stopped = true;
        std::cout << "overfit  -> " << kindName(a.kind) << "  \"" << a.message << "\"\n";
      }
    }
    check(stopped, "rising held-out loss stops training early");
    check(at.finished(), "the coach reports itself finished after stopping");
    check(at.outcome() == AutoTrainer::Outcome::Overfit, "overfitting is reported as the outcome");
  }

  // 4) Perfect accuracy should stop, not keep grinding.
  {
    AutoTrainer at;
    at.begin(0.1f, false, true);
    at.step(mk(1.0f));
    AutoTrainer::Action a = at.step(mk(0.9f, 0.0f, 1.0f));  // no loss improvement, but perfect
    check(a.kind == AutoTrainer::Action::Stop, "perfect accuracy stops training");
    check(a.isBest, "the winning weights are snapshotted on the tick it finishes");
    check(at.outcome() == AutoTrainer::Outcome::Converged, "success is reported as the outcome");
    std::cout << "perfect  -> " << kindName(a.kind) << " isBest=" << a.isBest << "  \"" << a.message
              << "\"\n";
  }

  // 5) A healthy run should just keep going, snapshotting as it improves.
  {
    AutoTrainer at;
    at.begin(0.1f, false, true);
    at.step(mk(1.0f));
    int snapshots = 0;
    float l = 1.0f;
    for (int i = 0; i < 20; ++i) {
      l *= 0.8f;
      AutoTrainer::Action a = at.step(mk(l));
      if (a.isBest) ++snapshots;
      check(a.kind != AutoTrainer::Action::Stop, "a steadily improving run is not stopped");
    }
    check(snapshots > 10, "improvement is snapshotted as the new best");
    std::cout << "healthy  -> " << snapshots << " snapshots, still running\n";
  }

  // 6) The case a beginner gets stranded by: it stops improving while still scoring badly.
  //    The outcome must say "stalled" (not "converged") and the message must quote the score,
  //    so the UI can offer somewhere to go from here.
  {
    AutoTrainer at;
    at.patience = 2;
    at.minLr = 0.05f;              // hit the floor quickly
    at.begin(0.1f, false, true);
    at.step(mk(2.0f, 0.0f, 0.30f));
    std::string last;
    for (int i = 0; i < 30 && !at.finished(); ++i) {
      AutoTrainer::Action a = at.step(mk(2.0f, 0.0f, 0.30f));   // flat, and only 30% right
      if (!a.message.empty()) last = a.message;
    }
    check(at.finished(), "a permanent plateau eventually stops");
    check(at.outcome() == AutoTrainer::Outcome::Stalled, "a poor plateau reports Stalled");
    check(last.find("30%") != std::string::npos,
          "the final message quotes how good it actually got");
    std::cout << "stalled  -> outcome=stalled  \"" << last << "\"\n";
  }

  // 7) Zero TRAINING error while held-out examples still fail is not success — it is the
  //    textbook definition of memorizing. The coach must not congratulate the user here.
  {
    AutoTrainer at;
    at.begin(0.1f, true, true);
    at.step(mk(1.0f, 1.0f, 0.2f, 0.2f));
    // train loss ~0 and every studied example right, but nothing generalizes
    AutoTrainer::Action a = at.step(mk(0.00001f, 3.0f, 1.0f, 0.0f));
    check(a.kind == AutoTrainer::Action::Stop, "zero training error stops");
    check(at.outcome() == AutoTrainer::Outcome::Overfit,
          "zero training error + failing held-out data is reported as memorizing, not success");
    check(a.message.find("memoriz") != std::string::npos,
          "the message says it memorized rather than congratulating");
    std::cout << "fake-win -> " << kindName(a.kind) << "  \"" << a.message << "\"\n";
  }

  if (failures) return 1;
  std::cout << "OK — auto-training policy verified\n";
  return 0;
}
