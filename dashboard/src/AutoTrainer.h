#pragma once
#include <string>

#include "synapse/metrics.hpp"

// The training coach. Watches the numbers each tick and decides what to do about them, so a
// beginner never has to know what a learning rate is or what "it stopped improving" means.
//
// Deliberately plain C++ with no Qt and no engine calls: it takes metrics in and returns a
// decision out, which makes the whole policy unit-testable against synthetic loss curves
// (tests/auto_trainer_smoke.cpp) instead of only being observable through the GUI.
class AutoTrainer {
 public:
  struct Action {
    enum Kind {
      Continue,   // nothing to do
      LowerLr,    // caller should apply `lr` and keep going
      Rewind,     // restore the best weights AND apply `lr` (training blew up)
      Stop        // done; restore the best weights
    };
    Kind kind = Continue;
    // Independent of `kind`: the weights as they stand right now are the best seen, so the
    // caller should snapshot them. Kept separate so a run can improve *and* finish on the
    // same tick (hitting 100% accuracy) without the Stop rewinding away that improvement.
    bool isBest = false;
    float lr = 0.0f;
    std::string message;  // plain language, shown verbatim to the user
  };

  // Why training ended. The UI turns this into concrete advice — "it stopped improving" is
  // useless on its own if the model still scores badly, which is exactly where a beginner
  // gets stranded.
  enum class Outcome {
    None,       // still running
    Converged,  // it got there
    Stalled,    // stopped improving — may or may not be any good
    Overfit,    // held-out score started getting worse
    Diverged    // training kept blowing up
  };

  // Called once when auto-training starts. `accuracyMeaningful` is false for regression-style
  // models, where an accuracy percentage would be nonsense to quote back at the user.
  void begin(float lr, bool hasValidation, bool accuracyMeaningful);

  // Called once per training tick with the freshly measured metrics.
  Action step(const synapse::Metrics& m);

  bool finished() const { return m_done; }
  float learningRate() const { return m_lr; }
  Outcome outcome() const { return m_outcome; }
  // Best score reached, and the accuracy at the end — the UI needs these to advise.
  float finalAccuracy() const { return m_finalAcc; }

  // Tunables (exposed mainly so the test can drive the policy quickly).
  int patience = 12;         // ticks without improvement before easing off the learning rate
  int overfitPatience = 6;   // consecutive ticks of rising held-out loss before stopping
  float minLr = 1e-4f;       // below this there is nothing left to gain
  int maxTicks = 4000;       // hard stop so it can never run forever

 private:
  float score(const synapse::Metrics& m) const;  // what "better" means right now
  bool finishedWell() const;                     // did it actually learn the task?

  float m_lr = 0.1f;
  bool m_hasVal = false;
  bool m_accMeaningful = false;
  bool m_done = false;
  bool m_started = false;
  Outcome m_outcome = Outcome::None;
  float m_finalAcc = 0.0f;
  float m_best = 0.0f;
  float m_lastVal = 0.0f;
  int m_sinceImproved = 0;
  int m_valRising = 0;
  int m_ticks = 0;
};
