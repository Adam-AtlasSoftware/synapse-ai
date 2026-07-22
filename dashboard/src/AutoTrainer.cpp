#include "AutoTrainer.h"

#include <cmath>
#include <sstream>

namespace {
// Format a number the way a person would say it, not the way a float prints.
std::string num(float v, int places = 4) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(places);
  os << v;
  return os.str();
}
std::string pct(float v) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(0);
  os << v * 100.0f << "%";
  return os.str();
}
}  // namespace

// "Well" means it actually learned the task, not merely that the numbers settled.
bool AutoTrainer::finishedWell() const {
  return m_accMeaningful ? m_finalAcc >= 0.9f : false;
}

void AutoTrainer::begin(float lr, bool hasValidation, bool accuracyMeaningful) {
  m_lr = lr;
  m_hasVal = hasValidation;
  m_accMeaningful = accuracyMeaningful;
  m_outcome = Outcome::None;
  m_finalAcc = 0.0f;
  m_done = false;
  m_started = false;
  m_best = 0.0f;
  m_lastVal = 0.0f;
  m_sinceImproved = 0;
  m_valRising = 0;
  m_ticks = 0;
}

// When data is held out, the honest measure of "better" is how it does on data it has never
// seen. Otherwise all we have is the training loss.
float AutoTrainer::score(const synapse::Metrics& m) const {
  return m_hasVal ? m.val_loss : m.train_loss;
}

AutoTrainer::Action AutoTrainer::step(const synapse::Metrics& m) {
  Action a;
  if (m_done) return a;

  // The score we quote back is the honest one: how it does on data it never trained on,
  // when such data exists.
  m_finalAcc = m_hasVal ? m.val_acc : m.train_acc;
  ++m_ticks;
  const float s = score(m);

  // ── first look: nothing to compare against yet ───────────────────────────
  if (!m_started) {
    m_started = true;
    m_best = s;
    m_lastVal = m.val_loss;
    a.isBest = true;
    a.message = "Training. I'll keep an eye on it and adjust as it goes.";
    return a;
  }

  // ── it blew up: NaN, or far worse than the best we've seen ───────────────
  if (!std::isfinite(s) || (std::isfinite(m_best) && s > m_best * 3.0f + 1e-6f)) {
    m_lr *= 0.3f;
    m_valRising = 0;
    m_sinceImproved = 0;
    a.kind = Action::Rewind;
    a.lr = m_lr;
    a.message = "That went unstable — the steps were too big. I've gone back to the best "
                "version and lowered the learning rate to " + num(m_lr) + ".";
    if (m_lr < minLr) {
      m_done = true;
      m_outcome = Outcome::Diverged;
      a.kind = Action::Stop;
      a.message = "Training kept destabilizing even at tiny steps, so I've stopped and kept "
                  "the best version.";
    }
    return a;
  }

  // ── overfitting: held-out loss climbing while it still fits the training set ──
  if (m_hasVal) {
    if (m.val_loss > m_lastVal + 1e-6f) ++m_valRising;
    else m_valRising = 0;
    m_lastVal = m.val_loss;

    if (m_valRising >= overfitPatience) {
      m_done = true;
      m_outcome = Outcome::Overfit;
      a.kind = Action::Stop;
      a.message = "It started doing worse on the examples it hasn't seen — that's memorizing "
                  "rather than learning. I stopped early and kept the best version"
                  " (" + num(m.val_acc * 100.0f, 0) + "% on held-out examples).";
      return a;
    }
  }

  // ── improving? record it as the best, but keep evaluating: a tick can both improve
  //    and be the moment we're finished. ─────────────────────────────────────
  const bool improved = s < m_best - 1e-5f;
  if (improved) {
    m_best = s;
    m_sinceImproved = 0;
    a.isBest = true;
  }

  // ── good enough to stop ──────────────────────────────────────────────────
  // Driving the TRAINING error to zero is not success on its own: if held-out examples are
  // still being failed, that is precisely memorizing. Declaring victory there would hand the
  // user a congratulation and a 0% score in the same breath.
  const bool perfect = m.train_acc >= 0.999f && (!m_hasVal || m.val_acc >= 0.999f);
  if (m.train_loss < 1e-4f || perfect) {
    m_done = true;
    a.kind = Action::Stop;
    const bool generalizes = !m_hasVal || !m_accMeaningful || m.val_acc >= 0.9f;
    if (!generalizes) {
      m_outcome = Outcome::Overfit;
      a.message = "It now gets everything it studied right, but still fails the held-out "
                  "examples (" + pct(m.val_acc) + ") — it memorized them rather than "
                  "learning the pattern.";
    } else {
      m_outcome = Outcome::Converged;
      a.message = perfect ? "Done — it now gets every example right."
                          : "Done — the error is about as low as it goes.";
    }
    return a;
  }

  if (improved) return a;  // healthy progress, nothing to adjust

  // ── stalled: ease off the step size and give it another go ───────────────
  ++m_sinceImproved;
  if (m_sinceImproved >= patience) {
    m_sinceImproved = 0;
    m_lr *= 0.5f;
    if (m_lr < minLr) {
      m_done = true;
      m_outcome = Outcome::Stalled;
      a.kind = Action::Stop;
      // Saying only "it stopped improving" strands the user when the score is poor. Say how
      // good it actually got, and let the UI offer them somewhere to go from here.
      a.message = finishedWell()
                      ? "Done — it stopped improving, and it's doing well."
                      : (m_accMeaningful
                             ? "I've taken this as far as I can — it stopped improving and is "
                               "still only getting " + pct(m_finalAcc) + " right."
                             : "I've taken this as far as I can — it stopped improving, but "
                               "there's still a lot of error left.");
      return a;
    }
    a.kind = Action::LowerLr;
    a.lr = m_lr;
    a.message = "Progress stalled, so I halved the learning rate to " + num(m_lr) +
                " — smaller steps can settle into a better answer.";
    return a;
  }

  // ── ran long enough ──────────────────────────────────────────────────────
  if (m_ticks >= maxTicks) {
    m_done = true;
    m_outcome = finishedWell() ? Outcome::Converged : Outcome::Stalled;
    a.kind = Action::Stop;
    a.message = "That's a long run — I've stopped and kept the best version.";
  }
  return a;
}
