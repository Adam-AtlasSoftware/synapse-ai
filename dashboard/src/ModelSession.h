#pragma once
#include <QJsonObject>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <memory>
#include <random>
#include <vector>

#include "synapse/metrics.hpp"
#include "synapse/model_spec.hpp"
#include "synapse/network.hpp"
#include "synapse/telemetry.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// The engine behind ONE interface, so the dashboard does not care whether it is
// linked in-process or running as a separate OS process speaking JSON over pipes.
//
// The API is synchronous on purpose. The bridge already does its engine work on the
// main thread in small time-budgeted slices, so a blocking round-trip over a local
// pipe costs no more than the in-process call it replaces — and it keeps the bridge's
// control flow identical between the two backends.
//
// Note the granularity: train() runs a whole budget's worth of epochs in ONE call.
// Per-sample train_step() over a pipe would be tens of thousands of round-trips a tick.
// ─────────────────────────────────────────────────────────────────────────────
class ModelSession {
 public:
  virtual ~ModelSession() = default;

  virtual bool usable() const { return true; }
  virtual QString kind() const = 0;  // "in-process" | "subprocess"

  virtual void setObserver(synapse::Observer* obs) = 0;
  virtual void build(const synapse::Topology& topo, unsigned seed) = 0;
  virtual void setDataset(const synapse::Dataset& ds) = 0;

  virtual std::vector<float> forward(const std::vector<float>& in) = 0;
  virtual void beginForward(const std::vector<float>& in) = 0;
  virtual bool stepForward() = 0;
  virtual bool forwardDone() = 0;

  virtual void beginLearn(const std::vector<float>& in, const std::vector<float>& target,
                          float lr) = 0;
  virtual bool learnAdvance() = 0;
  virtual bool learnActive() = 0;

  // As many epochs as fit `budgetMs` (capped at maxEpochs), on the TRAINING split only.
  // Returns loss/accuracy for both splits plus the running epoch count.
  virtual synapse::Metrics train(float lr, int budgetMs, int maxEpochs, bool gpu,
                                 const std::vector<float>& refreshInput) = 0;

  // Score the current weights without training (the loss curve's first point).
  virtual synapse::Metrics metrics() = 0;

  // Hold out this fraction of the dataset for validation; 0 = train on everything.
  virtual void setSplit(float valFraction) = 0;

  // How weights move once backprop knows the direction: "sgd" | "momentum" | "adam" | custom.
  virtual void setOptimizer(const QString& name) = 0;
  virtual QStringList optimizerNames() = 0;

  // Flat parameter vector — lets training be saved and survive an engine hot-swap.
  virtual std::vector<float> parameters() = 0;
  virtual bool setParameters(const std::vector<float>& p) = 0;

  virtual QString deviceName() = 0;
  virtual QStringList activationNames() = 0;
};

// ── the engine linked straight into the GUI (the original, still the default) ──
class InProcessSession : public ModelSession {
 public:
  QString kind() const override { return QStringLiteral("in-process"); }
  void setObserver(synapse::Observer* obs) override { m_net.set_observer(obs); }
  void build(const synapse::Topology& topo, unsigned seed) override;
  void setDataset(const synapse::Dataset& ds) override;

  std::vector<float> forward(const std::vector<float>& in) override { return m_net.forward(in); }
  void beginForward(const std::vector<float>& in) override { m_net.begin_forward(in); }
  bool stepForward() override { return m_net.step_forward(); }
  bool forwardDone() override { return m_net.forward_done(); }

  void beginLearn(const std::vector<float>& in, const std::vector<float>& target,
                  float lr) override {
    m_net.begin_learn_step(in, target, lr);
  }
  bool learnAdvance() override { return m_net.learn_step_advance(); }
  bool learnActive() override { return m_net.learn_active(); }

  synapse::Metrics train(float lr, int budgetMs, int maxEpochs, bool gpu,
                         const std::vector<float>& refreshInput) override;
  synapse::Metrics metrics() override;
  void setSplit(float valFraction) override;
  void setOptimizer(const QString& name) override;
  QStringList optimizerNames() override;
  std::vector<float> parameters() override { return m_net.parameters(); }
  bool setParameters(const std::vector<float>& p) override { return m_net.set_parameters(p); }

  QString deviceName() override;
  QStringList activationNames() override;

 private:
  float runOneEpoch(float lr);
  void ensureFlat();
  void reslice();
  void score(const std::vector<int>& idx, float* loss, float* acc);

  synapse::Network m_net;
  synapse::Dataset m_dataset;
  synapse::DatasetSplit m_split;
  float m_valFraction = 0.0f;
  std::mt19937 m_rng{1};
  int m_epoch = 0;
  std::vector<float> m_flatIn, m_flatOut;
  bool m_flatDirty = true;
};

// ── the engine as its own process, driven over stdin/stdout JSON ──────────────
class SubprocessSession : public ModelSession {
 public:
  explicit SubprocessSession(const QString& exePath);
  ~SubprocessSession() override;

  bool usable() const override { return m_ok; }
  QString kind() const override { return QStringLiteral("subprocess"); }
  void setObserver(synapse::Observer* obs) override { m_obs = obs; }
  void build(const synapse::Topology& topo, unsigned seed) override;
  void setDataset(const synapse::Dataset& ds) override;

  std::vector<float> forward(const std::vector<float>& in) override;
  void beginForward(const std::vector<float>& in) override;
  bool stepForward() override;
  bool forwardDone() override;

  void beginLearn(const std::vector<float>& in, const std::vector<float>& target,
                  float lr) override;
  bool learnAdvance() override;
  bool learnActive() override;

  synapse::Metrics train(float lr, int budgetMs, int maxEpochs, bool gpu,
                         const std::vector<float>& refreshInput) override;
  synapse::Metrics metrics() override;
  void setSplit(float valFraction) override;
  void setOptimizer(const QString& name) override;
  QStringList optimizerNames() override;
  std::vector<float> parameters() override;
  bool setParameters(const std::vector<float>& p) override;

  QString deviceName() override;
  QStringList activationNames() override;

 private:
  // Send one command and pump the pipe until its reply arrives, delivering any
  // topology/step events to the observer on the way.
  QJsonObject request(QJsonObject cmd, int timeoutMs = 20000);

  QProcess m_proc;
  synapse::Observer* m_obs = nullptr;
  long m_nextId = 0;
  bool m_ok = false;
};
