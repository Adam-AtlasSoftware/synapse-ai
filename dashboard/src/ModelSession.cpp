#include "ModelSession.h"

#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonDocument>
#include <algorithm>
#include <numeric>

#include "synapse/activation.hpp"
#include "synapse/device_info.hpp"
#include "synapse/optimizer.hpp"

namespace {

QJsonArray toArray(const std::vector<float>& v) {
  QJsonArray a;
  for (float f : v) a.append(static_cast<double>(f));
  return a;
}

std::vector<float> toVec(const QJsonArray& a) {
  std::vector<float> v;
  v.reserve(a.size());
  for (const QJsonValue& e : a) v.push_back(static_cast<float>(e.toDouble()));
  return v;
}

// ── the wire format, read back into the plain telemetry structs ──────────────
synapse::Topology parseTopology(const QJsonObject& o) {
  synapse::Topology t;
  t.name = o.value("name").toString().toStdString();
  t.input_dim = o.value("input_dim").toInt();
  for (const QJsonValue& lv : o.value("layers").toArray()) {
    const QJsonObject lo = lv.toObject();
    synapse::LayerInfo L;
    L.name = lo.value("name").toString().toStdString();
    L.type = lo.value("type").toString("dense").toStdString();
    L.input_dim = lo.value("input_dim").toInt();
    L.output_dim = lo.value("output_dim").toInt();
    L.activation = lo.value("activation").toString("sigmoid").toStdString();
    t.layers.push_back(L);
  }
  return t;
}

synapse::StepSnapshot parseStep(const QJsonObject& o) {
  synapse::StepSnapshot s;
  s.step = static_cast<long>(o.value("step").toDouble());
  s.epoch = o.value("epoch").toInt();
  s.loss = static_cast<float>(o.value("loss").toDouble());
  s.phase = o.value("phase").toString().toStdString();
  s.active_layer = o.value("active_layer").toInt(-1);
  for (const QJsonValue& tv : o.value("tensors").toArray()) {
    const QJsonObject to = tv.toObject();
    synapse::TensorView t;
    t.name = to.value("name").toString().toStdString();
    t.rows = to.value("rows").toInt();
    t.cols = to.value("cols").toInt();
    t.data = toVec(to.value("data").toArray());
    s.tensors.push_back(std::move(t));
  }
  return s;
}

}  // namespace

// ═══════════════════════════ in-process ═══════════════════════════════════════

void InProcessSession::build(const synapse::Topology& topo, unsigned seed) {
  m_net.build(topo, seed);
  m_epoch = 0;
  m_flatDirty = true;
  reslice();
}

void InProcessSession::setDataset(const synapse::Dataset& ds) {
  m_dataset = ds;
  m_flatDirty = true;
  reslice();
}

void InProcessSession::setSplit(float valFraction) {
  m_valFraction = valFraction;
  m_flatDirty = true;
  reslice();
}

void InProcessSession::reslice() { m_split.rebuild(m_dataset.size(), m_valFraction); }

// Only the TRAINING split is ever flattened — held-out data must not reach the optimizer.
void InProcessSession::ensureFlat() {
  if (!m_flatDirty) return;
  m_flatIn.clear();
  m_flatOut.clear();
  for (int i : m_split.train) {
    const synapse::Sample& s = m_dataset.samples[static_cast<size_t>(i)];
    m_flatIn.insert(m_flatIn.end(), s.input.begin(), s.input.end());
    m_flatOut.insert(m_flatOut.end(), s.target.begin(), s.target.end());
  }
  m_flatDirty = false;
}

float InProcessSession::runOneEpoch(float lr) {
  if (m_split.train.empty()) return 0.0f;
  std::vector<int> order = m_split.train;
  std::shuffle(order.begin(), order.end(), m_rng);
  float sum = 0.0f;
  for (int idx : order) {
    const synapse::Sample& s = m_dataset.samples[static_cast<size_t>(idx)];
    sum += m_net.train_step(s.input, s.target, lr);
  }
  ++m_epoch;
  return sum / static_cast<float>(order.size());
}

// Identical scoring to the out-of-process host (both use synapse/metrics.hpp), so the two
// backends can never report different numbers for the same model.
void InProcessSession::score(const std::vector<int>& idx, float* loss, float* acc) {
  if (idx.empty()) {
    *loss = 0.0f;
    *acc = 0.0f;
    return;
  }
  float l = 0.0f;
  int correct = 0;
  for (int i : idx) {
    const synapse::Sample& s = m_dataset.samples[static_cast<size_t>(i)];
    const std::vector<float> out = m_net.predict(s.input);
    l += m_net.evaluate_loss(s.input, s.target);
    if (synapse::prediction_correct(out, s.target)) ++correct;
  }
  *loss = l / static_cast<float>(idx.size());
  *acc = static_cast<float>(correct) / static_cast<float>(idx.size());
}

synapse::Metrics InProcessSession::metrics() {
  synapse::Metrics m;
  score(m_split.train, &m.train_loss, &m.train_acc);
  score(m_split.val, &m.val_loss, &m.val_acc);
  m.train_n = static_cast<int>(m_split.train.size());
  m.val_n = static_cast<int>(m_split.val.size());
  m.epoch = m_epoch;
  return m;
}

synapse::Metrics InProcessSession::train(float lr, int budgetMs, int maxEpochs, bool gpu,
                                         const std::vector<float>& refreshInput) {
  if (!m_split.train.empty()) {
    QElapsedTimer t;
    t.start();
    int did = 0;
    do {
      if (gpu) {
        ensureFlat();
        m_net.train_epoch_batched(m_flatIn, m_flatOut, static_cast<int>(m_split.train.size()), lr);
        ++m_epoch;
      } else {
        runOneEpoch(lr);
      }
      ++did;
    } while (t.elapsed() < budgetMs && did < maxEpochs);
  }
  if (!refreshInput.empty()) {
    try {
      m_net.forward(refreshInput);
    } catch (const std::exception&) {
    }
  }
  return metrics();
}

void InProcessSession::setOptimizer(const QString& name) {
  m_net.set_optimizer(name.toStdString());
}

QStringList InProcessSession::optimizerNames() {
  QStringList names;
  for (const std::string& s : synapse::optimizer_names()) names << QString::fromStdString(s);
  return names;
}

QString InProcessSession::deviceName() {
  return QString::fromStdString(synapse::active_device_name());
}

QStringList InProcessSession::activationNames() {
  QStringList names;
  for (const std::string& s : synapse::activation_names()) names << QString::fromStdString(s);
  return names;
}

// ═══════════════════════════ subprocess ═══════════════════════════════════════

SubprocessSession::SubprocessSession(const QString& exePath) {
  m_proc.setProcessChannelMode(QProcess::SeparateChannels);  // stderr must not corrupt the stream
  m_proc.start(exePath, QStringList{});
  m_ok = m_proc.waitForStarted(5000);
}

SubprocessSession::~SubprocessSession() {
  if (m_proc.state() == QProcess::Running) {
    m_proc.write("{\"cmd\":\"quit\"}\n");
    m_proc.closeWriteChannel();
    if (!m_proc.waitForFinished(2000)) m_proc.kill();
  }
}

QJsonObject SubprocessSession::request(QJsonObject cmd, int timeoutMs) {
  if (!m_ok || m_proc.state() != QProcess::Running) return {};
  const long id = ++m_nextId;
  cmd.insert(QStringLiteral("id"), static_cast<double>(id));
  m_proc.write(QJsonDocument(cmd).toJson(QJsonDocument::Compact));
  m_proc.write("\n");

  QElapsedTimer timer;
  timer.start();
  while (timer.elapsed() < timeoutMs) {
    if (!m_proc.canReadLine() && !m_proc.waitForReadyRead(100)) {
      if (m_proc.state() != QProcess::Running) break;  // engine died
      continue;
    }
    while (m_proc.canReadLine()) {
      const QByteArray line = m_proc.readLine().trimmed();
      if (line.isEmpty()) continue;
      const QJsonObject o = QJsonDocument::fromJson(line).object();
      const QString ev = o.value("ev").toString();
      if (ev == QLatin1String("topology")) {
        if (m_obs) m_obs->on_topology(parseTopology(o.value("topology").toObject()));
      } else if (ev == QLatin1String("step")) {
        if (m_obs) m_obs->on_step(parseStep(o.value("step").toObject()));
      } else if (static_cast<long>(o.value("id").toDouble()) == id) {
        return o;  // "result" or "error" for this command
      }
    }
  }
  return {};
}

void SubprocessSession::build(const synapse::Topology& topo, unsigned seed) {
  // The model-spec fields ride at the top level so the engine's own parser reads them.
  QJsonObject cmd{{"cmd", "build"}, {"seed", static_cast<double>(seed)}};
  cmd.insert("name", QString::fromStdString(topo.name));
  cmd.insert("input_dim", topo.input_dim);
  QJsonArray layers;
  for (const synapse::LayerInfo& L : topo.layers) {
    layers.append(QJsonObject{{"type", QString::fromStdString(L.type)},
                              {"units", L.output_dim},
                              {"activation", QString::fromStdString(L.activation)}});
  }
  cmd.insert("layers", layers);
  request(cmd);
}

void SubprocessSession::setDataset(const synapse::Dataset& ds) {
  QJsonArray samples;
  for (const synapse::Sample& s : ds.samples)
    samples.append(QJsonObject{{"input", toArray(s.input)}, {"target", toArray(s.target)}});
  request(QJsonObject{{"cmd", "set_dataset"}, {"samples", samples}});
}

std::vector<float> SubprocessSession::forward(const std::vector<float>& in) {
  const QJsonObject r = request(QJsonObject{{"cmd", "forward"}, {"input", toArray(in)}});
  return toVec(r.value("output").toArray());
}

void SubprocessSession::beginForward(const std::vector<float>& in) {
  request(QJsonObject{{"cmd", "begin_forward"}, {"input", toArray(in)}});
}

bool SubprocessSession::stepForward() {
  return request(QJsonObject{{"cmd", "step_forward"}}).value("more").toBool();
}

bool SubprocessSession::forwardDone() {
  return request(QJsonObject{{"cmd", "forward_done"}}).value("done").toBool(true);
}

void SubprocessSession::beginLearn(const std::vector<float>& in, const std::vector<float>& target,
                                   float lr) {
  request(QJsonObject{{"cmd", "begin_learn"},
                      {"input", toArray(in)},
                      {"target", toArray(target)},
                      {"lr", static_cast<double>(lr)}});
}

bool SubprocessSession::learnAdvance() {
  return request(QJsonObject{{"cmd", "learn_advance"}}).value("more").toBool();
}

bool SubprocessSession::learnActive() {
  return request(QJsonObject{{"cmd", "learn_active"}}).value("active").toBool();
}

namespace {
synapse::Metrics parseMetrics(const QJsonObject& r) {
  synapse::Metrics m;
  m.train_loss = static_cast<float>(r.value("train_loss").toDouble());
  m.val_loss = static_cast<float>(r.value("val_loss").toDouble());
  m.train_acc = static_cast<float>(r.value("train_acc").toDouble());
  m.val_acc = static_cast<float>(r.value("val_acc").toDouble());
  m.train_n = r.value("train_n").toInt();
  m.val_n = r.value("val_n").toInt();
  m.epoch = r.value("epoch").toInt();
  return m;
}
}  // namespace

synapse::Metrics SubprocessSession::train(float lr, int budgetMs, int maxEpochs, bool gpu,
                                          const std::vector<float>& refreshInput) {
  QJsonObject cmd{{"cmd", "train"},
                  {"lr", static_cast<double>(lr)},
                  {"budget_ms", budgetMs},
                  {"max_epochs", maxEpochs},
                  {"mode", gpu ? "gpu" : "cpu"}};
  if (!refreshInput.empty()) cmd.insert("refresh_input", toArray(refreshInput));
  return parseMetrics(request(cmd, std::max(20000, budgetMs * 4)));
}

synapse::Metrics SubprocessSession::metrics() {
  return parseMetrics(request(QJsonObject{{"cmd", "metrics"}}));
}

void SubprocessSession::setSplit(float valFraction) {
  request(QJsonObject{{"cmd", "set_split"}, {"val_fraction", static_cast<double>(valFraction)}});
}

void SubprocessSession::setOptimizer(const QString& name) {
  request(QJsonObject{{"cmd", "set_optimizer"}, {"name", name}});
}

QStringList SubprocessSession::optimizerNames() {
  QStringList names;
  for (const QJsonValue& v : request(QJsonObject{{"cmd", "optimizers"}}).value("names").toArray())
    names << v.toString();
  return names;
}

std::vector<float> SubprocessSession::parameters() {
  return toVec(request(QJsonObject{{"cmd", "get_params"}}).value("params").toArray());
}

bool SubprocessSession::setParameters(const std::vector<float>& p) {
  return request(QJsonObject{{"cmd", "set_params"}, {"params", toArray(p)}})
      .value("ok")
      .toBool();
}

QString SubprocessSession::deviceName() {
  return request(QJsonObject{{"cmd", "device"}}).value("device").toString();
}

QStringList SubprocessSession::activationNames() {
  QStringList names;
  for (const QJsonValue& v : request(QJsonObject{{"cmd", "activations"}}).value("names").toArray())
    names << v.toString();
  return names;
}
