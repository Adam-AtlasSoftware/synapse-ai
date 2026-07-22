#include "EngineBridge.h"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QSettings>
#include <QTextStream>
#include <algorithm>
#include <exception>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "synapse/activation.hpp"
#include "synapse/device_info.hpp"
#include "synapse/model_spec.hpp"

#ifndef SYNAPSE_MODELS_DIR
#define SYNAPSE_MODELS_DIR "models"
#endif
#ifndef SYNAPSE_SOURCE_DIR
#define SYNAPSE_SOURCE_DIR "."
#endif
#ifndef SYNAPSE_BUILD_DIR
#define SYNAPSE_BUILD_DIR "build"
#endif
#ifndef SYNAPSE_HOST_EXE
#define SYNAPSE_HOST_EXE "synapse_engine_host"
#endif

using synapse::LayerInfo;
using synapse::StepSnapshot;
using synapse::TensorView;
using synapse::Topology;

namespace {
QVariantList toVariantList(const std::vector<float>& v) {
  QVariantList out;
  out.reserve(static_cast<int>(v.size()));
  for (float x : v) out.push_back(static_cast<double>(x));
  return out;
}

QString slugify(const QString& name) {
  QString s;
  for (QChar c : name.toLower()) {
    if (c.isLetterOrNumber()) s += c;
    else if (c == ' ' || c == '-' || c == '_') s += '_';
  }
  return s.isEmpty() ? QStringLiteral("model") : s;
}
}  // namespace

EngineBridge::EngineBridge(QObject* parent) : QObject(parent) {
  m_timer.setInterval(m_speedMs);
  connect(&m_timer, &QTimer::timeout, this, &EngineBridge::onTick);
  m_trainTimer.setInterval(50);
  connect(&m_trainTimer, &QTimer::timeout, this, &EngineBridge::onTrainTick);

  // The engine runs as its OWN PROCESS by default: a crash in the C++ you're writing in
  // the Code Lab takes down only the engine, and a rebuild swaps it in without restarting
  // the app. It speaks the same telemetry contract over pipes, so nothing else changes.
  //
  // Measured cost on the largest blueprint (ocr_5x5, 65 examples): ~1030 epochs/s
  // out-of-process vs ~1265 in-process — the round-trip and JSON snapshot eat into each
  // tick's fixed 20ms budget. Both converge in a second or two, so it isn't a cost you
  // can perceive; SYNAPSE_ENGINE_INPROCESS=1 opts out if you want the last 19% anyway.
  // If the host can't start we fall back to the linked-in engine rather than dying.
  if (qgetenv("SYNAPSE_ENGINE_INPROCESS").isEmpty()) {
    auto sub = std::make_unique<SubprocessSession>(QString::fromUtf8(SYNAPSE_HOST_EXE));
    if (sub->usable()) m_session = std::move(sub);
  }
  if (!m_session) m_session = std::make_unique<InProcessSession>();
  m_session->setObserver(this);  // set before any build so topology reaches us

  // Guidance preferences survive restarts — being told "turn coaching off" and having it
  // come back next launch would be its own annoyance.
  QSettings prefs;
  m_coachingEnabled = prefs.value(QStringLiteral("coachingEnabled"), true).toBool();
  m_autoTrainEnabled = prefs.value(QStringLiteral("autoTrainEnabled"), true).toBool();

  m_deviceName = m_session->deviceName();
  scanBlueprints();

  const QString xorPath = QString::fromUtf8(SYNAPSE_MODELS_DIR) + "/blueprints/xor.json";
  if (QFile::exists(xorPath)) {
    loadBlueprint(xorPath);
  } else {
    // Fall back to an in-code XOR net if the blueprints can't be found at runtime.
    m_topo = Topology{};
    m_topo.name = "XOR";
    m_topo.input_dim = 2;
    m_topo.layers = {LayerInfo{"L0", "dense", 2, 4, "tanh"},
                     LayerInfo{"L1", "dense", 4, 1, "sigmoid"}};
    m_inputLayout = "labels";
    m_inputLabels = QVariantList{"A", "B"};
    m_outputLabels = QVariantList{"A XOR B"};
    m_outputKind = "value";
    emit blueprintChanged();
    rebuild();
  }
}

void EngineBridge::rebuild() {
  m_timer.stop();       // editing the model cancels any in-flight playback / training
  m_trainTimer.stop();
  setPlaying(false);
  setTraining(false);
  m_epoch = 0;
  m_currentLoss = 0.0;
  m_lossHistory.clear();
  m_valLossHistory.clear();
  emit trainingProgress();
  try {
    m_session->build(m_topo, m_seed);  // re-initializes weights, emits topology
    syncDataset();                     // the engine owns the training loop, so it needs the data
    m_session->setOptimizer(m_optimizer);
    m_session->setSplit(static_cast<float>(m_valFraction));
    runForwardZeros();                 // populate activations + weights so nothing is blank
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
  }
}

// The engine runs training itself (one call per time budget rather than per sample),
// so it needs its own copy of the dataset whenever ours changes.
void EngineBridge::syncDataset() {
  if (m_session) m_session->setDataset(m_dataset);
}

// ── synapse::Observer ────────────────────────────────────────────────────────

void EngineBridge::on_topology(const Topology& topo) {
  m_topo = topo;
  m_modelName = QString::fromStdString(topo.name);  // set BEFORE topologyChanged fires

  QVariantList cols;
  QVariantMap input;
  input["title"] = "Input";
  input["activation"] = "input";
  input["count"] = topo.input_dim;
  input["kind"] = "input";
  cols.push_back(input);

  QVariantList lys;
  for (int k = 0; k < static_cast<int>(topo.layers.size()); ++k) {
    const LayerInfo& L = topo.layers[k];
    QVariantMap col;
    col["title"] = QString::fromStdString(L.name);
    col["activation"] = QString::fromStdString(L.activation);
    col["count"] = L.output_dim;
    col["kind"] = "dense";
    cols.push_back(col);

    QVariantMap ly;
    ly["index"] = k;
    ly["units"] = L.output_dim;
    ly["activation"] = QString::fromStdString(L.activation);
    ly["name"] = QString::fromStdString(L.name);
    lys.push_back(ly);
  }

  m_columns = cols;
  m_layers = lys;
  emit topologyChanged();
}

void EngineBridge::on_step(const StepSnapshot& snap) {
  // Index the snapshot's tensors by name for aligned assembly.
  std::unordered_map<std::string, const TensorView*> byName;
  for (const TensorView& tv : snap.tensors) byName[tv.name] = &tv;

  const int nCols = static_cast<int>(m_topo.layers.size()) + 1;
  QVariantList acts, dts;
  for (int i = 0; i < nCols; ++i) {
    acts.push_back(QVariantList{});
    dts.push_back(QVariantList{});
  }
  QVariantList wts, grs, bs;

  if (auto it = byName.find("input"); it != byName.end())
    acts[0] = toVariantList(it->second->data);

  for (int k = 0; k < static_cast<int>(m_topo.layers.size()); ++k) {
    const std::string& name = m_topo.layers[k].name;

    if (auto it = byName.find("activations." + name); it != byName.end())
      acts[k + 1] = toVariantList(it->second->data);
    if (auto it = byName.find("delta." + name); it != byName.end())
      dts[k + 1] = toVariantList(it->second->data);

    QVariantMap w;
    if (auto it = byName.find("weights." + name); it != byName.end()) {
      const TensorView* tv = it->second;
      w["rows"] = tv->rows;
      w["cols"] = tv->cols;
      w["values"] = toVariantList(tv->data);
    }
    wts.push_back(w);

    QVariantMap g;
    if (auto it = byName.find("grad." + name); it != byName.end()) {
      const TensorView* tv = it->second;
      g["rows"] = tv->rows;
      g["cols"] = tv->cols;
      g["values"] = toVariantList(tv->data);
    }
    grs.push_back(g);

    QVariantList bvec;
    if (auto it = byName.find("biases." + name); it != byName.end())
      bvec = toVariantList(it->second->data);
    bs.push_back(bvec);
  }

  // Predicted class = argmax over the output column's activations.
  m_predictedIndex = -1;
  if (!acts.isEmpty()) {
    const QVariantList last = acts.last().toList();
    double best = -1e30;
    for (int i = 0; i < last.size(); ++i) {
      const double v = last[i].toDouble();
      if (v > best) {
        best = v;
        m_predictedIndex = i;
      }
    }
  }

  m_activations = acts;
  m_weights = wts;
  m_deltas = dts;
  m_grads = grs;
  m_biases = bs;
  m_activeLayer = snap.active_layer;
  m_phase = QString::fromStdString(snap.phase);
  emit activationsChanged();
}

// ── inference ────────────────────────────────────────────────────────────────

std::vector<float> EngineBridge::toVector(const QVariantList& input) const {
  std::vector<float> v;
  v.reserve(input.size());
  for (const QVariant& x : input) v.push_back(static_cast<float>(x.toDouble()));
  return v;
}

void EngineBridge::runForward(const QVariantList& input) {
  if (input.size() != m_topo.input_dim) {
    emit errorOccurred(QStringLiteral("expected %1 inputs, got %2")
                           .arg(m_topo.input_dim)
                           .arg(input.size()));
    return;
  }
  m_lastInput = toVector(input);  // remember the current input so training re-runs IT
  try {
    m_session->forward(m_lastInput);   // instant: emits on_step once
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
  }
  emit inputChanged();
}

void EngineBridge::runForwardZeros() {
  runForward(QVariantList(m_topo.input_dim, QVariant(0.0)));
}

// ── current input (owned by the bridge) ──────────────────────────────────────

QVariantList EngineBridge::input() const {
  QVariantList out;
  out.reserve(static_cast<int>(m_lastInput.size()));
  for (float v : m_lastInput) out.push_back(static_cast<double>(v));
  return out;
}

void EngineBridge::setInput(int index, double value) {
  if (static_cast<int>(m_lastInput.size()) != m_topo.input_dim)
    m_lastInput.assign(m_topo.input_dim, 0.0f);
  if (index < 0 || index >= m_topo.input_dim) return;
  m_lastInput[index] = static_cast<float>(value);
  forwardCurrent();
  noteInputTried();
  emit inputChanged();
}

void EngineBridge::setInputVector(const QVariantList& values) {
  m_lastInput = toVector(values);
  m_lastInput.resize(m_topo.input_dim, 0.0f);
  forwardCurrent();
  emit inputChanged();
}

void EngineBridge::forwardCurrent() {
  if (static_cast<int>(m_lastInput.size()) != m_topo.input_dim)
    m_lastInput.assign(m_topo.input_dim, 0.0f);
  try {
    m_session->forward(m_lastInput);
  } catch (const std::exception&) {
  }
}

void EngineBridge::playCurrent() { play(input()); }

void EngineBridge::clearInput() { setInputVector(QVariantList(m_topo.input_dim, QVariant(0.0))); }

void EngineBridge::randomizeInput() {
  std::uniform_real_distribution<float> d(0.0f, 1.0f);
  const bool grid = (m_inputLayout == "grid");
  QVariantList v;
  for (int i = 0; i < m_topo.input_dim; ++i)
    v.push_back(grid ? (d(m_rng) < 0.35 ? 1.0 : 0.0) : static_cast<double>(d(m_rng)));
  setInputVector(v);
  noteInputTried();
}

// ── dataset browsing + manual examples ───────────────────────────────────────

QVariantList EngineBridge::exampleInput(int i) const {
  QVariantList out;
  if (i < 0 || i >= m_dataset.size()) return out;
  for (float v : m_dataset.samples[i].input) out.push_back(static_cast<double>(v));
  return out;
}

QString EngineBridge::exampleLabel(int i) const {
  if (i < 0 || i >= m_dataset.size()) return QString();
  const std::vector<float>& t = m_dataset.samples[i].target;
  if (m_outputKind == "class") {
    int best = 0;
    for (int k = 1; k < static_cast<int>(t.size()); ++k)
      if (t[k] > t[best]) best = k;
    return (best < m_outputLabels.size()) ? m_outputLabels[best].toString()
                                          : QString::number(best);
  }
  QString s;
  for (int k = 0; k < static_cast<int>(t.size()); ++k) {
    const QString name = (k < m_outputLabels.size()) ? m_outputLabels[k].toString()
                                                     : QString::number(k);
    if (!s.isEmpty()) s += ", ";
    s += name + "=" + QString::number(t[k]);
  }
  return s;
}

void EngineBridge::loadExample(int i) { setInputVector(exampleInput(i)); }

void EngineBridge::addExample(const QVariantList& in, int labelIndex) {
  if (m_outputKind != "class") {
    emit errorOccurred(QStringLiteral("manual examples need a classifier blueprint"));
    return;
  }
  const int outDim = m_topo.output_dim();
  if (labelIndex < 0 || labelIndex >= outDim) return;
  synapse::Sample s;
  s.input = toVector(in);
  s.input.resize(m_topo.input_dim, 0.0f);
  s.target.assign(outDim, 0.0f);
  s.target[labelIndex] = 1.0f;
  m_dataset.samples.push_back(std::move(s));
  syncDataset();  // engine owns the training loop, keep its copy current
  emit datasetChanged();
  emit trainingProgress();  // hasDataset / dataset size shown in the UI
}

QVariantList EngineBridge::exampleTarget(int i) const {
  QVariantList out;
  if (i < 0 || i >= m_dataset.size()) return out;
  for (float v : m_dataset.samples[i].target) out.push_back(static_cast<double>(v));
  return out;
}

void EngineBridge::addExampleRaw(const QVariantList& in, const QVariantList& target) {
  synapse::Sample s;
  s.input = toVector(in);
  s.input.resize(m_topo.input_dim, 0.0f);
  s.target = toVector(target);
  s.target.resize(m_topo.output_dim(), 0.0f);
  m_dataset.samples.push_back(std::move(s));
  syncDataset();  // engine owns the training loop, keep its copy current
  emit datasetChanged();
  emit trainingProgress();
}

void EngineBridge::updateExample(int i, const QVariantList& in, const QVariantList& target) {
  if (i < 0 || i >= m_dataset.size()) return;
  synapse::Sample& s = m_dataset.samples[i];
  s.input = toVector(in);
  s.input.resize(m_topo.input_dim, 0.0f);
  s.target = toVector(target);
  s.target.resize(m_topo.output_dim(), 0.0f);
  syncDataset();  // engine owns the training loop, keep its copy current
  emit datasetChanged();
}

void EngineBridge::removeExample(int i) {
  if (i < 0 || i >= m_dataset.size()) return;
  m_dataset.samples.erase(m_dataset.samples.begin() + i);
  syncDataset();  // engine owns the training loop, keep its copy current
  emit datasetChanged();
  emit trainingProgress();
}

void EngineBridge::saveDataset() {
  if (m_blueprintPath.isEmpty()) return;
  QFile in(m_blueprintPath);
  if (!in.open(QIODevice::ReadOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("cannot read blueprint to save"));
    return;
  }
  QJsonObject root = QJsonDocument::fromJson(in.readAll()).object();
  in.close();

  QJsonArray arr;
  for (const synapse::Sample& s : m_dataset.samples) {
    QJsonArray ji, jt;
    for (float v : s.input) ji.push_back(v);
    for (float v : s.target) jt.push_back(v);
    QJsonObject o;
    o["input"] = ji;
    o["target"] = jt;
    arr.push_back(o);
  }
  root["dataset"] = arr;

  QFile out(m_blueprintPath);
  if (!out.open(QIODevice::WriteOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("cannot write blueprint"));
    return;
  }
  out.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
  out.close();
}

// ── stepped playback ─────────────────────────────────────────────────────────

void EngineBridge::setPlaying(bool playing) {
  if (m_playing == playing) return;
  m_playing = playing;
  emit playbackChanged();
}

void EngineBridge::setSpeedMs(int ms) {
  ms = std::max(16, std::min(ms, 2000));  // 16ms (~60fps) .. 2s per layer
  if (m_speedMs == ms) return;
  m_speedMs = ms;
  m_timer.setInterval(ms);
  emit speedChanged();
}

void EngineBridge::onTick() {
  // Advance one sub-step of whichever animation is running (forward pass or a
  // full learn-step). Stop when it reports there is nothing left to do.
  const bool more = m_learnMode ? m_session->learnAdvance() : m_session->stepForward();
  if (!more) {
    m_timer.stop();
    setPlaying(false);
    m_learnMode = false;
  }
}

void EngineBridge::play(const QVariantList& input) {
  if (input.size() != m_topo.input_dim) {
    emit errorOccurred(QStringLiteral("expected %1 inputs, got %2")
                           .arg(m_topo.input_dim)
                           .arg(input.size()));
    return;
  }
  m_trainTimer.stop();  // playback and training are mutually exclusive
  setTraining(false);
  m_learnMode = false;
  m_lastInput = toVector(input);
  try {
    m_session->beginForward(m_lastInput);  // reset + show input column
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
    return;
  }
  setPlaying(true);
  m_timer.start(m_speedMs);
}

void EngineBridge::pause() {
  m_timer.stop();
  setPlaying(false);
}

void EngineBridge::beginForward(const QVariantList& input) {
  m_timer.stop();
  setPlaying(false);
  m_learnMode = false;
  m_lastInput = toVector(input);
  try {
    m_session->beginForward(m_lastInput);
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
  }
}

// ── gradient-flow animation: one SGD step on a chosen example, in slow motion ──

void EngineBridge::playLearnExample(int i) {
  if (i < 0 || i >= m_dataset.size()) {
    emit errorOccurred(QStringLiteral("no such example"));
    return;
  }
  m_trainTimer.stop();
  setTraining(false);
  loadExample(i);  // draw the example into the input
  const synapse::Sample& s = m_dataset.samples[i];
  try {
    m_session->beginLearn(s.input, s.target, static_cast<float>(m_lr));
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
    return;
  }
  m_learnMode = true;
  setPlaying(true);
  m_timer.start(m_speedMs);
}

void EngineBridge::stepLearnExample(int i) {
  if (i < 0 || i >= m_dataset.size()) return;
  m_timer.stop();
  m_trainTimer.stop();
  setPlaying(false);
  setTraining(false);
  m_learnMode = true;
  try {
    if (!m_session->learnActive()) {
      loadExample(i);
      const synapse::Sample& s = m_dataset.samples[i];
      m_session->beginLearn(s.input, s.target, static_cast<float>(m_lr));
    } else {
      m_session->learnAdvance();
    }
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
  }
}

void EngineBridge::stepOnce() {
  m_timer.stop();
  setPlaying(false);
  m_learnMode = false;
  if (static_cast<int>(m_lastInput.size()) != m_topo.input_dim)
    m_lastInput.assign(m_topo.input_dim, 0.0f);
  try {
    if (m_session->forwardDone())
      m_session->beginForward(m_lastInput);  // finished (or fresh) → reset to input
    else
      m_session->stepForward();              // otherwise advance one layer
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
  }
}

// ── Tier-1 model editing ─────────────────────────────────────────────────────

void EngineBridge::loadModel(const QString& path) {
  try {
    m_topo = synapse::load_topology_file(path.toStdString());
    rebuild();
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
  }
}

void EngineBridge::saveModel(const QString& path) {
  try {
    synapse::save_topology_file(m_topo, path.toStdString());
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
  }
}

QString EngineBridge::modelJson() const {
  return QString::fromStdString(synapse::topology_to_json(m_topo));
}

void EngineBridge::setInputDim(int dim) {
  if (dim < 1) dim = 1;
  m_topo.input_dim = dim;
  if (!m_topo.layers.empty()) m_topo.layers.front().input_dim = dim;
  rebuild();
}

void EngineBridge::setInputLayout(const QString& layout, int rows, int cols) {
  if (layout == "grid") {
    m_inputLayout = "grid";
    m_inputRows = std::max(1, rows);
    m_inputCols = std::max(1, cols);
    m_inputLabels.clear();
    m_topo.input_dim = m_inputRows * m_inputCols;
  } else if (layout == "segments") {
    // A fixed seven-segment display: seven inputs, named a–g in standard order.
    m_inputLayout = "segments";
    m_inputRows = 1;
    m_inputCols = 7;
    m_inputLabels = QVariantList{"a", "b", "c", "d", "e", "f", "g"};
    m_topo.input_dim = 7;
  } else {
    m_inputLayout = "labels";
    m_inputRows = 1;
    m_inputCols = 0;
    // Give every slider a generic name unless the current labels already fit.
    if (m_inputLabels.size() != m_topo.input_dim) {
      m_inputLabels.clear();
      for (int i = 0; i < m_topo.input_dim; ++i)
        m_inputLabels.push_back(QStringLiteral("x%1").arg(i));
    }
  }
  if (!m_topo.layers.empty()) m_topo.layers.front().input_dim = m_topo.input_dim;
  emit blueprintChanged();  // the panel swaps to the new input widget
  rebuild();                // re-inits the net for the new input dimension
  syncDataset();  // engine owns the training loop, keep its copy current
  if (!m_dataset.empty()) loadExample(0);
}

void EngineBridge::addLayer(int units, const QString& activation) {
  if (units < 1) units = 1;
  LayerInfo L;
  L.type = "dense";
  L.output_dim = units;
  L.input_dim = m_topo.layers.empty() ? m_topo.input_dim : m_topo.layers.back().output_dim;
  L.activation = activation.toStdString();
  L.name = "L" + std::to_string(m_topo.layers.size());
  m_topo.layers.push_back(L);
  rebuild();
}

void EngineBridge::removeLayer(int index) {
  if (index < 0 || index >= static_cast<int>(m_topo.layers.size())) return;
  if (m_topo.layers.size() <= 1) {
    emit errorOccurred(QStringLiteral("a network needs at least one layer"));
    return;
  }
  m_topo.layers.erase(m_topo.layers.begin() + index);
  // Re-chain input dims and names so the topology stays consistent.
  int prev = m_topo.input_dim;
  for (int k = 0; k < static_cast<int>(m_topo.layers.size()); ++k) {
    m_topo.layers[k].input_dim = prev;
    m_topo.layers[k].name = "L" + std::to_string(k);
    prev = m_topo.layers[k].output_dim;
  }
  rebuild();
}

void EngineBridge::setLayerUnits(int index, int units) {
  if (index < 0 || index >= static_cast<int>(m_topo.layers.size())) return;
  if (units < 1) units = 1;
  m_topo.layers[index].output_dim = units;
  if (index + 1 < static_cast<int>(m_topo.layers.size()))
    m_topo.layers[index + 1].input_dim = units;
  rebuild();
}

void EngineBridge::setLayerActivation(int index, const QString& activation) {
  if (index < 0 || index >= static_cast<int>(m_topo.layers.size())) return;
  m_topo.layers[index].activation = activation.toStdString();
  rebuild();
}

// ── trained weights: a sidecar next to the blueprint ─────────────────────────
// Blueprints stay pure templates (architecture + I/O + data); the weights you trained
// are their own artifact in models/weights/, so "Restore default" never fights them.

QString EngineBridge::weightsPath() const {
  if (m_blueprintPath.isEmpty()) return QString();
  return QString::fromUtf8(SYNAPSE_MODELS_DIR) + "/weights/" +
         QFileInfo(m_blueprintPath).completeBaseName() + ".json";
}

bool EngineBridge::hasSavedWeights() const {
  const QString p = weightsPath();
  return !p.isEmpty() && QFile::exists(p);
}

void EngineBridge::saveWeights() {
  const QString path = weightsPath();
  if (path.isEmpty()) {
    emit errorOccurred(QStringLiteral("save the model as a blueprint first"));
    return;
  }
  const std::vector<float> params = m_session->parameters();
  if (params.empty()) {
    emit errorOccurred(QStringLiteral("nothing to save"));
    return;
  }
  QJsonArray arr;
  for (float v : params) arr.append(static_cast<double>(v));
  QJsonObject root;
  root["model"] = m_modelName;
  root["count"] = static_cast<int>(params.size());
  root["epoch"] = m_epoch;
  root["params"] = arr;

  QDir().mkpath(QFileInfo(path).absolutePath());
  QFile f(path);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("cannot write %1").arg(path));
    return;
  }
  f.write(QJsonDocument(root).toJson(QJsonDocument::Compact));
  f.close();
  emit weightsChanged();
}

void EngineBridge::loadWeights() {
  const QString path = weightsPath();
  QFile f(path);
  if (path.isEmpty() || !f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("no saved weights for this blueprint"));
    return;
  }
  const QJsonObject root = QJsonDocument::fromJson(f.readAll()).object();
  f.close();

  std::vector<float> params;
  for (const QJsonValue& v : root.value("params").toArray())
    params.push_back(static_cast<float>(v.toDouble()));

  // Guard against weights saved for a different architecture (you edited the layers).
  if (params.empty() || !m_session->setParameters(params)) {
    emit errorOccurred(QStringLiteral("saved weights don't fit this architecture — ignoring"));
    return;
  }
  m_epoch = root.value("epoch").toInt();
  m_lossHistory.clear();
  m_valLossHistory.clear();
  forwardCurrent();
  applyMetrics(m_session->metrics(), false);
  m_epoch = root.value("epoch").toInt();  // metrics reports the engine's counter (0)
  emit trainingProgress();
}

// ── Tier-2 Code Lab: edit engine C++, recompile, relaunch ─────────────────────

QStringList EngineBridge::activationNames() const {
  QStringList names;
  for (const std::string& s : synapse::activation_names())
    names << QString::fromStdString(s);
  return names;
}

// The one user-editable engine source the Code Lab reads and writes.
static QString customActivationPath() {
  return QString::fromUtf8(SYNAPSE_SOURCE_DIR) + "/engine/src/custom_activations.cpp";
}

QString EngineBridge::customActivationSource() const {
  QFile f(customActivationPath());
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return QString();
  const QString s = QString::fromUtf8(f.readAll());
  f.close();
  return s;
}

void EngineBridge::saveCustomActivationSource(const QString& src) {
  QFile f(customActivationPath());
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("cannot write %1").arg(customActivationPath()));
    return;
  }
  f.write(src.toUtf8());
  f.close();
}

// Rebuild the engine (via the dashboard target, which relinks it in) and, on success,
// relaunch this exact binary re-opening the current blueprint — the "recompile & run"
// loop. On failure the compiler output stays on screen and the app keeps running.
bool EngineBridge::usingSubprocess() const {
  return m_session && m_session->kind() == QLatin1String("subprocess");
}

// Tear down the engine process and start a freshly-built one, then put it back exactly
// where we were: same architecture, same dataset, same input. The GUI never restarts —
// this is what the process boundary bought us.
void EngineBridge::reloadEngineProcess() {
  m_timer.stop();
  m_trainTimer.stop();
  setPlaying(false);
  setTraining(false);

  // Keep the training across the swap — same architecture, so the parameters still fit.
  const std::vector<float> saved = m_session ? m_session->parameters() : std::vector<float>{};
  const int savedEpoch = m_epoch;

  m_session.reset();  // quits the old engine
  auto sub = std::make_unique<SubprocessSession>(QString::fromUtf8(SYNAPSE_HOST_EXE));
  if (!sub->usable()) {
    m_buildOutput += QStringLiteral("\n✗ The rebuilt engine would not start.\n");
    emit buildOutputChanged();
    emit errorOccurred(QStringLiteral("could not restart the engine process"));
    return;
  }
  m_session = std::move(sub);
  m_session->setObserver(this);

  m_epoch = 0;
  m_currentLoss = 0.0;
  m_lossHistory.clear();
  m_session->build(m_topo, m_seed);  // fresh weights; emits topology
  syncDataset();
  m_session->setOptimizer(m_optimizer);
  m_session->setSplit(static_cast<float>(m_valFraction));
  if (!saved.empty() && m_session->setParameters(saved)) {
    m_epoch = savedEpoch;  // the weights really are that trained
    m_buildOutput += QStringLiteral("✓ Kept your trained weights across the swap.\n");
  }
  forwardCurrent();

  emit activationNamesChanged();  // your new activation shows up in the dropdowns
  emit optimizerNamesChanged();   // ...and any new optimizer too
  emit trainingProgress();
}

// Recompile the engine from the C++ you just edited. Out-of-process, only the engine
// binary needs rebuilding and we hot-swap it. Linked in-process, the engine is part of
// this executable, so the app has to relink and relaunch itself.
void EngineBridge::rebuildEngine() {
  if (m_building) return;
  m_building = true;
  emit buildingChanged();

  const bool hotSwap = usingSubprocess();
  const QString target = hotSwap ? QStringLiteral("synapse_engine_host")
                                 : QStringLiteral("synapse_dashboard");
  const QString buildDir = QString::fromUtf8(SYNAPSE_BUILD_DIR);
  m_buildOutput = QStringLiteral("$ cmake --build \"%1\" --target %2\n\n").arg(buildDir, target);
  emit buildOutputChanged();

  QProcess* proc = new QProcess(this);
  proc->setProcessChannelMode(QProcess::MergedChannels);
  connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc]() {
    m_buildOutput += QString::fromUtf8(proc->readAllStandardOutput());
    emit buildOutputChanged();
  });
  connect(proc, &QProcess::finished, this,
          [this, proc, hotSwap](int code, QProcess::ExitStatus status) {
            proc->deleteLater();
            m_building = false;
            emit buildingChanged();
            if (code != 0 || status != QProcess::NormalExit) {
              m_buildOutput +=
                  QStringLiteral("\n✗ Build failed (exit %1). Fix the errors above and Rebuild.\n")
                      .arg(code);
              emit buildOutputChanged();
              return;  // the running engine is untouched — keep working
            }
            if (hotSwap) {
              m_buildOutput += QStringLiteral(
                  "\n✓ Build succeeded — swapping in the new engine (weights re-initialized)…\n");
              emit buildOutputChanged();
              reloadEngineProcess();
              m_buildOutput += QStringLiteral("✓ Engine reloaded. Your activation is now in the "
                                              "layer dropdowns — no restart needed.\n");
              emit buildOutputChanged();
            } else {
              m_buildOutput += QStringLiteral("\n✓ Build succeeded — relaunching with your new code…\n");
              emit buildOutputChanged();
              // Re-open the current blueprint in the new process (env is inherited).
              if (!m_blueprintPath.isEmpty())
                qputenv("SYNAPSE_BLUEPRINT", QFileInfo(m_blueprintPath).completeBaseName().toUtf8());
              QProcess::startDetached(QCoreApplication::applicationFilePath(), QStringList{});
              QCoreApplication::quit();
            }
          });
  proc->setWorkingDirectory(buildDir);
  proc->start(QStringLiteral("cmake"),
              {QStringLiteral("--build"), buildDir, QStringLiteral("--target"), target});
}

// ── blueprints ───────────────────────────────────────────────────────────────

QString EngineBridge::predictedLabel() const {
  if (m_predictedIndex < 0 || m_predictedIndex >= m_outputLabels.size()) return QString();
  return m_outputLabels[m_predictedIndex].toString();
}

void EngineBridge::scanBlueprints() {
  m_blueprints.clear();
  QDir dir(QString::fromUtf8(SYNAPSE_MODELS_DIR) + "/blueprints");
  const QStringList files = dir.entryList({"*.json"}, QDir::Files, QDir::Name);
  for (const QString& file : files) {
    QFile f(dir.absoluteFilePath(file));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) continue;
    const QJsonObject root = QJsonDocument::fromJson(f.readAll()).object();
    QVariantMap bp;
    bp["name"] = root.value("name").toString(file);
    bp["description"] = root.value("description").toString();
    bp["path"] = dir.absoluteFilePath(file);
    bp["inputDim"] = root.value("input_dim").toInt();
    bp["blank"] = (QFileInfo(file).completeBaseName() == "blank");
    m_blueprints.push_back(bp);
  }
  // Blank first, then by input size (which tracks the ladder of complexity).
  std::sort(m_blueprints.begin(), m_blueprints.end(), [](const QVariant& a, const QVariant& b) {
    const QVariantMap ma = a.toMap(), mb = b.toMap();
    if (ma.value("blank").toBool() != mb.value("blank").toBool())
      return ma.value("blank").toBool();
    return ma.value("inputDim").toInt() < mb.value("inputDim").toInt();
  });
  emit blueprintsListChanged();
}

bool EngineBridge::loadBlueprintByName(const QString& name) {
  for (const QVariant& v : m_blueprints) {
    const QVariantMap bp = v.toMap();
    const QString path = bp.value("path").toString();
    const QString stem = QFileInfo(path).completeBaseName();
    if (bp.value("name").toString().compare(name, Qt::CaseInsensitive) == 0 ||
        stem.compare(name, Qt::CaseInsensitive) == 0) {
      loadBlueprint(path);
      return true;
    }
  }
  return false;
}

// ── training ─────────────────────────────────────────────────────────────────

void EngineBridge::setTraining(bool t) {
  if (m_training == t) return;
  m_training = t;
  emit trainingChanged();
}

void EngineBridge::setLearningRate(double lr) {
  lr = std::max(0.001, std::min(lr, 5.0));
  if (m_lr == lr) return;
  m_lr = lr;
  emit learningRateChanged();
}

QStringList EngineBridge::optimizerNames() const {
  return m_session ? const_cast<ModelSession*>(m_session.get())->optimizerNames() : QStringList{};
}

// Hold out a slice of the data. Training never sees it, so its loss/accuracy show whether
// the network is genuinely learning the pattern or just memorizing the examples.
void EngineBridge::setValFraction(double f) {
  f = std::max(0.0, std::min(f, 0.9));
  if (qFuzzyCompare(m_valFraction + 1.0, f + 1.0)) return;
  m_valFraction = f;
  if (m_session) m_session->setSplit(static_cast<float>(f));
  m_lossHistory.clear();  // the curves are no longer comparable across a split change
  m_valLossHistory.clear();
  emit validationChanged();
  if (m_session && !m_dataset.empty()) applyMetrics(m_session->metrics(), false);
}

void EngineBridge::setOptimizer(const QString& name) {
  if (m_optimizer == name) return;
  m_optimizer = name;
  if (m_session) m_session->setOptimizer(name);
  emit optimizerChanged();
}

void EngineBridge::setGpuTraining(bool on) {
  if (m_gpuTraining == on) return;
  m_gpuTraining = on;
  emit gpuTrainingChanged();
}

// Re-run the forward pass on the input the user is looking at, so the graph and
// the prediction update live as the weights change during training.
void EngineBridge::refreshDisplay() {
  if (static_cast<int>(m_lastInput.size()) == m_topo.input_dim && m_topo.input_dim > 0) {
    try {
      m_session->forward(m_lastInput);
    } catch (const std::exception&) {
    }
  }
}

// Record the average loss over the dataset before any weight update, so the loss
// curve starts at the true peak and shows the whole descent.
void EngineBridge::seedInitialLoss() {
  if (m_dataset.empty() || !m_lossHistory.isEmpty()) return;
  applyMetrics(m_session->metrics(), true);  // the engine scores its own copy of the data
  refreshDisplay();  // after scoring, so the graph shows the user's input
}

// Fold a fresh score into the properties the Training panel binds to. `record` also
// appends to the curves — the two histories stay index-aligned so the chart can draw
// training and validation on the same x axis.
void EngineBridge::applyMetrics(const synapse::Metrics& m, bool record) {
  m_metrics = m;
  m_epoch = m.epoch;
  m_currentLoss = static_cast<double>(m.train_loss);
  if (record) {
    m_lossHistory.push_back(m_currentLoss);
    m_valLossHistory.push_back(static_cast<double>(m.val_loss));
    if (m_lossHistory.size() > 1000) m_lossHistory.removeFirst();
    if (m_valLossHistory.size() > 1000) m_valLossHistory.removeFirst();
  }
  emit trainingProgress();
}

void EngineBridge::onTrainTick() {
  if (m_dataset.empty()) {
    trainStop();
    return;
  }
  // One call runs as many epochs as fit a small time budget, so a tick never stalls the
  // UI — and, crucially, it is ONE round-trip even when the engine is another process.
  // The engine also re-runs the current input, so the graph updates as the weights move.
  applyMetrics(m_session->train(static_cast<float>(m_lr), 20, 2000, m_gpuTraining, m_lastInput),
               true);
  if (m_autoTraining) applyAutoAction();
}

void EngineBridge::trainStart() {
  if (m_dataset.empty()) {
    emit errorOccurred(QStringLiteral("this blueprint has no training data"));
    return;
  }
  m_timer.stop();  // stop forward playback
  setPlaying(false);
  resetJourney();  // the old test tally described the previous weights
  seedInitialLoss();
  setTraining(true);
  m_trainTimer.start();
}

void EngineBridge::trainStop() {
  m_trainTimer.stop();
  setTraining(false);
  if (m_autoTraining) {
    m_autoTraining = false;
    emit autoChanged();
  }
}

// Once it has been trained, any hand-edited input is effectively a test of what it
// learned — so ask the user whether the answer was right.
void EngineBridge::noteInputTried() {
  if (m_epoch <= 0 || m_awaitingVerdict) return;
  m_awaitingVerdict = true;
  emit journeyChanged();
}

void EngineBridge::recordTest(bool correct) {
  if (correct) ++m_testsPassed;
  else ++m_testsFailed;
  m_awaitingVerdict = false;
  emit journeyChanged();
}

void EngineBridge::resetJourney() {
  m_awaitingVerdict = false;
  m_testsPassed = 0;
  m_testsFailed = 0;
  emit journeyChanged();
}

// ── the coach ────────────────────────────────────────────────────────────────

void EngineBridge::autoTrainStart() {
  if (m_dataset.empty()) {
    emit errorOccurred(QStringLiteral("teach it at least one example first"));
    return;
  }
  m_autoLog.clear();
  m_bestParams.clear();
  m_autoOutcome.clear();

  // Without held-out data there is no way to notice memorizing, so set some aside —
  // but not on a handful of examples, where holding any back would just starve it.
  if (m_dataset.size() >= 10 && !validationOn()) {
    setValFraction(0.2);
    m_autoLog.push_back(QStringLiteral(
        "Set aside 20% of your examples so I can tell real learning from memorizing."));
  }

  m_auto.begin(static_cast<float>(m_lr), validationOn(), m_outputKind == "class");
  trainStart();  // starts the timer; only trainStop() clears the auto flag
  m_autoTraining = true;
  m_autoStatus = tr("Starting…");
  emit autoChanged();
}

void EngineBridge::autoTrainStop() {
  if (!m_bestParams.empty()) m_session->setParameters(m_bestParams);
  trainStop();
  applyMetrics(m_session->metrics(), false);
  refreshDisplay();  // last, so the graph ends up showing the user's input
}

void EngineBridge::setCoachingEnabled(bool on) {
  if (m_coachingEnabled == on) return;
  m_coachingEnabled = on;
  QSettings().setValue(QStringLiteral("coachingEnabled"), on);
  emit guidanceChanged();
}

void EngineBridge::setAutoTrainEnabled(bool on) {
  if (m_autoTrainEnabled == on) return;
  m_autoTrainEnabled = on;
  QSettings().setValue(QStringLiteral("autoTrainEnabled"), on);
  emit guidanceChanged();
}

// "Give it more capacity" — the usual fix when a network stalls well short of the mark
// because it is simply too small to represent the pattern.
void EngineBridge::growNetwork() {
  if (m_topo.layers.empty()) return;
  const int n = static_cast<int>(m_topo.layers.size());
  if (n == 1) {
    // No hidden layer at all: give it one.
    synapse::LayerInfo h;
    h.type = "dense";
    h.output_dim = std::max(6, m_topo.input_dim);
    h.activation = "relu";
    m_topo.layers.insert(m_topo.layers.begin(), h);
  } else {
    for (int k = 0; k < n - 1; ++k)
      m_topo.layers[k].output_dim = std::min(128, m_topo.layers[k].output_dim * 2);
  }
  // Re-chain dimensions and names so the topology stays consistent.
  int prev = m_topo.input_dim;
  for (int k = 0; k < static_cast<int>(m_topo.layers.size()); ++k) {
    m_topo.layers[k].input_dim = prev;
    m_topo.layers[k].name = "L" + std::to_string(k);
    prev = m_topo.layers[k].output_dim;
  }
  rebuild();  // a different architecture means starting the learning over
}

// Feed the latest numbers to the coach and carry out whatever it decided.
void EngineBridge::applyAutoAction() {
  const AutoTrainer::Action a = m_auto.step(m_metrics);

  if (a.isBest) m_bestParams = m_session->parameters();

  switch (a.kind) {
    case AutoTrainer::Action::LowerLr:
      setLearningRate(static_cast<double>(a.lr));
      break;
    case AutoTrainer::Action::Rewind:
      if (!m_bestParams.empty()) m_session->setParameters(m_bestParams);
      setLearningRate(static_cast<double>(a.lr));
      break;
    case AutoTrainer::Action::Stop:
      switch (m_auto.outcome()) {
        case AutoTrainer::Outcome::Converged: m_autoOutcome = QStringLiteral("converged"); break;
        case AutoTrainer::Outcome::Stalled:   m_autoOutcome = QStringLiteral("stalled"); break;
        case AutoTrainer::Outcome::Overfit:   m_autoOutcome = QStringLiteral("overfit"); break;
        case AutoTrainer::Outcome::Diverged:  m_autoOutcome = QStringLiteral("diverged"); break;
        case AutoTrainer::Outcome::None:      m_autoOutcome.clear(); break;
      }
      if (!m_bestParams.empty()) m_session->setParameters(m_bestParams);
      trainStop();
      applyMetrics(m_session->metrics(), false);
      refreshDisplay();  // last, so the graph ends up showing the user's input
      break;
    case AutoTrainer::Action::Continue:
      break;
  }

  if (!a.message.empty()) {
    m_autoStatus = QString::fromStdString(a.message);
    m_autoLog.push_back(m_autoStatus);
    if (m_autoLog.size() > 20) m_autoLog.removeFirst();
    emit autoChanged();
  }
}

void EngineBridge::trainEpoch() {
  if (m_dataset.empty()) {
    emit errorOccurred(QStringLiteral("this blueprint has no training data"));
    return;
  }
  m_timer.stop();
  setPlaying(false);
  seedInitialLoss();
  // budget 0 → exactly one epoch (the loop always runs once).
  applyMetrics(m_session->train(static_cast<float>(m_lr), 0, 1, m_gpuTraining, m_lastInput), true);
}

void EngineBridge::resetWeights() {
  ++m_seed;   // a fresh random initialization each reset
  rebuild();  // re-inits weights, clears the loss curve, runs a zeros pass
}

bool EngineBridge::isBuiltIn() const {
  if (m_blueprintPath.isEmpty()) return false;
  const QString def = QString::fromUtf8(SYNAPSE_MODELS_DIR) + "/defaults/" +
                      QFileInfo(m_blueprintPath).fileName();
  return QFile::exists(def);
}

// Serialize the current model — architecture, I/O semantics, and dataset — to a new
// blueprint file, then load it. This is how you keep a custom design without ever
// touching the built-in blueprints.
void EngineBridge::saveBlueprintAs(const QString& name) {
  const QString trimmed = name.trimmed();
  if (trimmed.isEmpty()) {
    emit errorOccurred(QStringLiteral("please enter a name"));
    return;
  }
  const QString dir = QString::fromUtf8(SYNAPSE_MODELS_DIR) + "/blueprints";
  const QString path = dir + "/" + slugify(trimmed) + ".json";

  QJsonObject root;
  root["name"] = trimmed;
  root["description"] = QStringLiteral("A custom blueprint.");
  root["input_dim"] = m_topo.input_dim;

  QJsonArray layers;
  for (const synapse::LayerInfo& L : m_topo.layers) {
    QJsonObject lo;
    lo["type"] = QString::fromStdString(L.type);
    lo["units"] = L.output_dim;
    lo["activation"] = QString::fromStdString(L.activation);
    layers.append(lo);
  }
  root["layers"] = layers;

  QJsonObject io, in, out;
  in["layout"] = m_inputLayout;
  if (m_inputLayout == "grid") {
    in["rows"] = m_inputRows;
    in["cols"] = m_inputCols;
  } else {
    QJsonArray labs;
    for (const QVariant& v : m_inputLabels) labs.append(v.toString());
    in["labels"] = labs;
  }
  QJsonArray range;
  range.append(0);
  range.append(1);
  in["range"] = range;
  QJsonArray outLabels;
  for (const QVariant& v : m_outputLabels) outLabels.append(v.toString());
  out["labels"] = outLabels;
  out["kind"] = m_outputKind;
  io["input"] = in;
  io["output"] = out;
  root["io"] = io;

  QJsonArray ds;
  for (const synapse::Sample& s : m_dataset.samples) {
    QJsonArray ji, jt;
    for (float v : s.input) ji.append(v);
    for (float v : s.target) jt.append(v);
    QJsonObject o;
    o["input"] = ji;
    o["target"] = jt;
    ds.append(o);
  }
  root["dataset"] = ds;

  QDir().mkpath(dir);
  QFile f(path);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("cannot write blueprint: %1").arg(path));
    return;
  }
  f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
  f.close();

  scanBlueprints();       // the new blueprint appears in the menu
  loadBlueprint(path);    // and becomes the active model
}

// Copy the pristine default back over the working file, then reload it.
void EngineBridge::restoreDefault() {
  if (!isBuiltIn()) return;
  const QString def = QString::fromUtf8(SYNAPSE_MODELS_DIR) + "/defaults/" +
                      QFileInfo(m_blueprintPath).fileName();
  QFile in(def);
  if (!in.open(QIODevice::ReadOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("cannot read default"));
    return;
  }
  const QByteArray bytes = in.readAll();
  in.close();
  QFile out(m_blueprintPath);
  if (!out.open(QIODevice::WriteOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("cannot restore blueprint"));
    return;
  }
  out.write(bytes);
  out.close();
  loadBlueprint(m_blueprintPath);
}

void EngineBridge::loadBlueprint(const QString& path) {
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    emit errorOccurred(QStringLiteral("cannot open blueprint: %1").arg(path));
    return;
  }
  const QByteArray bytes = f.readAll();
  f.close();

  // 1) Architecture — reuse the engine's parser, which ignores the "io" block.
  try {
    m_topo = synapse::parse_topology_json(bytes.toStdString());
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
    return;
  }

  // 2) Presentation — the "io" block. Dashboard-only; the engine never sees it.
  m_inputLayout = "labels";
  m_inputRows = 1;
  m_inputCols = 0;
  m_inputLabels.clear();
  m_outputLabels.clear();
  m_outputKind = "value";

  const QJsonObject root = QJsonDocument::fromJson(bytes).object();
  const QJsonObject io = root.value("io").toObject();
  const QJsonObject in = io.value("input").toObject();
  const QJsonObject out = io.value("output").toObject();

  m_inputLayout = in.value("layout").toString("labels");
  if (m_inputLayout == "grid") {
    m_inputRows = in.value("rows").toInt(1);
    m_inputCols = in.value("cols").toInt(m_topo.input_dim);
  } else {
    for (const QJsonValue& v : in.value("labels").toArray()) m_inputLabels.push_back(v.toString());
  }
  for (const QJsonValue& v : out.value("labels").toArray()) m_outputLabels.push_back(v.toString());
  m_outputKind = out.value("kind").toString("value");

  // 3) Training data — the optional "dataset" block (engine's parser again).
  m_dataset = synapse::parse_dataset_json(bytes.toStdString());
  m_blueprintPath = path;
  m_seed = 42;  // deterministic fresh start per blueprint

  emit blueprintChanged();
  syncDataset();  // engine owns the training loop, keep its copy current
  emit datasetChanged();
  rebuild();  // build the net + run a zeros pass (emits topology + activations)

  // Show a real training example on load instead of a blank input — the graph is
  // meaningful immediately, and training then visibly changes it.
  if (!m_dataset.empty()) loadExample(0);

  // If this blueprint has trained weights saved alongside it, pick up where you left off.
  resetJourney();
  emit weightsChanged();
  if (hasSavedWeights()) loadWeights();
}
