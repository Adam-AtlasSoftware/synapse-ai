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

  m_deviceName = QString::fromStdString(synapse::active_device_name());
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
  m_flatDirty = true;   // output_dim / dataset shape may have changed
  emit trainingProgress();
  try {
    m_net.build(m_topo, m_seed);  // re-initializes weights
    m_net.set_observer(this);     // fires on_topology -> refreshes name/columns/layers
    runForwardZeros();            // populate activations + weights so nothing is blank
  } catch (const std::exception& e) {
    emit errorOccurred(QString::fromUtf8(e.what()));
  }
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
    m_net.forward(m_lastInput);   // instant: emits on_step once
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
    m_net.forward(m_lastInput);
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
  m_flatDirty = true;
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
  m_flatDirty = true;
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
  m_flatDirty = true;
  emit datasetChanged();
}

void EngineBridge::removeExample(int i) {
  if (i < 0 || i >= m_dataset.size()) return;
  m_dataset.samples.erase(m_dataset.samples.begin() + i);
  m_flatDirty = true;
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
  const bool more = m_learnMode ? m_net.learn_step_advance() : m_net.step_forward();
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
    m_net.begin_forward(m_lastInput);  // reset + show input column
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
    m_net.begin_forward(m_lastInput);
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
    m_net.begin_learn_step(s.input, s.target, static_cast<float>(m_lr));
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
    if (!m_net.learn_active()) {
      loadExample(i);
      const synapse::Sample& s = m_dataset.samples[i];
      m_net.begin_learn_step(s.input, s.target, static_cast<float>(m_lr));
    } else {
      m_net.learn_step_advance();
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
    if (m_net.forward_done())
      m_net.begin_forward(m_lastInput);  // finished (or fresh) → reset to input
    else
      m_net.step_forward();              // otherwise advance one layer
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
  m_flatDirty = true;
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
void EngineBridge::rebuildEngine() {
  if (m_building) return;
  m_building = true;
  emit buildingChanged();

  const QString buildDir = QString::fromUtf8(SYNAPSE_BUILD_DIR);
  m_buildOutput = QStringLiteral("$ cmake --build \"%1\" --target synapse_dashboard\n\n").arg(buildDir);
  emit buildOutputChanged();

  QProcess* proc = new QProcess(this);
  proc->setProcessChannelMode(QProcess::MergedChannels);
  connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc]() {
    m_buildOutput += QString::fromUtf8(proc->readAllStandardOutput());
    emit buildOutputChanged();
  });
  connect(proc, &QProcess::finished, this, [this, proc](int code, QProcess::ExitStatus status) {
    proc->deleteLater();
    m_building = false;
    emit buildingChanged();
    if (code == 0 && status == QProcess::NormalExit) {
      m_buildOutput += QStringLiteral("\n✓ Build succeeded — relaunching with your new code…\n");
      emit buildOutputChanged();
      // Re-open the current blueprint in the new process (env is inherited by the child).
      if (!m_blueprintPath.isEmpty())
        qputenv("SYNAPSE_BLUEPRINT", QFileInfo(m_blueprintPath).completeBaseName().toUtf8());
      QProcess::startDetached(QCoreApplication::applicationFilePath(), QStringList{});
      QCoreApplication::quit();
    } else {
      m_buildOutput += QStringLiteral("\n✗ Build failed (exit %1). Fix the errors above and Rebuild.\n").arg(code);
      emit buildOutputChanged();
    }
  });
  proc->setWorkingDirectory(buildDir);
  proc->start(QStringLiteral("cmake"), {QStringLiteral("--build"), buildDir,
                                        QStringLiteral("--target"), QStringLiteral("synapse_dashboard")});
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

// One shuffled SGD pass over the dataset. Returns the average loss for the epoch.
float EngineBridge::run_one_epoch() {
  const int n = m_dataset.size();
  if (n == 0) return 0.0f;
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), m_rng);
  float sum = 0.0f;
  for (int idx : order) {
    const synapse::Sample& s = m_dataset.samples[idx];
    sum += m_net.train_step(s.input, s.target, static_cast<float>(m_lr));
  }
  ++m_epoch;
  return sum / static_cast<float>(n);
}

void EngineBridge::setGpuTraining(bool on) {
  if (m_gpuTraining == on) return;
  m_gpuTraining = on;
  emit gpuTrainingChanged();
}

// Flatten the dataset into contiguous input/target arrays for the batched GPU path.
void EngineBridge::ensureFlatDataset() {
  if (!m_flatDirty) return;
  const int inDim = m_topo.input_dim, outDim = m_topo.output_dim();
  m_flatIn.clear();
  m_flatOut.clear();
  m_flatIn.reserve(static_cast<size_t>(m_dataset.size()) * inDim);
  m_flatOut.reserve(static_cast<size_t>(m_dataset.size()) * outDim);
  for (const synapse::Sample& s : m_dataset.samples) {
    for (int i = 0; i < inDim; ++i)
      m_flatIn.push_back(i < static_cast<int>(s.input.size()) ? s.input[i] : 0.0f);
    for (int o = 0; o < outDim; ++o)
      m_flatOut.push_back(o < static_cast<int>(s.target.size()) ? s.target[o] : 0.0f);
  }
  m_flatDirty = false;
}

// One full-batch gradient-descent step on the GPU.
float EngineBridge::run_one_epoch_gpu() {
  const int n = m_dataset.size();
  if (n == 0) return 0.0f;
  ensureFlatDataset();
  const float l = m_net.train_epoch_batched(m_flatIn, m_flatOut, n, static_cast<float>(m_lr));
  ++m_epoch;
  return l;
}

// Re-run the forward pass on the input the user is looking at, so the graph and
// the prediction update live as the weights change during training.
void EngineBridge::refreshDisplay() {
  if (static_cast<int>(m_lastInput.size()) == m_topo.input_dim && m_topo.input_dim > 0) {
    try {
      m_net.forward(m_lastInput);
    } catch (const std::exception&) {
    }
  }
}

// Record the average loss over the dataset before any weight update, so the loss
// curve starts at the true peak and shows the whole descent.
void EngineBridge::seedInitialLoss() {
  if (m_dataset.empty() || !m_lossHistory.isEmpty()) return;
  float l0 = 0.0f;
  for (const synapse::Sample& s : m_dataset.samples) l0 += m_net.evaluate_loss(s.input, s.target);
  l0 /= static_cast<float>(m_dataset.size());
  m_currentLoss = l0;
  m_lossHistory.push_back(l0);
  refreshDisplay();
  emit trainingProgress();
}

void EngineBridge::onTrainTick() {
  if (m_dataset.empty()) {
    trainStop();
    return;
  }
  // Run as many epochs as fit in a small time budget, so a tick never stalls the UI:
  // tiny nets do hundreds of epochs per tick, big nets do a few.
  QElapsedTimer t;
  t.start();
  float last = static_cast<float>(m_currentLoss);
  int did = 0;
  do {
    last = m_gpuTraining ? run_one_epoch_gpu() : run_one_epoch();
    ++did;
  } while (t.elapsed() < 20 && did < 2000);

  m_currentLoss = last;
  m_lossHistory.push_back(last);
  if (m_lossHistory.size() > 1000) m_lossHistory.removeFirst();
  refreshDisplay();
  emit trainingProgress();
}

void EngineBridge::trainStart() {
  if (m_dataset.empty()) {
    emit errorOccurred(QStringLiteral("this blueprint has no training data"));
    return;
  }
  m_timer.stop();  // stop forward playback
  setPlaying(false);
  seedInitialLoss();
  setTraining(true);
  m_trainTimer.start();
}

void EngineBridge::trainStop() {
  m_trainTimer.stop();
  setTraining(false);
}

void EngineBridge::trainEpoch() {
  if (m_dataset.empty()) {
    emit errorOccurred(QStringLiteral("this blueprint has no training data"));
    return;
  }
  m_timer.stop();
  setPlaying(false);
  seedInitialLoss();
  m_currentLoss = m_gpuTraining ? run_one_epoch_gpu() : run_one_epoch();
  m_lossHistory.push_back(m_currentLoss);
  if (m_lossHistory.size() > 1000) m_lossHistory.removeFirst();
  refreshDisplay();
  emit trainingProgress();
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
  m_flatDirty = true;
  emit datasetChanged();
  rebuild();  // build the net + run a zeros pass (emits topology + activations)

  // Show a real training example on load instead of a blank input — the graph is
  // meaningful immediately, and training then visibly changes it.
  if (!m_dataset.empty()) loadExample(0);
}
