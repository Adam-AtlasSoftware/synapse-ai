#pragma once
#include <QObject>
#include <QString>
#include <QTimer>
#include <QVariantList>
#include <random>
#include <vector>

#include "synapse/activation.hpp"
#include "synapse/model_spec.hpp"
#include "synapse/network.hpp"
#include "synapse/telemetry.hpp"

// The bridge between the pure-C++ engine and QML. It IS a synapse::Observer: when
// the engine emits topology or a forward-pass snapshot, the bridge repackages that
// generic data into QML-friendly properties and fires change signals. QML binds to
// those properties and re-lays-out automatically — so the view knows nothing about
// any specific model, exactly as the telemetry contract intends.
//
// The bridge also owns the "current input" vector — the single source of truth the
// input controls (pixel grid / sliders) bind to — so that training can re-run that
// exact input each step and you see the activations and prediction change live.
class EngineBridge : public QObject, public synapse::Observer {
  Q_OBJECT
  Q_PROPERTY(QString modelName READ modelName NOTIFY topologyChanged)
  Q_PROPERTY(QString deviceName READ deviceName CONSTANT)
  Q_PROPERTY(int inputDim READ inputDim NOTIFY topologyChanged)
  Q_PROPERTY(int outputDim READ outputDim NOTIFY topologyChanged)
  // columns: [{title, activation, count, kind}] — one per visible neuron column.
  Q_PROPERTY(QVariantList columns READ columns NOTIFY topologyChanged)
  // layers: [{index, units, activation, name}] — the editable dense layers only.
  Q_PROPERTY(QVariantList layers READ layers NOTIFY topologyChanged)
  // activationNames: every activation the engine knows (built-ins + custom C++), for
  // the layer dropdowns. CONSTANT within a run — new custom ones appear after a rebuild.
  Q_PROPERTY(QStringList activationNames READ activationNames CONSTANT)
  // --- Tier-2 "Code Lab": edit the engine's C++, recompile, relaunch --------
  Q_PROPERTY(QString buildOutput READ buildOutput NOTIFY buildOutputChanged)
  Q_PROPERTY(bool building READ building NOTIFY buildingChanged)
  // activations: [[...], [...], ...] aligned with columns (current forward pass).
  Q_PROPERTY(QVariantList activations READ activations NOTIFY activationsChanged)
  // weights: [{rows, cols, values}] — one per layer, for drawing edges.
  Q_PROPERTY(QVariantList weights READ weights NOTIFY activationsChanged)
  // deltas: [[...] per column] = ∂loss/∂pre (node gradient signal, backprop viz).
  Q_PROPERTY(QVariantList deltas READ deltas NOTIFY activationsChanged)
  // grads: [{rows,cols,values} per layer] = ∂loss/∂W (edge gradient, backprop viz).
  Q_PROPERTY(QVariantList grads READ grads NOTIFY activationsChanged)
  // biases: [[...] per layer] — one bias per neuron (for the neuron inspector).
  Q_PROPERTY(QVariantList biases READ biases NOTIFY activationsChanged)
  // --- playback state (watch the forward pass move one layer at a time) ----
  Q_PROPERTY(int speedMs READ speedMs WRITE setSpeedMs NOTIFY speedChanged)
  Q_PROPERTY(bool playing READ playing NOTIFY playbackChanged)
  // activeLayer: index of the layer just computed (-1 = none / instant run).
  Q_PROPERTY(int activeLayer READ activeLayer NOTIFY activationsChanged)
  // phase: "forward" (instant), or "begin"/"step"/"done" while stepping.
  Q_PROPERTY(QString phase READ phase NOTIFY activationsChanged)

  // --- blueprints: template architectures with meaning assigned to I/O ------
  // blueprints: [{name, description, path}] discovered on disk.
  Q_PROPERTY(QVariantList blueprints READ blueprints NOTIFY blueprintsListChanged)
  // input layout: "labels" (named sliders) or "grid" (paintable pixels).
  Q_PROPERTY(QString inputLayout READ inputLayout NOTIFY blueprintChanged)
  Q_PROPERTY(int inputRows READ inputRows NOTIFY blueprintChanged)
  Q_PROPERTY(int inputCols READ inputCols NOTIFY blueprintChanged)
  Q_PROPERTY(QVariantList inputLabels READ inputLabels NOTIFY blueprintChanged)
  // output: labels per output neuron, and kind = "class" (argmax) or "value".
  Q_PROPERTY(QVariantList outputLabels READ outputLabels NOTIFY blueprintChanged)
  Q_PROPERTY(QString outputKind READ outputKind NOTIFY blueprintChanged)
  // isBuiltIn: this blueprint ships a pristine default that "Restore default" reverts to.
  Q_PROPERTY(bool isBuiltIn READ isBuiltIn NOTIFY blueprintChanged)
  // predicted class index/label (argmax of the output column); -1 if none.
  Q_PROPERTY(int predictedIndex READ predictedIndex NOTIFY activationsChanged)
  Q_PROPERTY(QString predictedLabel READ predictedLabel NOTIFY activationsChanged)

  // --- training ------------------------------------------------------------
  // datasetChanged (not blueprintChanged) so the UI flips the moment the first
  // example is added — loadBlueprint emits datasetChanged too, so loads still update.
  Q_PROPERTY(bool hasDataset READ hasDataset NOTIFY datasetChanged)
  Q_PROPERTY(bool training READ training NOTIFY trainingChanged)
  Q_PROPERTY(double learningRate READ learningRate WRITE setLearningRate NOTIFY learningRateChanged)
  Q_PROPERTY(int epoch READ epoch NOTIFY trainingProgress)
  Q_PROPERTY(double currentLoss READ currentLoss NOTIFY trainingProgress)
  Q_PROPERTY(QVariantList lossHistory READ lossHistory NOTIFY trainingProgress)
  Q_PROPERTY(bool trained READ trained NOTIFY trainingProgress)
  // gpuTraining: full-batch gradient descent on the GPU vs per-sample SGD on the CPU.
  Q_PROPERTY(bool gpuTraining READ gpuTraining WRITE setGpuTraining NOTIFY gpuTrainingChanged)

  // --- input + dataset -----------------------------------------------------
  // current input vector — the input controls bind to this and write it back.
  Q_PROPERTY(QVariantList input READ input NOTIFY inputChanged)
  Q_PROPERTY(int datasetSize READ datasetSize NOTIFY datasetChanged)

 public:
  explicit EngineBridge(QObject* parent = nullptr);

  QString modelName() const { return m_modelName; }
  QString deviceName() const { return m_deviceName; }
  int inputDim() const { return m_topo.input_dim; }
  int outputDim() const { return m_topo.output_dim(); }
  QVariantList columns() const { return m_columns; }
  QVariantList layers() const { return m_layers; }
  QVariantList activations() const { return m_activations; }
  QVariantList weights() const { return m_weights; }
  QVariantList deltas() const { return m_deltas; }
  QVariantList grads() const { return m_grads; }
  QVariantList biases() const { return m_biases; }
  int speedMs() const { return m_speedMs; }
  void setSpeedMs(int ms);
  bool playing() const { return m_playing; }
  int activeLayer() const { return m_activeLayer; }
  QString phase() const { return m_phase; }
  QVariantList blueprints() const { return m_blueprints; }
  QStringList activationNames() const;
  QString buildOutput() const { return m_buildOutput; }
  bool building() const { return m_building; }
  QString inputLayout() const { return m_inputLayout; }
  int inputRows() const { return m_inputRows; }
  int inputCols() const { return m_inputCols; }
  QVariantList inputLabels() const { return m_inputLabels; }
  QVariantList outputLabels() const { return m_outputLabels; }
  QString outputKind() const { return m_outputKind; }
  bool isBuiltIn() const;
  int predictedIndex() const { return m_predictedIndex; }
  QString predictedLabel() const;
  bool hasDataset() const { return !m_dataset.empty(); }
  bool training() const { return m_training; }
  double learningRate() const { return m_lr; }
  void setLearningRate(double lr);
  int epoch() const { return m_epoch; }
  double currentLoss() const { return m_currentLoss; }
  QVariantList lossHistory() const { return m_lossHistory; }
  bool trained() const { return m_epoch > 0; }
  bool gpuTraining() const { return m_gpuTraining; }
  void setGpuTraining(bool on);
  QVariantList input() const;
  int datasetSize() const { return m_dataset.size(); }

  // --- input controls (bridge owns the current input vector) ---------------
  Q_INVOKABLE void setInput(int index, double value);       // set one component + forward
  Q_INVOKABLE void setInputVector(const QVariantList& values);
  Q_INVOKABLE void forwardCurrent();   // run inference on the current input
  Q_INVOKABLE void playCurrent();      // animate a forward pass on the current input
  Q_INVOKABLE void clearInput();       // all zeros
  Q_INVOKABLE void randomizeInput();

  // --- dataset browsing + manual examples ----------------------------------
  Q_INVOKABLE QVariantList exampleInput(int i) const;
  Q_INVOKABLE QVariantList exampleTarget(int i) const;
  Q_INVOKABLE QString exampleLabel(int i) const;   // human label of example i's target
  Q_INVOKABLE void loadExample(int i);             // load example i into the input
  Q_INVOKABLE void addExample(const QVariantList& in, int labelIndex);  // classifiers
  Q_INVOKABLE void addExampleRaw(const QVariantList& in, const QVariantList& target);
  Q_INVOKABLE void updateExample(int i, const QVariantList& in, const QVariantList& target);
  Q_INVOKABLE void removeExample(int i);
  Q_INVOKABLE void saveDataset();                  // persist the dataset to the blueprint

  // --- inference -----------------------------------------------------------
  Q_INVOKABLE void runForward(const QVariantList& input);
  Q_INVOKABLE void runForwardZeros();

  // --- stepped playback (watch the forward pass in motion) -----------------
  Q_INVOKABLE void play(const QVariantList& input);  // reset + auto-advance
  Q_INVOKABLE void pause();                          // freeze at current step
  Q_INVOKABLE void stepOnce();                       // advance exactly one layer
  Q_INVOKABLE void beginForward(const QVariantList& input);  // reset, show input

  // --- gradient-flow animation (one SGD step in slow motion) ---------------
  Q_INVOKABLE void playLearnExample(int i);  // animate forward→loss→backward→update
  Q_INVOKABLE void stepLearnExample(int i);  // advance the learn-step by hand

  // --- blueprints ----------------------------------------------------------
  Q_INVOKABLE void loadBlueprint(const QString& path);       // load a template file
  Q_INVOKABLE bool loadBlueprintByName(const QString& name); // by display name or file stem
  Q_INVOKABLE void saveBlueprintAs(const QString& name);     // persist current as a new blueprint
  Q_INVOKABLE void restoreDefault();                         // revert a built-in to its original

  // --- training controls ---------------------------------------------------
  Q_INVOKABLE void trainStart();    // run SGD continuously
  Q_INVOKABLE void trainStop();     // pause training
  Q_INVOKABLE void trainEpoch();    // one epoch (manual step)
  Q_INVOKABLE void resetWeights();  // re-initialize weights, clear the loss curve

  // --- Tier-1 model editing (no recompile; rebuilds the in-memory net) ------
  Q_INVOKABLE void loadModel(const QString& path);
  Q_INVOKABLE void saveModel(const QString& path);
  Q_INVOKABLE QString modelJson() const;
  Q_INVOKABLE void setInputDim(int dim);
  // Switch the input widget ("labels", "grid", or "segments"); rows/cols apply to grid.
  // Adjusts input_dim to match (grid → rows·cols, segments → 7) and rebuilds.
  Q_INVOKABLE void setInputLayout(const QString& layout, int rows, int cols);
  Q_INVOKABLE void addLayer(int units, const QString& activation);
  Q_INVOKABLE void removeLayer(int index);
  Q_INVOKABLE void setLayerUnits(int index, int units);
  Q_INVOKABLE void setLayerActivation(int index, const QString& activation);

  // --- Tier-2: edit real engine C++ (custom activations), recompile, relaunch ---
  // Read/write the user-editable source; rebuildEngine() saves it, runs the build,
  // streams output to buildOutput, and (on success) relaunches the app.
  Q_INVOKABLE QString customActivationSource() const;
  Q_INVOKABLE void saveCustomActivationSource(const QString& src);
  Q_INVOKABLE void rebuildEngine();

  // --- synapse::Observer ---------------------------------------------------
  void on_topology(const synapse::Topology& topo) override;
  void on_step(const synapse::StepSnapshot& snap) override;

 signals:
  void topologyChanged();
  void activationsChanged();
  void speedChanged();
  void playbackChanged();
  void blueprintChanged();
  void blueprintsListChanged();
  void trainingChanged();
  void trainingProgress();
  void learningRateChanged();
  void gpuTrainingChanged();
  void inputChanged();
  void datasetChanged();
  void errorOccurred(const QString& message);
  void buildOutputChanged();
  void buildingChanged();

 private slots:
  void onTick();       // forward-playback timer: advance one layer
  void onTrainTick();  // training timer: run a chunk of SGD epochs

 private:
  void rebuild();  // (re)build the net from m_topo, then run a zeros pass
  void setPlaying(bool playing);
  void setTraining(bool training);
  std::vector<float> toVector(const QVariantList& input) const;
  void scanBlueprints();      // discover the template files on disk
  float run_one_epoch();      // shuffle + one CPU SGD pass over the dataset; returns avg loss
  float run_one_epoch_gpu();  // one full-batch GPU gradient-descent step
  void ensureFlatDataset();   // (re)build the flattened input/target arrays for the GPU path
  void refreshDisplay();      // re-run forward on the current input to update the graph
  void seedInitialLoss();     // record the pre-training loss as the curve's first point

  synapse::Network m_net;
  synapse::Topology m_topo;
  QString m_modelName;
  QString m_deviceName;
  QVariantList m_columns;
  QVariantList m_layers;
  QVariantList m_activations;
  QVariantList m_weights;
  QVariantList m_deltas;
  QVariantList m_grads;
  QVariantList m_biases;

  QTimer m_timer;
  int m_speedMs = 600;
  bool m_playing = false;
  bool m_learnMode = false;  // m_timer is driving a learn-step (not a forward pass)
  int m_activeLayer = -1;
  QString m_phase;
  std::vector<float> m_lastInput;

  // blueprint / semantic I/O metadata
  QVariantList m_blueprints;
  QString m_inputLayout = "labels";
  int m_inputRows = 1;
  int m_inputCols = 0;
  QVariantList m_inputLabels;
  QVariantList m_outputLabels;
  QString m_outputKind = "value";
  int m_predictedIndex = -1;

  // training state
  QTimer m_trainTimer;
  synapse::Dataset m_dataset;
  QString m_blueprintPath;  // for persisting added examples
  std::mt19937 m_rng{1};
  bool m_training = false;
  double m_lr = 0.1;   // a safe default that converges reliably for both MSE and softmax
  int m_epoch = 0;
  double m_currentLoss = 0.0;
  QVariantList m_lossHistory;
  unsigned m_seed = 42;

  bool m_gpuTraining = false;
  std::vector<float> m_flatIn;   // N*input_dim, cached for the GPU batched path
  std::vector<float> m_flatOut;  // N*output_dim
  bool m_flatDirty = true;

  // Tier-2 "Code Lab": rebuild the engine from edited C++, then relaunch.
  QString m_buildOutput;
  bool m_building = false;
};
