# Synapse-AI

A from-scratch neural-network engine **and** a live visual dashboard for watching it
think. Built to *learn how AI actually works* — every neuron, weight, activation, and
(soon) gradient is visible and adjustable.

The engine is written in **SYCL** (via [AdaptiveCpp](https://adaptivecpp.github.io/)) so
it runs on any hardware — your GPU if you have one, the CPU otherwise. The dashboard is
**Qt 6 / QML**. The two are deliberately kept apart by a small, generic data contract, so
you can change the model however you like and the visualization adapts on its own.

![dashboard](docs/dashboard.png)

---

## What makes it tick: the decoupled design

The whole project hangs on one idea: **the visualization never knows about any specific
model.** The engine describes itself and streams what it computes through a handful of
plain C++ structs; the dashboard renders whatever it is handed.

```
  ┌─────────────────────────┐     Topology + StepSnapshot      ┌────────────────────────┐
  │  synapse_engine (SYCL)   │  ── (plain C++ structs) ──▶      │  synapse_dashboard      │
  │  tensors, layers, net,    │                                  │  (Qt 6 / QML)          │
  │  forward pass, JSON model │  ◀── load / edit / step ──       │  auto-laid-out graph,   │
  └─────────────────────────┘                                  │  controls, model editor │
        no Qt in here                                          └────────────────────────┘
                                                                   no SYCL in here
```

- **`Topology`** — the network's shape (layers, sizes, activations). The dashboard uses it
  to lay out the columns of neurons. Change it and the graph re-lays-out itself.
- **`StepSnapshot`** — a moment of computation: the current activations, weights, biases,
  and which layer just fired. The dashboard colors nodes and edges from this.

Because that's *all* the dashboard sees, a 2-4-1 XOR net and a 784-128-10 digit classifier
render with **zero** changes to the visualization code. The contract lives in
[`engine/include/synapse/telemetry.hpp`](engine/include/synapse/telemetry.hpp).

### Two ways to change the model — all C++ or raw data, no scripting language

1. **Data (no recompile).** The network's structure and hyperparameters live in a small
   **JSON** file the dashboard edits through nice controls. The engine rebuilds the
   in-memory network instantly. This covers layer count, sizes, and activation choices.
2. **Logic (recompile).** New *math* — a novel activation, a new layer type, a custom
   kernel — is real C++ you edit and recompile. *(Planned; see the roadmap.)*

### Why the engine and dashboard are separate libraries

SYCL (AdaptiveCpp) is compiled by its own Clang toolchain; Qt is compiled by the ordinary
one. Keeping the engine in its own library with **pure-C++ public headers** (no `sycl.hpp`
leaks — the device details hide behind a PIMPL) lets the two toolchains coexist cleanly and
keeps the boundary honest.

---

## Directory layout

```
synapse-ai/
  CMakeLists.txt          top-level build
  CMakePresets.json       the "default" preset: Clang + Ninja + AdaptiveCpp + Qt
  engine/
    include/synapse/      PUBLIC headers — pure C++ (telemetry, network, model_spec,
                          activation, optimizer, metrics)
    src/                  SYCL kernels + logic; the registries and the two files you
                          edit from the GUI: custom_activations.cpp, custom_optimizers.cpp
  host/                   synapse_engine_host — the engine as its own process, speaking
                          the telemetry contract as JSON over stdin/stdout
  dashboard/
    src/                  main.cpp, EngineBridge (C++ ↔ QML), ModelSession (in-process
                          vs subprocess engine behind one interface)
    qml/                  Main.qml, NetworkView.qml, InputWidget/PixelGrid/SegmentDisplay,
                          LossChart, StatsPanel, DataManager, CodeLab
  models/
    blueprints/           the models you load (architecture + I/O meaning + dataset)
    defaults/             pristine copies for "Restore default"
    weights/              trained weights, saved beside their blueprint
  res/                    brand artwork (app icon, marks) compiled into the binary
  examples/               the original SYCL sandbox files, still buildable
  tests/                  headless engine tests (run via ctest)
```

---

## Prerequisites

These are already installed on the dev machine; listed here for portability.

| Tool          | Why                                  |
|---------------|--------------------------------------|
| AdaptiveCpp   | the SYCL implementation (`acpp`)     |
| Qt 6.5+       | Quick, Qml, QuickControls2, Charts   |
| CMake 3.25+   | build system                         |
| Ninja         | fast generator                       |
| Clang         | C++17 compiler                       |

---

## Build & run

From the repo root:

```bash
cmake --preset default          # configure (Clang + Ninja + AdaptiveCpp + Qt)
cmake --build --preset default  # build everything
```

Then:

```bash
./build/dashboard/synapse_dashboard    # the visual dashboard
ctest --test-dir build                 # run the headless engine tests
./build/examples/matrix_gpu            # the original SYCL sandbox
```

**The engine runs as its own process.** The dashboard launches `synapse_engine_host` and
talks to it over pipes, so a crash in C++ you wrote in the Code Lab takes down only the
engine, and rebuilding swaps it in without restarting the app. It costs about 19% of
training throughput (~1030 vs ~1265 epochs/s on the 65-example OCR set) — imperceptible,
since both converge in a second or two. Set `SYNAPSE_ENGINE_INPROCESS=1` to link the
engine in directly instead. If the host binary can't start, it falls back automatically.

You can drive the engine yourself — it is just newline-delimited JSON:

```bash
printf '%s\n' '{"id":1,"cmd":"load","path":"models/blueprints/xor.json"}' \
               '{"id":2,"cmd":"train","lr":0.5,"budget_ms":300}' \
               '{"id":3,"cmd":"forward","input":[1,0]}' \
               '{"id":4,"cmd":"quit"}' | ./build/host/synapse_engine_host
```

In **VS Code**, the CMake Tools extension picks up `CMakePresets.json` automatically —
choose the "default" preset and hit Build/Run. `clangd` reads the generated
`build/compile_commands.json`.

> **Note:** On startup you may see AdaptiveCpp `[ze_backend]` / `zeInit failed` warnings.
> These are **harmless** — AdaptiveCpp is just probing for an Intel GPU and finding none; it
> then uses whatever device you *do* have (e.g. an NVIDIA GPU via CUDA).

---

## Using the dashboard

When it opens you'll see the network as columns of neurons, with the model name and your
compute device in the header. The controls on the right are organized into tabs:
**Run** (inputs + playback), **Train** (training + data), **Model** (architecture), and
**Info** (legend + per-layer stats).

**Blueprints** — *pick a model with meaning already assigned*

![blueprints](docs/blueprints.png)

Click the model name in the header (e.g. **"XOR ▾"**) to open the **Blueprints** menu — a
ladder of ready-made architectures where the inputs and outputs *mean* something:

| Blueprint            | Inputs                     | Outputs                 |
|----------------------|----------------------------|-------------------------|
| XOR                  | 2 labeled bits (A, B)      | A XOR B                 |
| Full adder           | 3 bits (A, B, carry-in)    | Sum, carry-out          |
| Seven-segment digit  | 7 segment toggles (a–g)    | digit 0–9               |
| Tiny shapes (3×3)    | a **paintable 3×3 grid**   | horizontal / vertical / diagonal |
| OCR digits (5×5)     | a **paintable 5×5 grid**   | digit 0–9               |

Picking one sets the input/output dimensions, the layers, *and* the semantics. Grid
blueprints give you a **pixel canvas to draw on**; labeled ones give you named sliders.
For classifiers, output neurons show their labels and the **predicted class** (argmax) is
ringed in green. A blueprint is just a JSON file (see below), so adding your own is a
drop-in — and you can still freely edit any blueprint in the Model panel afterward.

**Make your own.** Pick **＋ New blank model** for a clean slate (so tweaking dimensions never
touches the built-ins), change the layers in the Model panel, add data in the Data manager,
then **💾 Save as new blueprint…** to persist it under a new name — it appears in the menu.
Every built-in also has **↺ Restore default**, which reverts it to its shipped architecture
and data, so you can experiment freely and always get the original back. (Edits to a built-in
live only in the working copy until you restore — save a copy first if you want to keep them.)

> Predictions are meaningful only **after training** (Phase 4). On an untrained net the
> "prediction" is the argmax of random weights — hence the *(untrained)* tag next to it.

You can also launch straight into one:
`SYNAPSE_BLUEPRINT=ocr_5x5 ./build/dashboard/synapse_dashboard`.

**Reading the picture**
- **Nodes** are neurons. Their color shows the current activation value: **amber** =
  positive, **slate/grey** = near zero, **blue** = negative. The number inside is the value.
- **Click any neuron** to open the inspector — it breaks down exactly how that neuron
  computes: `value = activation(pre-activation)`, and `pre-activation = bias + Σ(input × weight)`
  with the biggest contributing inputs listed. Click ✕ (or another neuron) to change it.
- When you **step** a forward or backward pass, a caption below the graph narrates each stage
  in plain language ("Forward · L0: each neuron multiplies its inputs by its weights…").
- **Edges** are weights. Same color meaning (amber positive, blue negative); thicker/brighter
  = larger magnitude.
- **Columns** are laid out straight from the model's dimensions — the leftmost is your input.

**Inputs** (right panel) — *the input widget matches the blueprint*
- Three input methods, chosen per blueprint and switchable in the Model tab:
  **Sliders** (named values), **Pixel grid** (paint the pixels — OCR, shapes), and
  **7-segment** (click segments on a real calculator-style display).
- **Run (instant)** does a full forward pass immediately. **Clear** / **Rand** set the inputs.
- Making your own model? Pick the input method in the **Model** tab and the grid size with
  it — the network's input dimension follows automatically (grid → rows×cols, 7-segment → 7).

**Playback** — *watch it happen one step at a time*
- **▶ Play** resets to the input, then propagates through the network **one layer per tick**.
  The active column lights up (white ring); columns not yet computed are dimmed.
- **⏸ Pause** freezes mid-pass.
- **⧐ Step** advances exactly one layer per click — press it repeatedly to walk through the
  computation by hand.
- The **Slow ↔ Fast** slider sets how long each layer lingers (16 ms – 2 s), so you can make
  the signal crawl across the network and actually see each stage.
- The status line tells you what just happened ("Computing L0 …", "output ready ✓").

**Training** — *watch it learn*
- Every blueprint ships with a dataset, so hit **▶ Train** and SGD runs continuously: the
  **loss curve** falls toward zero and, for classifiers, the green prediction ring migrates
  to the correct answer. **⏸ Pause** stops it, **+1 epoch** steps by hand, **Reset**
  re-randomizes the weights to start over.
- The **learn rate** slider sets how big each gradient step is — too small crawls, too large
  diverges. Try it and watch the curve.
- **Optimizer** picks *how* each weight moves once backprop knows the direction:
  **sgd** steps straight downhill; **momentum** keeps a velocity so consistent directions
  accelerate; **adam** gives every weight its own adaptive step size. On a fixed budget the
  difference is stark — in the test suite, 150 epochs give sgd `0.0067`, momentum `0.00033`,
  adam `0.00015`. (Adam adapts its own scale, so pair it with a smaller learn rate, ≈0.02.)
- **Hold out** reserves a slice of the data that training never sees — the only way to tell
  *learning* from *memorizing*. Turn it on and the loss chart draws a second **amber
  validation curve** beside the blue training one, and for classifiers you get
  **train vs held-out accuracy**. On the 10-digit seven-segment set, holding out 30% gives
  the classic picture: training loss → 0 and 100% train accuracy, while validation loss
  *climbs* and held-out accuracy sits at 0% — it memorized the 7 digits it saw and never
  learned the idea. That is overfitting, and now you can watch it happen.
- **💾 Save trained weights** keeps the training. Weights go to
  `models/weights/<blueprint>.json` — a sidecar, so blueprints stay clean templates and
  **Restore default** never fights them. They reload automatically next time you open that
  blueprint (and survive an engine hot-swap), so a trained model is a thing you keep.
- Under the hood: **backpropagation** computes the gradient of the loss with respect to every
  weight (the chain rule, applied layer by layer), then each weight moves a little downhill.
  That math is verified against finite differences in the test suite (`ctest`).
- `SYNAPSE_AUTOTRAIN=1 ./build/dashboard/synapse_dashboard` starts training on launch.
- **Compute** (toggle in the Training panel): **CPU (SGD)** does per-sample stochastic gradient
  descent on the host; **GPU (full-batch)** does full-batch gradient descent as real matrix
  operations on the GPU (in-order queue, two syncs per epoch). Both are gradient-checked in the
  tests. On this AdaptiveCpp-JIT + GPU combo the two are roughly **on par** for these dataset
  sizes — the GPU's edge only shows with large batches, since per-kernel launch overhead caps
  the gains at small scale. Each training tick is time-bounded so the UI stays smooth either way.

**Data** — *see what it learns from, and teach it yourself*
- The **Data** panel browses the training examples (◀ / ▶). **Show this example** loads it into
  the input, so you can see each digit/pattern the network is being taught.
- **Watch it learn (slow)** animates one gradient-descent step on the current example: the
  forward pass lights up left-to-right, the loss is measured, then the **gradient flows
  backward** (magenta) one layer at a time, and finally the weights update. Paced by the
  Playback speed slider; **⧐ Step** walks it by hand.
- **Manual training** (classifier blueprints): draw/set an input, pick the right **answer**, and
  **+ Add current input** appends it as a new training example. **Save** persists it into the
  blueprint's JSON. Draw a few 7s, add them, and train.
- **Manage training data…** opens a dedicated window: a gallery of every example (click to
  edit, ✕ to delete) beside an editor for adding/changing pairs. Built for entering data fast —
  draw a digit and **press its number key** to label-and-commit in one keystroke; **Enter**
  commits, **Delete** removes the selected example, **Save to file** writes it all to the JSON.

**Model editor** (Tier-1 editing — no recompile)
- Change the **input method**, a layer's **units**, or its **activation**, or **✕**/**＋ Add
  layer**. Every edit rebuilds the network and the graph re-lays-out itself. (Weights
  re-initialize on a structural change, which also clears the loss curve.)

**Code Lab** (Tier-2 editing — *write new math in C++*)

This is the part the whole design was built for: when config isn't enough, you write real
C++ and the app recompiles the engine around it.

- **Model tab → ⚙ Custom C++ (activations)…** opens an editor on a real engine source file,
  `engine/src/custom_activations.cpp`.
- An activation is two small functions: `forward(pre, act, n)` and `derivative(pre, post)`.
  Pick one of six worked **examples** — `leaky_relu`, `swish`, `elu`, `softplus`, `gelu`,
  `sine` — and **＋ Insert at cursor**. Every one is gradient-checked in the test suite, so
  the math you copy is verified.
- **⚙ Rebuild & Apply** compiles it. The build log streams into the window; compiler errors
  land there and the running engine is untouched, so you just fix and rebuild. On success the
  new engine is **swapped in live** — your session, your data, and your trained weights stay
  put — and your activation appears in every layer's activation dropdown.
- Optimizers are pluggable the same way: `engine/src/custom_optimizers.cpp` takes a name, how
  many floats of per-parameter state you need (0 for SGD, 1 for a velocity, 2 for Adam's
  moments), and the update rule. There's a commented `rmsprop` to start from.

---

## The JSON model spec

A model is just data. Blueprints live in `models/blueprints/*.json` and are discovered
automatically. `models/blueprints/xor.json`:

```json
{
  "name": "XOR",
  "description": "The classic. Two binary inputs, one output = A XOR B.",
  "input_dim": 2,
  "layers": [
    { "type": "dense", "units": 4, "activation": "tanh" },
    { "type": "dense", "units": 1, "activation": "sigmoid" }
  ],
  "io": {
    "input":  { "layout": "labels", "labels": ["A", "B"], "range": [0, 1] },
    "output": { "layout": "labels", "labels": ["A XOR B"], "kind": "value" }
  }
}
```

- **Architecture** (`input_dim`, `layers`) is read by the *engine*. Each layer's input size
  is inferred from the previous layer. Built-in activations: `linear`, `sigmoid`, `relu`,
  `tanh`, `softmax` — plus **any activation you wrote** in the Code Lab, referenced by name.
- **`io`** is read only by the *dashboard* (the engine ignores it) and assigns meaning:
  - `input.layout`: `"labels"` (named sliders, with a `labels` array), `"grid"` (a paintable
    canvas, with `rows`/`cols`), or `"segments"` (a clickable seven-segment display; the 7
    inputs are the segments a–g in the standard order).
  - `output.labels`: a name per output neuron. `output.kind`: `"class"` highlights the
    argmax as a prediction; `"value"` just shows the numbers.

Drop a new file in `models/blueprints/` and it appears in the menu — no code, no rebuild.

---

## How the AI works (the learning arc)

The code is built up in the same order the concepts build on each other:

1. **Tensors** — blocks of numbers on the GPU (`engine/src/tensor.hpp`).
2. **A dense layer** — `output = activation(W · input + b)` (`engine/src/network.cpp`).
3. **Activations** — the non-linearities that let a network learn curves, not just lines.
4. **A forward pass** — stacking layers to turn an input into a prediction.
5. **Loss** — measuring how wrong the prediction is (MSE, or cross-entropy for softmax).
6. **Backpropagation** — the chain rule computing how each weight affected the error.
7. **An optimizer** — nudging weights down the gradient. Repeat = learning.
8. **Generalization** — holding data back to find out whether it learned the pattern or
   just memorized the answers. *(this is where the interesting questions start)*

---

## Roadmap

- [x] **Phase 0** — build system + toolchain (SYCL engine links into the Qt app)
- [x] **Phase 1** — engine forward path + the telemetry contract
- [x] **Phase 2** — live network + activations dashboard, JSON model editor
- [x] **Playback** — speed control + step-through of the forward pass
- [x] **Phase 3** — Blueprints: template architectures with meaning assigned to the inputs
      and outputs (paintable pixel grids, labeled I/O, predicted-class highlight)
- [x] **Phase 4** — training: loss (MSE + cross-entropy), **backprop from scratch**
      (finite-difference gradient-checked), SGD, a **live loss curve**, learning-rate
      control, a **dataset for every blueprint**, **dataset browsing + draw-your-own
      examples**, and a **gradient-flow animation** — watch one SGD step in slow motion
      (forward → loss → backprop layer-by-layer → weight update).
- [x] **Phase 5** — beginner vs advanced views: an **Advanced** toggle reveals per-layer
      activation/weight/gradient μ·σ with live **weight and activation histograms**; a
      **click-any-neuron inspector** breaks down `value = activation(bias + Σ inputs×weights)`;
      stepping is narrated in plain language.
- [x] **Phase 6** — the full "edit code in the GUI, recompile, run" end-state:
      **Tier-2 Code Lab** (write a custom activation in real C++, rebuild, use it) and the
      **engine as a separate process** streaming the telemetry contract as JSON over pipes,
      so a rebuild hot-swaps the engine without restarting the app.
- [x] **Phase 7** — keep what you train and find out if it generalizes: **pluggable
      optimizers** (sgd/momentum/adam, and write your own), an **opt-in validation split**
      with train-vs-held-out **accuracy** and a dual loss curve that makes **overfitting**
      visible, and **trained-weight persistence** that survives reloads and hot-swaps.

### Env hooks (handy for demos and screenshots)

| Variable | Effect |
|---|---|
| `SYNAPSE_BLUEPRINT=ocr_5x5` | launch straight into a blueprint |
| `SYNAPSE_ENGINE_INPROCESS=1` | link the engine in instead of running it as a process |
| `SYNAPSE_AUTOTRAIN=1` | start training immediately |
| `SYNAPSE_HOLDOUT=0.3` | hold out 30% for validation at launch |
| `SYNAPSE_OPTIMIZER=adam` | pick the update rule at launch |
| `SYNAPSE_TAB=1` | open a tab (0 Run, 1 Train, 2 Model, 3 Info) |
| `SYNAPSE_ADVANCED=1` | start in the advanced view |
| `SYNAPSE_CODELAB=1` / `SYNAPSE_DATAMANAGER=1` | open a tool window at launch |
| `SYNAPSE_SELECT="1,3"` | pre-select a neuron in the inspector |
| `SYNAPSE_LEARNSTEP=N` | advance the gradient-flow animation N sub-steps |
| `SYNAPSE_SAVEWEIGHTS=1` | write the weights sidecar shortly after launch |
| `SYNAPSE_GRAB=out.png` | render the window to a PNG and quit |

### What the tests cover (`ctest --test-dir build`)

| Test | Proves |
|---|---|
| `forward_smoke` | builds any JSON net and runs inference; public headers stay SYCL-free |
| `train_xor` / `train_classifier` | backprop matches finite differences, and the net learns (sigmoid+MSE and softmax+cross-entropy) |
| `train_gpu_batched` | the full-batch GPU training path agrees with the CPU one |
| `custom_activation` | all six Code Lab activation templates are gradient-correct and train |
| `optimizers` | sgd/momentum/adam all train, don't disturb backprop, and parameters round-trip exactly |
| `engine_host_ipc` | the engine works as a separate process: streams telemetry, trains XOR, survives a bad command |

---

## Troubleshooting

- **`[ze_backend] zeInit failed` on startup** — harmless (see the build note above).
- **`Main.qml` reset to a "Qt Design" stub** — a Qt visual designer overwrote it. It replaces
  the whole file, so hand-edit the QML directly and don't open `Main.qml` in the designer.
- **First run is slow / "JIT-compiled" warning** — AdaptiveCpp's `generic` target compiles
  kernels on first use; subsequent runs are cached and faster.
- **"Rebuild & Apply" fails to compile** — that's just your C++; the errors are in the Code
  Lab's build log and the running engine is untouched. Fix and rebuild.
- **Saved weights don't load** — they're rejected when the architecture changed (the
  parameter count no longer matches). Retrain and save again.
- **Training feels slower than you remember** — the engine now runs out-of-process by
  default (~19% fewer epochs/sec). `SYNAPSE_ENGINE_INPROCESS=1` gets it back.
