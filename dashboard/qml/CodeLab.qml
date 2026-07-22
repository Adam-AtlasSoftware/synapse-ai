pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Tier-2 "Code Lab": edit real engine C++ — a custom activation function — then
// Rebuild & Apply. The app recompiles the engine and relaunches with your new math
// available in every layer's activation dropdown. This is the "edit code in the GUI,
// recompile, run" end-state: new *math* is genuine C++, not config.
ApplicationWindow {
    id: root
    width: 1000
    height: 720
    minimumWidth: 760
    minimumHeight: 520
    title: qsTr("Code Lab — Custom C++ activations")
    color: "#0d1117"

    // Worked, ready-to-compile examples. Every one of these is gradient-checked by
    // tests/custom_activation_smoke.cpp, so the math you copy here is verified.
    readonly property var templates: [
    {
        name: "leaky_relu",
        blurb: qsTr("ReLU with a small slope for negatives, so units never fully \"die\". The simplest place to start."),
        code: `  // "leaky_relu" — ReLU with a small slope for negatives, so units never fully die.
  register_activation({
      "leaky_relu", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) act[i] = pre[i] > 0.0f ? pre[i] : 0.01f * pre[i];
      },
      [](float pre, float post) { return pre > 0.0f ? 1.0f : 0.01f; }});
`
    },
    {
        name: "swish",
        blurb: qsTr("x · sigmoid(x) — smooth and self-gating; often trains a little better than ReLU."),
        code: `  // "swish" (a.k.a. SiLU) = x * sigmoid(x): a smooth, self-gated activation.
  register_activation({
      "swish", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) {
          float s = 1.0f / (1.0f + std::exp(-pre[i]));
          act[i] = pre[i] * s;                    // x · sigmoid(x)
        }
      },
      [](float pre, float post) {
        float s = 1.0f / (1.0f + std::exp(-pre));
        return s + pre * s * (1.0f - s);          // sigmoid(x) + x·sigmoid'(x)
      }});
`
    },
    {
        name: "elu",
        blurb: qsTr("Linear above zero, saturating exponential below — keeps mean activations near zero. Shows how to reuse `post` in the derivative."),
        code: `  // "elu" — linear above zero, saturating exponential below.
  register_activation({
      "elu", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) act[i] = pre[i] > 0.0f ? pre[i] : (std::exp(pre[i]) - 1.0f);
      },
      // For x <= 0 the output is e^x - 1, so the slope is e^x = post + 1.
      [](float pre, float post) { return pre > 0.0f ? 1.0f : post + 1.0f; }});
`
    },
    {
        name: "softplus",
        blurb: qsTr("ln(1 + eˣ), a smooth ReLU. Its derivative is exactly sigmoid(x) — an elegant one to study."),
        code: `  // "softplus" — ln(1 + e^x), a smooth ReLU. Its derivative is exactly sigmoid(x).
  register_activation({
      "softplus", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i)
          act[i] = pre[i] > 20.0f ? pre[i] : std::log1p(std::exp(pre[i]));  // guard overflow
      },
      [](float pre, float post) { return 1.0f / (1.0f + std::exp(-pre)); }});
`
    },
    {
        name: "gelu",
        blurb: qsTr("The activation used in most transformers (tanh approximation). A good example of a longer derivative."),
        code: `  // "gelu" (tanh approximation) — the activation used in most transformers.
  register_activation({
      "gelu", false,
      [](const float* pre, float* act, int n) {
        const float c = 0.7978845608f;            // sqrt(2/pi)
        for (int i = 0; i < n; ++i) {
          float x = pre[i];
          float u = c * (x + 0.044715f * x * x * x);
          act[i] = 0.5f * x * (1.0f + std::tanh(u));
        }
      },
      [](float pre, float post) {
        const float c = 0.7978845608f;
        float x = pre;
        float u = c * (x + 0.044715f * x * x * x);
        float t = std::tanh(u);
        float du = c * (1.0f + 0.134145f * x * x);  // d/dx of the inner term
        return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
      }});
`
    },
    {
        name: "sine",
        blurb: qsTr("A periodic activation (SIREN). Surprising and fun — good at fitting smooth, repeating signals."),
        code: `  // "sine" — a periodic activation (SIREN); good at smooth, repeating signals.
  register_activation({
      "sine", false,
      [](const float* pre, float* act, int n) {
        for (int i = 0; i < n; ++i) act[i] = std::sin(pre[i]);
      },
      [](float pre, float post) { return std::cos(pre); }});
`
    }]

    onVisibleChanged: if (visible) editor.text = bridge.customActivationSource()

    // Dev/demo hook: SYNAPSE_CODELAB + SYNAPSE_GRAB renders this window to a PNG.
    Timer {
        running: root.visible && (typeof grabPath !== 'undefined' && grabPath && grabPath.length > 0)
        interval: 1400
        onTriggered: {
            if (!bodyCol.grabToImage(function(r) { r.saveToFile(grabPath); Qt.quit() }))
                Qt.quit()
        }
    }

    ColumnLayout {
        id: bodyCol
        anchors.fill: parent
        anchors.margins: 14
        spacing: 10

        Label {
            Layout.fillWidth: true
            text: qsTr("Write a new activation function in C++, then Rebuild & Apply. It compiles into "
                     + "the engine and appears in every layer's activation dropdown. Each activation is a "
                     + "forward(pre, act, n) and its derivative(pre, post).")
            color: "#8b949e"; font.pixelSize: 12; wrapMode: Text.WordWrap
        }

        Label {
            Layout.fillWidth: true
            text: bridge.engineKind === "subprocess"
                  ? qsTr("⚡ Engine: separate process — Rebuild swaps in the new engine live; this window and your session stay open.")
                  : qsTr("Engine: linked in-process — Rebuild relinks and relaunches the app. Unset SYNAPSE_ENGINE_INPROCESS to hot-swap instead.")
            color: bridge.engineKind === "subprocess" ? "#3fb950" : "#8b949e"
            font.pixelSize: 11; wrapMode: Text.WordWrap
        }

        // ── source editor ────────────────────────────────────────────────────
        Label { text: qsTr("engine/src/custom_activations.cpp"); color: "#58a6ff"; font.pixelSize: 12; font.bold: true }
        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: 320
            background: Rectangle { color: "#0b0f14"; border.color: "#30363d"; radius: 6 }
            padding: 2
            ScrollView {
                anchors.fill: parent
                clip: true
                TextArea {
                    id: editor
                    wrapMode: TextEdit.NoWrap
                    font.family: "monospace"
                    font.pixelSize: 12
                    color: "#e6edf3"
                    selectByMouse: true
                    tabStopDistance: 28
                    background: null
                }
            }
        }

        // ── worked examples ──────────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Label { text: qsTr("Examples:"); color: "#8b949e"; font.pixelSize: 12 }
            ComboBox {
                id: templatePicker
                Layout.preferredWidth: 150
                model: root.templates.map(function (t) { return t.name })
            }
            Button {
                text: qsTr("＋ Insert at cursor")
                enabled: !bridge.building
                onClicked: {
                    editor.forceActiveFocus()
                    editor.insert(editor.cursorPosition, root.templates[templatePicker.currentIndex].code)
                }
            }
            Button {
                text: qsTr("Reload from disk")
                enabled: !bridge.building
                onClicked: editor.text = bridge.customActivationSource()
            }
            Item { Layout.fillWidth: true }
            BusyIndicator { running: bridge.building; visible: bridge.building; implicitWidth: 24; implicitHeight: 24 }
            Label {
                visible: bridge.building
                text: qsTr("Compiling…")
                color: "#f2a33c"; font.pixelSize: 12
            }
            Button {
                text: qsTr("⚙ Rebuild & Apply")
                enabled: !bridge.building
                highlighted: true
                onClicked: {
                    bridge.saveCustomActivationSource(editor.text)
                    bridge.rebuildEngine()
                }
            }
        }
        Label {
            Layout.fillWidth: true
            text: "↳ " + root.templates[templatePicker.currentIndex].blurb
                  + qsTr("  ·  Insert drops it at your cursor — put that inside register_custom_activations().")
            color: "#8b949e"; font.pixelSize: 11; wrapMode: Text.WordWrap
        }

        // ── build output console ─────────────────────────────────────────────
        Label { text: qsTr("Build output"); color: "#8b949e"; font.pixelSize: 11 }
        Frame {
            Layout.fillWidth: true
            Layout.preferredHeight: 190
            background: Rectangle { color: "#0b0f14"; border.color: "#30363d"; radius: 6 }
            padding: 2
            ScrollView {
                id: consoleScroll
                anchors.fill: parent
                clip: true
                TextArea {
                    id: buildConsole
                    readOnly: true
                    wrapMode: TextEdit.NoWrap
                    font.family: "monospace"
                    font.pixelSize: 11
                    color: "#9db3c0"
                    background: null
                    text: bridge.buildOutput
                    // Keep the newest output in view as it streams in.
                    onTextChanged: cursorPosition = length
                }
            }
        }
    }
}
