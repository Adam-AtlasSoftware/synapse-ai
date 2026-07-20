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

    // A ready-to-build example: the same file with the "swish" activation enabled.
    readonly property string swishExample: `#include <cmath>

#include "activation_registry.hpp"

// Custom activation functions — real C++ the engine compiles. Edit, then Rebuild.
namespace synapse {

void register_custom_activations() {
  // "swish" (a.k.a. SiLU) = x * sigmoid(x): a smooth, self-gated activation.
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
}

}  // namespace synapse
`

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

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Button {
                text: qsTr("Insert swish example")
                enabled: !bridge.building
                onClicked: editor.text = root.swishExample
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
