pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: win
    visible: true
    width: 1240
    height: 760
    title: appTitle
    color: "#0d1117"

    readonly property var activationOptions: ["linear", "sigmoid", "relu", "tanh", "softmax"]

    // Beginner view keeps it simple; advanced view reveals per-layer stats/histograms.
    property bool advanced: false

    // Which dataset example the Data panel is browsing.
    property int exampleIndex: 0

    // Human-readable description of the current forward/backward step.
    function phaseText() {
        var p = bridge.phase
        var ly = bridge.layers[bridge.activeLayer]
        var name = ly ? ly.name : "layer"
        if (p === "begin")   return "Input loaded — press Step or Play to propagate"
        if (p === "step")    return "Forward · computing " + name + " …"
        if (p === "done")    return "Forward · " + name + " — output ready ✓"
        if (p === "loss")    return "Loss measured at the output"
        if (p === "backward") return "Backward · gradient at " + name
        if (p === "update")  return "Weights updated by gradient descent ✓"
        if (p === "forward") return "Forward pass complete (instant)"
        return "Ready"
    }

    // Are we mid-way through a stepped forward/backward pass?
    function isStepping() {
        var p = bridge.phase
        return p === "begin" || p === "step" || p === "loss" || p === "backward"
               || p === "update" || p === "done"
    }
    // Plain-language narration of the current step — the guided tutorial voice.
    function stepNarration() {
        var p = bridge.phase
        var ly = bridge.layers[bridge.activeLayer]
        var name = ly ? ly.name : "this layer"
        var act = ly ? ly.activation : ""
        if (p === "begin")    return qsTr("Input loaded. Press Step to push it through the first layer.")
        if (p === "step")     return qsTr("Forward · ") + name + qsTr(": each neuron multiplies its inputs by its weights, sums them (plus a bias), then applies ") + act + "."
        if (p === "loss")     return qsTr("Loss: the output is compared to the correct answer — the gap is the error to reduce.")
        if (p === "backward") return qsTr("Backward · ") + name + qsTr(": working out how much each neuron here contributed to the error (its gradient).")
        if (p === "update")   return qsTr("Update: every weight nudges a little against its gradient — the network just learned a bit.")
        if (p === "done")     return qsTr("Forward pass complete — that's the network's prediction.")
        return ""
    }

    Connections {
        target: bridge
        function onDatasetChanged() { win.exampleIndex = 0 }
        function onErrorOccurred(message) { errorBar.flash(message) }
    }

    // Dev/demo hook: SYNAPSE_GRAB=<path> renders the main window to a PNG and quits.
    // (SYNAPSE_DATAMANAGER + SYNAPSE_GRAB captures the data-manager window instead.)
    Timer {
        running: (typeof grabPath !== 'undefined' && grabPath && grabPath.length > 0)
                 && !(typeof autoOpenDataManager !== 'undefined' && autoOpenDataManager)
        interval: 1200
        onTriggered: {
            if (!rootRow.grabToImage(function(r) { r.saveToFile(grabPath); Qt.quit() }))
                Qt.quit()
        }
    }

    Component.onCompleted: {
        if (autoOpenDataManager) dataManager.show()
        if (typeof startAdvanced !== 'undefined' && startAdvanced) win.advanced = true
        if (typeof selectNeuron !== 'undefined' && selectNeuron && selectNeuron.length > 0) {
            var parts = selectNeuron.split(",")
            netView.selCol = parseInt(parts[0])
            netView.selNeuron = parseInt(parts[1])
        }
    }

    RowLayout {
        id: rootRow
        anchors.fill: parent
        spacing: 0

        // ── left: header + network view ──────────────────────────────────────
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 56
                color: "#161b22"
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 18
                    anchors.rightMargin: 18
                    spacing: 12
                    Label { text: qsTr("Synapse-AI"); color: "#e6edf3"; font.pixelSize: 20; font.bold: true }
                    Rectangle { implicitWidth: 1; implicitHeight: 24; color: "#30363d" }
                    Label {
                        id: bpLabel
                        text: bridge.modelName + "  ▾"
                        color: bpHover.hovered ? "#79b8ff" : "#58a6ff"
                        font.pixelSize: 15
                        TapHandler { onTapped: bpMenu.popup() }
                        HoverHandler { id: bpHover; cursorShape: Qt.PointingHandCursor }
                        Menu {
                            id: bpMenu
                            Instantiator {
                                model: bridge.blueprints
                                delegate: MenuItem {
                                    required property var modelData
                                    text: modelData.name
                                    onTriggered: bridge.loadBlueprint(modelData.path)
                                }
                                onObjectAdded: (index, object) => bpMenu.insertItem(index, object)
                                onObjectRemoved: (index, object) => bpMenu.removeItem(object)
                            }
                        }
                    }
                    // Predicted class (argmax of the output column).
                    Rectangle {
                        visible: bridge.outputKind === "class" && bridge.predictedLabel !== ""
                        implicitWidth: predRow.implicitWidth + 18
                        implicitHeight: 26
                        radius: 13
                        color: "#12261a"
                        border.color: "#238636"
                        border.width: 1
                        RowLayout {
                            id: predRow
                            anchors.centerIn: parent
                            spacing: 6
                            Label { text: qsTr("prediction:"); color: "#8b949e"; font.pixelSize: 12 }
                            Label { text: bridge.predictedLabel; color: "#3fb950"; font.pixelSize: 14; font.bold: true }
                            Label {
                                text: qsTr("(untrained)")
                                visible: bridge.epoch === 0
                                color: "#8b949e"; font.pixelSize: 10
                            }
                        }
                    }
                    Item { Layout.fillWidth: true }
                    Label { text: qsTr("device:"); color: "#8b949e"; font.pixelSize: 12 }
                    Label { text: bridge.deviceName; color: "#3fb950"; font.pixelSize: 12 }
                    Rectangle { implicitWidth: 1; implicitHeight: 24; color: "#30363d" }
                    Label { text: qsTr("Advanced"); color: "#8b949e"; font.pixelSize: 12 }
                    Switch { checked: win.advanced; onToggled: win.advanced = checked }
                }
            }

            NetworkView {
                id: netView
                Layout.fillWidth: true
                Layout.fillHeight: true
            }

            // Guided narration while stepping; a beginner explainer otherwise.
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 42
                visible: win.isStepping() || !win.advanced
                color: win.isStepping() ? "#1b2230" : "#161b22"
                Label {
                    anchors.fill: parent
                    anchors.margins: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    wrapMode: Text.WordWrap
                    elide: Text.ElideRight
                    text: win.isStepping()
                          ? win.stepNarration()
                          : qsTr("Each circle is a neuron (its color = its current value); the lines are weights. "
                                 + "Set the input and press Run — or Train to teach it. Flip “Advanced” for the numbers.")
                    color: win.isStepping() ? "#e6edf3" : "#8b949e"
                    font.pixelSize: 12
                }
            }
        }

        // ── right: controls ──────────────────────────────────────────────────
        Rectangle {
            Layout.preferredWidth: 340
            Layout.fillHeight: true
            color: "#12161c"

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                TabBar {
                    id: actionsTab
                    Layout.fillWidth: true
                    TabButton { text: qsTr("Run") }
                    TabButton { text: qsTr("Train") }
                    TabButton { text: qsTr("Model") }
                    TabButton { text: qsTr("Info") }
                }

                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: actionsTab.currentIndex

                    // ══ Run tab: inputs + playback ══════════════════════════
                    ScrollView {
                        id: sv0
                        contentWidth: availableWidth
                        clip: true

                        ColumnLayout {
                            width: sv0.availableWidth
                            spacing: 16

                    // ── Inputs ──────────────────────────────────────────────
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        title: qsTr("Inputs")

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            // GRID blueprints (OCR / shapes): paint the pixels.
                            PixelGrid {
                                visible: bridge.inputLayout === "grid"
                                Layout.alignment: Qt.AlignHCenter
                                rows: bridge.inputRows
                                cols: bridge.inputCols
                                values: bridge.input
                                onPainted: (index, value) => bridge.setInput(index, value)
                            }
                            Label {
                                visible: bridge.inputLayout === "grid"
                                Layout.fillWidth: true
                                text: qsTr("Left-drag to draw · right-drag to erase")
                                color: "#8b949e"; font.pixelSize: 11
                                horizontalAlignment: Text.AlignHCenter
                            }

                            // LABELED blueprints (XOR / adder / segments): named sliders.
                            Repeater {
                                model: bridge.inputLayout === "labels" ? bridge.inputDim : 0
                                delegate: RowLayout {
                                    id: inputRow
                                    required property int index
                                    Layout.fillWidth: true
                                    spacing: 8
                                    Label {
                                        text: (bridge.inputLabels[inputRow.index] !== undefined)
                                              ? bridge.inputLabels[inputRow.index] : ("x" + inputRow.index)
                                        color: "#8b949e"
                                        Layout.preferredWidth: 34
                                        elide: Text.ElideRight
                                    }
                                    Slider {
                                        Layout.fillWidth: true
                                        from: 0; to: 1; stepSize: 0.05
                                        value: (bridge.input[inputRow.index] !== undefined)
                                               ? bridge.input[inputRow.index] : 0
                                        onMoved: bridge.setInput(inputRow.index, value)
                                    }
                                    Label {
                                        text: ((bridge.input[inputRow.index] !== undefined)
                                               ? bridge.input[inputRow.index] : 0).toFixed(2)
                                        color: "#e6edf3"
                                        Layout.preferredWidth: 40
                                        horizontalAlignment: Text.AlignRight
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Button {
                                    text: qsTr("Run (instant)")
                                    Layout.fillWidth: true
                                    onClicked: bridge.forwardCurrent()
                                }
                                Button {
                                    text: qsTr("Clear")
                                    onClicked: bridge.clearInput()
                                }
                                Button {
                                    text: qsTr("Rand")
                                    onClicked: bridge.randomizeInput()
                                }
                            }
                        }
                    }

                    // ── Playback: watch the forward pass move one layer at a time ──
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        title: qsTr("Playback")

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            Label {
                                Layout.fillWidth: true
                                text: win.phaseText()
                                color: "#e6edf3"
                                font.pixelSize: 13
                                wrapMode: Text.WordWrap
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 6
                                Label { text: qsTr("Slow"); color: "#8b949e"; font.pixelSize: 11 }
                                Slider {
                                    id: speedSlider
                                    Layout.fillWidth: true
                                    from: 0; to: 1
                                    // Right = faster. Map to 1500ms (slow) … 60ms (fast) per layer.
                                    value: (1500 - bridge.speedMs) / (1500 - 60)
                                    onMoved: bridge.speedMs = Math.round(1500 - value * (1500 - 60))
                                }
                                Label { text: qsTr("Fast"); color: "#8b949e"; font.pixelSize: 11 }
                            }
                            Label {
                                Layout.fillWidth: true
                                text: bridge.speedMs + qsTr(" ms / layer")
                                color: "#8b949e"; font.pixelSize: 11
                                horizontalAlignment: Text.AlignRight
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Button {
                                    text: bridge.playing ? qsTr("⏸ Pause") : qsTr("▶ Play")
                                    Layout.fillWidth: true
                                    onClicked: bridge.playing ? bridge.pause()
                                                              : bridge.playCurrent()
                                }
                                Button {
                                    text: qsTr("⧐ Step")
                                    onClicked: bridge.stepOnce()
                                }
                            }
                        }
                    }

                            Item { Layout.fillHeight: true }
                        }
                    }

                    // ══ Train tab: training + data ══════════════════════════
                    ScrollView {
                        id: sv1
                        contentWidth: availableWidth
                        clip: true

                        ColumnLayout {
                            width: sv1.availableWidth
                            spacing: 16

                    // ── Training: backprop + SGD, watch the loss fall ──────────
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        title: qsTr("Training")

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            Label {
                                visible: !bridge.hasDataset
                                Layout.fillWidth: true
                                text: qsTr("This blueprint has no training data.")
                                color: "#8b949e"; font.pixelSize: 12; wrapMode: Text.WordWrap
                            }

                            ColumnLayout {
                                visible: bridge.hasDataset
                                Layout.fillWidth: true
                                spacing: 8

                                Label {
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    text: bridge.training
                                          ? qsTr("● Training — the model's weights are updating live")
                                          : (bridge.trained
                                             ? (qsTr("✓ Trained — ") + bridge.epoch + qsTr(" epochs applied to the live model"))
                                             : qsTr("Untrained — press ▶ Train to teach the network"))
                                    color: bridge.training ? "#f2a33c" : (bridge.trained ? "#3fb950" : "#8b949e")
                                    font.pixelSize: 13
                                    font.bold: true
                                }
                                Label {
                                    Layout.fillWidth: true
                                    text: qsTr("epoch ") + bridge.epoch
                                          + qsTr("   ·   loss ") + bridge.currentLoss.toFixed(4)
                                    color: "#8b949e"; font.pixelSize: 12
                                }

                                LossChart { Layout.fillWidth: true }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 6
                                    Label { text: qsTr("learn rate"); color: "#8b949e"; font.pixelSize: 11 }
                                    Slider {
                                        Layout.fillWidth: true
                                        from: 0.01; to: 1.0; stepSize: 0.01
                                        value: bridge.learningRate
                                        onMoved: bridge.learningRate = value
                                    }
                                    Label {
                                        text: bridge.learningRate.toFixed(2)
                                        color: "#e6edf3"; Layout.preferredWidth: 34
                                        horizontalAlignment: Text.AlignRight
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 6
                                    Label { text: qsTr("compute"); color: "#8b949e"; font.pixelSize: 11 }
                                    Item { Layout.fillWidth: true }
                                    Label {
                                        text: bridge.gpuTraining ? qsTr("GPU (full-batch)") : qsTr("CPU (SGD)")
                                        color: "#e6edf3"; font.pixelSize: 11
                                    }
                                    Switch {
                                        checked: bridge.gpuTraining
                                        onToggled: bridge.gpuTraining = checked
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8
                                    Button {
                                        text: bridge.training ? qsTr("⏸ Pause") : qsTr("▶ Train")
                                        Layout.fillWidth: true
                                        onClicked: bridge.training ? bridge.trainStop() : bridge.trainStart()
                                    }
                                    Button { text: qsTr("+1 epoch"); onClicked: bridge.trainEpoch() }
                                    Button { text: qsTr("Reset"); onClicked: bridge.resetWeights() }
                                }
                            }
                        }
                    }

                    // ── Data: browse the training examples, or add your own ────
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        visible: bridge.hasDataset || bridge.outputKind === "class"
                        title: qsTr("Data (") + bridge.datasetSize + qsTr(" examples)")

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            Label {
                                Layout.fillWidth: true
                                text: qsTr("What the network learns from. Browse the examples, or teach it your own.")
                                color: "#8b949e"; font.pixelSize: 11; wrapMode: Text.WordWrap
                            }
                            Button {
                                Layout.fillWidth: true
                                text: qsTr("⤢ Manage training data…")
                                onClicked: dataManager.show()
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                visible: bridge.datasetSize > 0
                                spacing: 6
                                Button {
                                    text: "◀"; implicitWidth: 34
                                    onClicked: win.exampleIndex = (win.exampleIndex + bridge.datasetSize - 1) % bridge.datasetSize
                                }
                                Label {
                                    Layout.fillWidth: true
                                    horizontalAlignment: Text.AlignHCenter
                                    text: (win.exampleIndex + 1) + " / " + bridge.datasetSize
                                          + "   →   " + bridge.exampleLabel(win.exampleIndex)
                                    color: "#e6edf3"; font.pixelSize: 12
                                }
                                Button {
                                    text: "▶"; implicitWidth: 34
                                    onClicked: win.exampleIndex = (win.exampleIndex + 1) % bridge.datasetSize
                                }
                            }
                            Button {
                                Layout.fillWidth: true
                                visible: bridge.datasetSize > 0
                                text: qsTr("Show this example in the network")
                                onClicked: bridge.loadExample(win.exampleIndex)
                            }

                            // Gradient-flow: one SGD step in slow motion on this example.
                            RowLayout {
                                Layout.fillWidth: true
                                visible: bridge.datasetSize > 0
                                spacing: 8
                                Button {
                                    Layout.fillWidth: true
                                    text: bridge.playing ? qsTr("⏸ Pause")
                                                         : qsTr("▶ Watch it learn (slow)")
                                    onClicked: bridge.playing ? bridge.pause()
                                                              : bridge.playLearnExample(win.exampleIndex)
                                }
                                Button {
                                    text: qsTr("⧐ Step")
                                    onClicked: bridge.stepLearnExample(win.exampleIndex)
                                }
                            }
                            Label {
                                Layout.fillWidth: true
                                visible: bridge.datasetSize > 0 && win.phaseText() !== "Ready"
                                text: win.phaseText()
                                color: "#d267ff"; font.pixelSize: 12; wrapMode: Text.WordWrap
                            }

                            // Manual training — classifiers only.
                            ColumnLayout {
                                visible: bridge.outputKind === "class"
                                Layout.fillWidth: true
                                spacing: 6
                                Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: "#30363d" }
                                Label {
                                    Layout.fillWidth: true
                                    text: qsTr("Teach your own: draw/set the input, pick the right answer, add it.")
                                    color: "#8b949e"; font.pixelSize: 11; wrapMode: Text.WordWrap
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 6
                                    Label { text: qsTr("answer:"); color: "#8b949e"; font.pixelSize: 12 }
                                    ComboBox {
                                        id: labelCombo
                                        Layout.fillWidth: true
                                        model: bridge.outputLabels
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8
                                    Button {
                                        Layout.fillWidth: true
                                        text: qsTr("+ Add current input")
                                        onClicked: bridge.addExample(bridge.input, labelCombo.currentIndex)
                                    }
                                    Button {
                                        text: qsTr("Save")
                                        onClicked: bridge.saveDataset()
                                    }
                                }
                            }
                        }
                    }

                            Item { Layout.fillHeight: true }
                        }
                    }

                    // ══ Model tab ═══════════════════════════════════════════
                    ScrollView {
                        id: sv2
                        contentWidth: availableWidth
                        clip: true

                        ColumnLayout {
                            width: sv2.availableWidth
                            spacing: 16

                    // ── Model editor (Tier-1: edits rebuild the net, no recompile) ──
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        title: qsTr("Model")

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: qsTr("Input dim"); color: "#8b949e"; Layout.fillWidth: true }
                                SpinBox {
                                    from: 1; to: 64
                                    value: bridge.inputDim
                                    onValueModified: bridge.setInputDim(value)
                                }
                            }

                            Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: "#30363d" }

                            Repeater {
                                model: bridge.layers
                                delegate: Frame {
                                    id: layerRow
                                    required property var modelData
                                    Layout.fillWidth: true

                                    RowLayout {
                                        anchors.fill: parent
                                        spacing: 6
                                        Label {
                                            text: layerRow.modelData.name
                                            color: "#58a6ff"
                                            Layout.preferredWidth: 28
                                        }
                                        SpinBox {
                                            from: 1; to: 128
                                            value: layerRow.modelData.units
                                            Layout.preferredWidth: 96
                                            onValueModified: bridge.setLayerUnits(layerRow.modelData.index, value)
                                        }
                                        ComboBox {
                                            Layout.fillWidth: true
                                            model: win.activationOptions
                                            currentIndex: win.activationOptions.indexOf(layerRow.modelData.activation)
                                            onActivated: bridge.setLayerActivation(layerRow.modelData.index, currentText)
                                        }
                                        Button {
                                            text: "✕"
                                            implicitWidth: 30
                                            onClicked: bridge.removeLayer(layerRow.modelData.index)
                                        }
                                    }
                                }
                            }

                            Button {
                                text: qsTr("+ Add layer")
                                Layout.fillWidth: true
                                onClicked: bridge.addLayer(4, "tanh")
                            }

                            Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: "#30363d" }

                            Button {
                                Layout.fillWidth: true
                                text: qsTr("💾 Save as new blueprint…")
                                onClicked: { bpNameField.text = bridge.modelName + " copy"; saveAsDialog.open() }
                            }
                            Button {
                                Layout.fillWidth: true
                                visible: bridge.isBuiltIn
                                text: qsTr("↺ Restore default")
                                onClicked: restoreDialog.open()
                            }
                        }
                    }

                            Item { Layout.fillHeight: true }
                        }
                    }

                    // ══ Info tab: legend + layer stats ══════════════════════
                    ScrollView {
                        id: sv3
                        contentWidth: availableWidth
                        clip: true

                        ColumnLayout {
                            width: sv3.availableWidth
                            spacing: 16

                    // ── Legend ──────────────────────────────────────────────
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        title: qsTr("Legend")
                        ColumnLayout {
                            spacing: 6
                            RowLayout {
                                spacing: 8
                                Rectangle { implicitWidth: 16; implicitHeight: 16; radius: 8; color: "#4c8dff" }
                                Label { text: qsTr("negative activation / weight"); color: "#8b949e"; font.pixelSize: 12 }
                            }
                            RowLayout {
                                spacing: 8
                                Rectangle { implicitWidth: 16; implicitHeight: 16; radius: 8; color: "#2b2f37" }
                                Label { text: qsTr("near zero"); color: "#8b949e"; font.pixelSize: 12 }
                            }
                            RowLayout {
                                spacing: 8
                                Rectangle { implicitWidth: 16; implicitHeight: 16; radius: 8; color: "#f2a33c" }
                                Label { text: qsTr("positive activation / weight"); color: "#8b949e"; font.pixelSize: 12 }
                            }
                        }
                    }

                    // ── Advanced: per-layer stats + weight histograms ──────────
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        visible: win.advanced
                        title: qsTr("Layer stats")
                        StatsPanel {
                            anchors.left: parent.left
                            anchors.right: parent.right
                        }
                    }

                            Item { Layout.fillHeight: true }
                        }
                    }
                }
            }
        }
    }

    // Dedicated training-data manager (a separate window, opened on demand).
    DataManager { id: dataManager; visible: false }

    // Save the current model as a new blueprint.
    Dialog {
        id: saveAsDialog
        title: qsTr("Save as new blueprint")
        modal: true
        anchors.centerIn: Overlay.overlay
        width: 360
        standardButtons: Dialog.Save | Dialog.Cancel
        onAccepted: bridge.saveBlueprintAs(bpNameField.text)
        ColumnLayout {
            anchors.fill: parent
            spacing: 8
            Label { text: qsTr("Name for your blueprint:"); color: "#e6edf3" }
            TextField {
                id: bpNameField
                Layout.fillWidth: true
                placeholderText: qsTr("My model")
                onAccepted: saveAsDialog.accept()
            }
        }
    }

    // Confirm reverting a built-in blueprint to its shipped default.
    Dialog {
        id: restoreDialog
        title: qsTr("Restore default?")
        modal: true
        anchors.centerIn: Overlay.overlay
        width: 380
        standardButtons: Dialog.Yes | Dialog.No
        onAccepted: bridge.restoreDefault()
        Label {
            width: 340
            wrapMode: Text.WordWrap
            color: "#e6edf3"
            text: qsTr("Reset “") + bridge.modelName
                  + qsTr("” to its original architecture and data? Your changes to this built-in "
                         + "will be discarded — use “Save as new blueprint” first if you want to keep them.")
        }
    }

    // ── transient error banner ──────────────────────────────────────────────
    Rectangle {
        id: errorBar
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 32
        color: "#a40e26"
        visible: false
        property alias text: errorText.text
        function flash(message) { errorText.text = message; errorBar.visible = true; errorTimer.restart() }
        Label {
            id: errorText
            anchors.centerIn: parent
            color: "white"
            font.pixelSize: 13
        }
        Timer { id: errorTimer; interval: 4000; onTriggered: errorBar.visible = false }
    }
}
