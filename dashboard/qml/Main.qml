pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SynapseDashboard

// The shell: the network on the left, the guided workflow on the right, and a coach line
// along the bottom that always says what to do next. All the panel content lives in
// WorkflowPanel.qml — this file just wires the pieces together.
ApplicationWindow {
    id: win
    visible: true
    width: 1320
    height: 820
    title: appTitle
    color: Theme.bg

    // Simple by default; Advanced opens every Disclosure and reveals the model editor.
    property bool advanced: false

    // Which dataset example the "watch it learn" animation uses.
    property int exampleIndex: 0

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
    // (SYNAPSE_DATAMANAGER / SYNAPSE_CODELAB + SYNAPSE_GRAB capture those windows instead.)
    Timer {
        running: (typeof grabPath !== 'undefined' && grabPath && grabPath.length > 0)
                 && !(typeof autoOpenDataManager !== 'undefined' && autoOpenDataManager)
                 && !(typeof autoOpenCodeLab !== 'undefined' && autoOpenCodeLab)
        interval: (typeof grabDelay !== 'undefined' && grabDelay > 0) ? grabDelay : 1200
        onTriggered: {
            if (!rootCol.grabToImage(function(r) { r.saveToFile(grabPath); Qt.quit() }))
                Qt.quit()
        }
    }

    Component.onCompleted: {
        if (autoOpenDataManager) dataManager.show()
        if (typeof autoOpenCodeLab !== 'undefined' && autoOpenCodeLab) codeLab.show()
        if (typeof startAdvanced !== 'undefined' && startAdvanced) win.advanced = true
        if (typeof selectNeuron !== 'undefined' && selectNeuron && selectNeuron.length > 0) {
            var parts = selectNeuron.split(",")
            netView.selCol = parseInt(parts[0])
            netView.selNeuron = parseInt(parts[1])
        }
    }

    ColumnLayout {
        id: rootCol
        anchors.fill: parent
        spacing: 0

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // ── left: header + network ───────────────────────────────────────
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 60
                    color: Theme.surface

                    Rectangle {
                        anchors.bottom: parent.bottom
                        width: parent.width; height: 1
                        color: Theme.borderSoft
                    }

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: Theme.lg
                        anchors.rightMargin: Theme.lg
                        spacing: Theme.md

                        Image {
                            source: "qrc:/branding/mark-color-1024.png"
                            sourceSize.width: 32; sourceSize.height: 32
                            Layout.preferredWidth: 32; Layout.preferredHeight: 32
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                        }
                        Label {
                            text: qsTr("Synapse-AI")
                            color: Theme.text
                            font.pixelSize: Theme.fLarge
                            font.bold: true
                        }
                        Rectangle { implicitWidth: 1; implicitHeight: 22; color: Theme.border }

                        // Blueprint picker
                        Label {
                            id: bpLabel
                            text: bridge.modelName + "  ▾"
                            color: bpHover.hovered ? Qt.lighter(Theme.accent, 1.2) : Theme.accent
                            font.pixelSize: Theme.fTitle
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

                        Item { Layout.fillWidth: true }

                        Label {
                            text: bridge.deviceName
                            color: Theme.textFaint
                            font.pixelSize: Theme.fSmall
                        }
                        Rectangle { implicitWidth: 1; implicitHeight: 22; color: Theme.border }
                        Label {
                            text: qsTr("Coaching")
                            color: bridge.coachingEnabled ? Theme.text : Theme.textMuted
                            font.pixelSize: Theme.fBody
                        }
                        Switch {
                            checked: bridge.coachingEnabled
                            onToggled: bridge.coachingEnabled = checked
                        }
                        Rectangle { implicitWidth: 1; implicitHeight: 22; color: Theme.border }
                        Label {
                            text: qsTr("Advanced")
                            color: win.advanced ? Theme.text : Theme.textMuted
                            font.pixelSize: Theme.fBody
                        }
                        Switch { checked: win.advanced; onToggled: win.advanced = checked }
                    }
                }

                NetworkView {
                    id: netView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                // Narration while stepping; a plain explainer of the picture otherwise.
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 40
                    visible: win.isStepping() || (bridge.coachingEnabled && !win.advanced)
                    color: win.isStepping() ? Theme.card : Theme.surface
                    Label {
                        anchors.fill: parent
                        anchors.margins: Theme.sm
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        wrapMode: Text.WordWrap
                        elide: Text.ElideRight
                        text: win.isStepping()
                              ? win.stepNarration()
                              : qsTr("Each circle is a neuron — its colour is the number it's holding. "
                                   + "The lines between them are weights. Click any neuron to see how it got its value.")
                        color: win.isStepping() ? Theme.text : Theme.textMuted
                        font.pixelSize: Theme.fBody
                    }
                }
            }

            // ── right: the guided workflow ───────────────────────────────────
            Rectangle {
                Layout.preferredWidth: 380
                Layout.fillHeight: true
                color: Theme.surface

                Rectangle { width: 1; height: parent.height; color: Theme.borderSoft }

                WorkflowPanel {
                    anchors.fill: parent
                    anchors.margins: Theme.md
                    advanced: win.advanced
                    activeStep: coach.suggestedStep
                    exampleIndex: win.exampleIndex
                    onExampleIndexChanged: win.exampleIndex = exampleIndex
                    onManageDataRequested: dataManager.show()
                    onCodeLabRequested: codeLab.show()
                    onSaveBlueprintRequested: { bpNameField.text = bridge.modelName + " copy"; saveAsDialog.open() }
                    onRestoreDefaultRequested: restoreDialog.open()
                }
            }
        }

        CoachLine { id: coach; Layout.fillWidth: true; visible: bridge.coachingEnabled }
    }

    // Dedicated tool windows, opened on demand.
    DataManager { id: dataManager; visible: false }
    CodeLab { id: codeLab; visible: false }

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
            spacing: Theme.sm
            Label { text: qsTr("Name for your blueprint:"); color: Theme.text }
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
            color: Theme.text
            text: qsTr("Reset “") + bridge.modelName
                  + qsTr("” to its original architecture and data? Your changes to this built-in "
                         + "will be discarded — use “Save as blueprint” first if you want to keep them.")
        }
    }

    // ── transient error banner ──────────────────────────────────────────────
    Rectangle {
        id: errorBar
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 32
        color: Theme.danger
        visible: false
        property alias text: errorText.text
        function flash(message) { errorText.text = message; errorBar.visible = true; errorTimer.restart() }
        Label {
            id: errorText
            anchors.centerIn: parent
            color: "white"
            font.pixelSize: Theme.fLabel
        }
        Timer { id: errorTimer; interval: 4000; onTriggered: errorBar.visible = false }
    }
}
