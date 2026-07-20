pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// A dedicated window for managing training data: a gallery of every example on the
// left, an editor on the right. Add, edit, and delete pairs; optimized for entering
// a lot of data fast — draw, press the label's number key, repeat.
ApplicationWindow {
    id: root
    width: 960
    height: 680
    minimumWidth: 720
    minimumHeight: 480
    title: qsTr("Training Data — ") + bridge.modelName
    color: "#0d1117"

    // ── editor state ─────────────────────────────────────────────────────────
    property int selectedIndex: -1     // -1 = composing a new example
    property var editInput: []
    property int editLabel: 0
    property var editTarget: []
    property int rev: 0                 // bumped on any dataset change to refresh thumbnails

    readonly property int gridRows: bridge.inputLayout === "grid" ? bridge.inputRows : 1
    readonly property int gridCols: bridge.inputLayout === "grid" ? bridge.inputCols : bridge.inputDim
    readonly property bool isClass: bridge.outputKind === "class"

    function zeros(n) { var a = []; for (var i = 0; i < n; ++i) a.push(0); return a }
    function oneHot(k, n) { var a = root.zeros(n); if (k >= 0 && k < n) a[k] = 1; return a }
    function initEdit() {
        root.editInput = root.zeros(bridge.inputDim)
        root.editTarget = root.zeros(bridge.outputDim)
        root.editLabel = 0
        root.selectedIndex = -1
    }
    function setEditInput(i, v) { var a = root.editInput.slice(); a[i] = v; root.editInput = a }
    function setEditTarget(i, v) { var a = root.editTarget.slice(); a[i] = v; root.editTarget = a }
    function buildTarget() { return root.isClass ? root.oneHot(root.editLabel, bridge.outputDim) : root.editTarget }
    function addCurrent() {
        bridge.addExampleRaw(root.editInput, root.buildTarget())
        root.editInput = root.zeros(bridge.inputDim)   // ready for the next one
        root.selectedIndex = -1
    }
    function updateCurrent() {
        if (root.selectedIndex >= 0)
            bridge.updateExample(root.selectedIndex, root.editInput, root.buildTarget())
    }
    function commit() { if (root.selectedIndex >= 0) root.updateCurrent(); else root.addCurrent() }
    function selectExample(i) {
        root.selectedIndex = i
        root.editInput = bridge.exampleInput(i)
        var t = bridge.exampleTarget(i)
        if (root.isClass) {
            var m = 0
            for (var k = 1; k < t.length; ++k) if (t[k] > t[m]) m = k
            root.editLabel = m
        } else {
            root.editTarget = t
        }
    }
    function deleteExample(i) {
        bridge.removeExample(i)
        if (root.selectedIndex === i) root.initEdit()
        else if (root.selectedIndex > i) root.selectedIndex -= 1
    }
    // Pressing a digit sets the label and commits — the fast path for entering data.
    function labelKey(d) {
        if (!root.isClass || d >= bridge.outputDim) return
        root.editLabel = d
        root.commit()
    }

    Component.onCompleted: {
        root.initEdit()
        // Dev/demo hook: SYNAPSE_GRAB=<path> renders this window to a PNG and quits.
        if (typeof grabPath !== 'undefined' && grabPath && grabPath.length > 0) {
            root.visible = true
            grabTimer.start()
        }
    }
    Timer {
        id: grabTimer
        interval: 1500
        onTriggered: {
            if (!body.grabToImage(function(result) { result.saveToFile(grabPath); Qt.quit(); }))
                Qt.quit()
        }
    }
    Connections {
        target: bridge
        function onTopologyChanged() { root.initEdit(); root.rev += 1 }
        function onDatasetChanged() { root.rev += 1 }
    }

    // number keys → label + commit; Enter → commit; Delete → remove selected
    Shortcut { sequences: ["0"]; onActivated: root.labelKey(0) }
    Shortcut { sequences: ["1"]; onActivated: root.labelKey(1) }
    Shortcut { sequences: ["2"]; onActivated: root.labelKey(2) }
    Shortcut { sequences: ["3"]; onActivated: root.labelKey(3) }
    Shortcut { sequences: ["4"]; onActivated: root.labelKey(4) }
    Shortcut { sequences: ["5"]; onActivated: root.labelKey(5) }
    Shortcut { sequences: ["6"]; onActivated: root.labelKey(6) }
    Shortcut { sequences: ["7"]; onActivated: root.labelKey(7) }
    Shortcut { sequences: ["8"]; onActivated: root.labelKey(8) }
    Shortcut { sequences: ["9"]; onActivated: root.labelKey(9) }
    Shortcut { sequences: ["Return", "Enter"]; onActivated: root.commit() }
    Shortcut { sequences: ["Delete"]; onActivated: if (root.selectedIndex >= 0) root.deleteExample(root.selectedIndex) }

    // ── header ────────────────────────────────────────────────────────────────
    header: ToolBar {
        background: Rectangle { color: "#161b22" }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 14
            anchors.rightMargin: 14
            Label {
                text: bridge.datasetSize + qsTr(" training examples")
                color: "#e6edf3"; font.pixelSize: 15; font.bold: true
            }
            Item { Layout.fillWidth: true }
            Button { text: qsTr("Save to file"); onClicked: bridge.saveDataset() }
        }
    }

    RowLayout {
        id: body
        anchors.fill: parent
        spacing: 0

        // ── gallery ────────────────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#0d1117"

            GridView {
                id: gallery
                anchors.fill: parent
                anchors.margins: 10
                clip: true
                cellWidth: root.gridCols * 14 + 28
                cellHeight: root.gridRows * 14 + 52
                model: bridge.datasetSize

                delegate: Item {
                    id: card
                    required property int index
                    width: gallery.cellWidth
                    height: gallery.cellHeight

                    Rectangle {
                        anchors.fill: parent
                        anchors.margins: 4
                        radius: 6
                        color: card.index === root.selectedIndex ? "#132e50" : "#161b22"
                        border.color: card.index === root.selectedIndex ? "#58a6ff" : "#30363d"
                        border.width: 1

                        ColumnLayout {
                            anchors.centerIn: parent
                            spacing: 4
                            PixelGrid {
                                Layout.alignment: Qt.AlignHCenter
                                interactive: false
                                rows: root.gridRows
                                cols: root.gridCols
                                cell: 12
                                gap: 1
                                values: { root.rev; return bridge.exampleInput(card.index) }
                                onClicked: root.selectExample(card.index)
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: { root.rev; return bridge.exampleLabel(card.index) }
                                color: "#e6edf3"; font.pixelSize: 12
                            }
                        }

                        MouseArea {
                            anchors.fill: parent
                            z: -1
                            onClicked: root.selectExample(card.index)
                        }

                        Button {
                            anchors.top: parent.top
                            anchors.right: parent.right
                            anchors.margins: 2
                            implicitWidth: 22
                            implicitHeight: 22
                            text: "✕"
                            onClicked: root.deleteExample(card.index)
                        }
                    }
                }

                Label {
                    anchors.centerIn: parent
                    visible: bridge.datasetSize === 0
                    text: qsTr("No examples yet — add some on the right →")
                    color: "#8b949e"
                }
            }
        }

        Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: "#30363d" }

        // ── editor ─────────────────────────────────────────────────────────────
        Rectangle {
            Layout.preferredWidth: 340
            Layout.fillHeight: true
            color: "#12161c"

            ScrollView {
                anchors.fill: parent
                contentWidth: availableWidth
                clip: true

                ColumnLayout {
                    width: 340
                    spacing: 12

                    Label {
                        Layout.margins: 16
                        Layout.fillWidth: true
                        text: root.selectedIndex >= 0
                              ? (qsTr("Editing example #") + (root.selectedIndex + 1))
                              : qsTr("New example")
                        color: root.selectedIndex >= 0 ? "#58a6ff" : "#3fb950"
                        font.pixelSize: 15; font.bold: true
                    }

                    // Input editor
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.leftMargin: 16
                        Layout.rightMargin: 16
                        title: qsTr("Input")

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            PixelGrid {
                                visible: bridge.inputLayout === "grid"
                                Layout.alignment: Qt.AlignHCenter
                                interactive: true
                                rows: root.gridRows
                                cols: root.gridCols
                                cell: 30
                                values: root.editInput
                                onPainted: (index, value) => root.setEditInput(index, value)
                            }
                            Repeater {
                                model: bridge.inputLayout === "labels" ? bridge.inputDim : 0
                                delegate: RowLayout {
                                    id: inRow
                                    required property int index
                                    Layout.fillWidth: true
                                    Label {
                                        text: (bridge.inputLabels[inRow.index] !== undefined)
                                              ? bridge.inputLabels[inRow.index] : ("x" + inRow.index)
                                        color: "#8b949e"; Layout.preferredWidth: 40; elide: Text.ElideRight
                                    }
                                    Slider {
                                        Layout.fillWidth: true
                                        from: 0; to: 1; stepSize: 0.05
                                        value: (root.editInput[inRow.index] !== undefined) ? root.editInput[inRow.index] : 0
                                        onMoved: root.setEditInput(inRow.index, value)
                                    }
                                }
                            }
                        }
                    }

                    // Target editor
                    GroupBox {
                        Layout.fillWidth: true
                        Layout.leftMargin: 16
                        Layout.rightMargin: 16
                        title: qsTr("Correct answer")

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            ComboBox {
                                visible: root.isClass
                                Layout.fillWidth: true
                                model: bridge.outputLabels
                                currentIndex: root.editLabel
                                onActivated: root.editLabel = currentIndex
                            }
                            Repeater {
                                model: root.isClass ? 0 : bridge.outputDim
                                delegate: RowLayout {
                                    id: tRow
                                    required property int index
                                    Layout.fillWidth: true
                                    Label {
                                        text: (bridge.outputLabels[tRow.index] !== undefined)
                                              ? bridge.outputLabels[tRow.index] : ("y" + tRow.index)
                                        color: "#8b949e"; Layout.preferredWidth: 60; elide: Text.ElideRight
                                    }
                                    Slider {
                                        Layout.fillWidth: true
                                        from: 0; to: 1; stepSize: 0.05
                                        value: (root.editTarget[tRow.index] !== undefined) ? root.editTarget[tRow.index] : 0
                                        onMoved: root.setEditTarget(tRow.index, value)
                                    }
                                }
                            }
                            Label {
                                visible: root.isClass
                                Layout.fillWidth: true
                                text: qsTr("Tip: press a number key to set the answer and commit.")
                                color: "#8b949e"; font.pixelSize: 11; wrapMode: Text.WordWrap
                            }
                        }
                    }

                    // Actions
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: 16
                        Layout.rightMargin: 16
                        Layout.bottomMargin: 16
                        spacing: 8

                        Button {
                            visible: root.selectedIndex < 0
                            Layout.fillWidth: true
                            text: qsTr("＋ Add example  (Enter)")
                            onClicked: root.addCurrent()
                        }
                        RowLayout {
                            visible: root.selectedIndex >= 0
                            Layout.fillWidth: true
                            spacing: 8
                            Button {
                                Layout.fillWidth: true
                                text: qsTr("Update  (Enter)")
                                onClicked: root.updateCurrent()
                            }
                            Button {
                                text: qsTr("Delete")
                                onClicked: root.deleteExample(root.selectedIndex)
                            }
                        }
                        Button {
                            Layout.fillWidth: true
                            text: qsTr("Clear / new")
                            onClicked: root.initEdit()
                        }
                    }
                }
            }
        }
    }
}
