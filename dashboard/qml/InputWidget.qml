pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// The one place that knows how to render an input vector for a blueprint. Given a
// `layout` ("labels", "grid", or "segments") plus the shape metadata, it draws the
// right editor and emits a uniform `edited(index, value)` — so the main panel and the
// training-data editor share one implementation. Adding a new input method means
// adding one case here (and a widget file), nothing else changes.
ColumnLayout {
    id: root

    property string layout: "labels"
    property int rows: 1
    property int cols: 0
    property var labels: []
    property int dim: 0
    property var values: []
    property bool interactive: true
    property real gridCell: 34
    property real segSize: 96

    signal edited(int index, real value)

    spacing: 8

    // GRID (OCR / shapes): paint the pixels.
    PixelGrid {
        visible: root.layout === "grid"
        Layout.alignment: Qt.AlignHCenter
        interactive: root.interactive
        rows: root.rows
        cols: root.cols
        cell: root.gridCell
        values: root.values
        onPainted: (index, value) => root.edited(index, value)
    }

    // SEGMENTS (seven-segment digit): toggle the segments.
    SegmentDisplay {
        visible: root.layout === "segments"
        Layout.alignment: Qt.AlignHCenter
        interactive: root.interactive
        size: root.segSize
        values: root.values
        onPainted: (index, value) => root.edited(index, value)
    }

    Label {
        visible: root.interactive && (root.layout === "grid" || root.layout === "segments")
        Layout.fillWidth: true
        horizontalAlignment: Text.AlignHCenter
        text: root.layout === "segments" ? qsTr("Click a segment to toggle it")
                                         : qsTr("Left-drag to draw · right-drag to erase")
        color: "#8b949e"; font.pixelSize: 11
    }

    // LABELS (XOR / adder / custom): named sliders.
    Repeater {
        model: root.layout === "labels" ? root.dim : 0
        delegate: RowLayout {
            id: inputRow
            required property int index
            Layout.fillWidth: true
            spacing: 8
            Label {
                text: (root.labels[inputRow.index] !== undefined)
                      ? root.labels[inputRow.index] : ("x" + inputRow.index)
                color: "#8b949e"
                Layout.preferredWidth: 40
                elide: Text.ElideRight
            }
            Slider {
                Layout.fillWidth: true
                enabled: root.interactive
                from: 0; to: 1; stepSize: 0.05
                value: (root.values[inputRow.index] !== undefined) ? root.values[inputRow.index] : 0
                onMoved: root.edited(inputRow.index, value)
            }
            Label {
                text: ((root.values[inputRow.index] !== undefined)
                       ? root.values[inputRow.index] : 0).toFixed(2)
                color: "#e6edf3"
                Layout.preferredWidth: 40
                horizontalAlignment: Text.AlignRight
            }
        }
    }
}
