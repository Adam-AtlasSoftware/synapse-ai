pragma ComponentBehavior: Bound

import QtQuick

// A paintable pixel grid. Click or drag to draw (left = ink, right = erase); each
// cell maps to one input neuron in row-major order. Emits `painted(index, value)`
// so the parent can update the input vector and re-run the network.
Item {
    id: root

    property int rows: 5
    property int cols: 5
    property var values: []       // length rows*cols, read for display
    property real cell: 34
    property real gap: 3
    property bool interactive: true   // false = read-only thumbnail that emits clicked()

    signal painted(int index, real value)
    signal clicked()

    implicitWidth: cols * cell + (cols - 1) * gap
    implicitHeight: rows * cell + (rows - 1) * gap

    function valueAt(i) {
        return (values && i < values.length && values[i] !== undefined) ? values[i] : 0
    }
    // slate (0) → amber (1), matching the network's positive-activation color.
    function cellColor(v) {
        return Qt.rgba((43 + (242 - 43) * v) / 255,
                       (47 + (163 - 47) * v) / 255,
                       (55 + (60 - 55) * v) / 255, 1)
    }
    function cellIndexAt(x, y) {
        var c = Math.floor(x / (cell + gap))
        var r = Math.floor(y / (cell + gap))
        if (r < 0 || r >= rows || c < 0 || c >= cols) return -1
        return r * cols + c
    }
    function paintAt(x, y, buttons) {
        var i = root.cellIndexAt(x, y)
        if (i < 0) return
        root.painted(i, (buttons & Qt.RightButton) ? 0 : 1)
    }

    Grid {
        rows: root.rows
        columns: root.cols
        rowSpacing: root.gap
        columnSpacing: root.gap

        Repeater {
            model: root.rows * root.cols
            delegate: Rectangle {
                id: pixel
                required property int index
                width: root.cell
                height: root.cell
                radius: 4
                color: root.cellColor(root.valueAt(index))
                border.color: "#0d1117"
                border.width: 1
                Behavior on color { ColorAnimation { duration: 90 } }
            }
        }
    }

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        onPressed: (mouse) => { if (root.interactive) root.paintAt(mouse.x, mouse.y, mouse.buttons); }
        onPositionChanged: (mouse) => { if (root.interactive) root.paintAt(mouse.x, mouse.y, mouse.buttons); }
        onClicked: (mouse) => { if (!root.interactive) root.clicked(); }
    }
}
