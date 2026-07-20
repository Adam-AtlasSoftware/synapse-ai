pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls

// Self-adapting network graph. It reads only the generic `bridge` properties
// (columns / activations / weights) and computes its own layout, so ANY model —
// any number of layers, any sizes — renders with no changes here. Neurons are
// colored by their current activation; edges by weight sign and magnitude.
Item {
    id: view

    property int pad: 72
    property var columns: bridge.columns
    property var activations: bridge.activations
    property var weights: bridge.weights
    property int activeLayer: bridge.activeLayer
    property string phase: bridge.phase
    property var outputLabels: bridge.outputLabels
    property string outputKind: bridge.outputKind
    property string inputLayout: bridge.inputLayout
    property var inputLabels: bridge.inputLabels
    property int predictedIndex: bridge.predictedIndex
    property var deltas: bridge.deltas
    property var grads: bridge.grads
    property color bg: "#0d1117"

    // ── layout helpers (shared by the edge Canvas and the node Repeater) ──────
    function numCols() { return view.columns ? view.columns.length : 0 }

    function colCount(k) {
        return (view.columns && view.columns[k]) ? view.columns[k].count : 0
    }

    function nodeX(k) {
        var n = view.numCols()
        if (n <= 1) return view.width / 2
        return view.pad + k * (view.width - 2 * view.pad) / (n - 1)
    }

    function colGap(count) {
        if (count <= 1) return 0
        return Math.min(60, (view.height - 2 * view.pad) / (count - 1))
    }

    function nodeY(k, i, count) {
        return view.height / 2 + (i - (count - 1) / 2) * view.colGap(count)
    }

    function nodeRadius(count) {
        if (count <= 1) return 18
        var g = view.colGap(count)
        return Math.max(6, Math.min(18, g * 0.36))
    }

    function activationAt(k, i) {
        var a = view.activations
        if (!a || k >= a.length || !a[k] || i >= a[k].length) return 0
        return a[k][i]
    }

    // ── stepping state: which columns are computed / active / pending ─────────
    function stepping() {
        return view.phase === "begin" || view.phase === "step" || view.phase === "done"
    }
    function computedCols() {
        if (!view.stepping()) return view.numCols()   // instant run: everything is "on"
        if (view.phase === "begin") return 1           // only the input column
        return view.activeLayer + 2                     // input + layers 0..activeLayer
    }
    function isPending(k) { return k >= view.computedCols() }
    function isActive(k) {
        return view.stepping() && view.activeLayer >= 0 && k === view.activeLayer + 1
    }

    // ── semantic labels (from the loaded blueprint) ──────────────────────────
    function isOutputCol(k) { return k === view.numCols() - 1 }
    function isInputCol(k) { return k === 0 }
    function outputLabelAt(i) {
        return (view.outputLabels && i < view.outputLabels.length) ? view.outputLabels[i] : ""
    }
    function inputLabelAt(i) {
        return (view.inputLabels && i < view.inputLabels.length) ? view.inputLabels[i] : ""
    }
    function isPredicted(k, i) {
        return view.outputKind === "class" && view.isOutputCol(k) && i === view.predictedIndex
    }

    // ── backprop / gradient-flow state ───────────────────────────────────────
    function isBackward() {
        return view.phase === "loss" || view.phase === "backward" || view.phase === "update"
    }
    // Columns whose delta (backprop signal) has been computed so far.
    function deltaReached(k) {
        if (view.phase === "update") return true
        return k >= view.activeLayer + 1
    }
    function deltaAt(k, i) {
        var d = view.deltas
        if (!d || k >= d.length || !d[k] || i >= d[k].length) return 0
        return d[k][i]
    }
    function colMaxAbsDelta(k) {
        var d = view.deltas
        if (!d || k >= d.length || !d[k]) return 1e-9
        var m = 1e-9
        for (var i = 0; i < d[k].length; ++i) { var a = Math.abs(d[k][i]); if (a > m) m = a }
        return m
    }
    // slate → magenta by gradient intensity (distinct from the amber/blue of values)
    function gradColor(t) {
        t = Math.max(0, Math.min(1, t))
        return view.mix("#2b2f37", "#d267ff", t)
    }
    // A neuron shows its activation normally, but during the backward pass it shows
    // the magnitude of its gradient signal (delta) in magenta instead.
    function nodeColor(k, i) {
        if (view.isBackward() && k > 0 && view.deltaReached(k))
            return view.gradColor(Math.abs(view.deltaAt(k, i)) / view.colMaxAbsDelta(k))
        return view.activationColor(view.activationAt(k, i))
    }

    // Diverging colormap: negative → blue, ~0 → slate, positive → amber.
    function mix(a, b, f) {
        var ar = parseInt(a.substr(1, 2), 16), ag = parseInt(a.substr(3, 2), 16), ab = parseInt(a.substr(5, 2), 16)
        var br = parseInt(b.substr(1, 2), 16), bg = parseInt(b.substr(3, 2), 16), bb = parseInt(b.substr(5, 2), 16)
        return Qt.rgba((ar + (br - ar) * f) / 255,
                       (ag + (bg - ag) * f) / 255,
                       (ab + (bb - ab) * f) / 255, 1)
    }
    function activationColor(v) {
        var t = Math.max(-1, Math.min(1, v))
        return t >= 0 ? view.mix("#2b2f37", "#f2a33c", t) : view.mix("#2b2f37", "#4c8dff", -t)
    }

    Rectangle { anchors.fill: parent; color: view.bg }

    // ── edges ────────────────────────────────────────────────────────────────
    Canvas {
        id: edges
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            var cols = view.columns
            var wts = view.weights
            if (!cols || cols.length < 2 || !wts) return

            var updating = view.phase === "update"
            var backwardish = view.isBackward()
            var grads = view.grads

            for (var k = 0; k < wts.length; ++k) {
                var W = wts[k]
                if (!W || !W.values) continue
                var inN = W.cols, outN = W.rows

                // UPDATE frame: draw the gradients (∂loss/∂W) that just moved the weights.
                if (updating && grads && grads[k] && grads[k].values) {
                    var g = grads[k].values
                    var mg = 1e-9
                    for (var qg = 0; qg < g.length; ++qg) { var agv = Math.abs(g[qg]); if (agv > mg) mg = agv }
                    for (var og = 0; og < outN; ++og) {
                        for (var ig = 0; ig < inN; ++ig) {
                            var t = Math.abs(g[og * inN + ig]) / mg
                            ctx.strokeStyle = "#d267ff"
                            ctx.globalAlpha = 0.08 + 0.8 * t
                            ctx.lineWidth = 0.4 + 3.0 * t
                            ctx.beginPath()
                            ctx.moveTo(view.nodeX(k), view.nodeY(k, ig, inN))
                            ctx.lineTo(view.nodeX(k + 1), view.nodeY(k + 1, og, outN))
                            ctx.stroke()
                        }
                    }
                    continue
                }

                // Otherwise draw the weights. During the backward pass they dim right
                // down so the magenta node gradient signal stands out.
                var vals = W.values
                var mx = 1e-6
                for (var q = 0; q < vals.length; ++q) { var av = Math.abs(vals[q]); if (av > mx) mx = av }
                var pending = view.isPending(k + 1)
                var active = view.isActive(k + 1)

                for (var o = 0; o < outN; ++o) {
                    for (var i = 0; i < inN; ++i) {
                        var w = vals[o * inN + i]
                        var f = w / mx
                        var alpha = 0.12 + 0.6 * Math.abs(f)
                        if (backwardish) alpha *= 0.14
                        else if (pending) alpha *= 0.22
                        else if (active) alpha = Math.min(1, alpha + 0.3)
                        ctx.strokeStyle = f >= 0 ? "#f2a33c" : "#4c8dff"
                        ctx.globalAlpha = alpha
                        ctx.lineWidth = 0.5 + 2.4 * Math.abs(f)
                        ctx.beginPath()
                        ctx.moveTo(view.nodeX(k), view.nodeY(k, i, inN))
                        ctx.lineTo(view.nodeX(k + 1), view.nodeY(k + 1, o, outN))
                        ctx.stroke()
                    }
                }
            }
            ctx.globalAlpha = 1
        }
        Connections {
            target: bridge
            function onActivationsChanged() { edges.requestPaint() }
            function onTopologyChanged() { edges.requestPaint() }
        }
        onWidthChanged: edges.requestPaint()
        onHeightChanged: edges.requestPaint()
    }

    // ── column headers ────────────────────────────────────────────────────────
    Repeater {
        model: view.columns
        delegate: Column {
            id: header
            required property int index
            required property var modelData
            x: view.nodeX(index) - width / 2
            y: 18
            width: 140
            spacing: 2
            opacity: view.isPending(index) ? 0.4 : 1.0
            Behavior on opacity { NumberAnimation { duration: 150 } }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: header.modelData.title
                color: "#e6edf3"; font.pixelSize: 14; font.bold: true
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: header.modelData.kind === "input"
                      ? (header.modelData.count + " inputs")
                      : (header.modelData.activation + " · " + header.modelData.count)
                color: "#8b949e"; font.pixelSize: 11
            }
        }
    }

    // ── neurons ────────────────────────────────────────────────────────────────
    Repeater {
        model: view.columns
        delegate: Item {
            id: columnItem
            required property int index
            required property var modelData
            readonly property int colIndex: index
            readonly property int count: modelData.count
            anchors.fill: parent

            Repeater {
                model: columnItem.count
                delegate: Rectangle {
                    id: neuron
                    required property int index
                    readonly property real value: view.activationAt(columnItem.colIndex, index)
                    readonly property real r: view.nodeRadius(columnItem.count)

                    width: 2 * r
                    height: 2 * r
                    radius: r
                    x: view.nodeX(columnItem.colIndex) - r
                    y: view.nodeY(columnItem.colIndex, index, columnItem.count) - r
                    color: view.nodeColor(columnItem.colIndex, neuron.index)
                    opacity: view.isBackward()
                             ? ((columnItem.colIndex === 0 || view.deltaReached(columnItem.colIndex)) ? 1.0 : 0.35)
                             : (view.isPending(columnItem.colIndex) ? 0.28 : 1.0)
                    border.color: view.isPredicted(columnItem.colIndex, neuron.index) ? "#3fb950"
                                  : (view.isActive(columnItem.colIndex)
                                     ? (view.isBackward() ? "#d267ff" : "#e6edf3") : "#0d1117")
                    border.width: (view.isPredicted(columnItem.colIndex, neuron.index)
                                   || view.isActive(columnItem.colIndex)) ? 3 : 2
                    Behavior on opacity { NumberAnimation { duration: 150 } }

                    Text {
                        anchors.centerIn: parent
                        visible: neuron.width > 26
                        text: neuron.value.toFixed(2)
                        color: "#0d1117"
                        font.pixelSize: 9
                        font.bold: true
                    }

                    // Output-neuron label (e.g. the digit), highlighted if predicted.
                    Text {
                        visible: view.isOutputCol(columnItem.colIndex)
                                 && view.outputLabelAt(neuron.index) !== ""
                        anchors.left: parent.right
                        anchors.leftMargin: 10
                        anchors.verticalCenter: parent.verticalCenter
                        text: view.outputLabelAt(neuron.index)
                        color: view.isPredicted(columnItem.colIndex, neuron.index) ? "#3fb950" : "#c9d1d9"
                        font.pixelSize: 13
                        font.bold: view.isPredicted(columnItem.colIndex, neuron.index)
                    }

                    // Input-neuron label (only for labeled blueprints, not grids).
                    Text {
                        visible: view.isInputCol(columnItem.colIndex)
                                 && view.inputLayout === "labels"
                                 && view.inputLabelAt(neuron.index) !== ""
                        anchors.right: parent.left
                        anchors.rightMargin: 10
                        anchors.verticalCenter: parent.verticalCenter
                        text: view.inputLabelAt(neuron.index)
                        color: "#c9d1d9"
                        font.pixelSize: 13
                    }

                    ToolTip.visible: hover.hovered
                    ToolTip.text: "value: " + neuron.value.toFixed(4)
                    HoverHandler { id: hover }
                }
            }
        }
    }
}
