pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Advanced view: per-layer statistics (mean/σ of activations, weights, and gradients)
// plus a live histogram of each layer's weight distribution. Reads only the generic
// bridge telemetry, so it works for any model.
ColumnLayout {
    id: root
    spacing: 12

    function stats(arr) {
        if (!arr || arr.length === 0) return { mean: 0, std: 0, min: 0, max: 0 }
        var m = 0
        for (var i = 0; i < arr.length; ++i) m += arr[i]
        m /= arr.length
        var s = 0, mn = arr[0], mx = arr[0]
        for (var j = 0; j < arr.length; ++j) {
            var v = arr[j]
            s += (v - m) * (v - m)
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
        return { mean: m, std: Math.sqrt(s / arr.length), min: mn, max: mx }
    }
    function l2(arr) {
        if (!arr) return 0
        var s = 0
        for (var i = 0; i < arr.length; ++i) s += arr[i] * arr[i]
        return Math.sqrt(s)
    }
    function fmt(x) { return (x >= 0 ? " " : "") + x.toFixed(3) }

    Repeater {
        model: bridge.layers
        delegate: Frame {
            id: card
            required property int index
            required property var modelData
            Layout.fillWidth: true

            readonly property var wvals: (bridge.weights[card.index] && bridge.weights[card.index].values)
                                         ? bridge.weights[card.index].values : []
            readonly property var avals: bridge.activations[card.index + 1] ? bridge.activations[card.index + 1] : []
            readonly property var gvals: (bridge.grads[card.index] && bridge.grads[card.index].values)
                                         ? bridge.grads[card.index].values : []
            readonly property var wstat: root.stats(wvals)
            readonly property var astat: root.stats(avals)

            ColumnLayout {
                anchors.fill: parent
                spacing: 4

                Label {
                    text: card.modelData.name + "  ·  " + card.modelData.activation
                          + "  ·  " + card.modelData.units + " neurons"
                    color: "#58a6ff"; font.pixelSize: 12; font.bold: true
                }
                Label {
                    text: "activations   μ " + root.fmt(card.astat.mean) + "   σ " + root.fmt(card.astat.std)
                    color: "#c9d1d9"; font.pixelSize: 11; font.family: "monospace"
                }
                Label {
                    text: "weights       μ " + root.fmt(card.wstat.mean) + "   σ " + root.fmt(card.wstat.std)
                          + "   [" + root.fmt(card.wstat.min) + "," + root.fmt(card.wstat.max) + "]"
                    color: "#c9d1d9"; font.pixelSize: 11; font.family: "monospace"
                }
                Label {
                    text: "‖gradient‖    " + root.fmt(root.l2(card.gvals))
                          + (bridge.trained ? "" : "   (train to populate)")
                    color: "#c9d1d9"; font.pixelSize: 11; font.family: "monospace"
                }

                // Weight-distribution histogram.
                Label { text: qsTr("weight distribution"); color: "#8b949e"; font.pixelSize: 10 }
                Canvas {
                    id: hist
                    Layout.fillWidth: true
                    implicitHeight: 46
                    property var values: card.wvals
                    onValuesChanged: requestPaint()
                    onWidthChanged: requestPaint()
                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.reset()
                        var v = values
                        if (!v || v.length < 2) return
                        var mn = v[0], mx = v[0]
                        for (var i = 0; i < v.length; ++i) { if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i] }
                        var span = (mx - mn) || 1e-6
                        var bins = 24
                        var counts = new Array(bins).fill(0)
                        for (var k = 0; k < v.length; ++k) {
                            var b = Math.floor((v[k] - mn) / span * (bins - 1))
                            counts[b] += 1
                        }
                        var cmax = 1
                        for (var c = 0; c < bins; ++c) if (counts[c] > cmax) cmax = counts[c]
                        var bw = width / bins
                        // zero line
                        var zx = (0 - mn) / span * width
                        ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1
                        ctx.beginPath(); ctx.moveTo(zx, 0); ctx.lineTo(zx, height); ctx.stroke()
                        // bars (blue = negative weights, amber = positive)
                        for (var bi = 0; bi < bins; ++bi) {
                            var h = counts[bi] / cmax * (height - 2)
                            var center = mn + (bi + 0.5) / bins * span
                            ctx.fillStyle = center >= 0 ? "#f2a33c" : "#4c8dff"
                            ctx.fillRect(bi * bw + 0.5, height - h, bw - 1, h)
                        }
                    }
                    Connections {
                        target: bridge
                        function onActivationsChanged() { hist.requestPaint() }
                    }
                }

                // Activation-distribution histogram.
                Label { text: qsTr("activation distribution"); color: "#8b949e"; font.pixelSize: 10 }
                Canvas {
                    id: ahist
                    Layout.fillWidth: true
                    implicitHeight: 40
                    property var values: card.avals
                    onValuesChanged: requestPaint()
                    onWidthChanged: requestPaint()
                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.reset()
                        var v = values
                        if (!v || v.length < 1) return
                        var mn = v[0], mx = v[0]
                        for (var i = 0; i < v.length; ++i) { if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i] }
                        var span = (mx - mn) || 1e-6
                        var bins = 20
                        var counts = new Array(bins).fill(0)
                        for (var k = 0; k < v.length; ++k) counts[Math.floor((v[k] - mn) / span * (bins - 1))] += 1
                        var cmax = 1
                        for (var c = 0; c < bins; ++c) if (counts[c] > cmax) cmax = counts[c]
                        var bw = width / bins
                        for (var bi = 0; bi < bins; ++bi) {
                            var h = counts[bi] / cmax * (height - 2)
                            ctx.fillStyle = "#39c5cf"
                            ctx.fillRect(bi * bw + 0.5, height - h, bw - 1, h)
                        }
                    }
                    Connections {
                        target: bridge
                        function onActivationsChanged() { ahist.requestPaint() }
                    }
                }
            }
        }
    }
}
