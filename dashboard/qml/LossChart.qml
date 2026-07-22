pragma ComponentBehavior: Bound

import QtQuick

// Live loss curves. Blue = training loss (bridge.lossHistory, one point per tick). When a
// validation split is held out, an amber curve is drawn on the same auto-scaled axes: while
// both fall the network is genuinely learning, but when the amber one turns back UP while
// blue keeps dropping, you are watching it overfit — memorizing instead of generalizing.
Item {
    id: chart
    property var history: bridge.lossHistory
    property var valHistory: bridge.valLossHistory
    property bool showVal: bridge.validationOn
    implicitHeight: 120

    Rectangle {
        anchors.fill: parent
        color: "#0d1117"
        border.color: "#30363d"
        border.width: 1
        radius: 6
    }

    Canvas {
        id: cv
        anchors.fill: parent
        anchors.margins: 8

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            var h = chart.history
            if (!h || h.length < 2) return

            var v = (chart.showVal && chart.valHistory && chart.valHistory.length === h.length)
                    ? chart.valHistory : null

            // Both curves share one scale, so their divergence is honest.
            var maxv = 1e-9
            for (var i = 0; i < h.length; ++i) if (h[i] > maxv) maxv = h[i]
            if (v) for (var k = 0; k < v.length; ++k) if (v[k] > maxv) maxv = v[k]

            var W = width, H = height

            // baseline (loss = 0)
            ctx.strokeStyle = "#30363d"
            ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(0, H - 0.5); ctx.lineTo(W, H - 0.5); ctx.stroke()

            function plot(series, color) {
                ctx.strokeStyle = color
                ctx.lineWidth = 2
                ctx.beginPath()
                for (var j = 0; j < series.length; ++j) {
                    var x = j * W / (series.length - 1)
                    var y = H - (series[j] / maxv) * H
                    if (j === 0) ctx.moveTo(x, y)
                    else ctx.lineTo(x, y)
                }
                ctx.stroke()
            }

            if (v) plot(v, "#f2a33c")   // validation (amber) — drawn under training
            plot(h, "#58a6ff")          // training (blue)
        }

        Connections {
            target: bridge
            function onTrainingProgress() { cv.requestPaint() }
        }
        onWidthChanged: cv.requestPaint()
        onHeightChanged: cv.requestPaint()
    }

    Text {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 10
        text: chart.history && chart.history.length ? qsTr("peak ") + chart.history[0].toFixed(2) : ""
        color: "#8b949e"
        font.pixelSize: 10
    }

    // Legend — only worth showing once there are two curves to tell apart.
    Row {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.margins: 10
        spacing: 10
        visible: chart.showVal
        Row {
            spacing: 4
            Rectangle { width: 10; height: 2; color: "#58a6ff"; anchors.verticalCenter: parent.verticalCenter }
            Text { text: qsTr("training"); color: "#8b949e"; font.pixelSize: 10 }
        }
        Row {
            spacing: 4
            Rectangle { width: 10; height: 2; color: "#f2a33c"; anchors.verticalCenter: parent.verticalCenter }
            Text { text: qsTr("validation"); color: "#8b949e"; font.pixelSize: 10 }
        }
    }
}
