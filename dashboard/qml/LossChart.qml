pragma ComponentBehavior: Bound

import QtQuick

// A minimal live loss curve. Reads bridge.lossHistory (one point per training tick)
// and draws a linearly auto-scaled line — you watch it plunge toward zero as the
// network learns.
Item {
    id: chart
    property var history: bridge.lossHistory
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

            var maxv = 1e-9
            for (var i = 0; i < h.length; ++i) if (h[i] > maxv) maxv = h[i]

            var W = width, H = height

            // baseline (loss = 0)
            ctx.strokeStyle = "#30363d"
            ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(0, H - 0.5); ctx.lineTo(W, H - 0.5); ctx.stroke()

            // the curve
            ctx.strokeStyle = "#58a6ff"
            ctx.lineWidth = 2
            ctx.beginPath()
            for (var j = 0; j < h.length; ++j) {
                var x = j * W / (h.length - 1)
                var y = H - (h[j] / maxv) * H
                if (j === 0) ctx.moveTo(x, y)
                else ctx.lineTo(x, y)
            }
            ctx.stroke()
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
}
