pragma ComponentBehavior: Bound

import QtQuick

// A clickable seven-segment display. Each of the 7 segments maps to one input
// neuron in the standard a–g order (a=top, b=upper-right, c=lower-right, d=bottom,
// e=lower-left, f=upper-left, g=middle) — exactly how the seven_segment dataset is
// encoded. Click a segment to toggle it on/off. Shares PixelGrid's interface
// (`values`, `interactive`, `painted(index,value)`, `clicked()`) so it drops into
// the same slots as the pixel grid.
Item {
    id: root

    property var values: []            // length 7, read for display
    property real size: 96             // width in px; height follows the aspect ratio
    property bool interactive: true    // false = read-only thumbnail that emits clicked()

    // Colors: lit = amber (matches positive activation), off = dim slate.
    property color litColor: "#f2a33c"
    property color offColor: "#2b2f36"

    signal painted(int index, real value)
    signal clicked()

    readonly property real thick: size * 0.19
    implicitWidth: size
    implicitHeight: size * 1.72

    function valueAt(i) {
        return (values && i < values.length && values[i] !== undefined) ? values[i] : 0
    }

    // Geometry for segment i (index order a,b,c,d,e,f,g) inside width W × height H.
    function segRect(i) {
        var W = width, H = height, t = root.thick
        var vh = (H - 3 * t) / 2          // length of a vertical segment
        switch (i) {
        case 0: return { x: t,     y: 0,           w: W - 2 * t, h: t  }  // a  top
        case 1: return { x: W - t, y: t,           w: t,         h: vh }  // b  upper-right
        case 2: return { x: W - t, y: (H + t) / 2, w: t,         h: vh }  // c  lower-right
        case 3: return { x: t,     y: H - t,       w: W - 2 * t, h: t  }  // d  bottom
        case 4: return { x: 0,     y: (H + t) / 2, w: t,         h: vh }  // e  lower-left
        case 5: return { x: 0,     y: t,           w: t,         h: vh }  // f  upper-left
        case 6: return { x: t,     y: (H - t) / 2, w: W - 2 * t, h: t  }  // g  middle
        }
        return { x: 0, y: 0, w: 0, h: 0 }
    }

    Repeater {
        model: 7
        delegate: Rectangle {
            id: seg
            required property int index
            readonly property var r: root.segRect(index)
            x: r.x; y: r.y
            width: r.w; height: r.h
            radius: root.thick / 2
            color: root.valueAt(index) > 0.5 ? root.litColor : root.offColor
            antialiasing: true
            Behavior on color { ColorAnimation { duration: 90 } }

            TapHandler {
                enabled: root.interactive
                onTapped: root.painted(seg.index, root.valueAt(seg.index) > 0.5 ? 0 : 1)
            }
        }
    }

    // Read-only thumbnails forward a single click to select the whole example.
    TapHandler {
        enabled: !root.interactive
        onTapped: root.clicked()
    }
}
