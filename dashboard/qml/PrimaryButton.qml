pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import SynapseDashboard

// The one obvious thing to do. There should be at most one of these visible per card —
// the old UI had 24 identical buttons and no hierarchy at all.
Button {
    id: root

    property color tone: Theme.accent

    implicitHeight: 40
    font.pixelSize: Theme.fLabel
    font.bold: true

    background: Rectangle {
        radius: Theme.rSm
        color: !root.enabled ? Theme.surface
             : root.down     ? Qt.darker(root.tone, 1.25)
             : root.hovered  ? Qt.lighter(root.tone, 1.12)
                             : root.tone
        Behavior on color { ColorAnimation { duration: Theme.quick } }
    }

    contentItem: Label {
        text: root.text
        // Dark ink on the bright fill reads better than white at this weight.
        color: root.enabled ? "#08111d" : Theme.textFaint
        font: root.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
