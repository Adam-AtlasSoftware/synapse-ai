pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import SynapseDashboard

// Everything that isn't the main action: quiet outline, same metrics as PrimaryButton
// so rows of mixed buttons still line up.
Button {
    id: root

    implicitHeight: 34
    font.pixelSize: Theme.fBody

    background: Rectangle {
        radius: Theme.rSm
        color: root.down ? Theme.cardHover : (root.hovered && root.enabled ? Theme.cardHover
                                                                          : "transparent")
        border.width: 1
        border.color: root.enabled ? (root.hovered ? Theme.border : Theme.borderSoft)
                                   : Theme.borderSoft
        Behavior on color { ColorAnimation { duration: Theme.quick } }
    }

    contentItem: Label {
        text: root.text
        color: root.enabled ? Theme.text : Theme.textFaint
        font: root.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
