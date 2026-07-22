pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import SynapseDashboard

// A small "?" next to a term that stops a newcomer. Click it and you get the idea in
// plain language — no jargon, no maths — without cluttering the interface for anyone
// who already knows.
Item {
    id: root

    property string term: ""
    property string explanation: ""

    implicitWidth: 16
    implicitHeight: 16

    Rectangle {
        anchors.fill: parent
        radius: width / 2
        color: hover.hovered ? Theme.accent : "transparent"
        border.width: 1
        border.color: hover.hovered ? Theme.accent : Theme.border
        Behavior on color { ColorAnimation { duration: Theme.quick } }

        Label {
            anchors.centerIn: parent
            text: "?"
            color: hover.hovered ? "#08111d" : Theme.textFaint
            font.pixelSize: Theme.fTiny
            font.bold: true
        }
    }

    HoverHandler { id: hover; cursorShape: Qt.PointingHandCursor }
    TapHandler { onTapped: popup.open() }

    Popup {
        id: popup
        parent: Overlay.overlay
        anchors.centerIn: Overlay.overlay
        width: 340
        modal: true
        dim: true
        padding: Theme.lg

        background: Rectangle {
            color: Theme.card
            radius: Theme.rMd
            border.width: 1
            border.color: Theme.border
        }

        contentItem: Column {
            spacing: Theme.sm
            Label {
                text: root.term
                color: Theme.accent
                font.pixelSize: Theme.fTitle
                font.bold: true
            }
            Label {
                width: popup.availableWidth
                text: root.explanation
                color: Theme.text
                font.pixelSize: Theme.fBody
                wrapMode: Text.WordWrap
                lineHeight: 1.25
            }
        }
    }
}
