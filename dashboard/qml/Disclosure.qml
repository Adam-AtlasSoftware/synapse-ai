pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SynapseDashboard

// How the expert controls stay out of a beginner's way without being taken away: a quiet
// "▸ label" row that expands in place. Advanced mode opens these by default, so a power
// user sees everything at once and a newcomer sees a single unobtrusive line.
ColumnLayout {
    id: root

    property string label: qsTr("Advanced options")
    property bool expanded: false
    default property alias content: body.data

    spacing: Theme.sm
    Layout.fillWidth: true

    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.xs

        Label {
            text: root.expanded ? "▾" : "▸"
            color: Theme.textFaint
            font.pixelSize: Theme.fSmall
        }
        Label {
            text: root.label
            color: hover.hovered ? Theme.accent : Theme.textMuted
            font.pixelSize: Theme.fSmall
        }
        Item { Layout.fillWidth: true }

        TapHandler { onTapped: root.expanded = !root.expanded }
        HoverHandler { id: hover; cursorShape: Qt.PointingHandCursor }
    }

    ColumnLayout {
        id: body
        Layout.fillWidth: true
        Layout.leftMargin: Theme.sm
        spacing: Theme.sm
        visible: root.expanded
    }
}
