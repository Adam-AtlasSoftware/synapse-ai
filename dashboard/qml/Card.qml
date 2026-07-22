pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SynapseDashboard

// One step of the workflow. A titled surface with an optional step number and a
// subtitle — the numbers are what turn a wall of controls into a path you follow.
Rectangle {
    id: root

    property string title: ""
    property string subtitle: ""
    property int step: 0            // 0 = no number badge
    property bool active: false     // highlights the step you should do next
    default property alias content: body.data

    color: Theme.card
    radius: Theme.rMd
    border.width: 1
    border.color: root.active ? Theme.accentDim : Theme.borderSoft
    Layout.fillWidth: true
    implicitHeight: layout.implicitHeight + Theme.lg * 2

    Behavior on border.color { ColorAnimation { duration: Theme.calm } }

    ColumnLayout {
        id: layout
        anchors.fill: parent
        anchors.margins: Theme.lg
        spacing: Theme.md

        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.sm
            visible: root.title !== ""

            // Step badge — fills in once this step is reachable.
            Rectangle {
                visible: root.step > 0
                implicitWidth: 22
                implicitHeight: 22
                radius: width / 2
                color: root.active ? Theme.accent : "transparent"
                border.width: 1
                border.color: root.active ? Theme.accent : Theme.border
                Label {
                    anchors.centerIn: parent
                    text: root.step
                    color: root.active ? "#08111d" : Theme.textFaint
                    font.pixelSize: Theme.fSmall
                    font.bold: true
                }
            }
            Label {
                text: root.title
                color: Theme.text
                font.pixelSize: Theme.fTitle
                font.bold: true
            }
            Item { Layout.fillWidth: true }
        }

        // Explanatory prose is coaching — it goes away when you switch coaching off.
        Label {
            visible: root.subtitle !== "" && bridge.coachingEnabled
            Layout.fillWidth: true
            text: root.subtitle
            color: Theme.textMuted
            font.pixelSize: Theme.fSmall
            wrapMode: Text.WordWrap
        }

        ColumnLayout {
            id: body
            Layout.fillWidth: true
            spacing: Theme.md
        }
    }
}
