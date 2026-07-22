pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SynapseDashboard

// The bar along the bottom that always answers "what do I do next?". It reads the app's
// actual state rather than following a fixed script, so it stays right no matter which
// order you do things in.
Rectangle {
    id: root

    // How many correct answers on unseen inputs before we call it convincing.
    readonly property int convincingStreak: 3

    // 0 Try · 1 Teach · 2 Train · 3 Understand — also drives which Card is highlighted.
    // This is the whole first-run journey: teach → train → TEST what it learned → understand.
    readonly property int suggestedStep: {
        if (bridge.datasetSize === 0) return 1               // nothing to learn from yet
        if (bridge.training) return 2
        if (bridge.epoch === 0) return 2                     // has data, never trained
        if (bridge.awaitingVerdict) return 0                 // waiting to hear if it was right
        if (bridge.testsFailed > 0) return 1                 // it got one wrong — teach it that
        if (bridge.testsPassed >= root.convincingStreak) return 3   // proven; go understand it
        return 0                                             // trained — go test it
    }

    readonly property string message: {
        if (bridge.datasetSize === 0)
            return qsTr("Start here: set an input on the right, then tell it the right answer to teach it.")
        if (bridge.training)
            return bridge.autoStatus !== "" ? bridge.autoStatus
                                            : qsTr("Training — watch the error fall.")
        if (bridge.epoch === 0)
            return qsTr("It knows %1 example%2 but hasn't learned from them yet — press “Train it for me”.")
                   .arg(bridge.datasetSize).arg(bridge.datasetSize === 1 ? "" : "s")

        // ── the testing phase: does it actually work on something new? ────────
        if (bridge.awaitingVerdict)
            return qsTr("You changed the input, so this is a real test — it has never seen this one. Was its answer right?")
        if (bridge.testsFailed > 0)
            return qsTr("It got that one wrong — which is exactly what more examples fix. Click the right answer in step 2, then train it again.")
        if (bridge.testsPassed >= root.convincingStreak)
            return qsTr("%1 new ones in a row, all correct — it genuinely learned the pattern rather than memorizing. Now look at step 4 to see *how* it does it.")
                   .arg(bridge.testsPassed)
        if (bridge.testsPassed > 0)
            return qsTr("Correct — and it had never seen that one. Try another to be sure.")

        if (bridge.validationOn && bridge.valAccuracy < bridge.trainAccuracy - 0.2)
            return qsTr("It aced the examples it studied but not the held-out ones — that's memorizing, not learning. Try more examples.")
        return qsTr("Now draw something new and press “Run it” — let's find out whether it really learned.")
    }

    color: Theme.surface
    implicitHeight: 44
    Layout.fillWidth: true

    Rectangle { width: parent.width; height: 1; color: Theme.borderSoft }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: Theme.lg
        anchors.rightMargin: Theme.lg
        spacing: Theme.sm

        Label {
            text: bridge.training ? "◐" : "→"
            color: bridge.training ? Theme.warn : Theme.accent
            font.pixelSize: Theme.fLabel
            font.bold: true
        }
        Label {
            Layout.fillWidth: true
            text: root.message
            color: Theme.text
            font.pixelSize: Theme.fBody
            elide: Text.ElideRight
        }
    }
}
