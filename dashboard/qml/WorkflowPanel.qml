pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SynapseDashboard

// The right-hand panel, reorganized from four static tabs into the path you actually walk:
// try it → teach it → train it → understand it. Every expert control that used to be on
// screen at all times still exists — it just lives inside a Disclosure until you ask for it,
// and Advanced mode opens them all at once.
ScrollView {
    id: root

    property bool advanced: false
    property int activeStep: 1        // which card to highlight (from the coach)
    property int exampleIndex: 0      // which dataset example the learn-step animates

    // Expert bench: Advanced with coaching off. The guided "try it / teach it / train it"
    // framing — step numbers, next-step highlighting, hand-holding button labels — drops
    // away and the same panels read as a dense set of tools for building a custom model.
    readonly property bool expert: advanced && !bridge.coachingEnabled

    signal manageDataRequested()
    signal codeLabRequested()
    signal saveBlueprintRequested()
    signal restoreDefaultRequested()

    contentWidth: availableWidth
    clip: true

    // Dev/demo hook (see SYNAPSE_SCROLL): jump the panel down just before the screenshot.
    // Late on purpose — cards appearing during training change the content height, which
    // clamps an early scroll back to the top.
    Timer {
        running: typeof startScroll !== 'undefined' && startScroll > 0
        interval: Math.max(400, ((typeof grabDelay !== 'undefined' ? grabDelay : 1200) - 500))
        onTriggered: if (root.contentItem) root.contentItem.contentY = startScroll
    }

    // Bring a card into view — used by the advice buttons so "teach it more" actually
    // takes you there instead of just telling you to go.
    function scrollTo(item) {
        if (contentItem) contentItem.contentY = Math.max(0, item.y - Theme.md)
    }

    function outcomeHeadline() {
        switch (bridge.autoOutcome) {
        case "converged": return qsTr("It learned this.")
        case "overfit":   return qsTr("It memorized instead of learning.")
        case "diverged":  return qsTr("Training kept blowing up.")
        case "stalled":   return bridge.trainAccuracy > 0.9
                                 ? qsTr("About as good as this gets.")
                                 : qsTr("It got stuck short of the mark.")
        }
        return ""
    }

    // The important half: what to actually do about it.
    function outcomeAdvice() {
        switch (bridge.autoOutcome) {
        case "converged":
            return qsTr("Draw something it has never seen and check it still gets it right.")
        case "overfit":
            return qsTr("It does well on what it studied but not on new examples. The cure is "
                      + "variety — more examples, drawn differently, rather than more training.")
        case "diverged":
            return qsTr("The steps were too big for this model. I've already lowered the "
                      + "learning rate, so another run has a good chance of settling.")
        case "stalled":
            return bridge.trainAccuracy > 0.9
                   ? qsTr("More training isn't buying much now. More examples would push it higher.")
                   : qsTr("It stopped improving while still getting a lot wrong. That usually means "
                        + "one of two things: it hasn't got enough examples to learn the pattern "
                        + "from, or the network is too small to represent it. Both are fixable.")
        }
        return ""
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: Theme.md

        // ─────────────────────────────── ① TRY IT ────────────────────────────
        Card {
            step: root.expert ? 0 : 1
            title: root.expert ? qsTr("Inference") : qsTr("Try it")
            subtitle: qsTr("Set an input, and see what the network makes of it.")
            active: !root.expert && (root.activeStep === 0 || root.activeStep === 1)

            InputWidget {
                Layout.fillWidth: true
                layout: bridge.inputLayout
                rows: bridge.inputRows
                cols: bridge.inputCols
                labels: bridge.inputLabels
                dim: bridge.inputDim
                values: bridge.input
                onEdited: (index, value) => bridge.setInput(index, value)
            }

            // The answer, stated plainly and large — this is the payoff of the whole app.
            Rectangle {
                Layout.fillWidth: true
                visible: bridge.outputKind === "class" && bridge.predictedLabel !== ""
                implicitHeight: 56
                radius: Theme.rSm
                color: Theme.inset
                border.width: 1
                border.color: bridge.epoch > 0 ? Theme.success : Theme.borderSoft

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: Theme.md
                    anchors.rightMargin: Theme.md
                    spacing: Theme.sm
                    Label {
                        text: root.expert ? qsTr("output") : qsTr("It says")
                        color: Theme.textMuted
                        font.pixelSize: Theme.fBody
                    }
                    Label {
                        text: bridge.predictedLabel
                        color: bridge.epoch > 0 ? Theme.success : Theme.textMuted
                        font.pixelSize: Theme.fHuge
                        font.bold: true
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        visible: bridge.epoch === 0 && !root.expert
                        Layout.maximumWidth: 150
                        text: qsTr("just guessing — it hasn't learned yet")
                        color: Theme.textFaint
                        font.pixelSize: Theme.fTiny
                        wrapMode: Text.WordWrap
                        horizontalAlignment: Text.AlignRight
                    }
                }
            }

            // The test loop. Once it is trained, anything you draw is a real test of whether
            // it learned the idea — so ask, and let the answer steer what happens next.
            Rectangle {
                Layout.fillWidth: true
                visible: bridge.awaitingVerdict && bridge.epoch > 0 && bridge.coachingEnabled
                implicitHeight: verdictCol.implicitHeight + Theme.md * 2
                radius: Theme.rSm
                color: Theme.inset
                border.width: 1
                border.color: Theme.accentDim

                ColumnLayout {
                    id: verdictCol
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.margins: Theme.md
                    spacing: Theme.sm

                    Label {
                        Layout.fillWidth: true
                        text: qsTr("Did it get that right?")
                        color: Theme.text
                        font.pixelSize: Theme.fLabel
                        font.bold: true
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.sm
                        SecondaryButton {
                            Layout.fillWidth: true
                            text: qsTr("✓ Yes")
                            onClicked: bridge.recordTest(true)
                        }
                        SecondaryButton {
                            Layout.fillWidth: true
                            text: qsTr("✗ No")
                            onClicked: { bridge.recordTest(false); root.scrollTo(teachCard) }
                        }
                    }
                }
            }

            // The reward for a correct answer on something it never studied.
            Label {
                Layout.fillWidth: true
                visible: bridge.coachingEnabled && !bridge.awaitingVerdict
                         && bridge.testsPassed > 0 && bridge.testsFailed === 0
                text: qsTr("✓ Right again — that's %1 new one%2 it got correct.")
                      .arg(bridge.testsPassed).arg(bridge.testsPassed === 1 ? "" : "s")
                color: Theme.success
                font.pixelSize: Theme.fBody
                wrapMode: Text.WordWrap
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: Theme.sm
                PrimaryButton {
                    Layout.fillWidth: true
                    text: root.expert ? qsTr("Forward pass") : qsTr("Run it")
                    onClicked: bridge.forwardCurrent()
                }
                SecondaryButton { text: qsTr("Clear"); onClicked: bridge.clearInput() }
                SecondaryButton { text: qsTr("Random"); onClicked: bridge.randomizeInput() }
            }

            Disclosure {
                label: root.expert ? qsTr("Stepped forward pass")
                                   : qsTr("Watch it step through, layer by layer")
                expanded: root.advanced
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.sm
                    SecondaryButton {
                        Layout.fillWidth: true
                        text: bridge.playing ? qsTr("⏸ Pause") : qsTr("▶ Play")
                        onClicked: bridge.playing ? bridge.pause() : bridge.playCurrent()
                    }
                    SecondaryButton { text: qsTr("⧐ Step"); onClicked: bridge.stepOnce() }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.sm
                    Label { text: qsTr("Slow"); color: Theme.textFaint; font.pixelSize: Theme.fTiny }
                    Slider {
                        Layout.fillWidth: true
                        from: 16; to: 2000
                        value: 2016 - bridge.speedMs
                        onMoved: bridge.speedMs = 2016 - value
                    }
                    Label { text: qsTr("Fast"); color: Theme.textFaint; font.pixelSize: Theme.fTiny }
                }
            }
        }

        // ────────────────────────────── ② TEACH IT ───────────────────────────
        Card {
            id: teachCard
            step: root.expert ? 0 : 2
            title: root.expert ? qsTr("Dataset") : qsTr("Teach it")
            subtitle: bridge.outputKind === "class"
                      ? qsTr("Set an input above, then click the answer it should have given.")
                      : qsTr("Set an input above, dial in the answer it should have given, then add it.")
            active: !root.expert && root.activeStep === 1

            // One click = one training example. This used to be three levels deep behind
            // a dropdown; now the answers themselves are the buttons.
            Flow {
                Layout.fillWidth: true
                visible: bridge.outputKind === "class"
                spacing: Theme.xs
                Repeater {
                    model: bridge.outputLabels
                    delegate: SecondaryButton {
                        required property int index
                        required property var modelData
                        text: modelData
                        implicitWidth: Math.max(44, implicitContentWidth + Theme.lg)
                        onClicked: {
                            bridge.addExample(bridge.input, index)
                            teachCard.lastAdded = modelData
                            teachCard.unsaved += 1
                        }
                    }
                }
            }

            property string lastAdded: ""
            property int unsaved: 0
            property int rev: 0     // bumped on dataset changes so thumbnails refresh

            Connections {
                target: bridge
                function onDatasetChanged() { teachCard.rev += 1 }
            }

            // Press the digit instead of clicking it — the fast path once you're entering
            // a lot of examples. (Same idiom the data manager already uses.) The wrapper is
            // zero-height on purpose: these are non-visual, so they must not take layout rows.
            Item {
                Layout.preferredHeight: 0
                Layout.preferredWidth: 0
                Repeater {
                    model: bridge.outputKind === "class" ? Math.min(10, bridge.outputLabels.length) : 0
                    delegate: Item {
                        required property int index
                        Shortcut {
                            sequence: String(index)
                            enabled: !bridge.training
                            onActivated: {
                                bridge.addExample(bridge.input, index)
                                teachCard.lastAdded = bridge.outputLabels[index]
                                teachCard.unsaved += 1
                            }
                        }
                    }
                }
            }

            // Value-output models get sliders instead of answer buttons.
            ColumnLayout {
                Layout.fillWidth: true
                visible: bridge.outputKind !== "class"
                spacing: Theme.xs
                Repeater {
                    model: bridge.outputKind !== "class" ? bridge.outputDim : 0
                    delegate: RowLayout {
                        id: tRow
                        required property int index
                        Layout.fillWidth: true
                        Label {
                            text: (bridge.outputLabels[tRow.index] !== undefined)
                                  ? bridge.outputLabels[tRow.index] : ("y" + tRow.index)
                            color: Theme.textMuted
                            font.pixelSize: Theme.fBody
                            Layout.preferredWidth: 70
                            elide: Text.ElideRight
                        }
                        Slider {
                            Layout.fillWidth: true
                            from: 0; to: 1; stepSize: 0.05
                            value: teachCard.targets[tRow.index] !== undefined
                                   ? teachCard.targets[tRow.index] : 0
                            onMoved: {
                                var t = teachCard.targets.slice()
                                while (t.length < bridge.outputDim) t.push(0)
                                t[tRow.index] = value
                                teachCard.targets = t
                            }
                        }
                    }
                }
                PrimaryButton {
                    Layout.fillWidth: true
                    text: qsTr("+ Add this example")
                    onClicked: {
                        bridge.addExampleRaw(bridge.input, teachCard.targets)
                        teachCard.lastAdded = qsTr("example")
                        teachCard.unsaved += 1
                    }
                }
            }
            property var targets: []

            // The most recent examples, newest first — instant proof the click landed,
            // and a chance to spot a mislabelled one straight away.
            Flow {
                Layout.fillWidth: true
                visible: bridge.datasetSize > 0
                spacing: Theme.xs
                Repeater {
                    model: Math.min(6, bridge.datasetSize)
                    delegate: Rectangle {
                        id: thumb
                        required property int index
                        readonly property int exIndex: bridge.datasetSize - 1 - index
                        readonly property bool isSegments: bridge.inputLayout === "segments"
                        width: 50
                        height: 58
                        radius: Theme.rSm
                        color: Theme.inset
                        border.width: 1
                        border.color: index === 0 ? Theme.success : Theme.borderSoft

                        Column {
                            anchors.centerIn: parent
                            spacing: 2
                            // Same thumbnail dispatch the data manager uses: a real
                            // seven-segment glyph, or a pixel strip for grids and sliders.
                            SegmentDisplay {
                                visible: thumb.isSegments
                                anchors.horizontalCenter: parent.horizontalCenter
                                interactive: false
                                size: 20
                                values: { teachCard.rev; return bridge.exampleInput(thumb.exIndex) }
                            }
                            PixelGrid {
                                visible: !thumb.isSegments
                                anchors.horizontalCenter: parent.horizontalCenter
                                interactive: false
                                rows: bridge.inputLayout === "grid" ? bridge.inputRows : 1
                                cols: bridge.inputLayout === "grid" ? bridge.inputCols : bridge.inputDim
                                cell: 6
                                gap: 1
                                values: { teachCard.rev; return bridge.exampleInput(thumb.exIndex) }
                            }
                            Label {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: { teachCard.rev; return bridge.exampleLabel(thumb.exIndex) }
                                color: Theme.textMuted
                                font.pixelSize: Theme.fTiny
                                width: 46
                                elide: Text.ElideRight
                                horizontalAlignment: Text.AlignHCenter
                            }
                        }
                    }
                }
            }

            // What it knows, and an easy way out of a mistake.
            RowLayout {
                Layout.fillWidth: true
                spacing: Theme.sm
                Label {
                    Layout.fillWidth: true
                    text: bridge.datasetSize === 0
                          ? qsTr("Nothing learned from yet.")
                          : qsTr("Knows %1 example%2").arg(bridge.datasetSize)
                                                      .arg(bridge.datasetSize === 1 ? "" : "s")
                            + (teachCard.lastAdded !== ""
                               ? qsTr(" · added “%1”").arg(teachCard.lastAdded) : "")
                    color: teachCard.lastAdded !== "" ? Theme.success : Theme.textMuted
                    font.pixelSize: Theme.fBody
                    elide: Text.ElideRight
                }
                SecondaryButton {
                    text: qsTr("Undo")
                    visible: teachCard.unsaved > 0 && bridge.datasetSize > 0
                    onClicked: {
                        bridge.removeExample(bridge.datasetSize - 1)
                        teachCard.unsaved = Math.max(0, teachCard.unsaved - 1)
                        teachCard.lastAdded = ""
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: Theme.sm
                SecondaryButton {
                    Layout.fillWidth: true
                    text: teachCard.unsaved > 0 ? qsTr("Save %1 new to file").arg(teachCard.unsaved)
                                                : qsTr("Saved")
                    enabled: teachCard.unsaved > 0
                    onClicked: { bridge.saveDataset(); teachCard.unsaved = 0 }
                }
                SecondaryButton {
                    text: qsTr("Manage all…")
                    onClicked: root.manageDataRequested()
                }
            }
        }

        // ────────────────────────────── ③ TRAIN IT ───────────────────────────
        Card {
            step: root.expert ? 0 : 3
            title: root.expert ? qsTr("Training") : qsTr("Train it")
            subtitle: qsTr("Let it learn from the examples until it gets them right.")
            active: !root.expert && root.activeStep === 2

            ColumnLayout {
                Layout.fillWidth: true
                visible: !bridge.hasDataset
                spacing: Theme.sm
                Label {
                    Layout.fillWidth: true
                    text: qsTr("Teach it at least one example first — there's nothing to learn from yet.")
                    color: Theme.textMuted
                    font.pixelSize: Theme.fBody
                    wrapMode: Text.WordWrap
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                visible: bridge.hasDataset
                spacing: Theme.md

                // With the coach on this is one button that picks the settings, watches the
                // numbers, adjusts and stops at the right moment. Switch it off and you drive.
                PrimaryButton {
                    Layout.fillWidth: true
                    text: bridge.training ? qsTr("⏸ Stop")
                                          : (bridge.autoTrainEnabled
                                             ? (root.expert ? qsTr("▶ Auto-train") : qsTr("▶ Train it for me"))
                                             : qsTr("▶ Train"))
                    tone: bridge.training ? Theme.warn : Theme.accent
                    onClicked: {
                        if (bridge.training) {
                            if (bridge.autoTrainEnabled) bridge.autoTrainStop()
                            else bridge.trainStop()
                        } else {
                            if (bridge.autoTrainEnabled) bridge.autoTrainStart()
                            else bridge.trainStart()
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.sm
                    Label {
                        Layout.fillWidth: true
                        text: root.expert ? qsTr("Auto-train")
                             : (bridge.autoTrainEnabled ? qsTr("Coach picks the settings and knows when to stop")
                                                        : qsTr("Manual — you set everything and decide when to stop"))
                        color: Theme.textFaint
                        font.pixelSize: Theme.fTiny
                        wrapMode: Text.WordWrap
                    }
                    Switch {
                        checked: bridge.autoTrainEnabled
                        enabled: !bridge.training
                        onToggled: bridge.autoTrainEnabled = checked
                    }
                }

                // What the coach is doing and why — this is where a beginner learns. It's
                // coaching, so it goes quiet on the expert bench (the log below still records it).
                Label {
                    Layout.fillWidth: true
                    visible: bridge.coachingEnabled && bridge.autoTrainEnabled
                             && bridge.autoStatus !== "" && (bridge.training || bridge.epoch > 0)
                    text: bridge.autoStatus
                    color: bridge.training ? Theme.text : Theme.success
                    font.pixelSize: Theme.fBody
                    wrapMode: Text.WordWrap
                }

                // When training ends, say what happened AND give somewhere to go. "It stopped
                // improving" is a dead end on its own when the score is still poor.
                Rectangle {
                    Layout.fillWidth: true
                    visible: !bridge.training && bridge.autoTrainEnabled
                             && bridge.coachingEnabled && bridge.autoOutcome !== ""
                    implicitHeight: adviceCol.implicitHeight + Theme.md * 2
                    radius: Theme.rSm
                    color: Theme.inset
                    border.width: 1
                    border.color: bridge.autoOutcome === "converged" ? Theme.success : Theme.warn

                    // Anchor to the TOP, not verticalCenter: the parent's height is derived
                    // from this column, so centring inside it is a binding loop that
                    // collapses the block to nothing.
                    ColumnLayout {
                        id: adviceCol
                        anchors.top: parent.top
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: Theme.md
                        spacing: Theme.sm

                        Label {
                            Layout.fillWidth: true
                            text: root.outcomeHeadline()
                            color: bridge.autoOutcome === "converged" ? Theme.success : Theme.warn
                            font.pixelSize: Theme.fLabel
                            font.bold: true
                            wrapMode: Text.WordWrap
                        }
                        Label {
                            Layout.fillWidth: true
                            text: root.outcomeAdvice()
                            color: Theme.text
                            font.pixelSize: Theme.fBody
                            wrapMode: Text.WordWrap
                        }

                        // Concrete next steps, not just a diagnosis.
                        Flow {
                            Layout.fillWidth: true
                            spacing: Theme.xs
                            SecondaryButton {
                                visible: bridge.autoOutcome !== "diverged"
                                text: qsTr("Teach it more")
                                onClicked: root.scrollTo(teachCard)
                            }
                            SecondaryButton {
                                visible: bridge.autoOutcome === "stalled"
                                         && bridge.trainAccuracy <= 0.9
                                text: qsTr("Give it a bigger brain")
                                onClicked: { bridge.growNetwork(); bridge.autoTrainStart() }
                            }
                            SecondaryButton {
                                visible: bridge.autoOutcome !== "converged"
                                text: qsTr("Train again")
                                onClicked: bridge.autoTrainStart()
                            }
                            SecondaryButton {
                                visible: bridge.autoOutcome === "diverged"
                                text: qsTr("Start over")
                                onClicked: bridge.resetWeights()
                            }
                        }
                    }
                }

                Label {
                    Layout.fillWidth: true
                    text: bridge.training
                          ? qsTr("epoch %1 · error %2").arg(bridge.epoch)
                                                       .arg(bridge.currentLoss.toFixed(4))
                          : (bridge.epoch > 0
                             ? qsTr("Trained for %1 epochs · error %2").arg(bridge.epoch)
                                                                       .arg(bridge.currentLoss.toFixed(4))
                             : qsTr("Not trained yet."))
                    color: Theme.textMuted
                    font.pixelSize: Theme.fSmall
                    wrapMode: Text.WordWrap
                }

                // How well it's doing, in the terms a beginner cares about.
                RowLayout {
                    Layout.fillWidth: true
                    visible: bridge.outputKind === "class" && bridge.epoch > 0
                    spacing: Theme.sm
                    Label {
                        text: qsTr("Gets %1% right").arg(Math.round(bridge.trainAccuracy * 100))
                        color: Theme.success
                        font.pixelSize: Theme.fLabel
                        font.bold: true
                    }
                    Label {
                        visible: bridge.validationOn
                        text: qsTr("· %1% on examples it never saw")
                              .arg(Math.round(bridge.valAccuracy * 100))
                        color: bridge.valAccuracy < bridge.trainAccuracy - 0.2 ? Theme.warn
                                                                               : Theme.textMuted
                        font.pixelSize: Theme.fBody
                    }
                    Item { Layout.fillWidth: true }
                    InfoTip {
                        visible: bridge.validationOn
                        term: qsTr("Held-out examples")
                        explanation: qsTr("Some examples are kept aside and never used for training. "
                                        + "If it does well on the ones it studied but badly on these, "
                                        + "it memorized the answers instead of learning the pattern. "
                                        + "That's called overfitting.")
                    }
                }

                LossChart { Layout.fillWidth: true }

                // Everything the coach decided, in order — the record of its reasoning.
                Disclosure {
                    label: qsTr("What the coach did (%1)").arg(bridge.autoLog.length)
                    visible: bridge.autoLog.length > 0
                    Repeater {
                        model: bridge.autoLog
                        delegate: Label {
                            required property var modelData
                            Layout.fillWidth: true
                            text: "· " + modelData
                            color: Theme.textMuted
                            font.pixelSize: Theme.fSmall
                            wrapMode: Text.WordWrap
                        }
                    }
                }

                Disclosure {
                    label: qsTr("Training options")
                    expanded: root.advanced || !bridge.autoTrainEnabled

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.sm
                        Label { text: qsTr("learn rate"); color: Theme.textMuted; font.pixelSize: Theme.fSmall }
                        Slider {
                            Layout.fillWidth: true
                            from: 0.01; to: 1.0; stepSize: 0.01
                            value: bridge.learningRate
                            onMoved: bridge.learningRate = value
                        }
                        Label {
                            text: bridge.learningRate.toFixed(2)
                            color: Theme.text; font.pixelSize: Theme.fSmall
                            Layout.preferredWidth: 30
                            horizontalAlignment: Text.AlignRight
                        }
                        InfoTip {
                            term: qsTr("Learning rate")
                            explanation: qsTr("How big a step it takes each time it corrects itself. "
                                            + "Too small and it crawls; too large and it overshoots and "
                                            + "never settles.")
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.sm
                        Label { text: qsTr("optimizer"); color: Theme.textMuted; font.pixelSize: Theme.fSmall }
                        ComboBox {
                            id: optimizerCombo
                            Layout.fillWidth: true
                            model: bridge.optimizerNames
                            Component.onCompleted: currentIndex = Math.max(0, model.indexOf(bridge.optimizer))
                            Connections {
                                target: bridge
                                function onOptimizerNamesChanged() {
                                    optimizerCombo.currentIndex =
                                        Math.max(0, bridge.optimizerNames.indexOf(bridge.optimizer))
                                }
                            }
                            onActivated: bridge.optimizer = currentText
                        }
                        InfoTip {
                            term: qsTr("Optimizer")
                            explanation: qsTr("The rule for how each weight moves once we know which "
                                            + "way is downhill. “sgd” steps straight down; “momentum” "
                                            + "builds up speed in a consistent direction; “adam” gives "
                                            + "every weight its own step size.")
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.sm
                        Label { text: qsTr("hold out"); color: Theme.textMuted; font.pixelSize: Theme.fSmall }
                        Slider {
                            Layout.fillWidth: true
                            from: 0.0; to: 0.5; stepSize: 0.05
                            value: bridge.valFraction
                            onMoved: bridge.valFraction = value
                        }
                        Label {
                            text: bridge.validationOn ? Math.round(bridge.valFraction * 100) + "%"
                                                      : qsTr("off")
                            color: bridge.validationOn ? Theme.warn : Theme.textFaint
                            font.pixelSize: Theme.fSmall
                            Layout.preferredWidth: 30
                            horizontalAlignment: Text.AlignRight
                        }
                        InfoTip {
                            term: qsTr("Holding data back")
                            explanation: qsTr("Keeps a slice of your examples out of training so you can "
                                            + "check whether it really learned the idea or just memorized "
                                            + "the answers it was shown.")
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.sm
                        Label { text: qsTr("compute"); color: Theme.textMuted; font.pixelSize: Theme.fSmall }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: bridge.gpuTraining ? qsTr("GPU (full batch)") : qsTr("CPU (one at a time)")
                            color: Theme.text; font.pixelSize: Theme.fSmall
                        }
                        Switch {
                            checked: bridge.gpuTraining
                            onToggled: bridge.gpuTraining = checked
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.sm
                        SecondaryButton { Layout.fillWidth: true; text: qsTr("+1 epoch"); onClicked: bridge.trainEpoch() }
                        SecondaryButton { Layout.fillWidth: true; text: qsTr("Start over"); onClicked: bridge.resetWeights() }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.sm
                        SecondaryButton {
                            Layout.fillWidth: true
                            text: qsTr("💾 Save what it learned")
                            enabled: bridge.epoch > 0
                            onClicked: bridge.saveWeights()
                        }
                        SecondaryButton {
                            Layout.fillWidth: true
                            text: qsTr("↺ Load saved")
                            enabled: bridge.hasSavedWeights
                            onClicked: bridge.loadWeights()
                        }
                    }
                }
            }
        }

        // ─────────────────────────── ④ UNDERSTAND IT ─────────────────────────
        Card {
            step: root.expert ? 0 : 4
            title: root.expert ? qsTr("Inspect") : qsTr("Understand it")
            subtitle: qsTr("Slow everything down and watch the machinery work.")
            active: !root.expert && root.activeStep === 3

            Label {
                Layout.fillWidth: true
                text: qsTr("Click any neuron in the diagram to see exactly how it got its number.")
                color: Theme.textMuted
                font.pixelSize: Theme.fSmall
                wrapMode: Text.WordWrap
            }

            // Pick which example to study, and put it in the input so you can see it.
            RowLayout {
                Layout.fillWidth: true
                spacing: Theme.xs
                visible: bridge.datasetSize > 0
                SecondaryButton {
                    text: "◀"
                    implicitWidth: 32
                    onClicked: root.exampleIndex =
                        (root.exampleIndex + bridge.datasetSize - 1) % bridge.datasetSize
                }
                Label {
                    Layout.fillWidth: true
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("example %1 of %2 → %3").arg(root.exampleIndex + 1)
                              .arg(bridge.datasetSize).arg(bridge.exampleLabel(root.exampleIndex))
                    color: Theme.textMuted
                    font.pixelSize: Theme.fSmall
                    elide: Text.ElideRight
                }
                SecondaryButton {
                    text: "▶"
                    implicitWidth: 32
                    onClicked: root.exampleIndex = (root.exampleIndex + 1) % bridge.datasetSize
                }
                SecondaryButton {
                    text: qsTr("Show")
                    onClicked: bridge.loadExample(root.exampleIndex)
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: Theme.sm
                visible: bridge.datasetSize > 0
                SecondaryButton {
                    Layout.fillWidth: true
                    text: bridge.playing ? qsTr("⏸ Pause") : qsTr("▶ Watch it learn")
                    onClicked: bridge.playing ? bridge.pause()
                                              : bridge.playLearnExample(root.exampleIndex)
                }
                SecondaryButton {
                    text: qsTr("⧐ Step")
                    onClicked: bridge.stepLearnExample(root.exampleIndex)
                }
            }
            Label {
                Layout.fillWidth: true
                visible: bridge.datasetSize > 0
                text: qsTr("Runs one learning step in slow motion: the signal goes forward, the error "
                         + "is measured, then the correction flows backward (magenta) and the weights move.")
                color: Theme.textFaint
                font.pixelSize: Theme.fTiny
                wrapMode: Text.WordWrap
            }

            Disclosure {
                label: qsTr("Layer statistics and histograms")
                expanded: root.advanced
                StatsPanel { Layout.fillWidth: true }
            }
        }

        // ───────────────────── ADVANCED ONLY · the model itself ──────────────
        Card {
            visible: root.advanced
            title: root.expert ? qsTr("Architecture") : qsTr("The model")
            subtitle: qsTr("Change the architecture, or write new maths for it.")

            RowLayout {
                Layout.fillWidth: true
                spacing: Theme.sm
                Label { text: qsTr("Input method"); color: Theme.textMuted; font.pixelSize: Theme.fSmall; Layout.fillWidth: true }
                ComboBox {
                    id: inputMethodCombo
                    Layout.preferredWidth: 140
                    model: [qsTr("Sliders"), qsTr("Pixel grid"), qsTr("7-segment")]
                    function layoutIndex() {
                        return bridge.inputLayout === "grid" ? 1
                             : (bridge.inputLayout === "segments" ? 2 : 0)
                    }
                    Component.onCompleted: currentIndex = layoutIndex()
                    Connections {
                        target: bridge
                        function onBlueprintChanged() {
                            inputMethodCombo.currentIndex = inputMethodCombo.layoutIndex()
                        }
                    }
                    onActivated: {
                        if (currentIndex === 1)
                            bridge.setInputLayout("grid", Math.max(2, bridge.inputRows),
                                                          Math.max(2, bridge.inputCols))
                        else if (currentIndex === 2) bridge.setInputLayout("segments", 1, 7)
                        else bridge.setInputLayout("labels", 1, 0)
                    }
                }
            }

            RowLayout {
                visible: bridge.inputLayout === "grid"
                Layout.fillWidth: true
                Label { text: qsTr("Grid size"); color: Theme.textMuted; font.pixelSize: Theme.fSmall; Layout.fillWidth: true }
                SpinBox {
                    from: 1; to: 16
                    value: bridge.inputRows
                    onValueModified: bridge.setInputLayout("grid", value, bridge.inputCols)
                }
                Label { text: "×"; color: Theme.textMuted }
                SpinBox {
                    from: 1; to: 16
                    value: bridge.inputCols
                    onValueModified: bridge.setInputLayout("grid", bridge.inputRows, value)
                }
            }
            RowLayout {
                visible: bridge.inputLayout === "labels"
                Layout.fillWidth: true
                Label { text: qsTr("Inputs"); color: Theme.textMuted; font.pixelSize: Theme.fSmall; Layout.fillWidth: true }
                SpinBox { from: 1; to: 64; value: bridge.inputDim; onValueModified: bridge.setInputDim(value) }
            }

            Repeater {
                model: bridge.layers
                delegate: RowLayout {
                    id: layerRow
                    required property var modelData
                    Layout.fillWidth: true
                    spacing: Theme.xs
                    Label {
                        text: layerRow.modelData.name
                        color: Theme.accent
                        font.pixelSize: Theme.fSmall
                        Layout.preferredWidth: 24
                    }
                    SpinBox {
                        from: 1; to: 128
                        value: layerRow.modelData.units
                        Layout.preferredWidth: 90
                        onValueModified: bridge.setLayerUnits(layerRow.modelData.index, value)
                    }
                    ComboBox {
                        Layout.fillWidth: true
                        model: bridge.activationNames
                        currentIndex: bridge.activationNames.indexOf(layerRow.modelData.activation)
                        onActivated: bridge.setLayerActivation(layerRow.modelData.index, currentText)
                    }
                    SecondaryButton {
                        text: "✕"
                        implicitWidth: 30
                        onClicked: bridge.removeLayer(layerRow.modelData.index)
                    }
                }
            }
            SecondaryButton {
                Layout.fillWidth: true
                text: qsTr("+ Add layer")
                onClicked: bridge.addLayer(4, "tanh")
            }

            Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: Theme.borderSoft }

            SecondaryButton {
                Layout.fillWidth: true
                text: qsTr("⚙ Write a custom activation in C++…")
                onClicked: root.codeLabRequested()
            }
            RowLayout {
                Layout.fillWidth: true
                spacing: Theme.sm
                SecondaryButton {
                    Layout.fillWidth: true
                    text: qsTr("💾 Save as blueprint…")
                    onClicked: root.saveBlueprintRequested()
                }
                SecondaryButton {
                    Layout.fillWidth: true
                    visible: bridge.isBuiltIn
                    text: qsTr("↺ Restore default")
                    onClicked: root.restoreDefaultRequested()
                }
            }
        }

        Item { Layout.fillHeight: true; implicitHeight: Theme.lg }
    }
}
