pragma Singleton

import QtQuick

// The single source of truth for how Synapse-AI looks. Every colour, gap, radius and
// text size comes from here, so the whole app restyles coherently from one file instead
// of the hex codes that used to be scattered across nine QML files.
QtObject {
    // ── surfaces ─────────────────────────────────────────────────────────────
    readonly property color bg:          "#0b0f16"   // app background / canvas
    readonly property color surface:     "#12181f"   // panels
    readonly property color card:        "#161d26"   // raised cards
    readonly property color cardHover:   "#1b2430"
    readonly property color inset:       "#0b0f14"   // wells: editors, charts, consoles
    readonly property color border:      "#283040"
    readonly property color borderSoft:  "#1e2530"

    // ── text ─────────────────────────────────────────────────────────────────
    readonly property color text:      "#e6edf3"
    readonly property color textMuted: "#8b98a9"
    readonly property color textFaint: "#6b7686"

    // ── meaning ──────────────────────────────────────────────────────────────
    // Kept consistent with the network view: amber = positive, blue = negative.
    readonly property color accent:    "#4c9aff"   // primary actions, links
    readonly property color accentDim: "#2b5f9e"
    readonly property color positive:  "#f2a33c"   // positive activation / validation curve
    readonly property color negative:  "#4c8dff"   // negative activation
    readonly property color success:   "#3fb950"
    readonly property color warn:      "#f2a33c"
    readonly property color danger:    "#f85149"
    readonly property color gradient:  "#d267ff"   // backprop
    readonly property color select:    "#39c5cf"   // selection ring

    // ── spacing (4-pt scale) ─────────────────────────────────────────────────
    readonly property int xs: 4
    readonly property int sm: 8
    readonly property int md: 12
    readonly property int lg: 16
    readonly property int xl: 24

    // ── radius ───────────────────────────────────────────────────────────────
    readonly property int rSm: 6
    readonly property int rMd: 10
    readonly property int rLg: 14
    readonly property int rPill: 999

    // ── type scale ───────────────────────────────────────────────────────────
    readonly property int fTiny:  10
    readonly property int fSmall: 11
    readonly property int fBody:  12
    readonly property int fLabel: 13
    readonly property int fTitle: 15
    readonly property int fLarge: 20
    readonly property int fHuge:  30

    readonly property string mono: "monospace"

    // Standard motion — short and unfussy.
    readonly property int quick: 110
    readonly property int calm: 180
}
