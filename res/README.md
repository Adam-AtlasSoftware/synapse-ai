# Synapse AI — brand asset package

The mark: a single path that begins as a thin line and grows as it travels the
circle — the learning journey — turning amber in its final stretch and arriving
at the spark: full AI understanding.

## Colors
- Background gradient: #060a12 -> #101c30
- Journey (blue gradient): #26496e -> #4076b2
- Breakthrough (amber): #c07e2e
- Spark (gold): #e6b64a

## Contents
- svg/ — resolution-independent masters. Edit these, re-export everything else.
  - icon-dark-rounded.svg — primary app icon (rounded tile, transparent corners)
  - icon-dark-fullbleed.svg — square, for platforms that apply their own mask (iOS)
  - icon-dark-maskable.svg — mark shrunk to the 80% safe zone (Android adaptive)
  - icon-light-rounded.svg — light-mode variant
  - mark-color.svg / mark-mono-white.svg / mark-mono-navy.svg — standalone marks, no tile
  - hero-1600x900.svg, og-1200x630.svg — wordmark artwork
- png/app-icon/ — rounded icon at 16–1024 px
- png/ios/ — full-bleed squares: 1024 (App Store), 180/120 (iPhone), 167/152 (iPad)
- png/android/ — legacy densities 48–512 plus maskable 192/512
- png/favicon/ — favicon.ico (16/32/48 bundled) and individual PNGs
- branding/ — 2048 px large artwork, light variant, standalone marks
  (white mono for dark surfaces, navy mono for print/light), hero and
  Open Graph social image

## Usage notes
- Use mark-mono-white inside the app UI (e.g., loading screens) so the mark
  sits directly on your existing dark navy without a tile-in-tile look.
- The maskable PNGs are for Android adaptive icons and PWA manifests
  ("purpose": "maskable").
- og-1200x630.png is sized for link previews (Open Graph / Twitter cards).
