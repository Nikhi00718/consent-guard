---
name: ConsentGuard
description: A darkroom light table for covering the private parts of one photo.
colors:
  marker: "#ff6a3d"
  marker-hover: "#ff8360"
  marker-press: "#ef5a2d"
  marker-ink: "#1d0b04"
  marker-soft: "rgb(255 106 61 / 0.14)"
  table: "#141312"
  rebate: "#0c0b0a"
  light-box: "#0e0d0c"
  tray: "#1c1a18"
  tray-raised: "#25221f"
  tray-hover: "#2f2c28"
  line: "#3a3631"
  line-soft: "#292623"
  paper: "#f1ebe0"
  text: "#eee8de"
  text-2: "#b5ada1"
  text-3: "#8f887d"
  cover-ink: "#000000"
  safelight: "#e9b851"
  safelight-text: "#f0dcaa"
  fixer: "#8fca9c"
  stop: "#f07189"
  stop-text: "#f7c4ce"
typography:
  display:
    fontFamily: "Archivo Variable, Segoe UI Variable Text, Segoe UI, system-ui, sans-serif"
    fontSize: "3rem"
    fontWeight: 720
    lineHeight: 1.02
    letterSpacing: "-0.028em"
    fontVariation: "'wdth' 86"
  headline:
    fontFamily: "Archivo Variable, Segoe UI Variable Text, Segoe UI, system-ui, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.02em"
    fontVariation: "'wdth' 88"
  title:
    fontFamily: "Archivo Variable, Segoe UI Variable Text, Segoe UI, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 650
    lineHeight: 1.4
  body:
    fontFamily: "Archivo Variable, Segoe UI Variable Text, Segoe UI, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Archivo Variable, Segoe UI Variable Text, Segoe UI, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 600
    letterSpacing: "0.08em"
    fontVariation: "'wdth' 78"
  data:
    fontFamily: "ui-monospace, Cascadia Mono, Consolas, monospace"
    fontSize: "0.92em"
    fontFeature: "'tnum' 1"
rounded:
  sm: "4px"
  md: "6px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  2xl: "32px"
  3xl: "64px"
components:
  button-primary:
    backgroundColor: "{colors.marker}"
    textColor: "{colors.marker-ink}"
    rounded: "{rounded.sm}"
    padding: "12px 20px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.marker-hover}"
  button-primary-active:
    backgroundColor: "{colors.marker-press}"
  button-primary-disabled:
    backgroundColor: "{colors.tray-raised}"
    textColor: "{colors.text-3}"
  button-secondary:
    backgroundColor: "{colors.tray-raised}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "12px 20px"
    height: "48px"
  button-secondary-hover:
    backgroundColor: "{colors.tray-hover}"
  chip-on:
    backgroundColor: "{colors.tray}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "5px 12px 5px 10px"
    height: "34px"
  chip-off:
    textColor: "{colors.text-3}"
    rounded: "{rounded.sm}"
    padding: "5px 12px 5px 10px"
    height: "34px"
  tool:
    textColor: "{colors.text-2}"
    rounded: "{rounded.sm}"
    size: "36px"
  tool-active:
    backgroundColor: "{colors.marker-soft}"
    textColor: "{colors.marker}"
  panel:
    backgroundColor: "{colors.tray}"
    rounded: "{rounded.md}"
    padding: "16px 18px 18px"
  input:
    backgroundColor: "{colors.light-box}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "40px"
  rebate:
    backgroundColor: "{colors.rebate}"
    textColor: "{colors.text-3}"
    height: "56px"
---

# Design System: ConsentGuard

## Overview

**Creative North Star: "The Darkroom Light Table"**

ConsentGuard is a light table for one photo. The owner lays a print on a warm charcoal table, marks what must be hidden the way a photographer marks a contact sheet with a china marker, inspects the edges through a loupe, and leaves with a finished print. The interface is photo-editor familiar: standard controls, one task per screen, nothing that competes with the picture.

The table is dark because the subject is a photograph, and a neutral surround keeps its colour honest. It is warm-neutral, not blue-black. Chrome recedes into tonal layers of the same charcoal. Colour is spent almost entirely on one china-marker red-orange, which means "marked" everywhere it appears. Density is moderate: generous around the photo, compact and legible in the side panels.

The system rejects the navy dashboard with neon glow, eyebrow labels over headings, stat cards, and decorative texture or gradients. Motion is limited to state changes plus one authored moment, the marker scan line that crosses the photo while detectors run.

**Key Characteristics:**
- Warm-neutral charcoal table, near-black film-rebate header, paper-white prints.
- One accent, china-marker red-orange, reserved for marks, the current step and the primary action.
- The cover is true black with marker hatching: what you edit is what you save.
- Archivo on its width axis is the only UI face; semi-condensed for headings, condensed caps for film-edge labels.
- Status speaks in darkroom lights: safelight amber warns, fixer green clears, stop red fails.

## Colors

One saturated accent over a tonal ladder of warm charcoals, with three quiet status hues.

### Primary
- **China Marker** (#ff6a3d): primary action fill, the current step, active tool, checked chips, crop corners, the brush ring, and the hatching on the cover. Hover lightens to #ff8360, press deepens to #ef5a2d. Text on it is Marker Ink (#1d0b04, 6.7:1).

### Neutral
- **Table** (#141312): the page ground.
- **Rebate** (#0c0b0a): the header strip, the darkest band, like the unexposed edge of film.
- **Light Box** (#0e0d0c): behind photos (easel, canvas, print table) and inside form fields.
- **Tray** (#1c1a18): side panels. **Tray Raised** (#25221f) for secondary buttons and keys; **Tray Hover** (#2f2c28) for their hover and the selected segment.
- **Line** (#3a3631) and **Line Soft** (#292623): 1px borders and dividers. There are no other borders.
- **Paper** (#f1ebe0): headings, the brand, and the white border of a finished print.
- **Text** (#eee8de), **Text 2** (#b5ada1), **Text 3** (#8f887d): primary, secondary and tertiary text. Text 3 holds 4.9:1 on Tray and is the floor; never go dimmer for readable text.
- **Cover Ink** (#000000): the redaction itself, drawn at 88% while editing so a trace of the photo shows through, and at 100% in the saved file.

### Status
- **Safelight** (#e9b851, text #f0dcaa): warnings, "hit and miss" detector types, the look-before-you-share reminder.
- **Fixer** (#8fca9c): checks that came back clear, the "ready" verdict.
- **Stop** (#f07189, text #f7c4ce): errors and failed checks.

### Named Rules
**The One Marker Rule.** Red-orange means "marked" or "do this next". It never decorates, never tints a background larger than a tool button, and never appears on two competing buttons at once.

**The Honest Table Rule.** Grounds stay warm-neutral charcoal. No blue-black, no gradients, no noise texture behind the photo.

## Typography

**Display and UI Font:** Archivo Variable (with Segoe UI Variable Text, Segoe UI, system-ui)
**Data Font:** the system monospace (ui-monospace, Cascadia Mono, Consolas)

**Character:** one grotesque family doing every job, stretched narrower as the voice gets louder: semi-condensed and heavy for headings, condensed tracked caps for film-edge labels, normal width for reading. Monospace appears only for measurements, sizes, zoom levels, hashes and keys.

### Hierarchy
- **Display** (720, 3rem, 1.02, wdth 86, -0.028em): the one headline on the upload screen. Its second sentence drops to Text 2 at weight 480. Steps down to 2.5rem under 1100px and 2.25rem under 560px.
- **Headline** (700, 1.75rem, 1.15, wdth 88): the review and result screen titles; the scanning title uses 2rem.
- **Title** (650, 0.9375rem): panel titles, list item names, check names. The verdict title is 1.25rem.
- **Body** (400, 0.875rem, 1.5): panel copy and controls. The intro paragraph is 1.0625rem at 1.6 with a 42ch measure.
- **Label** (600, 0.8125rem, wdth 78, +0.08em, uppercase): film-edge step labels and check status words only.
- **Minimum size** is 0.75rem, used only for keys, status words, reason codes and the canvas readout.

### Named Rules
**The Stretch, Don't Swap Rule.** Hierarchy comes from width, weight and size inside Archivo. Never add a second display family.

**The Mono Means Measured Rule.** Monospace is for numbers you might compare or copy. Never for labels, headings or mood.

## Layout

A centred workspace up to 1360px wide with 24px side padding (16px under 860px). The upload screen is a 5/12 and 7/12 split: intro and "looks for" options on the left, the easel and its actions on the right. Review and result screens are a flexible main column beside a 360px inspector (320px under 1100px). The photo column stays in view while the inspector scrolls.

Spacing follows 4, 8, 12, 16, 24, 32 and 64px. Related items sit 8 to 12px apart; panels are 12px apart; major regions are 24px or more apart. Photo frames size from the photo's own aspect ratio and cap at the viewport height minus the chrome, so there is no dead letterbox.

Breakpoints are set in em so the layout follows the browser text size: 68.75em, 53.75em and 35em (1100, 860 and 560px at default size). At 53.75em everything stacks to one column (intro, easel, options; heading, editor, inspector, save bar), the save bar becomes sticky at the bottom, and step labels collapse to numbers except the current one. At 35em the header splits into two rows, actions stack, and the toolbar wraps with the brush size on its own row. On touch screens every control grows to at least 44px.

## Elevation & Depth

Flat by default. Depth comes from the tonal ladder: Rebate, Light Box, Table, Tray, Tray Raised, each a step lighter, separated by 1px lines. One shadow exists, and it belongs to the finished print.

### Shadow Vocabulary
- **Print** (`box-shadow: 0 2px 3px rgb(0 0 0 / 0.4), 0 22px 44px -18px rgb(0 0 0 / 0.85)`): only under the saved photo on the result screen, so the output reads as a physical print lying on the table.

### Named Rules
**The Only the Print Casts a Shadow Rule.** Panels, buttons and chips stay flat. No glows, no coloured halos, no offset shadows.

## Shapes

Small, nearly square corners: 4px on controls, chips, keys and fields; 6px on panels, the editor and the easel. The print itself is square-cornered, like trimmed photo paper. The recurring silhouettes are photographic: L-shaped crop corners marking the easel, a triangular film-edge arrow before the current step, a circular loupe with a paper rim, and a round brush ring the true size of the brush.

## Components

### Buttons
Confident, square-shouldered, no ornament.
- **Shape:** 4px corners, 48px tall, 12px by 20px padding, weight 650.
- **Primary:** China Marker fill with Marker Ink text. One per screen: Erase and save, Save my version, Download.
- **Hover / Active:** lighter marker on hover, deeper marker and a 1px drop on press, 160ms ease-out. Disabled turns to Tray Raised with Text 3, never a faded marker.
- **Secondary:** Tray Raised with a Line border and Text. **Quiet:** borderless Text 2 with an icon, for Start over, Replace and Delete.

### Chips
- **Style:** 34px tall, 4px corners, a 10px square mark before the label.
- **State:** on is Tray with a Line border and a filled marker square; off is transparent with a Line Soft border, Text 3 and an empty square. Weak detector types carry a safelight warning icon.

### Panels
- **Corner Style:** 6px. **Background:** Tray with a Line Soft border. **Shadow:** none. **Padding:** 16px by 18px.

### Inputs / Fields
- **Style:** Light Box fill, Line border, 4px corners, 40px tall.
- **Focus:** Marker border plus a 3px Marker Soft ring. **Error:** Stop border, message in a Stop-tinted box below.

### Navigation
The film rebate: a 56px Rebate strip with the brand at left, four frame steps in the centre, Start over at right. Steps are condensed caps with a number; done steps show a check in Text 2, the current step is Marker with a film-edge arrow, upcoming steps are Text 3.

### The Easel (signature)
The upload target is a dark Light Box field held by four L-shaped marker crop corners. Empty, it invites a drop, a paste or a click; while dragging, the corners close in by 16px. Loaded, it shows the photo itself and the corners fall back to Line.

### The Cover Editor (signature)
Photo on Light Box. Covered areas are Cover Ink with a 12px china-marker hatch that stays the same screen size at any zoom. The brush shows a marker ring at true size (a dashed paper ring for the eraser); the loupe is a 176px paper-rimmed circle magnifying photo and cover together. Toolbar: Cover and Found views, brush, eraser, loupe, move, undo, redo, discard, size, zoom. Standard shortcuts: B, E, L, H, Space to pan, [ and ], Ctrl+Z.

### The Print (signature)
The saved photo is shown with an 8px Paper border and the Print shadow, entering with a 420ms fade-up. The checks list beside it gives every check a plain name, a one-sentence result and a condensed status word coloured by Fixer, Safelight or Stop.

## Do's and Don'ts

### Do:
- **Do** keep China Marker (#ff6a3d) for marks, the current step and the single primary action on a screen.
- **Do** show the cover as black with marker hatching wherever the user edits it, so the preview matches the saved file.
- **Do** write every status, check and warning as a plain sentence; keep system codes out of personal mode.
- **Do** put limits where the decision happens: the look-before-you-share reminder sits beside Download.
- **Do** keep readable text at Text 3 (#8f887d) or brighter and at least 0.75rem.

### Don't:
- **Don't** put small tracked capitals above headings; the heading speaks for itself.
- **Don't** add glows, gradients, glass, noise textures or coloured halos.
- **Don't** use a coloured left or right border thicker than 1px on alerts, panels or list items.
- **Don't** use monospace for anything that is not a measurement, size, hash or key.
- **Don't** use a teal or blue accent; the previous navy-and-teal look is retired.
