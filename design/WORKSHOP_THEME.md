# The Workshop — this fork's editor identity

This fork of the Godot editor wears its own face. Instead of the stock dark‑grey
IDE with the electric‑blue accent, the editor opens as **a calm personal
workshop**: warm paper and dark ink, a single muted accent used *like a marker in
a notebook*, and **Lato** carrying the words. Mostly monochrome, with colour used
on purpose. The interface is stage crew — your work is the star.

There is a rendered preview of everything below — the actual editor chrome in both
themes, the palette, and the type — in **[`workshop-editor-identity.html`](workshop-editor-identity.html)**
(open it in any browser; it is self‑contained).

---

## What you get

Two signature colour presets ship with the editor, and **Workshop Paper is the
default** on first launch:

| Preset | Feel | Base | Accent | Contrast |
|--------|------|------|--------|----------|
| **Workshop Paper** *(default)* | Warm paper, dark ink, one ink‑teal marker | `#EAE7E0` · `Color(0.917, 0.905, 0.878)` | `#216B66` · `Color(0.13, 0.42, 0.40)` | −0.06 |
| **Workshop Ink** | Charcoal and ink after dark — never black‑with‑neon | `#22211F` · `Color(0.133, 0.129, 0.122)` | `#5CA899` · `Color(0.36, 0.66, 0.60)` | 0.26 |

Both use the same accent hue, so the identity is cohesive whether the lights are
on or low. If you turn on **Follow System Theme**, the OS light/dark preference now
maps to Workshop Paper / Workshop Ink instead of the stock themes.

Alongside the presets, three fork‑wide defaults changed:

- **Typeface → Lato.** The whole editor UI is set in Lato (regular + bold),
  replacing Inter. It is bundled into the engine, so it works with no setup.
- **Corners eased 4 → 3.** A slightly crisper, less “pill”‑shaped edge — the
  manifesto asks to avoid over‑rounding.
- **Icons calmed 2.0 → 1.0.** Icon saturation is halved so the toolbars read as
  quiet monochrome rather than candy. The interface stops competing with the work.

Everything here is a **default, not a lock** — palette, accent, corners, spacing
and font all remain fully editable in *Editor Settings*.

---

## Every choice traces to a line in the brief

> “A clean, human sans‑serif such as **Lato** as the base.”

→ The editor's UI font is now Lato, bundled and set as the built‑in default.

> “**One muted accent** colour at a time … like a marker in a notebook, not a carnival.”

→ A single restrained ink‑teal drives every selection, focus ring and active tab.
No rainbow of category colours.

> “Soft off‑white backgrounds … dark mode: **charcoal and ink, not pure black with neon.**”

→ Workshop Paper (warm paper) is the default; Workshop Ink is a charcoal room, not
black with neon.

> “Avoid **overuse of rounded** cards and pill buttons.”

→ Corner radius eased from 4 to 3.

> “**Content is the star; the interface is the stage crew.**”

→ Icon saturation dropped from 2.0 to 1.0 so the chrome recedes.

---

## Files changed

| File | Change |
|------|--------|
| `editor/themes/editor_fonts.cpp` | Editor UI font `Inter` → `Lato` (4 references: regular + bold, incl. MSDF). |
| `editor/themes/editor_theme_manager.cpp` | Added the `Workshop Ink` and `Workshop Paper` colour presets; pointed *Follow System Theme* at them. |
| `editor/themes/editor_theme_manager.h` | `default_corner_radius` 4 → 3. |
| `editor/settings/editor_settings.cpp` | Default preset → `Workshop Paper`; base/accent/contrast/icon‑saturation/corner defaults updated; both presets registered in the picker enum. |
| `thirdparty/fonts/Lato_Regular.ttf`, `Lato_Bold.ttf` | Bundled Lato (SIL Open Font License). |
| `thirdparty/fonts/LICENSE.Lato.txt` | Lato OFL licence text. |

The font swap works because the editor builds its font header from a glob over
`thirdparty/fonts/*.ttf`; the two new files become `_font_Lato_Regular` /
`_font_Lato_Bold` automatically, and `editor_fonts.cpp` points at them.

---

## Build it and wear it

```sh
# Build the editor for your platform (Festival module included).
scons platform=linuxbsd target=editor

# Run it — first launch already wears Workshop Paper.
bin/godot.linuxbsd.editor.x86_64
```

To switch themes, or to go back to a stock Godot look at any time:

```
Editor  ›  Editor Settings  ›  Interface  ›  Theme  ›  Color Preset
        →  Workshop Paper  /  Workshop Ink  (or any built‑in preset)
```

The two Workshop presets sit at the top of that list, and of the Project
Manager's quick theme picker.

---

## A note on scope

This is the identity of the **editor** — the tool you work inside. It is a visual
theme (palette, typeface, spacing, corners), so it changes how the editor *looks
and feels* without touching how it works: every panel, shortcut and workflow is
exactly where Godot puts it. The larger layout ideas in the brief — one dominant
action per screen, editorial page titles, generous empty states — belong to
whole‑screen redesigns rather than a theme, so they live in the preview as
direction, not in the editor's C++ yet.
