# Goline — Identity & Branding (Stage 1)

This document records Goline's chosen identity and maps it onto the places in
the Godot Engine source where identity/branding is currently handled. It is a
**plan and a reference** — nothing here is implemented yet, and no upstream
Godot file is modified until an explicit change is approved.

## The brand

- **Short name / identifier:** `goline`
- **Display name:** `Goline`
- **Relationship:** Goline is a fork of the **Godot Engine**
  (https://godotengine.org). Godot's attribution and MIT license are
  preserved; Goline does not claim Godot's logo as its own.
- **Website:** not yet assigned (placeholder).

The canonical values live in `goline/branding/goline_branding.hpp` as
`GOLINE_*` constants.

## Where Godot holds its identity (reference)

Below is every place Godot 4.8-dev stores or renders its own name, so a later
re-brand can be comprehensive.

| # | Location | What it controls | Fallback |
|---|---|---|---|
| 1 | `version.py` (+ SCons `name=`/`short_name=`/`website=` override) | Engine/editor display name, short name, website → `version_generated.gen.h` | Build-time args |
| 2 | `core/core_builders.py` (`version_info_builder`) | Generates `GODOT_VERSION_*` defines from `version.py` | — |
| 3 | `core/version.h` | `GODOT_VERSION_FULL_NAME`, `GODOT_VERSION_FULL_BUILD` macros | from generated header |
| 4 | `main/main.cpp` (`:380-388`) | Console header: name + version + website | reads `GODOT_VERSION_NAME`/`WEBSITE` |
| 5 | `editor/editor_node.cpp` `_update_title()` (`:395`) | Editor window title `"... - <NAME>"` | reads `GODOT_VERSION_NAME` |
| 6 | `editor/project_manager/project_manager.cpp` (`:120`) | Project Manager window title | reads `GODOT_VERSION_NAME` |
| 7 | `editor/docks/editor_dock_manager.cpp` (`:294`) | Dock window title `"%s - Godot Engine"` | **hardcoded** |
| 8 | `editor/gui/editor_about.cpp` (`:59-60`) | About title + "Godot Engine contributors" | **hardcoded** |
| 9 | `editor/gui/editor_version_button.cpp` (`:48`, `:86`) | About version string; "Copied Godot editor version." | reads version macros / hardcoded |
| 10 | `platform/windows/display_server_windows.cpp` (`:874-907`) | Windows app/window-class name prefix `"Godot."` | **hardcoded** |
| 11 | `core/core_builders.py` (`:32`) | `GODOT_VERSION_DOCS_URL` → `docs.godotengine.org` | hardcoded docs host |
| 12 | `main/splash.png`, `main/splash_editor.png` + `main_builders.py` | Boot splash images/colors (behind SCons `no_editor_splash`) | assets |

## Branding assets (sources of the logos)

- `misc/logo/icon{.svg,.png, _outlined.*}`, `logo{.svg,.png, _outlined.*}` — the upstream Godot logos.
- `editor/icons/Logo.svg`, `editor/icons/TitleBarLogo.svg` — used in the About dialog and editor window.
- `editor/icons/editor_icons_builders.py` — embeds icons into a generated header.
- Goline's own assets will live under `goline/assets/` (not yet populated).

## How a re-brand could be applied (later, opt-in)

1. **Lowest risk, no C++ edits:** pass `name="Goline"`, `short_name="goline"`,
   `website="..."` to the SCons build. This overrides `GODOT_VERSION_NAME` and
   cascades to items 4, 5, 6 and the About-button version (item 9) with a
   single, non-destructive mechanism.
2. **Hardcoded strings** (items 7, 8, 10, 11) still say "Godot" and need
   individual, explicit edits if a full re-brand is desired.
3. **Assets** (item 12, logo files) behind an explicit branding decision.

## Current status

- **Category A (additive, no Godot-edit risk):** DONE — `goline_branding.hpp`
  established as the source of truth; this document created; `goline/assets/`
  contains the chosen G+L monogram logo (`icon.svg`).
- **Category B (visible re-brand):** DONE for editor display identity —
  `version.py` `name = "Goline"` (cascades via `GODOT_VERSION_NAME` to window
  titles, console header, About version button), plus the hardcoded editor
  strings updated: dock + game-workspace window titles, About dialog
  (contributors / community / third-party intro), "About Goline" menu command
  and project-manager tooltip, "Copied Goline editor version", and Windows
  app-id/app-name prefixes.
  - Intentionally **not** changed: `short_name = "godot"` (keeps user data/
    config/cache dirs and binary naming stable), `website` and the docs URL
    (no Goline docs host yet; Godot docs remain the correct reference), and
    the `.po` translation files (out of scope here).
- **Category C (splash, packaging, official logo replacement):** deferred to
  later stages.
