# Goline

Goline is a **fork of the Godot Engine** focused on making game development
more **AI-assisted**.

## What Goline is

- A fork of the Godot Engine (currently 4.8.0-dev / `master`).
- An engine and editor workflow that integrates **AI assistance** — AI-driven
  coding/script help, debugging support, and project/context awareness.
- A codebase where **AI CLI agents** such as **OpenCode** are a first-class
  part of the development workflow.

## Guiding principles

1. **Preserve Godot.** Existing Godot functionality is preserved unless a
   change is intentional and documented.
2. **Isolate Goline code.** Goline-specific code lives under `goline/` and
   `docs/goline/` so it can never be confused with upstream Godot code.
3. **Additive and compatible.** Changes are additive and consider upstream
   Godot compatibility wherever practical.
4. **Small, reviewable changes.** Build capability incrementally instead of
   large rewrites.

## Repository layout

```
GOLINE.md            Project charter (purpose, principles, constraints)
goline/              Goline-specific code and infrastructure
docs/goline/         Goline documentation
  ARCHITECTURE.md    Planned architecture (not yet implemented)
  ROADMAP.md         Staged development plan
  AI_DEVELOPMENT.md  Rules for AI coding agents
```

All upstream Godot code remains in its original location and is modified only
when a change is intentional. See `GOLINE.md` and `docs/goline/*` for details.
