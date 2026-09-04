# GOLINE — Project Charter

Goline is a **fork of the Godot Engine** whose purpose is to make game
development more **AI-assisted**.

This document is the canonical statement of Goline's purpose, principles, and
constraints. It applies to every human and every AI agent working on this
repository.

---

## 1. Goline is a fork of Godot Engine

Goline starts from the official Godot Engine source (currently 4.8.0-dev /
`master`). It is not a rewrite: it builds on top of Godot and aims to remain
compatible with upstream Godot wherever practical.

## 2. Purpose: AI-assisted game development

Goline's primary goal is to make game development more productive through AI
assistance — including AI-driven code/script assistance, debugging support,
project and context awareness, and agent-driven tooling — all integrated into
the Godot workflow.

## 3. AI agents/CLIs are a core part of development

AI coding agents and CLIs, such as **OpenCode**, are a first-class part of the
Goline development workflow. They are used to inspect, generate, debug, and
extend the codebase. To keep that safe and reliable, agents must follow the
rules in `docs/goline/AI_DEVELOPMENT.md`.

## 4. Preserve existing Godot functionality

Godot's existing functionality should be preserved. Goline changes are
additive and must not silently break, remove, or regress upstream behavior.
If a behavior change is intentional, it must be deliberate and documented.

## 5. Goline-specific changes are clearly identifiable

Goline-specific code and documentation live in clearly identifiable,
separated locations — primarily `goline/` and `docs/goline/`. Any
Goline-specific change to an existing Godot file must be marked clearly so it
can be isolated and reviewed.

## 6. Upstream Godot compatibility

Upstream Godot compatibility should be considered whenever practical.
Changes should be additive and backward-compatible, and should take into
account how they interact with Godot's build system, engine core, editor,
scene system, and renderer — even when those systems are not the focus of a
change.

---

See `docs/goline/ARCHITECTURE.md` for the planned architecture and
`docs/goline/ROADMAP.md` for the staged plan. No AI functionality is
implemented yet.
