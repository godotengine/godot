# Goline — Planned Architecture

> **Status: PLANNED — NOT IMPLEMENTED.** This document describes the intended
> target architecture for Goline. None of the layers below are implemented
> yet, and no upstream Godot functionality is modified until a roadmap stage
> authorizes it.

The architecture organizes Goline into layered, clearly separated areas that
build on top of the unmodified Godot Engine rather than replacing it.

---

## 1. Godot Engine

The upstream Godot Engine core — engine, renderer, editor, scene system,
scripting, and build system. This is the **base layer** and is preserved.

- Upstream Godot functionality is retained unless a specific, documented
  change is intentional.
- Goline does not rewrite the engine's architecture.
- Additions integrate through clear seams, keeping upstream compatibility in
  mind.

## 2. Goline Editor Layer

The editor-facing surface where Goline-specific UI and workflows appear.

- Editor extensions/plugins that add Goline capability inside the Godot
  editor.
- Goline panels, docks, and toolbars for AI-assisted workflows.
- Opt-in; upstream editor behavior remains intact.

## 3. AI Agent Integration Layer

The core abstraction that connects the engine/editor to AI capabilities.

- Defines the interfaces and data flow for AI-driven operations.
- Centralized services: prompting, context assembly, result handling.
- Model-agnostic where practical (local and remote backends).

## 4. AI CLI Adapters

Adapters that connect external AI CLI tools (such as **OpenCode**) to Goline.

- Bridges between the Goline layer and external agent CLIs.
- Maps engine/project context and commands in both directions.
- Abstracts CLI-specific details behind a common interface.

*Status: CLI foundation.* `goline/cli/providers.py` implements a
`ProviderDriver` SPI (port of T3 Code's provider model) with OpenCode and
Claude drivers, a normalized event vocabulary, and a `--handover` command.

## 5. Project Context System

Provides AI agents with accurate, scoped awareness of the project.

- Collects project, file, and structural context for the AI.
- Indexes / queries the project to answer "what is here" reliably.
- Scopes context to the current task to keep prompts accurate.

## 6. Code/Script Assistance

AI-assisted code and script generation for the engine and GDScript/C#.

- Generate and refactor code with human review and validation.
- Suggestions that integrate with the editor workflow.
- Follows the AI development rules (small, reviewable, behavioral).

## 7. AI Debugging

AI-assisted debugging support.

- Help analyze build failures, runtime errors, and regressions.
- Surface diagnostics and candidate explanations to the developer.
- Preserves test integrity; never removes functionality to pass a build.

## 8. Tool/Command Execution

Infrastructure for AI agents to act on the repository.

- Controlled execution of build/test/analysis commands.
- Sandboxed and governed by the security layer below.
- Commands are invoked only within defined permissions.

*Status: CLI foundation.* Goline shells out via `goline/cli/goline_cli.py`
(discovery + launch); the permission gate is `goline/cli/policy.py`.

## 9. Security and Permission Controls

The governance layer for all agent actions.

- Explicit permission model for tool/command execution.
- Restriction of destructive or out-of-scope operations.
- Audit trail and review of agent-driven changes.

*Status: CLI foundation.* `Policy.classify()` gives allow/deny/error with a
reason (default deny-destructive); `AuditLog` is an append-only JSONL trail.
Gating is currently an inspection step (`--gate`) and is not yet enforced
inline inside the agent launch prompt.

---

This architecture is presented as a plan. Implementation proceeds stage by
stage as described in `docs/goline/ROADMAP.md`.
