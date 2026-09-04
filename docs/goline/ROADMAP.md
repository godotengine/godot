# Goline — Roadmap

The staged plan for turning the Godot Engine fork into **Goline**, an
AI-assisted game development engine.

> **Status:** Stage 0 is complete. All later stages are planned but
> **NOT IMPLEMENTED**. No upstream Godot code is modified before a stage
> explicitly authorizes it.

---

## Stage 0 — Godot repository foundation ✅

*Implemented.* Clone the official Godot Engine repository, verify it, and
establish the Goline foundation (charter, roadmap, architecture, AI rules).
No engine code is modified; the foundation is documentation only.

## Stage 1 — Goline identity and branding ✅

*Implemented.* Re-branded the engine's visible identity to **Goline** across
`version.py` (display `name`), the editor display strings (dock/game-workspace
window titles, About dialog, "About Goline" command + tooltip, version-copy
toast), and the Windows app-id/app-name prefixes. Additive brand assets live in
`goline/branding/` and `goline/assets/` (`goline_branding.hpp`, the G+L
monogram `icon.svg`). `short_name = "godot"` and the docs URL are intentionally
unchanged (keep data dirs and working docs links). Full details and the
intentional exclusions are in `docs/goline/IDENTITY.md`.

## Stage 2 — Goline editor integration — PARTIAL

- **Done (external-CLI model):** the agent-agnostic **CLI orchestration seam** —
  `goline/cli/goline_cli.py` (discovery of installed AI CLIs on `PATH`,
  interactive launch against the repo root, Windows `.ps1`/`.cmd` wrapper
  handling) plus `docs/goline/CLI_INTEGRATION.md` and a pure test suite
  (`goline/cli/tests/`). No upstream Godot code is touched; nothing needs an
  engine build.
- **Deferred:** the in-editor dock/toolbar surface (ARCHITECTURE §2) requires
  compiling an editor build with an EditorPlugin and is deliberately not done
  until a C++ toolchain exists. The CLI layer is exercised from the developer
  shell — the primary model for Goline anyway.

## Stage 3 — AI integration architecture — PARTIAL

- **Done (external-CLI reality):** grounded **context packs** in
  `goline/cli/context.py` — engine pack (`--context engine`: repo root, git
  state, tools, key dirs, `version.py` identity) and game pack
  (`--context game --project <dir>`: reads `project.godot`, bounded file list).
  `goline_cli.py` gained `--context`/`--print-context`; a full offline test
  suite guards `goline/cli/` (`test_context.py` + `test_goline_cli.py`, 22
  tests). This implements the ARCHITECTURE §5 "Project Context System" for the
  CLI path.
- **Still to define:** safety boundaries and a permission/audit model for agent
  actions (ARCHITECTURE §8/§9); deeper architecture doc refresh to match the
  external-CLI-only decision.

## Stage 4 — OpenCode integration — PARTIAL

- **Done:** a **ProviderDriver SPI** and normalized event vocabulary in
  `goline/cli/providers.py`, with concrete **OpenCode** (`opencode run
  --format json`, grounded via `--file`) and **Claude** (`claude -p`) drivers,
  plus a **`handover`** command (`--handover`) that resolves the workspace
  root, assembles a context pack, and dispatches the first prompt while
  streaming   normalized events — a lightweight port of T3 Code's provider model
  and `t3code handover`. `opencode` CLI `1.18.27` installed; discovered as a
  `file-edit` agent. **Live-validated end-to-end**: a real handover with a
  grounded engine-context pack was dispatched to a free model
  (`opencode/nemotron-3.5-lightning-free`) and returned the correct verdict
  ("Goline") with normalized events (`step.start`/`content.delta`/`done`) and
  exit 0. No provider login was required for the free tier.
- **Done:** an **audit/gate hook on handover** — `--audit <file>` classifies any
  command-bearing events the agent emits through `goline.cli.policy.Policy()`,
  surfaces the ALLOW/DENY verdict in the transcript, and records it to the
  JSONL audit log (pure, append-only). Verified live: `--handover --audit`
  runs clean and writes the log.
- **Still ahead:** richer event surface (session/thread persistence); validate a
  real gameplayed handover (`--context game`) end-to-end.

## Stage 5 — AI-assisted coding — NOT IMPLEMENTED

- Add AI-assisted code/script generation assistance.
- Generate engine/editor code with human review, following the AI rules.

## Stage 6 — AI-assisted debugging — NOT IMPLEMENTED

- Add AI-assisted debugging support.
- Help diagnose build failures, runtime errors, and regressions with AI
  assistance while preserving test integrity.

## Stage 7 — Project/context awareness — NOT IMPLEMENTED

- Give AI agents awareness of the project, files, and surrounding context.
- Provide accurate, scoped context to improve AI assistance quality.

## Stage 8 — Agent tools and permissions — PARTIAL

- **Done (permission/audit gate, ARCHITECTURE §8/§9):** `goline/cli/policy.py` —
  pure command **classification** (default deny-destructive: `rm`/`del`/`rd`/
  `shred`/`dd`/`Remove-Item`/`format`/`wipefs`, `sudo`, dangerous `git` incl.
  force-push in every form and branch/tag/stash/rm/prune/gc, `curl|wget|… | sh`
  supply-chain pipelines, shutdown/reboot/init, `kill -9`/`taskkill /f`, storage
  tools `fdisk`/`parted`/`mkfs`, registry deletes, and unknown executables;
  allow read-only `git` + known toolchain) and an **append-only JSONL audit log**.
  Exposed as `--gate "command"` (never executes; exit 0 allow / 1 deny) with
  optional `--audit <path>`, and wired as a post-dispatch audit hook on
  `--handover`. Offline-tested (59 tests).
- **Still ahead:** enforcing the gate inside the launch path (agents holding
  the policy as an instruction), a richer rule file, and review/approval UX.

## Stage 9 — Testing, performance and polish — NOT IMPLEMENTED

- Harden testing and reliability.
- Measure and optimize Goline workflows; polish usability and readiness.
