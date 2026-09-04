# Goline — CLI Agent Integration (Stage 2)

> **Status:** DESIGN + foundation implemented (additive, no upstream Godot
> edits). The agent-agnostic orchestration layer and the discovery mechanism
> are real and testable. The in-editor UI surface is deferred until an editor
> build exists (see "Editor surface" below).

## Why "external CLIs"

Goline's AI strategy is **agent-agnostic external CLIs**, not AI living inside
the engine. The AI agents (OpenCode, Claude Code, codex, ...) run as ordinary
developer tools on the repo; Goline discovers, invokes, and (later) surfaces
them. This matches `ARCHITECTURE.md` §3/§4 and requires no heavy in-engine C++
AI systems.

Two consequences:

1. **No deep C++ work is required** for the AI story. The engine can be rebuilt
   only when smoke-testing an editor build.
2. **Agent-agnostic by design** — nothing here is tied to one CLI. Adapters
   normalize each CLI's quirks behind a common interface.

## Target architecture

```
   Goline editor UI (future)          Goline CLI tools             Developer shell
   ┌──────────────────────┐     ┌─────────────────────────┐    ┌──────────────────┐
   │ docks / toolbars     │     │  cli/ (discover, run,   │    │  "claude"        │
   │ (deferred)           │ ──► │  context, gate)         │ ──►│  "opencode"      │
   └──────────────────────┘     └─────────────────────────┘    │  "codex" ...     │
                                        │                      └──────────────────┘
                                        ▼
                              Agent-agnostic contract:
                              discovery / launch / capability /
                              scoped-context / permission gate
```

The seam is a small, dependency-free module under `goline/cli/` that:

1. **Discovers** installed CLI agents on `PATH` (`opencode`, `claude`, `codex`,
   `gemini`, `aichat`, ...).
2. **Reports** each CLI's identity, version, and rough capability (generic,
   natural-language, or file-editing agent) via a `--help`/`--version` sniff.
3. **Launches** a chosen CLI against the repo working directory.
4. **Scopes context** (Stage 7) and **gates permissions** (Stage 8) later.

This is deliberately the "AI CLI Adapters" (§4) and "Tool/Command Execution"
(§8) layers of the architecture, delivered now as plain repo tooling.

## Discovery contract

A CLI is a "Goline agent" if it satisfies these checks in order:

1. **On PATH**: `command -v` / `where` finds an executable of that name.
2. **Responds**: running `<cli> --version` (or `--help` for CLIs that lack a
   version flag) exits 0 and prints something.
3. **Versioned**: a plausible version string is parsed for display.

The adapter table maps each known CLI name to:
- `version_flag` — the flag to probe (`--version`, or `--help`).
- `label` — human name.
- `kind` — `file-edit` (can modify files, e.g. Claude Code, OpenCode) vs
  `chat` (conversational only).

Unknown CLIs are reported as "generic" rather than refused, so the system stays
open.

## Adapter table (v1)

| Name      | Version flag | Kind       | Notes |
|-----------|--------------|------------|-------|
| `opencode`| `--version`  | file-edit  | Primary target (roadmap Stage 4) |
| `claude`  | `--version`  | file-edit  | Claude Code |
| `codex`   | `--version`  | file-edit  | OpenAI Codex |
| `gemini`  | `--version`  | file-edit  | Gemini CLI |
| `aichat`  | `--version`  | chat       | Generic chat backend |

## Launch semantics

- Runs the CLI in the **repo root** (`C:\Users\USER\Documents\Goline\godot`)
  as the working directory, so the agent sees the whole tree.
- Passes through `stdin/stdout/stderr` so the CLI is fully interactive.
- Never mutates the repo or the CLI on its own — the agent (and the human)
  own all file changes.
- Extensible via `GOLINE_AGENT` env var to override which CLI is used.

## Repository context (grounded packs)

`GOLINE.md` at the repo root is the agent handoff document. On top of that,
Goline now assembles **grounded context packs** (`goline/cli/context.py`) so
agents work from accurate, scoped state rather than guesses:

- **Engine pack** (`--context engine`): repo root, git branch/commit/clean
  state, engine markers, tools detected on PATH, key dirs present, the
  `version.py` identity (name/short_name/major/minor/patch), and Goline handoff
  docs.
- **Game pack** (`--context game --project <dir>`): requires `project.godot`;
  surfaces project name, `config_version`, main scene, and a bounded (≤60
  files, ≤4 deep) listing of `.gd`/`.cs`/`.tscn`/`.tres` files.

`--context <kind>` writes the pack to a temp file and launches the chosen agent
with a prompt pointing at it (plus any `--` prompt you supply).
`--print-context <kind>` prints the pack without launching an agent or creating
a temp file — useful for review and tests.

## Editor surface (deferred)

The `docs/goline/ARCHITECTURE.md` "Goline Editor Layer" (§2) implies docks and
toolbars inside the editor. That requires compiling an editor build with a
Godot module / EditorPlugin. That is **not** delivered here — it needs the C++
toolchain. Until then, Goline's CLI integration is exercised from the
**developer shell** (the primary model anyway) and will later get an optional
in-editor launcher.

## Security posture

- Discovery and launch are read-only against `PATH` and subprocesses.
- `--version` sniffing runs with no working-dir writes.
- The helper refuses to shell out to anything not discovered via the adapter
  table unless explicitly allowed.
- Destructive operations are never automated here; real agent actions follow
  the `docs/goline/AI_DEVELOPMENT.md` rules and the Stage 8 permission layer.

## Permission & audit gate (ARCHITECTURE §8/§9)

`goline/cli/policy.py` is a pure (never-executes) decision layer:

- **Classification** — `Policy.classify(command)` returns `allow`/`deny`/
  `error` with a reason. Default posture is **deny-destructive**:
  - **Denied:** file/filesystem destruction (`rm`/`del`/`rd`/`rmdir`/`shred`/
    `dd`/`Remove-Item`/`clear-*`/`format`/`wipefs`), `sudo`, dangerous `git`
    (`reset`/`clean`/`rebase`/`merge`/`prune`/`gc`, `checkout --`, `rm`,
    `stash drop`/`clear`, `branch`/`tag -d`, and *any* force push including
    `--force`/`-f`/`+refspec`), shell redirections, remote-code install
    pipelines (`curl|wget|iwr … | sh|bash|python|node|iex`), system/power
    mutation (`shutdown`/`reboot`/`init 0`/`Stop-Computer`), process killing
    (`kill -9`/`taskkill /f`/`Stop-Process -Force`), low-level storage tools
    (`fdisk`/`parted`/`mkfs`/`cryptsetup`), registry deletes, and any
    **unrecognized executable** (default-deny).
  - **Allowed:** read-only `git`, the known toolchain (`python`, `node`,
    `scons`, `cl`/`g++`, ...), and known agent CLIs.
  - Callers can add `custom_allow` / `custom_deny` regexes, or set `deny_all`.
- **Audit log** — `AuditLog` is **append-only**: each decision is a JSONL line
  with a UTC timestamp, decision, reason, and command. A write failure never
  crashes the caller.

CLI: `--gate "command"` classifies a command (does **not** run it) and returns
0 on allow / 1 on deny; `--audit <path>` appends the decision to a JSONL file.

## T3 Code patterns (adopted via Option A port)

We studied **T3 Code** (`pingdotgg/t3code` — an agent "harness control surface"
that wraps Codex/Claude/Cursor/Grok/OpenCode behind a WebSocket server + web/
desktop/mobile clients). It is a large Node 24 + Effect + pnpm + vite-plus
monorepo; we deliberately did **not** vendor it. Instead we ported its clean
ideas into `goline/cli/providers.py`:

- **ProviderDriver SPI** — a common `ProviderDriver` interface
  (`driver_kind` / `display_name` / `dispatch`) mirroring T3's
  `ProviderDriver`/`ProviderInstance` model, reduced to a one-shot dispatch.
- **Normalized event vocabulary** — a small discriminated-union of
  `ProviderEvent`s (`content.delta`, `reasoning`, `prompt.recorded`,
  `session.updated`, `permission`, `error`, `done`), inspired by T3's
  `ProviderRuntimeEventV2`.
- **Drivers** (native headless CLI modes, no hand-rolled HTTP SDK):
  - `OpenCodeDriver` → `opencode run --format json [--file <ctx>] [--model <m>] <prompt>`
    (matches T3's `opencode` provider; `--file` attaches the grounded context).
  - `ClaudeDriver` → `claude -p` (headless print mode) for parity.
- **`handover` command** — our lightweight port of `t3code handover`: resolve
  the workspace/git root, assemble a grounded context pack, write it to a temp
  file, and dispatch the first prompt to a provider, streaming normalized
  events. Usage:

  ```
  python goline/cli/goline_cli.py --handover --provider opencode \
      --context engine --model opencode/<model> -- "your prompt"
  python goline/cli/goline_cli.py --handover --provider claude \
      --context game --project <game> -- "help with player.gd"
  ```

  > **Auth note (real-world):** the standalone `opencode` CLI reports **0
  > credentials** via `opencode providers list`, **and yet its free-tier models
  > (verified live: `opencode/nemotron-3.5-lightning-free`, etc. from
  > `opencode models`) work with no login** — a real handover was run against
  > one successfully. Paid/model-specific providers will additionally need
  > `opencode providers login`. Verified quirks: the executable resolves to a
  > PowerShell `.ps1` wrapper (handled by `providers._executable_argv`), and
  > `opencode run` requires the prompt **before** `--file` (a trailing
  > positional after `--file` is misread as a file path, not inline text).

## Verified on the dev machine

- **OpenCode CLI `1.18.27`** installed globally (`npm i -g --allow-scripts=opencode-ai opencode-ai`; the npm package's postinstall script must be allowed so the correct Windows `bin/opencode.exe` is fetched — the default blocked-script warning leaves a non-Windows stub otherwise).
- **Claude Code `2.1.220`** installed globally (npm).
- `python goline/cli/goline_cli.py --self-test` discovers both as `file-edit` agents.
- **Live-validated end-to-end**: an engine handover
  (`--handover --context engine`) correctly answered "Goline" from `version.py`,
  and a game handover (`--handover --context game --project
  goline/examples/sample_game`) read the real game files from the context pack
  and returned the correct answers (`player.gd extends CharacterBody2D`; Player
  node at `Vector2(576, 324)`). Both streamed normalized events and exited 0.

> The user's ARE you using OpenCode now is the **desktop app** (Electron,
> `%LOCALAPPDATA%\Programs\@opencode-aidesktop\OpenCode.exe`), which is a GUI
> and NOT a headless CLI — it cannot be shelled out to by Goline. Goline
> automation requires the standalone CLI (installed above), which is a distinct
> binary from the desktop app.

## Usage

```
python goline/cli/goline_cli.py --list          # list discovered agents
python goline/cli/goline_cli.py --agent claude -- "<args>"         # launch one
python goline/cli/goline_cli.py --self-test     # verify discovery on this machine
python goline/cli/goline_cli.py --print-context engine             # show engine pack
python goline/cli/goline_cli.py --context engine -- "<prompt>"     # engine agent
python goline/cli/goline_cli.py --context game --project <game> -- "<prompt>"
python goline/cli/goline_cli.py --gate "git status"                # allow (exit 0)
python goline/cli/goline_cli.py --gate "rm -rf /tmp" --audit a.jsonl   # deny (exit 1)
python goline/cli/goline_cli.py --handover --provider opencode \
    --context engine --model opencode/<model> -- "prompt"          # t3-style handover
python goline/cli/goline_cli.py --handover --provider opencode \
    --model opencode/<model> --audit handover.jsonl -- "prompt"    # + audit gate
```

## Tests

```
python -m unittest discover -s goline/cli/tests -v   # pure, no network
```
