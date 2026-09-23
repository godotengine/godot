# SlimeEngine — AI Creation System Implementation Plan

**Version:** 1.0  
**Prepared:** September 23, 2026  
**Implementation workspace:** `E:\SlimeEngine`  
**Audience:** One lead coding agent, with no more than two concurrent subagents.  
**Document status:** Implementation specification. No features or passing tests are implied by this document.

## 0. Mission and first assignment

Build SlimeEngine's native AI-assisted game-development foundation on the existing Godot fork. It must let a human, a language model, or a deterministic recipe inspect real project state, propose changes, preview their effects, apply authorized operations, inspect results, verify behavior, and recover from failures.

Do not build only a chat panel. Do not build a mouse-click automation layer. Do not start a renderer rewrite, a new scripting language, or a large multi-agent framework.

**First execution assignment: complete P00–P03.** This means a documented baseline, a real native subsystem, working inspection, and a tested preview/apply/undo transaction for native scene content. Deliver a visible, functioning result without needing an API key. The first handoff must contain implementation and test evidence, not another architecture document alone.

The full roadmap continues through P10. Work phase by phase; do not scaffold every later feature before making the first vertical slice work. At the first assignment boundary, report the implemented gate and preserve a precise next task in `NEXT_TASK.md`.

### 0.1 Ground truth and limitations

The remote `master` branch inspected while preparing this plan was at:

```text
ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f
```

At that revision, `version.py` identifies Godot 4.8.0-dev. The source files inspected include `editor/SCsub`, `editor/register_editor_types.cpp`, `editor/editor_undo_redo_manager.h`, `tests/SCsub`, and `tests/test_main.cpp`. [R1–R6]

The user's local drive was not inspected. **Local source, existing instructions, modifications, and actual build output are authoritative.** Record local HEAD; do not reset or downgrade it to the remote revision above. Do not claim remote inspection establishes the state of `E:\SlimeEngine`.

All new filenames, protocols, tool names, test names, and phase identifiers below are proposed implementation targets unless explicitly identified as existing Godot facilities.

### 0.2 Non-negotiable outcomes

- Ordinary editor functionality continues to work without the AI service or network access.
- Initial platform is Windows x86-64. Keep platform boundaries suitable for later Linux support, but do not claim Linux support without testing it.
- Engine/editor implementation remains C++. Provider adapters and orchestration use a small TypeScript service. Initial generated game code is GDScript.
- Scene and resource writes go through typed native commands, not arbitrary LLM-generated method calls.
- Every advertised write operation has validation, a preview, permission checks, conflict checks, tests, and a documented recovery contract.
- The exported game does not require the orchestration service, API credentials, or a model connection.
- The implementation remains usable with a deterministic fake provider for tests.
- New code follows local Godot conventions and preserves upstream licensing notices.

## 1. Agent operating rules

### 1.1 Distinguish the coding agent from the product's AI

The coding agent is implementing SlimeEngine and may use its authorized local development tools to edit source, invoke compilers, and run trusted tests. The future in-product AI is a separate actor that receives only the capabilities granted through SlimeEngine.

Do not confuse these authority levels. A restriction on the product's AI calling an arbitrary shell does not prohibit the coding agent from invoking SCons. Conversely, the coding agent's ability to edit engine code is not a capability to expose to the in-product AI.

### 1.2 Lead plus at most two subagents

The limit is **one lead and at most two simultaneously active subagents**, including nested descendants. Subagents must not spawn further agents. Reuse those two slots; do not create hidden reviewers or provider-hosted agent trees outside the limit. The product itself should use one serial lead workflow initially; implementing its own multi-agent scheduler is deferred.

| Role | Primary ownership | Restrictions |
|---|---|---|
| Lead | Contracts, permission model, phase gates, integration, native registration/build hooks, final verification | Sole integration authority; reviews evidence and shared-file changes |
| Subagent A | Native inspection/editing components and focused C++ tests | Changes only assigned paths; cannot independently alter shared protocol or engine registration |
| Subagent B | Service/transport tests, mock provider, harness, later real providers | Changes only assigned paths; cannot independently alter native command semantics or acceptance criteria |

Do not use both slots merely because they exist. The lead works serially where a task has no clean boundary.

### 1.3 Work coordination

Before delegation, write a task record containing its ID, objective, allowed paths, forbidden paths, dependency revision, acceptance checks, and expected return format.

Use separate worktrees only after checking disk space and local status. Suggested sibling paths are `E:\SlimeEngine-worktrees\native` and `E:\SlimeEngine-worktrees\service`; they are proposals, not paths assumed to exist. Never nest a worktree inside another worktree.

The engine checkout is large. Share immutable tools/caches only when safe; do not share writable SCons output directories across simultaneous builds. Permit one full engine build at a time unless resource measurements justify otherwise. TypeScript tests may run alongside that build.

When worktrees are impractical, use explicit file ownership in the existing checkout and serialize any overlapping writes. Do not allow two agents to edit the same file. The lead owns `editor/SCsub`, registration/lifecycle hooks, protocol schema changes, and common test configuration.

A subagent handoff must include:

```text
Task ID:
Base revision:
Files changed:
Behavior implemented:
Exact commands run:
Exit codes and test counts:
Evidence paths:
Known limitations:
Unresolved integration requirements:
Commit or patch reference:
```

The lead reruns integration checks after merging; a subagent's statement of success is not the gate.

### 1.4 Repository hygiene

Read applicable `AGENTS.md`, contribution instructions, and existing project documentation before editing. Do not replace an existing `AGENTS.md` with this plan.

Inspect the working tree before branch operations. Preserve unrelated edits. Do not automatically stash, discard, overwrite, or commit the user's pre-existing work. Never use destructive cleanup/reset commands as a shortcut. Stage named paths, not all files indiscriminately.

Create a feature branch when safe, or continue an appropriate existing branch. Use small local commits for independently passing changes. Do not push, publish releases, buy services, change system security settings, or install privileged components without the user's corresponding authorization.

No production credentials in commits, fixtures, screenshots, command arguments, diagnostics, or handoff reports. Do not enumerate secret environment values when checking the environment.

### 1.5 Persistent implementation state

Create or extend these files under `docs/slime_ai/` without replacing existing unrelated documentation:

| File | Purpose |
|---|---|
| `IMPLEMENTATION_PLAN.md` | Repository copy of this specification, with justified amendments |
| `STATUS.md` | Gate state, local baseline, last verified integration revision |
| `TASKS.md` | Task IDs, owners, dependencies, states, evidence pointers |
| `DECISIONS.md` | Short architectural decisions and rejected alternatives |
| `NEXT_TASK.md` | Exact resumable next assignment, including reproduction commands |
| `KNOWN_ISSUES.md` | Reproducible unresolved defects and limitations |
| `BASELINE.md` | Toolchain, build configuration, initial failures, source revision |
| `TEST_MATRIX.md` | Required checks and current evidence status |
| `SOURCE_MAP.md` | Confirmed local integration points and relevant symbols |

Task states: `not_started`, `in_progress`, `implemented_unverified`, `verified`, `blocked`, `deferred`.

Never mark a missing-key live provider test as passed. Never mark a zero-test filter as a passing suite. Never convert a failed assertion into a warning to advance a gate.

## 2. Scope and architecture

### 2.1 Build now

The initial implementation covers native editor integration; typed inspection and edit tools; change previews, approval, undo and recovery; provider-independent sessions; cloud API adapters; structured verification; deterministic content recipes; and a bounded autonomous prototype workflow.

### 2.2 Defer deliberately

Defer image-to-3D pipelines, voice interaction, marketplace integration, cloud account synchronization, runtime NPC language models, autonomous engine-source modification, multiplayer development, console/mobile certification, broad C# support, arbitrary shell access, general plugin installation, and parallel game-building agents. Do not silently remove these from the long-term vision; list them as deferred.

Use existing Godot capabilities rather than rewriting rendering, physics, animation, scene serialization, asset imports, or the editor's entire command system.

### 2.3 Components and ownership

```text
Native SlimeEngine editor UI
  -> session controller
  -> versioned local transport
  -> TypeScript orchestration/provider service
       -> model responses become untrusted action proposals
  -> native capability/permission gate
  -> native command dispatcher + revision checks + edit journal
  -> scene/resource/file operations
  -> isolated worker job submission where execution is required
  <- structured results, diffs, test evidence, artifact references
```

**Native C++ owns** editor state, live scene changes, capability enforcement, authoritative revision checks, application of approved changes, and the edit/recovery journal.

**The TypeScript service owns** provider protocols, streaming, task/session state, context assembly, budgeting, and a separate session event log. It must not bypass native policy by directly rewriting project files.

**The worker supervisor owns** execution environments, time/resource limits, process lifetime, immutable job inputs, and collection of untrusted worker outputs. The supervisor accepts a typed job, not an arbitrary shell string from a model.

**The trusted UI owns** approval events. Text saying “approved” in a model response or project file has no authority.

### 2.4 Proposed source layout

Create directories when their first real implementation is needed. Do not fill them with empty placeholders.

```text
E:\SlimeEngine\
  editor\slime_ai\
    SCsub
    slime_ai_editor_plugin.{h,cpp}
    slime_ai_session.{h,cpp}
    protocol\
    inspection\
    commands\
    transactions\
    permissions\
    ui\
  tools\slime_ai\
    protocol\                 # canonical schemas, fixtures, protocol documentation
    agent_service\            # TypeScript package, lockfile, src/, tests/
    harness\                  # integration runner and Windows launch/build helpers
    worker\                   # trusted worker/probe code and isolation configuration
    recipes\                  # only after P08 needs shared recipe specifications
  tests\editor\slime_ai\     # focused native .cpp tests following local conventions
  tests\projects\slime_ai\   # clean 2D/3D fixtures; copy before modifying during tests
  docs\slime_ai\
```

Keep generated logs, backups, binaries, API transcripts, worker outputs, and user-specific paths out of version control. Prefer an editor-owned local data directory outside the game project for journals/artifacts. Use a project identity and workspace identity in the directory structure; never key a project solely by its display name.

### 2.5 Existing integration points to inspect locally

`editor/SCsub` currently builds the editor through named subdirectories. Add a small, conditional `slime_ai/SCsub` entry rather than introducing editor dependencies into `core/`. `editor/register_editor_types.cpp` is a relevant type-registration location, but type registration and editor plugin instantiation are separate steps. Find the actual current lifecycle/plugin construction path locally and register/unregister cleanly. [R2, R3]

`EditorUndoRedoManager` supplies per-history action creation, references, do/undo operations, and history tracking. Reuse it; do not assume it supplies a durable cross-file transaction system. [R4]

The inspected tests use `.cpp` discovery and force-linking. Follow the current `TEST_FORCE_LINK()` convention; do not copy an obsolete header-only test-registration pattern. [R5, R6, S2]

Native editor changes must be compiled out of non-editor targets. Add a discoverable `slime_ai` build option if the baseline's conventions permit it, document its default, and test enabled/disabled builds. Do not assume the flag already exists.

## 3. Windows environment and build discipline

### 3.1 First commands: observation, not mutation

Run from the user's actual checkout:

```powershell
Set-Location -LiteralPath 'E:\SlimeEngine'
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
git worktree list
Get-Content -LiteralPath '.\version.py'
python --version
python -m SCons --version
node --version
npm --version
```

Missing optional tools are a recorded prerequisite, not a reason to overwrite the checkout or install unrelated software. Inspect the existing C++ compiler/Windows SDK and available memory/disk space without assuming hardware details from another machine. Use the user's configured compatible toolchain when available.

Godot's current Windows documentation describes MSVC or supported MinGW toolchains, Python, and SCons, and distinguishes editor/template targets. The exact local source and `python -m SCons -h` output remain authoritative. [S1]

### 3.2 Baseline build

An initial build command to adapt to the local environment is:

```powershell
python -m SCons platform=windows target=editor arch=x86_64 tests=yes dev_build=yes -j4
```

`-j4` is a conservative starting configuration, not a performance recommendation for unknown hardware. Choose and record a sustainable job count after checking RAM and CPU availability.

When the checkout lacks optional Direct3D 12 or ANGLE dependencies, an explicitly recorded reduced-dependency baseline may use `d3d12=no angle=no` after confirming both options locally. Do not disable unrelated features to hide SlimeEngine build failures. The documentation states that disabling D3D12 does not remove Vulkan/OpenGL. [S1]

Record the actual output executable path. Do not select an unrelated installed Godot binary or blindly choose the newest `.exe` in `bin/`. Keep compiler configuration, engine revision, and binary hash together in a build manifest.

### 3.3 Verification command templates

These templates require `$Editor` to be set to the verified current build. The harness must preserve nonzero exit codes, capture logs, enforce timeouts, and identify which checks actually ran.

```powershell
& $Editor --version
& $Editor --help
& $Editor --test --help
& $Editor --test --test-case='*[SlimeAI]*'
```

The last filter is meaningful only after matching tests exist; validate that the executed count is greater than zero. Keep baseline/upstream checks separate from the new SlimeAI filter. Doctest options after `--test` are supported by Godot's test runner. [S2, R6]

For trusted fixture projects, or inside an approved isolated worker:

```powershell
& $Editor --headless --path $Fixture --import
& $Editor --headless --path $Fixture --script 'res://scripts/example.gd' --check-only
& $Editor --headless --path $Fixture --quit-after 120
```

`--check-only` parses a specified script; it is not a whole-game test. `--quit-after` is a bounded smoke-run mechanism, not proof of correct gameplay. Importing starts editor-side activity and is not automatically safe for untrusted projects. [S3, S4]

Implement `build_editor.ps1`, `run_native_tests.ps1`, and later `run_integration.ps1` as real, documented wrappers. Do not include fake successful exit paths. A wrapper must record command arguments, source revision, exit status, timeout state, test count, and artifact location.

### 3.4 Reproducibility

Pin the service runtime/tooling and package lockfile after discovering a compatible local environment. Prefer a small dependency set and the runtime's built-in test facilities where sufficient. Keep dependency installation explicit, use lockfile-based installs, and review lifecycle scripts. Do not globally update every installed tool.

A baseline compiler failure must be diagnosed separately from new failures. An isolated pre-existing unrelated failure may be documented, but all targeted SlimeAI checks must pass and the relevant broader subset must show no new regressions.

## 4. Protocol and command contract

### 4.1 Local transport

Use private redirected process I/O for v1, not a network listener. The editor launches a fixed, user-configured or packaged service executable with a fixed argument list. The model cannot change that executable or its arguments.

Godot documents `OS.execute_with_pipe()`, including non-blocking pipe access and a separate stderr stream. It also documents that the child does not automatically terminate with Godot; implement lifetime handling rather than assuming it. Confirm the local native signature before use. [S5]

Specify UTF-8 newline-delimited JSON frames. Newlines inside strings must be escaped. Standard output carries protocol frames only; diagnostics go to stderr. Buffer partial frames, handle split multibyte characters, bound frame size, implement backpressure, and reject malformed or oversized messages without editor failure.

Start with a 1 MiB maximum control frame, 256 operations per patch, and paginated inspection. These are initial design limits, not measured performance guarantees. Large assets use opaque artifact IDs. The broker resolves those IDs inside its own artifact store; callers cannot smuggle arbitrary paths.

Use a handshake containing protocol version, service build, engine revision, workspace identity, supported command versions, and capabilities. Unsupported major versions fail with a visible compatibility error. Shutdown on EOF; add heartbeat/lifetime supervision and cancellation. Do not mutate engine objects from pipe-reader threads. [S6]

### 4.2 Public tool arguments versus trusted envelope

The model supplies tool arguments. The host constructs authoritative session/workspace/request metadata and binds the request to a user-controlled grant. An illustrative internal request is:

```json
{
  "protocol_version": "1.0",
  "request_id": "req-000184",
  "session_id": "session-07",
  "workspace_id": "workspace-a",
  "method": "scene_patch_preview",
  "params": {
    "scene_ref": "scene-a",
    "base_revision": "scene-rev-42",
    "operations": [
      {
        "op": "set_property",
        "node_ref": "node-player",
        "property": "position",
        "value": {"type": "Vector3", "value": [0.0, 1.0, 0.0]}
      }
    ]
  }
}
```

This is a proposed protocol example, not an already implemented command. Approval fields do not belong in model arguments. Unknown fields that imply authority must be rejected, not ignored into accidental behavior.

A result must include operation status, workspace/revision identity, relevant changeset/job ID, structured errors, and evidence references. Never collapse `applied`, `verified`, and `accepted` into one generic success value.

### 4.3 Initial tool set

Implement tools incrementally in this order:

| Phase | Tool names | Minimum behavior |
|---|---|---|
| P01 | `engine_capabilities`, `session_status` | Real versions, registered capabilities, service state |
| P02 | `project_inspect`, `scene_inspect`, `object_inspect`, `api_describe` | Structured, scoped, paginated, revision-bearing observations |
| P03 | `scene_patch_preview`, `changeset_apply`, `changeset_status`, `changeset_revert` | Native allowlisted changes, immutable preview, authorization and recovery |
| P04 | `project_search`, `code_read` | Exact file/symbol/context retrieval; no vector database required |
| P06 | `code_patch_preview`, `resource_patch_preview`, `asset_import_request` | Project-scoped changes with execution-risk classification |
| P07 | `job_start`, `job_status`, `job_cancel`, `test_run`, `viewport_capture`, `project_export` | Typed worker jobs and evidence-backed outcomes |
| P08 | `recipe_list`, `recipe_preview`, `recipe_apply` | Seeded generation and protected manual overrides |
| P09 | `task_status`, `evidence_query` | Task completion based on required evidence, not model assertion |

Only advertise implemented and enabled capabilities. Deferred operations must return `UNSUPPORTED_OPERATION`, not a placeholder success. Expose small task-relevant tool groups rather than all schemas on every request.

### 4.4 Data types and identifiers

Encode vectors, colors, transforms, resource references, node references, and large integers explicitly. Reject non-finite numbers and invalid types/ranges. Preserve integers that exceed JavaScript's safe numeric range with string-backed typed encodings.

Use session-local opaque node handles for v1. They are stable only within the relevant scene/session lifetime and must be invalidated after deletion/reload. Do not pretend an object pointer or NodePath is a permanent ID. Return a clear stale-reference error with enough context to reinspect.

Use saved-resource identity where available, plus normalized resource path and content revision as needed. Read-only inspection must not dirty a scene by inserting persistent UUID metadata. If persistent node identities are later required, design them explicitly with duplication and inheritance rules.

### 4.5 Revisions and trust boundaries

Distinguish disk state, unsaved editor buffers, staged changes, and runtime state. A response says which one it describes. Use per-scene/file revisions and conservative invalidation when necessary. A v1 full-scene content/revision comparison is preferable to an incomplete sophisticated tracker.

Treat project text, comments, resource metadata, screenshots, and tool outputs as untrusted data. None can change system policy, provider routes, spending limits, or approved roots. A project file named `AGENTS.md` may supply user-authorized development context, but cannot grant new host permissions to the product's AI.

### 4.6 Required error codes

At minimum define and test:

```text
INVALID_ARGUMENT
UNKNOWN_TOOL
UNSUPPORTED_OPERATION
UNSUPPORTED_SCHEMA
CAPABILITY_UNAVAILABLE
PERMISSION_REQUIRED
PERMISSION_DENIED
STALE_REFERENCE
REVISION_CONFLICT
PATH_OUTSIDE_SCOPE
READ_ONLY_RESOURCE
EXECUTION_ISOLATION_REQUIRED
BUDGET_EXHAUSTED
REQUEST_ALREADY_RECORDED
JOB_TIMEOUT
CANCELLED
APPLY_FAILED_RECOVERY_REQUIRED
PROVIDER_PROTOCOL_ERROR
```

Errors include a safe explanation and recovery action, not raw credentials, private filesystem contents, or an unbounded exception dump.

## 5. Change sets, permissions, and recovery

### 5.1 Define the supported transaction boundary honestly

P03 supports one loaded scene and native built-in data only. Later phases add multi-file staged operations. Do not claim cross-scene atomic undo before implementing it.

A change set contains its objective, exact operations, base revisions, before-state, affected objects/files, computed risk, required grant, validation requirements, preview hash, and application status.

Recommended states:

```text
proposed -> validated -> awaiting_approval -> authorized -> applying
 -> applied -> verifying -> verified -> accepted
                 |              |
                 +-> failed <----+
                      -> recovery_required -> restored
```

Verification failure does not mean the apply never happened. Cancellation during application must complete or compensate to a recorded safe state; do not interrupt a half-written operation and claim cancellation restored it.

### 5.2 Immutable previews and permissions

Bind approval to the preview hash, base revisions, affected paths, requested capabilities, and policy revision. Revalidate all of them immediately before applying. Any meaningful change invalidates approval.

Permission modes:

- **Manual:** Explicit approval for reads in a scoped batch and each mutation/execution group.
- **Protected:** Routine in-scope creation and editing can proceed; deletions, file moves/renames, node reparenting, and destructive replacement require approval. Classify the actual operation effect, not the friendly tool name. Ordinary removal of code lines during a normal script edit is not automatically a filesystem deletion, but protected files and capability changes remain protected.
- **Freedom:** No per-action prompts within the run's explicitly granted scope. Logs, cancellation, recovery, budgets, and execution boundaries continue to apply.

Mode is not scope. Editing one game project does not authorize changing `E:\SlimeEngine`, system files, the credential store, provider configuration, or acceptance tests. User controls—not model-generated arguments—set grants.

### 5.3 Native mutation correctness

Validate classes/properties against an explicit safe command registry. Introspection is for discovery, not blanket permission to instantiate arbitrary classes or invoke arbitrary methods/getters.

For supported scene operations, handle ownership, parent/sibling position, references, saved state, undo history, and object lifetime. Start with native primitives and simple scalar/vector properties. Reject inherited-scene edits, custom script setters, shared-resource mutations, or other unsupported cases explicitly until their dedicated tests exist.

Register one coherent undo entry per supported change set using the existing undo manager. Test redo and repeated undo/redo before closing the scene/editor. Ordinary undo history is session-scoped for this implementation; do not promise it survives reload or restart. Durable change-set recovery is a separate feature. Avoid accidentally executing the change once manually and again when committing its undo action. [R4]

### 5.4 Concurrency and conflict rules

One mutation lane per workspace. Reads may run concurrently only against valid snapshots. Never assume an earlier inspected node or file still exists.

Immediately before applying, compare current unsaved buffer versions and disk hashes against the proposal. A user edit produces `REVISION_CONFLICT`; do not silently save, reload, overwrite, or merge it away. A user switching scenes or closing the editor also requires safe cancellation/rebinding.

Restoration after later user work must preserve that work. In v1, refusing an unsafe restore and presenting recovery choices is better than overwriting newer edits.

### 5.5 Crash recovery and deduplication

Implement a native write-ahead edit journal outside the game project, with content-addressed before-state and immutable operation IDs. Keep the service's task log separate from this authority.

The P03 durability claim is process-crash recovery, not a universal power-loss guarantee. Document flush behavior. Use checksummed/framed records or another design that detects truncated writes. Exercise recovery after deliberately terminating the editor at defined fault points.

Journal before mutating; record completed operations and terminal results. On restart reconcile the journal with actual file/state hashes before deciding whether to finish or compensate. Do not replay blindly.

For the same operation ID and identical payload, return the recorded result or in-progress state. Reject reuse with a different payload. An uncertain transport result triggers status reconciliation, not a new independent write. Do not claim arbitrary provider API calls or external effects are exactly-once.

Add fault points before apply, after the first effect, before final save, after save but before result delivery, and during compensation. Recovery must either reproduce the before-state or state exactly what remains unresolved.

### 5.6 Windows path safety

Centralize all project-path handling. Reject out-of-root traversal, unauthorized absolute paths, UNC/device paths, alternate data streams, and paths that escape through symlinks/junctions/reparse points. Handle case-insensitive comparisons, normalized separators, trailing characters, and long paths consistently.

Resolve actual targets, not only lexical prefixes. Validate again at use time to avoid replacement races. Avoid in-place writes through hardlinks; use safe staging/replacement or reject unsupported cases. Do not follow links found in worker output. New import names and archive entries must pass the same policy.

Set quotas for file count, output bytes, texture dimensions, and job duration. A generated project or imported archive must not bypass scope through an expansion bomb or oversized resource.

## 6. Phase P00 — Establish the baseline

**Dependencies:** None.  
**Primary owner:** Lead. Subagent B may perform a read-only source-map review; do not begin overlapping implementation.

### Tasks

**P00-01 — Inspect instructions and checkout.** Read local guidance; record HEAD, branch, dirty paths, remotes, and existing SlimeEngine work. Do not overwrite user changes.

**P00-02 — Confirm toolchain.** Record compiler, SDK, Python/SCons, available Node/npm, disk and RAM constraints, and actual compatible build options. Missing optional provider credentials are not blockers for P00–P04's offline path.

**P00-03 — Build and smoke-test the unmodified baseline.** Capture output and binary identity. Start/close the ordinary editor on a trusted fixture; run a meaningful existing test subset or full available suite and document exclusions.

**P00-04 — Build a local source map.** Find actual editor plugin lifecycle, dock APIs, scene selection, unsaved buffer access, undo histories, process pipes, and test registration. Record symbols rather than guessing based on an older Godot release.

**P00-05 — Prepare small trusted fixtures.** Add minimal 2D and 3D projects with a few native nodes, basic resources, and documented initial hashes. Test runners copy them before modification. Keep executable fixture scripts simple, reviewed, and separate from future untrusted generated projects.

**P00-06 — Establish status files and task ownership.** Copy/adapt this plan; record decisions; assign the next tasks to at most two subagents.

### Gate P00

The current baseline builds, its executable is identified, a trusted fixture opens, pre-existing failures are recorded, and no user's work was discarded. Missing mandatory compiler prerequisites leave this gate blocked; read-only planning/testing work may continue, but do not pretend a modified unbuildable engine is verified.

## 7. Phase P01 — Native shell, protocol, and fake service

**Dependencies:** P00.  
**Lead:** Contracts, lifecycle hooks, integration.  
**Subagent A:** Native plugin/dock/session skeleton.  
**Subagent B:** Service skeleton, fake provider, framing and schema tests.

### Tasks

**P01-01 — Write the minimal contracts.** Define handshake, envelope, status/error/result schemas, capability discovery, and first typed values. Include golden valid/invalid fixtures shared by both implementations. Use a documented JSON Schema subset and matching strict C++ validation; do not claim support for features the native validator ignores.

**P01-02 — Add an editor-only subsystem.** Hook `editor/slime_ai/SCsub`, type registration as required, and actual plugin startup/shutdown. Avoid side effects at static initialization. Start disabled/offline; ordinary editor startup cannot depend on Node or an API.

**P01-03 — Add a small native dock.** Show service state, current project/selection, a diagnostic connect/disconnect action, and an activity list. Use existing editor themes, localization patterns, keyboard focus, and scaling. Do not build a browser-based replacement UI.

**P01-04 — Implement the fixed service launch and private transport.** Handle fragmented frames, Unicode, stderr, missing executable, service crash, close/reopen, and parent shutdown. No network listener. Read pipes without blocking the editor.

**P01-05 — Implement fake sessions.** The fake provider can emit deterministic text, an intended inspection call, malformed arguments, a delayed response, and a forced disconnect. Label mock mode visibly; never label fake output as a live model response.

**P01-06 — Add lifecycle and protocol tests.** New native tests use current test-registration conventions. TypeScript tests exercise golden fixtures and frame parsing. Integration verifies that enabled and disabled configurations start safely.

### Gate P01

The real editor shows its AI dock, exchanges a versioned handshake with the fake service, reports capabilities, survives disconnect/malformed input, and closes without leaving a continuously running orphan service. The editor works with AI disabled. Protocol tests execute nonzero counts and pass.

## 8. Phase P02 — Real engine/project inspection

**Dependencies:** P01.  
**Subagent A:** Inspectors and native tests.  
**Subagent B:** Inspection client, structured output rendering, fixtures.  
**Lead:** Revision semantics and permissions review.

### Tasks

**P02-01 — Implement `project_inspect`.** Return engine/build identity, project root reference, open scenes, unsaved scene/buffer indicators, registered tools, and scoped asset summaries. Do not send all project content automatically.

**P02-02 — Implement `scene_inspect`.** Return node hierarchy, opaque references, native type, ownership, instancing flags, supported property summaries, and revision. Paginate large scenes and enforce depth/count limits.

**P02-03 — Implement `object_inspect`.** Query safe supported properties and resources with typed encoding. Avoid invoking untrusted custom script getters on the host. Clearly mark unsupported properties rather than fabricating their values.

**P02-04 — Implement `api_describe`.** Describe a requested native class/property/method from the current build's metadata; separate discoverable metadata from enabled actions.

**P02-05 — Add explicit context attachments.** Selected scene/node references appear as inspectable chips in the dock. Display the outgoing context manifest, including whether unsaved content is included.

**P02-06 — Implement conservative invalidation.** Selection changes, scene edits, reloads, resource changes, undo, and external file modifications invalidate affected references/snapshots. Return revisions with every relevant result.

**P02-07 — Add inspection tests.** Verify a known tree, unsaved movement, stale node after deletion/reload, scoped asset enumeration, Unicode names, a shared resource, and no dirtying caused by inspection.

### Gate P02

The fake client can inspect the actual selected fixture, accurately distinguish saved and unsaved state, and receive a stale/conflict response rather than outdated fabricated data. Read-only inspection leaves scene/file state unchanged. The UI remains interactive during bounded large-scene inspection.

## 9. Phase P03 — Preview, apply, undo, and recover

**Dependencies:** P02.  
**Lead:** Transaction/grant design and fault-injection review.  
**Subagent A:** Native command handlers and undo integration.  
**Subagent B:** Changes UI/harness fixtures and crash-recovery scenarios, within assigned paths.

### Tasks

**P03-01 — Support a deliberately small mutation set.** Create a built-in Node2D/Node3D child, set safe scalar/vector properties, rename a node, and remove a node with proper approval. Use one scene per change set. Mesh/material creation can be added for the visual demonstration only with tested native resource handling.

**P03-02 — Build the preview.** Show exact property deltas, node additions/removals, scope, risks, required permission, and base revision. Freeze it to a preview hash. Preview itself cannot mutate the scene.

**P03-03 — Implement modes and grants.** Native policy classifies actions. Trusted user controls approve a particular change set. A model-supplied approval token or a changed payload must fail.

**P03-04 — Apply through the dispatcher.** Serialize writes on the appropriate editor execution path. Check references and revisions just before application. Preserve node ownership and dirty-state semantics. Record effects in the edit journal.

**P03-05 — Integrate undo/redo.** One coherent undo entry for the supported scene transaction. Reopen/save round trips must retain the created nodes. Removal undo restores properties and structure, not merely a name.

**P03-06 — Implement operation deduplication and status queries.** Repeated requests cannot create a duplicate node. Disconnect after apply must allow the client to retrieve the original result.

**P03-07 — Implement restart recovery.** Durable before-state and journal enable process-crash reconciliation. Restore only if current state matches a supported recovery precondition; protect newer user edits.

**P03-08 — Exercise negative paths.** Permission denial; stale revision; bad property/type; unsupported custom script setter; scene closed mid-request; invalid node owner; locked/read-only file; partial application; service disconnect; duplicate request; cancel.

**P03-09 — Demonstrate the complete first slice.** In the real editor, select the fixture scene, request a deterministic native addition/property edit through the fake service, preview, authorize, apply, undo/redo within the same editor session, then save/reopen to verify persistence, and demonstrate one rejected conflicting edit.

### Gate P03: first assignment complete

The demonstration above works against a real built editor. It has nonzero native/service/integration tests, a verified conflict rejection, a deduplication test, and a real fault-injection/recovery result. Unsupported operations fail explicitly. Report the supported transaction boundary and any recovery limitations.

**Stop expanding scope at this first handoff.** Record P04's exact next task. Do not substitute provider integration or a larger UI for any missing transaction test.

## 10. Phase P04 — One real provider and the daily-use interface

**Dependencies:** P03.  
**Lead:** Human workflow and context policy.  
**Subagent A:** Dock/changes/selection integration.  
**Subagent B:** First live provider adapter and session loop.

### Tasks

**P04-01 — Define the provider-neutral adapter.** The adapter streams normalized events such as `text_delta`, `tool_call_ready`, `usage_update`, `completed`, and `failed`. Tool arguments are accumulated and validated before `tool_call_ready`; partial JSON never causes an action. Preserve opaque provider continuation state separately from the visible conversation.

**P04-02 — Implement one live adapter.** Default to OpenAI Responses. If the user has explicitly configured only another intended provider, start with its compatible adapter and record the decision; do not demand new purchases. Offline tests remain runnable without keys. Do not hardcode a speculative model name or pricing table.

OpenAI's function-calling documentation describes application-executed tools and call-ID-linked results, with provider-specific response items that must be handled correctly. Use the current official contract. [S7]

**P04-03 — Add the bounded agent loop.** Inspect -> propose -> preview -> native authorization -> apply -> inspect actual result -> report. Cap turns, tool calls, output tokens, and retries. The model cannot approve itself or redefine scope.

**P04-04 — Add Discuss / Propose / Execute.** Discuss is read-only, Propose cannot apply, and Execute follows the active grant. Derive available actions from capability state, not from which tab is visible.

**P04-05 — Add clear task controls.** Show provider/model profile, mode, scope, estimated/reported usage, active operation, pause, cancel, and relevant evidence. Pause means no new dispatch after a safe boundary; cancel must reconcile in-flight mutation status. Do not imply already-consumed API usage was refunded.

**P04-06 — Add search and code reading.** Begin with exact filename/text/symbol lookup and dependency references. Paginate and bound results. Do not require embeddings or a vector database for the first working product.

**P04-07 — Add a safe credential path.** Prefer credentials held by the service through an OS-protected store. The editor receives a profile ID/status, not raw keys. A temporary environment-key development path must be explicitly documented, not leak values, and cannot pass provider keys to project workers. No plaintext saved key in `project.godot`.

**P04-08 — Perform the first live scene edit when authorized.** Repeat the P03 demonstration with the selected API and record model/endpoint, tool calls, preview, resulting state, and usage. A missing key leaves live verification `not_run`; fake-provider success must not impersonate it.

### Gate P04

The editor completes a bounded real-model scene-edit loop when configured, or explicitly distinguishes an implemented-but-not-live-tested adapter. Discuss and Propose remain non-mutating. Malformed arguments, provider failure, and cancellation do not damage the project. The UI provides usable context and meaningful change review.

## 11. Phase P05 — Complete the five provider profiles

**Dependencies:** P04 contracts and test harness. Native feature development may continue independently once its contracts are stable.

Implement three protocol families, not five separate orchestration engines:

| Profile | Contract | Specific requirement |
|---|---|---|
| OpenAI | Responses | Preserve call IDs and required continuation/output items |
| Anthropic / Claude | Messages | Handle `tool_use`, matching `tool_result`, content blocks, and stop conditions |
| DeepSeek | Compatible Chat Completions initially | Preserve documented thinking-mode continuation fields and endpoint-specific tool restrictions |
| Moonshot / Kimi | Compatible Chat Completions initially | Test actual tool-call/result and streaming behavior on the configured endpoint |
| OpenRouter | Compatible routed profile | Pin allowed routes/capabilities; no silent incompatible provider fallback |

The official documentation supports these tool-use interfaces, but compatibility does not mean identical behavior. DeepSeek currently documents `reasoning_content` replay requirements for tool-enabled thinking sessions; Anthropic uses tool content blocks; OpenRouter exposes routing restrictions such as `require_parameters`. Recheck the configured endpoint before implementation. [S8–S11]

### Tasks

**P05-01 — Introduce a capability profile.** Store supported tools, schema subset, streaming modes, image input, context/output limits, continuation behavior, routing policy, and whether usage is reported or estimated. Defaults must be conservative. A configuration entry is not a verified capability.

**P05-02 — Add Anthropic adapter.** Preserve content-block order and tool-result linkage. Handle multiple tool calls without assuming they should execute concurrently. Mutation serialization still applies.

**P05-03 — Add the compatible base adapter and explicit profiles.** Do not erase provider-specific fields through an overaggressive “normalization” function. Test DeepSeek reasoning continuation, Moonshot/Kimi tool messages, and OpenRouter routing separately.

**P05-04 — Preserve routing and privacy consent.** Provider/model switches and fallback routes may send project content to another party. Only use previously authorized routes. A failed request must not silently expand the set of providers receiving data.

**P05-05 — Implement usage and budget accounting.** Reserve estimated cost/tokens before dispatch, reconcile reported usage afterward, and handle missing usage. Unknown pricing must be visibly unknown, never free. With a monetary cap and no usable rate, block dispatch or require an explicit alternative token/request limit. In-flight requests can still incur cost; distinguish local budget enforcement from provider billing guarantees.

**P05-06 — Add conformance fixtures.** For every profile, test fragmented tool arguments, invalid JSON, unknown tools, multiple calls, incomplete streams, rate limits, authentication failure, retry-after, disconnect after tool application, cancellation, context limits, and missing usage. Sanitize recorded fixtures before committing them.

**P05-07 — Record live verification separately.** Each provider is labeled `implemented`, `fixture_verified`, or `live_verified` with endpoint/model and date. No API credentials means no live test, not a failing architecture or a fake success.

### Gate P05

All five profiles have passing offline conformance tests and explicit capability records. At least the configured primary provider completes the bounded edit loop live for a live-release claim. Other profiles retain truthful verification labels. No provider can bypass native grants or cause duplicate writes through retries.

## 12. Phase P06 — Broader editing and safe execution preparation

**Dependencies:** P03 transaction core. P04/P05 are useful clients but not prerequisites for offline testing.

### 12.1 Execution isolation is a release gate, not a label

Godot's `@tool` scripts run in the editor, and import/loading paths can involve code. A copied folder, Git worktree, process timeout, or ordinary child process is not a security sandbox. Do not automatically load generated tool scripts, native extensions, custom importers, or editor plugins into the credential-bearing host editor. [S4]

For the Windows-first isolated worker, implement a backend interface and a concrete Windows Sandbox backend when available. Detect platform availability in P00/P06; do not enable optional Windows features or reboot without authorization. An already-authorized dedicated VM backend is an acceptable documented alternative.

The Windows Sandbox documentation provides controls for networking, shared folders, clipboard, and other devices. It specifically warns that mapped host folders expose data and writable state. Use deliberate configuration rather than defaults. [S12]

Minimum job configuration:

- Networking off; clipboard, microphone, camera, and printer sharing off.
- Only an immutable per-job input bundle and trusted tool bundle mapped read-only.
- One empty, quota-limited per-job output directory mapped writable; never map the engine checkout, home directory, credential store, or entire project parent.
- Copy the editable project into the guest's local storage before opening/importing it.
- No provider keys, session authority, Git credentials, signing secrets, or unrestricted host IPC inside the guest.
- Validate output names, formats, links, sizes, and hashes before copying them anywhere else. Do not execute returned host scripts.
- Use software rendering initially where viable. A GPU-enabled profile is separate, explicit, and measured; do not claim it has the same attack surface or performance behavior.
- A worker timeout/cancel terminates the owned environment and descendants. Do not kill unrelated user processes.

When isolation is unavailable, keep static/offline development and explicitly trusted fixture tests available, but report `EXECUTION_ISOLATION_REQUIRED` for unattended untrusted code execution. Do not silently downgrade to running it on the user's host. A user-supervised trusted project path may exist with explicit, accurately described trust; it is not the autonomous isolation gate.

### 12.2 Editing tasks

**P06-01 — Implement content-addressed file patching.** Code edits target project-scoped files with exact base hashes and explicit patches. Support UTF-8/newlines consistently. A patch against the wrong revision fails; no fuzzy whole-file replacement as a fallback.

**P06-02 — Support ordinary GDScript creation/editing.** Update unsaved buffer/file semantics carefully. Diagnostic inspection and script loading take place in the appropriate trusted or isolated context. `@tool`, editor plugin, autoload, and native extension changes trigger execution-risk handling; a text scan is defense-in-depth, not proof of safety.

**P06-03 — Extend resource changes.** Create/modify native materials and simple resources. Detect shared references, offer “make unique,” preview the dependency impact, and test serialization. Inherited scene overrides and instantiated subscenes remain explicit supported/unsupported cases.

**P06-04 — Add safe scene instancing and signals.** Use the current engine APIs and ownership rules. Resolve references through validated handles. Preview both source and destination impacts. Preserve references across save/reload.

**P06-05 — Add staged multi-file transactions.** Build in an isolated candidate workspace. Journal before-state and publish only after revisions are still valid. Refuse cross-scene unsaved-buffer situations that are not implemented correctly. Do not advertise a single global Ctrl-Z action if only per-file recovery is supported; expose the actual change-set restore behavior.

**P06-06 — Implement the worker supervisor and isolation backend.** Typed job manifests include immutable input hashes, engine/template identity, allowed output types, time/memory/output limits, and test specification hashes. Execute reviewed command templates with argument arrays, not model-supplied shell strings.

**P06-07 — Add containment tests.** A harmless test project attempts to read a host sentinel outside the mapping, reach a prohibited network endpoint, write outside its assigned output, access a fake credential sentinel, and survive cancellation. The assertions must demonstrate denial. Do not use real credentials as sentinels.

**P06-08 — Prevent accidental host execution during integration.** Untrusted candidates stay in isolated editor/worker sessions. Passing functional tests does not prove malicious code is safe to promote. Loading custom editor code into the host requires an explicit trust decision. Reject or hold risky changes for review.

### Gate P06

Supported scripts/resources/scenes can be changed with accurate diffs, revision checks, and recovery. A concrete isolation backend passes containment tests before untrusted unattended execution is enabled. Any unsupported platform or risky host-load path remains visibly blocked instead of mislabeled safe.

## 13. Phase P07 — Verification, runtime inspection, and builds

**Dependencies:** P06 for untrusted execution. Trusted fixture harness development can begin earlier.  
**Lead:** Protected acceptance criteria and evidence gate.  
**Subagent A:** Debug/runtime bridge and editor evidence views.  
**Subagent B:** Worker harness, test scenarios, export jobs.

### Tasks

**P07-01 — Define evidence records.** Every record includes project/workspace identity, source content hash, engine and template builds, scenario version/hash, seed, environment/renderer, actual command, exit status, timeout state, assertions, and artifact references. A record from an older relevant revision becomes stale.

**P07-02 — Implement six distinct checks.** Structural integrity, script/import diagnostics, runtime smoke, behavioral assertions, rendered/visual checks, and exported-build tests. Report `passed`, `failed`, `not_run`, `unsupported`, or `inconclusive` per check; never infer missing checks passed.

**P07-03 — Add a development-only runtime probe.** Inspect allowlisted runtime properties, observe registered events, play bounded input sequences, and capture actual viewports. Discover local debugger interfaces before implementing. Avoid a general remote `eval` or unrestricted method-call endpoint.

**P07-04 — Build a scenario runner.** Scenarios assert externally meaningful state changes using frame/time bounds, not fragile blind sleeps. Control seeds and starting saves. Test an actual input route at least once; calling a gameplay function directly does not prove the input mapping works.

**P07-05 — Protect test oracles.** Required acceptance specifications live outside the model-writable candidate project and are identified by hash. The agent may propose new tests, but cannot weaken or delete required ones to complete a task. Worker output remains untrusted; collect independent process/exit observations and keep the authoritative gate outside the generated game. Do not claim this fully proves correctness against deliberately malicious code.

**P07-06 — Add real rendered captures.** Use a rendering-enabled worker for visual evidence. Empty/black/zero-size captures fail capture validation. Test resolution, camera, frame number, and renderer metadata. Headless dummy rendering does not count. Visual model assessment is advisory unless a precisely specified visual criterion is independently checked.

**P07-07 — Add export jobs.** Build matching Windows debug/release templates from the intended engine revision or use verified matching artifacts. Configure explicit preset/template paths and output directories. Never download an arbitrary stable template and assume compatibility with the development fork.

**P07-08 — Test the exported executable.** Run it in the supported worker, check the core interaction path, and inspect package contents. The game must start without the agent service, provider key, or editor being present. Use unique test save directories.

**P07-09 — Separate test probes from production exports.** Use a development-only injection/export mechanism with an explicit allowlist. Verify that release packages contain no credentials, journals, private prompts, orchestration code, active test bridge, or unnecessary development fixtures. A manifest inspection alone is insufficient if the content was accidentally embedded into the pack; inspect the package's actual included paths/data.

**P07-10 — Add honest performance sampling.** Separate editor-disabled overhead, idle AI overhead, active inspection latency, tool dispatch, and game frame-time measurements. Report hardware and test conditions. Use fixed sample protocols, not invented global performance promises.

Godot's CLI supports import, script parsing, export modes, and scripting-solution builds, but these are distinct checks. An exported artifact existing on disk is not proof it runs correctly. [S3]

### Core behavioral fixture

```text
Start the clean interaction fixture.
Move the player into range through an input sequence.
Press the configured interaction action.
Assert the collectible count increases by one.
Assert the world object can no longer be collected a second time.
Save and quit.
Restart with the same isolated test save directory.
Assert the count and collection state persist.
```

Also run the negative case: an intentionally broken pickup handler must make the scenario fail. The implementing model must not be able to report the task complete while that required failure exists.

### Gate P07

A known-good fixture passes, a known-bad fixture fails for the expected assertion, runtime evidence matches the tested revision, a real rendered capture exists, and a matching exported Windows build passes its designated smoke/behavior check. Missing GPU, templates, or isolation produces a specific blocked/not-run status—not a green result.

## 14. Phase P08 — Algorithm-assisted Content Lab

**Dependencies:** P03/P06 transactions and P07 validation.  
**Lead:** Recipe ownership and regeneration semantics.  
**Subagent A:** Native preview/application and content UI.  
**Subagent B:** Recipe specifications, generators where appropriate, property tests.

### 14.1 Start with three recipes

1. **Constrained 3D prop scatter.** Approved asset list, region, exclusion zones, seed, density/spacing, scale range, surface normal/slope limit, and optional alignment.
2. **Simple UI composition.** An inventory/hotbar layout built from native controls, with slot count, spacing, anchors/container layout, and readable labels.
3. **Native material variants.** Bounded color/roughness/metallic or other supported material parameters applied as unique resources when required.

Do not implement an unrestricted procedural graph editor first. Use a recipe registry, typed parameters, previews, and transaction-backed application. These recipes work without an AI connection; the model only selects inputs and parameters.

### Tasks

**P08-01 — Define the recipe format.** Include generator ID/version, schema version, input hashes, parameters, seed, constraints, output IDs, provenance, and manual overrides. Persist it as a project asset only after user authorization.

**P08-02 — Implement generator/validator separation.** Generate candidates, then check constraints independently. No infinite resampling: bounded attempts and explicit unsatisfied-constraint results.

**P08-03 — Handle coordinate systems.** Define local/world-space conversion, transforms, radians/degrees, units, distribution functions, and numerical tolerances. Test degenerate regions, zero count, invalid negative values, slope boundaries, and deterministic output under a fixed generator version/environment.

**P08-04 — Add previews and variants.** Generate multiple seeded candidates, show counts/constraint failures, and apply only a chosen result. Preview generation cannot silently modify the source scene.

**P08-05 — Preserve user work.** Track generated ownership by recipe/output identity. Pinning an object or overriding a property prevents regeneration from destroying it. Detach makes output ordinary manual content. Resolve deleted input assets and duplicate recipe IDs explicitly.

**P08-06 — Add provenance.** Record original inputs, generator version/seed, and user-approved asset sources. Future remote asset providers use a separate capability interface from text-model adapters. Preserve actual remote outputs; do not promise seed-based byte-identical regeneration for a remote API.

### Gate P08

All three recipes run manually and through tool calls, validate their outputs, produce reviewable changes, and preserve protected edits on regeneration. Geometry/math tests include analytic cases and randomized invariant checks. Unsupported constraints produce a failure, not an aesthetically plausible but invalid result.

## 15. Phase P09 — Bounded autonomous prototype builder

**Dependencies:** P04 primary provider, P06 isolation, P07 verification, and the required P08 recipes. P05 supplies additional provider choices but missing secondary live credentials need not block the primary route.

### 15.1 First autonomous benchmark: Collect-and-Exit

Use a deliberately small 3D game so success tests are unambiguous. Start from a blank reviewed project template, not a precompleted game.

The requested game contains a title screen, controllable player, small traversable arena, five collectible objects, a counter HUD, an exit that remains locked until all five are collected, a victory screen, and restart. Use primitive meshes and native materials. A save/load variant is a subsequent scenario, not permission to expand the initial game into a survival sandbox.

Required acceptance criteria:

- Title screen starts the game and relevant controls are visible.
- Actual configured movement/input routes work.
- Each collectible increments the count once and cannot be counted repeatedly.
- The exit cannot win early, then works after all five collectibles.
- Restart resets the required state.
- Mandatory scenarios report no relevant parse/runtime errors.
- A rendering-enabled capture shows the expected game view/HUD.
- The exported Windows build runs without the editor, service, or credentials.

These checks are protected. A generated README saying the game works is not evidence.

### Tasks

**P09-01 — Define a structured Game Brief.** Record required mechanics, platform, controls, scene/art constraints, approved assets, disallowed features, budget, and acceptance criteria. Separate user requirements from agent assumptions.

**P09-02 — Create dependency-aware tasks.** Each task has outputs, scope, dependencies, required tests, budget allocation, status, and relevant evidence. Start with one lead model and serial implementation; do not add a swarm.

**P09-03 — Implement the controller-owned state machine.** Planning, inspecting, proposing, awaiting approval, applying, verifying, repairing, accepted, blocked, and cancelled are explicit states. State transitions are determined by results and policy, not merely model text.

**P09-04 — Add bounded repair.** Initial test configuration: at most three repair attempts for the same failing acceptance condition, a bounded total turn count, and an explicit per-run API allowance. These are tunable product defaults, not promises of model success. Detect repeated equivalent patches/no improvement and stop with a blocker report.

**P09-05 — Add durable session resumption.** Recover task state, approved scope, last confirmed changeset, pending jobs, provider continuation state, and usage. Reconcile actual project/operation status before resuming. Do not replay uncertain writes or repeat already-billed requests automatically.

**P09-06 — Add model handoff summaries.** A provider switch transfers verified task facts, project references, accepted decisions, and evidence. Preserve protocol-specific opaque state only within the compatible adapter; do not blindly send it to another provider.

**P09-07 — Run multiple clean benchmark attempts.** Record all attempts, failures, repair counts, tool-call counts, cost/usage where known, and final gate results. Use different specified layout seeds. A successful attempt is demonstrated capability, not a universal autonomous-game-building guarantee.

**P09-08 — Test user interruption.** Pause/resume, scene conflicts, denied destructive actions, exhausted budgets, cancelled worker jobs, and service restart must produce resumable or explicitly blocked states.

### Gate P09

The primary live model completes at least one recorded clean benchmark through the Action API with mandatory tests and exported-build checks passing. Run a small repeatability set (initial target: three clean attempts) and publish all outcomes. Any failed required attempt remains visible; do not quietly omit it from reliability claims. Offline fake-provider rehearsals do not count as live autonomous performance.

## 16. Phase P10 — Hardening and developer-preview packaging

**Dependencies:** Earlier gates needed by the advertised feature set.

### Tasks

**P10-01 — Package the service.** The developer preview has a known runtime/service build, dependency lock, compatible protocol, and executable discovery. Missing service files produce a recoverable setup state, not editor startup failure.

**P10-02 — Verify non-editor templates.** Compile debug and release export templates without editor AI dependencies. Confirm release games do not launch a sidecar or listen for privileged commands.

**P10-03 — Harden error recovery.** Inject disconnects, partial frames, disk-full/locked-file failures, malformed resources, worker timeouts, truncated journal records, bad provider credentials, and unavailable renderers.

**P10-04 — Review secret handling and data exposure.** Search generated artifacts/logs using fake sentinel credentials. Confirm context previews, redaction, credential-store boundaries, provider routing consent, and configurable retention. Document that automatic secret detection cannot guarantee discovering every secret in user content.

**P10-05 — Review UX.** Keyboard navigation, scaling, contrast from editor themes, context clarity, operation status, accessible error messages, before/after diffs, clear mock mode, and predictable cancel behavior. Screenshots should come from a real build, not a mock claiming implementation.

**P10-06 — Add CI/offline checks.** Native tests, service typecheck/tests, schema conformance, deterministic integration fixtures, and static packaging checks. Live provider tests are opt-in, budgeted, and secret-aware. Mark unavailable platform/GPU tests as such.

**P10-07 — Maintain upstream compatibility.** Document the small upstream touch points. Keep rebasing/upstream updates separate from feature work. Re-run the required gate suite after an upstream update; do not mix it into an unrelated defect fix.

**P10-08 — Publish a truthful support matrix.** Supported tools/operations, single-/multi-scene limitations, provider fixture/live status, worker isolation platforms, tested renderer configurations, supported exports, and known issues.

### Gate P10

A documented local developer preview installs/starts, supports its advertised workflows, remains usable offline, contains no secrets, passes its required test matrix, and provides clear unsupported-feature behavior. Do not call it production-ready merely because one prototype benchmark passed.

## 17. Required test matrix

Every test needs an ID, owner, relevant phase, actual command, result, revision, and evidence path. The following are minimum scenarios, not a mandate to hardcode a particular test framework.

| ID | Scenario | Required assertion |
|---|---|---|
| BASE-01 | AI disabled | Editor opens/closes and ordinary fixture operations work |
| IPC-01 | Split frames and Unicode | Correct reconstruction; no partial execution |
| IPC-02 | Wrong protocol/oversized frame | Rejected; editor remains responsive |
| IPC-03 | Sidecar dies/editor closes | Visible state; bounded child cleanup; no duplicate operation |
| INS-01 | Saved versus unsaved scene | Correct source/revision and current values |
| INS-02 | Read-only inspection | No changed file hash/dirty flag |
| INS-03 | Delete/reload node | Old handle fails safely |
| TX-01 | Create -> save -> reopen | Owner and structure persist |
| TX-02 | Apply -> undo -> redo | State matches expected snapshots |
| TX-03 | Human edit after preview | Revision conflict; human edit preserved |
| TX-04 | Same operation twice | Exactly one recorded scene effect |
| TX-05 | Same ID/different payload | Rejected |
| TX-06 | Crash after effect/before response | Restart reconciles; no blind duplicate |
| TX-07 | Recovery after later human change | No destructive overwrite |
| POL-01 | Model says approved | No grant produced |
| POL-02 | Protected deletion/move | Native approval required |
| POL-03 | Freedom outside scope | Denied despite no per-action prompts |
| PATH-01 | Traversal/UNC/device/ADS | Denied |
| PATH-02 | Junction/link escape and race | No out-of-scope access |
| EDIT-01 | Shared material change | Impact shown; unique-copy path works |
| EDIT-02 | Unsupported inherited/custom setter | Explicit unsupported/risk result |
| EDIT-03 | Wrong-base script patch | No file/buffer overwritten |
| PROV-01 | Invalid/missing tool args | No native mutation |
| PROV-02 | Tool continuation across requests | Correct linkage and required provider state |
| PROV-03 | Rate limit/disconnect/usage absent | Bounded retry; truthful accounting |
| PROV-04 | Disallowed fallback provider | No unauthorized data transmission |
| ISO-01 | Read host sentinel | Denied outside mapped scope |
| ISO-02 | Prohibited network/credential access | Denied; no real secrets used |
| ISO-03 | Cancel runaway worker | Owned processes stop; unrelated processes survive |
| VER-01 | Known-good fixture | Required assertions pass |
| VER-02 | Known-broken fixture | Expected assertion fails |
| VER-03 | Headless run only | Visual check remains not_run |
| VER-04 | Change after test | Prior evidence becomes stale |
| VER-05 | Agent modifies required tests | Blocked; task cannot complete by weakening gate |
| EXP-01 | Matching export executable | Runs on designated Windows worker |
| EXP-02 | No service/key installed | Exported game still runs |
| EXP-03 | Release artifact inspection | No keys, journals, prompts, active test bridge |
| GEN-01 | Same seed/version/input | Expected deterministic result under stated environment |
| GEN-02 | Impossible constraints | Bounded failure, no infinite generation |
| GEN-03 | Regenerate with pinned edits | Manual work preserved |
| AUTO-01 | Budget exhausted | No new unauthorized requests/jobs |
| AUTO-02 | Interrupted/resumed task | Correct reconciliation, no duplicate effects |
| AUTO-03 | Collect-and-Exit | All protected required scenarios and export checks pass |

### Mathematical correctness rules

For geometry, transforms, placement, layout, tolerances, and budget calculations, specify units and bounds. Use small analytic examples in addition to randomized tests. Check invariants such as minimum distances, containment, count bounds, normalization, and unchanged protected outputs.

A subagent's arithmetic explanation is not a test. When a failure depends on random input, record the seed and convert it into a stable regression case. Do not use an LLM as the sole correctness oracle for deterministic math.

## 18. Done criteria for every implementation task

A task is verified only when its code is integrated, it compiles/typechecks where relevant, required positive and negative tests execute and pass, it does not introduce relevant baseline regressions, its permission and recovery behavior are documented, and its status has evidence pointers.

For UI tasks, inspect the real built UI at the supported scaling/window configuration. For behavioral tasks, run the behavior. For export tasks, run the exported artifact. Static reading does not replace those checks.

Incomplete APIs should not be exposed to models as working tools. Missing dependencies and credentials must produce explicit status. There must be no success-shaped placeholder, always-true validator, disabled assertion, fake test report, or hardcoded screenshot masquerading as runtime evidence.

The lead's handoff format:

```text
Completed gate:
Local source baseline and resulting revision:
Implemented behavior:
User-visible demonstration:
Exact build/test commands:
Executed/passed/failed/skipped counts:
Evidence locations:
Provider live verification status:
Unresolved defects and limitations:
Unrelated user changes preserved:
Next exact task:
```

## 19. First assignment checklist for immediate execution

1. Enter `E:\SlimeEngine`, read instructions, and establish P00 without destructive Git operations.
2. Build the baseline and record actual binary/toolchain identity.
3. Freeze the small P01 protocol with golden fixtures.
4. Assign Subagent A native plugin/inspection tasks and Subagent B fake-service/protocol tests only when the paths are disjoint.
5. Implement P01's real dock and service connection, including startup/shutdown tests.
6. Implement P02's real scene inspection and unsaved-state/revision semantics.
7. Implement P03's bounded native transaction with preview, grant checks, apply, undo/redo, deduplication, conflict rejection, and restart recovery.
8. Run and inspect the first editor demonstration and required negative tests.
9. Commit only intended local changes when appropriate; do not push.
10. Update `STATUS.md`, `TEST_MATRIX.md`, and `NEXT_TASK.md`; hand off actual results, not a restatement of this plan.

If a prerequisite blocks progress, report the exact missing prerequisite and what was still implemented/tested. Continue independent safe tasks where useful, but do not leap to unrelated later features or fabricate a passed gate.

## 20. Source notes and implementation references

These references establish existing facilities and protocol behavior, not proof that the proposed SlimeEngine features exist. Checked September 23, 2026. Live documentation can change; use the pinned/local source for engine integration and recheck API contracts for the configured endpoint.

### Repository sources

- **R1 — Version metadata:** `https://github.com/slimedragonair/SlimeEngine/blob/ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f/version.py`
- **R2 — Editor build:** `https://github.com/slimedragonair/SlimeEngine/blob/ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f/editor/SCsub`
- **R3 — Editor registration:** `https://github.com/slimedragonair/SlimeEngine/blob/ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f/editor/register_editor_types.cpp`
- **R4 — Undo manager:** `https://github.com/slimedragonair/SlimeEngine/blob/ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f/editor/editor_undo_redo_manager.h`
- **R5 — Test build:** `https://github.com/slimedragonair/SlimeEngine/blob/ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f/tests/SCsub`
- **R6 — Test runner:** `https://github.com/slimedragonair/SlimeEngine/blob/ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f/tests/test_main.cpp`

### Official documentation

- **S1 — Windows compilation:** `https://docs.godotengine.org/en/latest/engine_details/development/compiling/compiling_for_windows.html`
- **S2 — Native unit testing:** `https://docs.godotengine.org/en/latest/engine_details/architecture/unit_testing.html`
- **S3 — Command-line tools:** `https://docs.godotengine.org/en/latest/tutorials/editor/command_line_tutorial.html`
- **S4 — Editor-executed scripts:** `https://docs.godotengine.org/en/latest/tutorials/plugins/running_code_in_the_editor.html`
- **S5 — Process/pipe APIs:** `https://docs.godotengine.org/en/latest/classes/class_os.html`
- **S6 — Thread-safe engine APIs:** `https://docs.godotengine.org/en/latest/tutorials/performance/thread_safe_apis.html`
- **S7 — OpenAI function calling:** `https://developers.openai.com/api/docs/guides/function-calling`
- **S8 — Anthropic tool use:** `https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview`
- **S9 — DeepSeek tools and thinking:** `https://api-docs.deepseek.com/guides/tool_calls/` and `https://api-docs.deepseek.com/guides/thinking_mode/`
- **S10 — Moonshot/Kimi tool calls:** `https://platform.kimi.ai/docs/guide/use-kimi-api-to-complete-tool-calls`
- **S11 — OpenRouter routing:** `https://openrouter.ai/docs/guides/routing/provider-selection`
- **S12 — Windows Sandbox configuration:** `https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-configure-using-wsb-file`
