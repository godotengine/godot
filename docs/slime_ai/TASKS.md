# P00–P03 task ledger

Baseline revision: `ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f`. Working branch: `codex/slime-ai-p00-p03`. Protocol dependency: `tools/slime_ai/protocol/CONTRACT.md`, native command revision 1.1 and service wire version 1.0 with golden fixtures. Initial user supplied plan files are unrelated, untracked inputs and must remain untouched.

| ID | Owner | State | Acceptance evidence |
|---|---|---|---|
| P00-01..04 | Lead | verified | `BASELINE.md`, `SOURCE_MAP.md`, `evidence/p00-*` |
| P00-05..06 | Lead | verified | Trusted 2D/3D fixtures, hashes, task records, `evidence/p00-fixture-*.log` |
| P01-01 | Lead | verified | Contract and golden fixtures; native/service tests and versioned handshake |
| P01-02..04 | Native worker and Lead integration | verified | Built-in dock, opt-in private child, IPC/lifecycle native tests and visible editor check |
| P01-05..06 | Service worker and Lead integration | verified | Service 16/16, typecheck exit 0, native IPC suite, final editor smokes |
| P02-01..07 | Lead integration | verified within bounded scope | Project/scene/object/API inspectors, selected context controls, explicit outgoing manifest, typed values/pages and 5 focused inspection tests |
| P03-01..08 | Native worker and Lead verification | verified within one-scene scope | Native transaction, negative-path, dedup, policy, and recovery tests; see `TEST_MATRIX.md` |
| P03-09 | Lead | verified | Real editor demonstration and process-crash restart; `evidence/EDITOR_DEMO.md` |

## Delegation record A: native, contract 1.0 then native 1.1

Allowed to create/edit `editor/slime_ai/**` except `editor/slime_ai/SCsub`, and `tests/editor/slime_ai/**`. No other paths. No shared registration/build hooks, protocol fixtures, docs, harness, or fixture edits. Do not spawn agents. Implement P01 native dock and private pipe client, P02 live read-only inspection, P03 one-scene native preview/grant/apply/undo/redo/dedup/recovery. Use the contract above; Lead added native command revision 1.1 after the initial wire contract was frozen, while keeping the service wire at 1.0. Engine mutations stay on editor main thread. Unsupported operations return structured errors. Coordinate any required shared hook change with Lead. Do not run a full engine build while Lead's build is active; Lead owns final build and tests.

Acceptance checks assigned: native C++ tests with nonzero cases for split/malformed frames, missing/disconnected service, saved versus unsaved inspection without mutation, node invalidation, create/save/reopen, undo/redo, stale preview, permission denial, duplicate ID/same and different payload, unsupported class/property, cancel, and injected process-crash recovery. Provide exact commands, exit codes, test counts, and evidence. A static or compile-only check is not sufficient for the gate.

## Delegation record B: service, contract 1.0

Allowed to create/edit only `tools/slime_ai/agent_service/**`. No other paths. Do not change the shared contract/fixtures, native files, docs, or harness. Do not spawn agents. Implement P01 deterministic fake service on private stdin/stdout with handshake and proposal scenarios, strict frame decoder, bounded UTF-8, stderr diagnostics, EOF shutdown, and explicit errors. No network listener, API calls, credentials, or direct project writes. Pin the package tooling and lockfile without global installs.

Acceptance checks assigned: `npm test` (nonzero test count) and typecheck, valid golden handshake/proposal, split UTF-8 and JSON frames, malformed/oversized/unsupported envelope, request ID echo, normal/delayed/malformed/disconnect scenarios, and EOF process exit. Return exact commands, exit codes, test counts, and evidence.

Both worker handoffs must state task ID, base revision, files changed, behavior, commands, exit codes/counts, evidence, limitations, unresolved integration, and patch/commit reference. Lead will independently rerun all gates and owns any shared changes.

## P03 closeout / P04 handoff, 2026-09-23/24

| ID | Owner | State | Acceptance evidence |
|---|---|---|---|
| C01 real denied save | Lead | verified | Built editor under exclusive Windows lock, old file preserved, unsaved edit retained, retry saved one effect; `evidence/P03_SAVE_FAILURE.md` |
| C02 unresolved reconciliation | Lead | verified within conservative scope | Fresh IDs blocked until exact operation/revision-bound user resolution; no replay; native cases 249 and 513 in `test_slime_ai.cpp` |
| C03 operation/evidence map | Lead | verified with documented gaps | `TEST_MATRIX.md` maps all native enabled operations and negative paths; P04 model tool remains create only |
| P04-01/02 provider events and OpenAI Responses | Service worker, Lead integration | offline verified; live not_run | `provider_events.ts`, `providers.ts`, `run_manager.ts`; 35/35 service cases, typecheck exit 0, `evidence/P04_LIVE_DEMO.md` |
| P04-03 bounded run and native authority | Lead | offline verified | `slime_ai_run_controller`, native P04 cases, exact 3-attempt/4-tool/1024-token/30s/120s limits |
| P04-04/05 dock and intents | Lead | GUI fixture verified; manual dock review unrun | `slime_ai_editor_plugin`, visible offline probe `evidence/P04_OFFLINE_DEMO.md` |
| P04-06 scoped project search/read | Native reads worker, Lead integration | verified | `slime_ai_project_search`, `slime_ai_code_reader`, native read cases and visible unsaved ScriptEditor proof |
| P04-07 credentials and consent | Service worker and Lead | offline verified; live not_run | Credential Manager lookup, no network at startup, one-run live authorization and synthetic failures |

For this continuation, the lead owned shared contracts, save/lifecycle hooks, native permissions, integration, documentation, and final verification. Native reads worker ownership was only `editor/slime_ai/slime_ai_project_search.{h,cpp}`, `slime_ai_code_reader.{h,cpp}`, `tests/editor/slime_ai/test_slime_ai_p04_read.cpp`, and `test_slime_ai_p03_3d.cpp`; acceptance was focused scope/read/3D tests plus final lead build. Service worker ownership was only `tools/slime_ai/agent_service/**`; acceptance was offline provider/event/limit/redaction tests, `npm test`, and `npm run typecheck`. Neither worker spawned another agent. Final lead rerun: build exit 0, native 33/33 and 393/393, service 35/35, typecheck 0, OS-lock probe 0, visible GUI probe 0. No live request or push.
