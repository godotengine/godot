# P00–P03 test matrix

`not_run` means no test execution has been observed. A compile or code review never changes this state to passed.

| ID | Scenario | State | Evidence |
|---|---|---|---|
| BASE-01 | Unmodified editor opens fixture without AI | passed | `evidence/p00-editor-smoke.log`, exit 0 |
| BASE-02 | Existing Node2D native subset | passed | `evidence/p00-native-tests.log`, 2/2 cases, 45/45 assertions |
| BASE-03 | Copied 2D and 3D fixtures open | passed | `evidence/p00-fixture-2d.log`, `evidence/p00-fixture-3d.log`, both exit 0 |
| SERVICE-01 | Fake service golden protocol and failure scenarios | passed | `evidence/p01-service-tests.log`, 16/16, exit 0 |
| SERVICE-02 | TypeScript typecheck | passed | `evidence/p01-service-typecheck.log`, exit 0 |
| BUILD-01 | First modified-editor compile checkpoint | failed | Compiler output captured in task transcript; new native header/include/type errors |
| BUILD-02 | Second modified-editor compile checkpoint | failed | `evidence/editor-build-checkpoint-02.log`, SCons exit 2; doctest/Variant comparison errors |
| BUILD-03 | Third modified-editor compile checkpoint | failed | `evidence/editor-build-20260923-220913-443.log`; Array::make unavailable |
| BUILD-04 | Fourth modified-editor compile checkpoint | failed | `evidence/editor-build-20260923-221030-228.log`; missing test force link |
| BUILD-05 | Modified editor after fixes | passed | `evidence/editor-build-20260923-224233-598.log`, exit 0; manifest has binary hash |
| NATIVE-01 | Targeted SlimeAI native tests | passed | `evidence/native-tests.log`, 24/24 cases, 271/271 assertions, exit 0 |
| IPC-01 | Split frames and Unicode | passed | Native IPC cases and service tests; `evidence/native-tests.log`, `evidence/p01-service-tests-final.log` |
| IPC-02 | Malformed/oversized/wrong version | passed | Native IPC cases and service tests; expected parser diagnostic appears in native log |
| IPC-03 | Missing/disconnected service and child cleanup | passed | Native handshake/missing/disconnect cases; final editor starts offline and connects on demand |
| INS-01 | Saved versus unsaved actual scene | passed | Native saved/unsaved cases; `evidence/editor-demo-inspect.png` |
| INS-02 | Read-only inspection leaves file unchanged | passed | Native saved-scene hash case; fixture file unchanged until apply/save |
| INS-03 | Stale node after deletion | passed | Native scene/object stale-reference cases |
| INS-04 | Bounded pages, typed values, Unicode, shared texture | passed | Native inspection cases; final dock evidence `evidence/editor-project-inspect.png`, `evidence/editor-object-inspect-confirmed.png` |
| TX-01 | Create, save, reopen | passed | Native test plus `evidence/editor-demo-saved.png`, `evidence/editor-demo-reopened.png`, `evidence/editor-final-binary-reopen.png` |
| TX-02 | Apply, undo, redo | passed | Native test plus `evidence/editor-demo-applied.png`, `evidence/editor-demo-undone.png`, `evidence/editor-demo-redone.png` |
| TX-03 | Human edit after preview rejected | passed | Native test plus `evidence/editor-conflict-human-edit.png`, `evidence/editor-conflict-rejected.png` |
| TX-04 | Duplicate ID and identical payload safe | passed | Native exact-ID test; repeat dock apply refused and left one node (`evidence/editor-final-duplicate-apply.png`) |
| TX-05 | Same ID with different payload rejected | passed | Native test returns `REQUEST_ALREADY_RECORDED` |
| TX-06 | Process terminated after effect / before confirmation | passed | `evidence/editor-crash-session.txt`, saved scene, journal copy, and `evidence/editor-crash-reconciled.png` |
| TX-07 | Recovery status preserves later human edit | passed | Native test verifies no replay and both nodes remain |
| TX-08 | Unwritable journal and closed scene leave no effect | passed | Native negative-path cases |
| POL-01 | Service text alone cannot grant approval | passed | Native permission case and `evidence/editor-demo-denied.png` |
| POL-02 | Protected remove needs native approval | passed | Native protected-removal case |
| EDIT-01 | Unsupported class/property and invalid owner rejected | passed | Native negative-path cases |
| CANCEL-01 | Cancel before effect has no mutation | passed | Native cancel case |
| FILE-01 | Editor save to an OS-locked copied scene | passed | `evidence/P03_SAVE_FAILURE.md`; real exclusive Windows lock, error 20, unchanged old hash, unsaved live edit, one-effect retry |

The table above records the original P00–P03 history; `NATIVE-01` and `SERVICE-01` retain their earlier counts. A failed first save-denial checkpoint is retained in `evidence/p03-denied-save-first-*` and was not counted as a pass.

## P03 closeout and P04 final checks

Final source: HEAD `b6112992a830487ba3a1f81c4cd125455eb8ff82` plus dirty working tree. Final binary SHA-256 `DD074DB9A8F501432CE8C2BC66B33E5091C072C24A01F122A534CBEF7BEBC6A5`. Build: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/slime_ai/harness/build_editor.ps1 -Jobs 8`, exit 0, `evidence/editor-build-20260924-002902-739-manifest.txt`. Native: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/slime_ai/harness/run_native_tests.ps1`, exit 0, 33/33 cases and 393/393 assertions, `evidence/native-tests.log` and `native-tests-manifest.txt`. Service: from `tools/slime_ai/agent_service`, `npm test` and `npm run typecheck`, both exit 0; 35/35 service cases, no skips, `evidence/p04-service-tests.txt`, `p04-service-typecheck.txt`, `p04-service-manifest.txt`. `git diff --check` exit 0.

| ID | Scenario | State | Evidence / limit |
|---|---|---|---|
| C-SAVE | Real denied save, old disk file, live unsaved state, duplicate retry, later save/reopen | passed | `powershell -NoProfile -ExecutionPolicy Bypass -File tools/slime_ai/harness/run_denied_save_probe.ps1`, exit 0; `evidence/p03-denied-save-{ready,denied,retry}.json`, manifest and stderr |
| C-SAVE-INJECT | Deterministic test-only save error, retry, reopen | passed | Same script with `-Injected`, exit 0; `evidence/p03-injected-save-{ready,denied,retry}.json` and manifest; separate from OS denial |
| C-RESOLVE | `present_unconfirmed` blocks a fresh ID; revision-bound explicit resolution; no replay; later human edit retained | passed | `test_slime_ai.cpp` lines 249 and 513; native 33-case log; earlier killed-editor record in `EDITOR_DEMO.md` |
| C-OPS | Native advertised operation map and negative paths | passed within documented scope | Operation map below; dedicated 3D property/rename/remove cases absent and those are not P04 model tools |
| A-STREAM | Split Unicode, terminal barrier, successful complete response | passed offline | `provider.test.ts` SSE/accumulator tests; native P04 run rejects an early tool event |
| A-INCOMPLETE | Failed, incomplete, truncated, closed, malformed or multiple tool calls never dispatch | passed offline | `provider.test.ts` accumulator/validator cases; no live stream exercised |
| A-SCHEMA | Strict supported schemas and local single-operation validation | passed offline | `provider.test.ts` schema case; `tool_schema.ts`; native `RunController` and transaction validation |
| A-LINK | `call_id` output continuation and duplicate service delivery | passed offline | `provider.test.ts` continuation case; `run_manager.ts` |
| A-AUTH | Missing key, 401/403, model/authorization required, redaction | passed offline | `provider.test.ts` injected responses; native P04 run case; live credential acceptance not_run |
| A-RETRY | 429, timeout, disconnect count within finite attempts | passed offline | `provider.test.ts` retry and limit cases |
| A-MULTI | One active serial service run, no parallel tool execution | passed offline | `run_manager.ts` and service limit tests; live provider behavior not_run |
| I-DISCUSS | Read-only fake run leaves scene unchanged | passed | `test_slime_ai_p04_run.cpp`; GUI `evidence/p04-offline-discuss.json` |
| I-PROPOSE | Immutable native preview, apply denied | passed | `test_slime_ai_p04_run.cpp`; GUI `evidence/p04-offline-propose.json` |
| I-GRANT | Explicit Execute transition, native grant/apply/undo/redo/save/reopen | passed | GUI `evidence/p04-offline-execute.json`, manifest, reopened editor exit 0 |
| I-READ | Scoped text search/read and actual unsaved ScriptEditor buffer | passed | `test_slime_ai_p04_read.cpp`; visible GUI `evidence/p04-offline-unsaved-read.json`; headless unsaved-buffer path not available |
| I-INJECT | Adversarial project text cannot switch provider or grant writes | passed for tested paths | `test_slime_ai_p04_read.cpp` and GUI unsaved-buffer fixture; service sentinel/redaction test |
| I-REVOKE | Mode change invalidates grant; old revision cannot Execute | passed native | `test_slime_ai.cpp` line 565; `test_slime_ai_p04_run.cpp` line 44; GUI revocation during a live model run not_run |
| R-CONFLICT | Human change makes preview/grant stale | passed | Native P03 test and prior GUI screenshot; P04 transition native test |
| R-CANCEL-1 | Cancel before effect suppresses later events | passed offline | Native P04 run and service cancellation tests |
| R-CANCEL-2 | Cancel after native dispatch reconciles result without undo/replay | partial | Native transaction status/recovery tested; full GUI timing case not_run |
| R-UNKNOWN | Uncertain native result stops continuation and new IDs | passed offline | `test_slime_ai.cpp` recovery cases; `provider.test.ts` unresolved continuation case |
| R-SAVE | Edit and save have separate facts | passed | Real lock probe and dock result; `evidence/p03-denied-save-denied.json` |
| R-LIMIT | Attempts, calls, timeout, deadline bounded | passed offline | `provider.test.ts` limit/retry cases; native fixed limits; no live usage |
| R-USAGE | Unknown usage remains unknown | passed offline | `evidence/p04-offline-discuss.json`; provider usage fixture tests |
| R-CLAIM | Model text cannot certify save/check/gameplay | passed for tested paths | Native run status axes and GUI fixture; gameplay explicitly not_run |
| R-OFFLINE | No key and no network needed for fixtures | passed | 35 service tests, GUI fake run and manifest `live_request=none` |
| R-SECRETS | OS credential lookup; inert sentinel not leaked | passed offline | `provider.test.ts`; `credentials.ts`; live secret use not_run |
| LIVE-01 | Authorized OpenAI Responses run against real account | not_run | No authorized model/key/scope/limits; `evidence/P04_LIVE_DEMO.md` |

### Enabled native operation map

| Operation | Actual tests | Scope / gap |
|---|---|---|
| `create_child` Node2D | `test_slime_ai.cpp` lines 180, 221, 249, 377, 410, 513; real GUI P03 and P04 probes | Preview/grant/apply/dedup/undo/redo/save/reopen/conflict/recovery; the only P04 model-facing edit |
| `create_child` Node3D | `test_slime_ai_p03_3d.cpp` line 18 | Native preview/grant/apply/undo/redo/save/reopen; no P04 GUI 3D model run |
| `set_property` Node2D position/visible | `test_slime_ai.cpp` lines 309 and 596 | Native preview/grant/apply/undo/redo; no dedicated 3D property test |
| `rename_node` | `test_slime_ai.cpp` lines 309 and 468 | Native preview/grant/apply/undo/redo and unowned-node rejection; no dedicated 3D test |
| `remove_node` subtree | `test_slime_ai.cpp` lines 309 and 486 | Native preview/grant/apply/undo, ownership and protected grant; no dedicated 3D test |

Other P03 negative cases: same ID/different payload line 377; stale grant/revision line 221 and mode change line 565; duplicate status line 180; scene closed line 549; cancellation line 249; private service disconnect line 133; unsupported class/property line 221/377; invalid owner line 468; journal write failure line 451; interrupted saved effect/reopen line 410. These are native/fixture tests; no claim of cross-file atomicity or power-loss durability. P06 and later phases remain deferred.
