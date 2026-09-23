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
| FILE-01 | Editor save to a locked/read-only scene file | not_run | No locked-file fixture was exercised; save errors are surfaced but not qualified here |

P04 provider, P06 isolation, P07 verification, and all later-phase cases remain deferred for this assignment.
