# Built-editor acceptance record, 2026-09-23

All projects below are GUID-named copies of `tools/slime_ai/fixtures/2d`, never the source fixture. The editor executable was `E:\SlimeEngine\bin\godot.windows.editor.dev.x86_64.console.exe` launched with `--editor --path <copy> --rendering-method gl_compatibility`. GUI actions were performed in the actual Windows editor, not headless. Screenshots are captured at 2560×1440. The final binary SHA-256 is `CF576A691A3F63567ACA829EABAA8BC629BB07ECAF73113CD93EBD47C98AC3BA`; the transaction screenshots were taken from the preceding successful modified-editor build with the same transaction core. `editor-final-binary-reopen.png` visually checks the final binary.

## Main flow

Fixture copy: `C:\Users\logan\AppData\Local\Temp\slime-ai-demo-2d-ce5ef7cb3983408ca82c9cd19f5e2fa2`; launch record `editor-demo-session.txt`.

1. Selected `FixtureRoot` and inspected the live saved scene (`editor-demo-inspect.png`). The scene tree contained `Marker` and no `AI_Marker`.
2. Connected the opt-in fake service and received the versioned handshake (`editor-demo-connected.png`). Requested the deterministic Node2D `AI_Marker` at Vector2 `(48,24)`; preview showed the before/after, revision, scope, and native grant requirement (`editor-demo-preview.png`). Preview did not change the scene tree.
3. Pressed apply without a grant; the dock returned `PERMISSION_REQUIRED` and the tree stayed unchanged (`editor-demo-denied.png`). Pressed native authorize, then apply; the tree showed one `AI_Marker` (`editor-demo-applied.png`).
4. Pressed undo, then redo; the tree lost and regained `AI_Marker` (`editor-demo-undone.png`, `editor-demo-redone.png`). Pressed save; `save_error: 0` (`editor-demo-saved.png`). Closed and reopened the project; the node persisted (`editor-demo-reopened.png`). The saved `main.tscn` contains exactly one AI marker with `position = Vector2(48, 24)` and `metadata/slime_ai_operation_id`; SHA-256 `0B808392FB8C3253B709B3E5A69FDE7A592F814645B3324BF0716A603C6A758E`.
5. After the later dock inspection work, the rebuilt editor exposed project/object/API inspection (`editor-project-inspect.png`, `editor-object-inspect-confirmed.png`, `editor-api-describe.png`) and a context manifest in the preview (`editor-final-preview.png`). It applied a fresh preview (`editor-final-applied.png`). A second click on apply refused the stale preview and the tree still had one AI marker (`editor-final-duplicate-apply.png`). The native exact-ID test separately verified that repeating the same ID/payload returns duplicate status and a different payload with the same ID is rejected.

## Human conflict

Fixture copy: `C:\Users\logan\AppData\Local\Temp\slime-ai-conflict-ee87535d1c354b088c4356484ef44a5b`; launch record `editor-conflict-session.txt`. Requested a preview (`editor-conflict-preview.png`), then renamed `Marker` to `HumanMarker` through the editor scene tree (`editor-conflict-human-edit.png`). Native grant followed by apply returned `REVISION_CONFLICT`, preserving the human rename and adding no AI node (`editor-conflict-rejected.png`).

## Process crash and restart

Fixture copy: `C:\Users\logan\AppData\Local\Temp\slime-ai-crash-ff6609b659974beea8d830f3d98fc380`; launch record `editor-crash-session.txt`. The launch process alone had `SLIME_AI_TEST_CRASH_AFTER_SAVE=1`. After preview and native grant, apply wrote a checksummed `prepared` journal record, performed the scene effect, saved the scene, then called `OS::kill` before confirmation. Process 12956 was no longer alive. The saved scene contained exactly one `AI_Marker` with operation ID `native-28400301-1`; SHA-256 `C1BA14A10E587AC1AC15FDE18E9E0AD91D13FF5D1A4C095D803A703135E1B223`.

Restarted the same project without the fault-injection environment variable (`editor-crash-restart.txt`). The node was present and the dock's recorded-operation query returned `status: prepared`, `effect: present_unconfirmed`, with a warning not to retry that ID (`editor-crash-reconciled.png`). The copied journal is `editor-crash-journal.json`, SHA-256 `34BE07516D536FDF3A266777CEA6134A91DFAC04F321FBAC7B82CBF1998786CD`. Its original path was under `%APPDATA%\Godot\app_userdata\Slime AI 2D Fixture\slime_ai\f5cf13c9a3aca828c7aab4a494f0f8effb8a170e3747aa1eed258237354b21b7\journal.json`.

This demonstrates process-crash detection and safe no-replay reconciliation. It does not demonstrate power-loss durability, automatic replay, or cross-file atomicity.

## Reproduction and verification commands

From `E:\SlimeEngine` in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/slime_ai/harness/build_editor.ps1 -Jobs 8
powershell -NoProfile -ExecutionPolicy Bypass -File tools/slime_ai/harness/run_native_tests.ps1
Push-Location tools/slime_ai/agent_service
npm test
npm run typecheck
Pop-Location
```

Final results: build exit 0; native 24/24 cases and 271/271 assertions, exit 0; service 16/16, exit 0; typecheck exit 0. Final copied 2D and 3D headless editor smokes each exited 0 (`final-editor-smoke-2d.log`, `final-editor-smoke-3d.log`). The malformed-frame native test deliberately writes a Unicode parser diagnostic; the test runner captures it as stderr data and reports the nonzero test count and exit code separately.

## P03 closeout and P04 continuation, 2026-09-23/24

The historical screenshots and counts above are preserved as the P00–P03 checkpoint, not relabeled as P04 evidence. The later final binary is SHA-256 `DD074DB9A8F501432CE8C2BC66B33E5091C072C24A01F122A534CBEF7BEBC6A5`. A real OS-locked save on a disposable fixture returned error 20 while the old file remained intact and the live edit stayed unsaved; the same logical operation could then be saved once after unlock. See `P03_SAVE_FAILURE.md`, including the failed first checkpoint.

The visible-editor **programmatic** P04 fixture flow with the offline fake provider ran Discuss, Propose, explicit Execute transition, native grant/apply/undo/redo/save/reopen, plus an unsaved ScriptEditor buffer read. See `P04_OFFLINE_DEMO.md`. Native tests increased to 33/33 cases and 393/393 assertions; service tests to 35/35. Manual clicks through all new dock controls and a paid provider run are not recorded as performed. Live status is `not_run` in `P04_LIVE_DEMO.md`.
