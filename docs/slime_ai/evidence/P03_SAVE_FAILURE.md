# P03 real save-denial closeout

Run from `E:\SlimeEngine` after the final build:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/slime_ai/harness/run_denied_save_probe.ps1
```

Exit 0 on local HEAD `b6112992a830487ba3a1f81c4cd125455eb8ff82` plus working tree; editor binary SHA-256 `DD074DB9A8F501432CE8C2BC66B33E5091C072C24A01F122A534CBEF7BEBC6A5`. Full manifest: `p03-denied-save-manifest.txt`. The harness copied the 2D fixture to `C:\Users\logan\AppData\Local\Temp\slime-ai-save-denial-c18c36b9a7b64b43838be7ef42bbdfdc`; source fixture was untouched. It applied one native `SaveDenialMarker` while unsaved, then held `main.tscn` open with an exclusive Windows `FileStream` (`FileShare.None`) during the real `EditorInterface::save_scene()` call. The lock was released in a `finally` block even on failure.

| Observation | Result |
|---|---|
| Original saved SHA-256 | `FE8557743069A5084B2A19C9DCBB9FBC2964D9BA729C1FA6DEDC15CE1674A88F` |
| Save under lock | Error `20`; `status=save_failed`; execution `applied`, persistence `failed`; marker present; two in-memory children; `editor_unsaved=true` |
| Disk after release, before retry | Exactly the original SHA-256; old scene still readable |
| Same logical operation redelivered | `duplicate_request=true`, `no_second_apply=true`, two children in memory |
| Explicit save after unlock | Error `0`; `editor_unsaved=false`; new SHA-256 `446B5516B8C49259FC01D5CEFD08D897DBC3AEF114E94D4106596E863D45F1ED`; exactly one saved marker; second editor reopen exit 0 |

Raw editor results: `p03-denied-save-ready.json`, `p03-denied-save-denied.json`, `p03-denied-save-retry.json`, `p03-denied-save-editor-stderr.txt`, and `p03-denied-save-reopen-stderr.txt`. The parent harness reads the disk hash only **after** releasing the lock; the editor could not read the locked file, so `denied.json` says `unreadable_while_exclusively_locked`. The harness parses the saved text, counts exactly one marker, and reopens this same fixture in a second editor process. Native undo/redo tests are mapped in `../TEST_MATRIX.md`.

The first OS-lock run was a **failed checkpoint**. It returned save error 0 and cleared the editor dirty flag while Windows safe replacement actually failed; stderr reported `Safe save failed`. Its original evidence remains in `p03-denied-save-first-ready.json`, `p03-denied-save-first-denied.json`, and `p03-denied-save-first-stderr.txt`. The fix propagates `FileAccessWindows::_close()` replacement/close failure through the text resource saver and `EditorNode` to `EditorInterface::save_scene()`.

A separate deterministic regression run used `powershell -NoProfile -ExecutionPolicy Bypass -File tools/slime_ai/harness/run_denied_save_probe.ps1 -Injected` and exited 0 on the same binary. This opt-in test-only path returned error 20 without calling save for the denied attempt, kept the old hash and live unsaved effect, then used the real save path on retry and reopened with exit 0. Its evidence is `p03-injected-save-{ready,denied,retry}.json`, `p03-injected-save-manifest.txt`, and stderr files. It is **not** the real OS-denial demonstration.

`present_unconfirmed` remains an unresolved historical operation, not a successful acknowledgement. Native tests show the same ID gives status without replay; a fresh ID in the same scene returns `RECONCILIATION_REQUIRED`; a stale revision cannot resolve; explicit reviewed resolution records `resolved_without_replay` and permits a later preview without altering the existing node. A human edit after the uncertain effect is retained. Earlier killed-editor GUI evidence remains in `EDITOR_DEMO.md`.
