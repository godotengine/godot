# P04 offline built-editor acceptance

Final command from `E:\SlimeEngine`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/slime_ai/harness/run_p04_offline_editor_probe.ps1 -Gui
```

Exit 0 with the final built editor binary SHA-256 `DD074DB9A8F501432CE8C2BC66B33E5091C072C24A01F122A534CBEF7BEBC6A5` at HEAD `b6112992a830487ba3a1f81c4cd125455eb8ff82` plus working tree. Manifest: `p04-offline-editor-manifest.txt`. Disposable copy: `C:\Users\logan\AppData\Local\Temp\slime-ai-p04-offline-75f49691d068466782e27b7ffa2fc8e1`. The command opens the real visible Windows editor, runs the opt-in native probe through its plugin/lifecycle, then reopens the saved scene in a second editor process. It uses the fake provider and makes no network request; model is `none`, usage is `unknown`.

1. `p04-offline-discuss.json`: Discuss completed with zero tool calls and no scene edit; save/check/gameplay facts are `not_requested`/`not_run`.
2. `p04-offline-propose.json`: the fake provider's tool call reached a native immutable preview of one `Node2D` `AI_Marker` at `(48,24)`; `apply` returned `PERMISSION_DENIED`, and the scene stayed unchanged. The preview records scene scope, disk/base revision, exact delta, hash, and native grant requirement.
3. `p04-offline-execute.json`: an explicit trusted transition to Execute revalidated the preview. Native grant and apply succeeded. Undo reduced children from two to one, redo restored two, save returned error 0, and gameplay verification stayed `not_run`. The disk changed from SHA-256 `FE8557743069A5084B2A19C9DCBB9FBC2964D9BA729C1FA6DEDC15CE1674A88F` to `2F67F14FB1AFBA55D6C8B23B52EAA66E2891A715A42FB194B3AA84EC1E948824`; the saved file has one AI marker. The second editor process reopened with exit 0.
4. `p04-offline-unsaved-read.json`: the visible ScriptEditor buffer was modified to `unsaved value` while the disk still held `saved value`. Native `code_read` reported `source=unsaved_editor_buffer`, `script_editor_unsaved=true`, and distinct buffer/disk revisions. The adversarial comment asking to change provider stayed plain text data. The probe temporarily disabled an external-editor setting and restored its prior value before exit; no system security setting changed.

The earlier headless and hidden-GUI attempts could not exercise ScriptEditor editing; their failed stdout/stderr checkpoints remain as `p04-unsaved-read-first-*` and `p04-gui-hidden-first-*`. A first visible-GUI probe exposed a reentrant save-notification bug in the **probe** and was fixed before this passing run; no earlier failure is counted as a pass. The final probe is programmatic inside a visible editor, not a manual usability review of every dock control. Offline native/service test counts are in `../TEST_MATRIX.md`.
