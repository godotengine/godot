# RichTextEdit display overflow regression

This project tests the actual renderer. The native C++ scene tests use a mock
rendering server and cannot establish whether overflow glyphs reached the canvas.
No external font, asset, network service, or desktop input automation is needed.

## Usage

Opt in on the same RichTextEdit used for editing and Typewriter playback:

```gdscript
editor.display_overflow_enabled = true
editor.editable = false
editor.mouse_filter = Control.MOUSE_FILTER_IGNORE
editor.focus_mode = Control.FOCUS_NONE
editor.caret_draw_when_editable_disabled = false
```

C# exposes the same option as `editor.DisplayOverflowEnabled = true` after
regenerating the Mono glue and building the GodotSharp assemblies.

The new property defaults to `false`. Its policy is owned by RichTextEdit; TextEdit
provides a protected clipping hook which defaults to the original behavior.
Setting `editable = true` restores the ordinary viewport without replacing the
editor or its document. Input and caret settings remain explicitly owned by the
application. Restore those settings when entering editing as usual.

Overflow does not enlarge the control or change its authored wrapping width.
It disables internal scrollbars, resets their offsets, and draws all visible
text rows. Ancestor `clip_contents`, the control's own `clip_contents`, and the
Viewport still apply. Content-fit sizing is a separate, existing feature; disable
it if the control itself must keep a fixed size. This mode processes all visible
rows and is not intended for very large scrolling documents.

`get_content_height()` (`GetContentHeight()` in C#) measures the native document
layout without enabling content-fit or resizing the control. It includes line
spacing and paragraph margins, but excludes style box margins. Before-shaping
reveal can change wrapping/font metrics; unrevealed logical lines remain empty
rows. Set `visible_characters = -1` when measuring full-document AUTO height.

## Run from the engine repository root (PowerShell)

Build an editor with `tests=yes`, then run:

```powershell
& .\bin\godot.windows.editor.dev.x86_64.mono.console.exe --headless --test '--test-case=*[TextEdit]*,*[CodeEdit]*,*[RichTextEdit]*' --no-colors
& .\bin\godot.windows.editor.dev.x86_64.mono.console.exe --path tests/scene/resources/rich_text_edit_overflow --script check.gd --rendering-method gl_compatibility
```

The second command requires a working graphics context (do not add `--headless`).
It renders to a fixed SubViewport, writes PNGs and `report.json` under
`bin/overflow_qa`, and exits nonzero on failure. Pixel comparisons allow a small
per-pixel RGB noise threshold, but require zero pixels above that threshold.

Coverage includes nowrap, multiline, authored-width wrapping, center/right and
mixed paragraph alignment, outline/background/underline, mixed font sizes,
inline images, RTL text, Unicode Typewriter prefixes, offscreen caret geometry,
parent/self clipping, and restoration of ordinary editable viewport pixels.
Scene packing/restoration preserves the option and passive input/caret settings.
The native tests also verify TextEdit/CodeEdit defaults, ten mode round trips,
document preservation and undo history.

Verified on Windows with the Compatibility renderer (Intel Iris Xe), 2026-08-27:
30 native cases / 6,769 assertions passed; all 21 renderer/state cases passed.
The 18 reference-image comparisons reported zero differing pixels above the
noise threshold, and both clipping cases reported zero leaked pixels.
The Mono editor, regenerated C# bindings, Debug/Release GodotSharp assemblies,
GodotTools and Godot.NET.Sdk also built successfully. Build logs are retained in
`bin/overflow_native_tests.log`, `bin/overflow_render_tests.log`,
`bin/overflow_mono_glue.log`, and `bin/overflow_mono_build.log`.

This standalone project validates the engine API. It does not replace
Persabrand's client, input, exported HTML, or full-site comparison gates, nor does
it merge this branch's implementation into another engine checkout.
