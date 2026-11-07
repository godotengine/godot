# RichTextEdit overflow regression

This project tests the actual renderer. The native C++ scene tests use a mock
rendering server and cannot establish whether overflow glyphs reached the canvas.
No external font, asset, network service, or desktop input automation is needed.

## Usage

`RichTextEdit.overflow_x` and `overflow_y` each select one of four modes (the
CSS `overflow-x` / `overflow-y` pair) and can be changed at run time on the same
RichTextEdit used for editing and Typewriter playback. Each mode governs its own
axis:

| Mode | Scrollbar of the axis | Content outside the control |
| --- | --- | --- |
| `OVERFLOW_AUTO` (default) | shown only when the content overflows that axis | clipped |
| `OVERFLOW_VISIBLE` | never shown | drawn |
| `OVERFLOW_HIDDEN` | never shown | clipped |
| `OVERFLOW_SCROLL` | always shown, even when the content fits | clipped |

```gdscript
editor.overflow_x = RichTextEdit.OVERFLOW_VISIBLE
editor.overflow_y = RichTextEdit.OVERFLOW_VISIBLE
editor.editable = false
editor.mouse_filter = Control.MOUSE_FILTER_IGNORE
editor.focus_mode = Control.FOCUS_NONE
editor.caret_draw_when_editable_disabled = false
```

C# exposes the same options as `editor.OverflowX` / `editor.OverflowY`
(`RichTextEdit.OverflowEnum.Visible`) after regenerating the Mono glue and
building the GodotSharp assemblies.

As in CSS, an axis set to `OVERFLOW_VISIBLE` computes to `OVERFLOW_AUTO` while the
other axis is not `OVERFLOW_VISIBLE`, and content is only drawn outside the control
when both axes are `OVERFLOW_VISIBLE`.

Both properties default to `OVERFLOW_AUTO`, which is the ordinary TextEdit
viewport. Overflow is independent from `editable`: neither one changes the
other, in either direction. Its policy is owned by RichTextEdit; TextEdit
provides protected clipping and scrollbar hooks which default to the original
behavior. Input and caret settings remain explicitly owned by the application.

No mode enlarges the control or changes its authored wrapping width. The modes
that hide the scrollbars also reset their offsets, and `OVERFLOW_VISIBLE` draws
all visible text rows. Ancestor `clip_contents`, the control's own
`clip_contents`, and the Viewport still apply. Content-fit sizing is a separate,
existing feature; disable it if the control itself must keep a fixed size.
`OVERFLOW_VISIBLE` processes all visible rows and is not intended for very large
documents.

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
It also covers all four overflow modes against content that fits and content
that overflows, in both editable states, including the forced scrollbars of
`OVERFLOW_SCROLL` over content that fits. Scene packing/restoration preserves
the mode and passive input/caret settings. The native tests also verify
TextEdit/CodeEdit defaults, mode round trips, document preservation and undo
history.

### Decoration pixel regression

The renderer script also checks underline, strikethrough and overline on the
first, middle and last logical rows. It samples the empty space between two
glyphs, independently requiring the expected number of colored strokes and
continuous coverage across glyph joins. This catches missing lines even when
both a small control and a large reference would render the same broken result.

The matrix contains 1,446 cases: font sizes 12/16/24/36/48/72, canvas scales
0.5/0.75/1/1.25/1.5/2, line heights normal/0.75/1/1.25/1.5/2/125%/150%/96px,
and quarter-pixel translations. Targeted cases additionally check node scaling,
all eight decoration combinations (including removing every style), bold,
italic and bold italic. Controls and transforms change on the same instance;
the test does not manually request a redraw when changing the canvas transform.

For the original two-case reproduction, run:

```powershell
& .\bin\godot.windows.editor.dev.x86_64.mono.console.exe --path tests/scene/resources/rich_text_edit_overflow --script check.gd --rendering-method gl_compatibility -- --decorations-only --smoke
```

Omit `--smoke` for the full decoration matrix, and omit `--decorations-only`
to include the existing overflow/state tests. Pass `--output=C:/absolute/path`
after `--` to keep separate logs and screenshots for each renderer. Vulkan can
be tested with `--rendering-method forward_plus --rendering-driver vulkan`.
The smoke cases fail on the old renderer (24px, scale 0.75, line height 1,
Y offsets 0/0.5) and pass with the antialiased, scale-aware decoration runs.

The two verification records below predate the `overflow` enum (they were run
against the older `display_overflow_enabled` boolean) and have not been re-run
since. Re-run both commands above before relying on them again.

On 2026-08-28, the 1,446 decoration cases and 21 existing renderer/state cases
passed with both Compatibility and Vulkan/Forward+ on Intel Iris Xe. This
verification used a dev Mono editor without native test support; it does not
re-run or supersede the native-test result recorded below.

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
