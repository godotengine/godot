# RichTextEdit document model checks

Two headless checks for the line-aligned RichTextEdit model, run from the
repository root with the built editor:

```
godot.windows.editor.dev.x86_64.mono.console.exe --headless --path tests/scene/resources/rich_text_edit_model --script check.gd
godot.windows.editor.dev.x86_64.mono.console.exe --headless --path tests/scene/resources/rich_text_edit_model --script perf.gd -- --mode=scroll
godot.windows.editor.dev.x86_64.mono.console.exe --headless --path tests/scene/resources/rich_text_edit_model --script perf.gd -- --mode=canvas
```

Both exit with a nonzero code on failure.

## check.gd

- Hundreds of seeded single operations (typing, paragraph breaks, deletion,
  inline styles, links, alignment, lists, quotes, line heights, rich copy and
  paste, indentation, images, rules) on a short document, and large-range ones
  on a ~300 paragraph document whose bulk changes defer shaping. After each,
  undoing everything must restore the canonical BBCode and content height
  exactly, and redoing everything must restore the edit.
- Formatting semantics: a typing style set at the caret applies on top of the
  paragraph formatting, Shift+Enter at the end of a list item starts a plain
  paragraph, a paragraph's line height travels with its first character, rich
  paste keeps formatting, and images survive undo and redo.
- Deferred shaping converges: once `has_pending_layout()` is false, the content
  height equals that of the same document with every line shaped.

## perf.gd

Times editing operations on a 500,000 character document with six styled runs
per paragraph (lists, quotes, centered paragraphs and line heights included).
Each operation is measured together with the two frames that draw its result,
so the floor is the duration of two frames. Single edits must stay under
100 ms; loading and document-wide changes, which reshape lines as they are drawn
and the rest in the background, have larger budgets.

## Model

`RichTextEdit` keeps one `Paragraph` per TextEdit line: style runs with
paragraph-relative columns referencing an interned style table, the style of
the paragraph break (empty paragraphs keep their list, quote and alignment from
it), inline objects, and the line height. TextEdit reports every insertion and
removal with its exact range before the lines change (`_text_inserting` /
`_text_removing`), so only the affected paragraphs are updated and reshaped.
Removed formatting is recorded so undo and redo restore it; style changes record
undo snapshots of the paragraphs they touch. BBCode is produced only when
requested.
