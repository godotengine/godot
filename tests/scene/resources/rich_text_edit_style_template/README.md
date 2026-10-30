# RichTextEdit style template probe

Run the built editor with:

```
godot.windows.editor.dev.x86_64.mono.console.exe --headless --path tests/scene/resources/rich_text_edit_style_template --script res://probe.gd --quit
```

The probe checks exact inline style replacement (including outline), then
verifies that quote, indent, lists, alignment, and line height only transfer
when the target selection covers complete lines.
