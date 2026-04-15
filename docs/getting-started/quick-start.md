# Public Evaluator

Use this page after [Installation](installation.md) when you want the shortest path to the public evaluator scene in the sample project.

Real editor screenshots for this flow are still pending, so this page stays text-first for now and keeps the diagram as a technical reference at the end.

## 1. Point at Your Editor

If you already have an editor built from this fork, set it directly:

```bash
export GODOT_BINARY=/absolute/path/to/your/godot-editor
```

```powershell
$env:GODOT_BINARY="C:\absolute\path\to\your\godot-editor.exe"
```

Need a binary first? Use the Linux nightly editor. The Windows release path exists in the workflow, but a public Windows release has not landed yet; if you are on Windows and need an editor today, use [Build from Source](../BUILDING.md). If you are on macOS, use [Build from Source](../BUILDING.md), then come back here and set `GODOT_BINARY` to the binary you have.

After a successful build, point `GODOT_BINARY` at the editor binary:

```bash
export GODOT_BINARY=/absolute/path/to/your/godot-editor
```

```powershell
$env:GODOT_BINARY="C:\absolute\path\to\your\godot-editor.exe"
```

You should have a working editor binary before continuing.

## 2. Open the Sample Project

```bash
$GODOT_BINARY --path tests/examples/godot/test_project
```

```powershell
& $env:GODOT_BINARY --path .\tests\examples\godot\test_project
```

You should see the sample project open in the editor.

## 3. Open the Public Evaluator

Press Play. The sample project now opens `res://scenes/public_evaluator.tscn` by default.

You should see:
- a visible splat in the viewport
- the evaluator scene already loaded

## If It Fails

- [Artist workflow overview](../user/quickstart.md)
- [Installation](installation.md)
- [Build from Source](../BUILDING.md)
- [Recurring issues](../troubleshooting/recurring-issues.md)

## Flow Reference

<figure markdown="1">
![Diagram of the public evaluator path from a fork-built editor to a visible sample splat](../assets/images/first-run-editor-path.svg){ .gs-diagram }
<figcaption>The public evaluator path is a short proof loop: point at your editor, open the sample project, and confirm a visible splat in the public evaluator scene.</figcaption>
</figure>
