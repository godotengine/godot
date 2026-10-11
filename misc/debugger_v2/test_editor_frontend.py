#!/usr/bin/env python3
"""Launch the custom editor and verify its protocol-v2 frontend pairs a child runtime."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def run(godot: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="godot-debug-v2-editor-") as directory:
        project = Path(directory)
        addon = project / "addons" / "auto_play"
        addon.mkdir(parents=True)
        (project / "project.godot").write_text(
            "[application]\n"
            'config/name="debug-v2-editor-smoke"\n'
            'run/main_scene="res://Main.tscn"\n\n'
            "[editor_plugins]\n"
            'enabled=PoolStringArray( "res://addons/auto_play/plugin.cfg" )\n',
            encoding="utf-8",
        )
        (project / "Main.tscn").write_text(
            "[gd_scene load_steps=2 format=2]\n\n"
            '[ext_resource path="res://main.gd" type="Script" id=1]\n\n'
            '[node name="Main" type="Node"]\nscript = ExtResource( 1 )\n',
            encoding="utf-8",
        )
        (project / "main.gd").write_text(
            "extends Node\n"
            "var total = 0\n"
            "func _ready():\n"
            "    breakpoint\n"
            "    for index in range(1000):\n"
            "        total += index\n"
            "    get_tree().quit()\n",
            encoding="utf-8",
        )
        (addon / "plugin.cfg").write_text(
            "[plugin]\n"
            'name="Protocol v2 smoke"\n'
            'description="Automatically runs and exits the test project."\n'
            'author="Godot contributors"\nversion="1.0"\n'
            'script="plugin.gd"\n',
            encoding="utf-8",
        )
        (addon / "plugin.gd").write_text(
            "tool\n"
            "extends EditorPlugin\n"
            "func _find_debugger(node):\n"
            "    if node.has_method(\"debug_next\") and node.has_method(\"debug_continue\"):\n"
            "        return node\n"
            "    for child in node.get_children():\n"
            "        var found = _find_debugger(child)\n"
            "        if found:\n"
            "            return found\n"
            "    return null\n"
            "func _enter_tree():\n"
            "    call_deferred(\"_run_smoke\")\n"
            "func _run_smoke():\n"
            "    get_editor_interface().play_main_scene()\n"
            "    yield(get_tree().create_timer(0.5), \"timeout\")\n"
            "    var debugger = _find_debugger(get_tree().root)\n"
            "    for _index in range(100):\n"
            "        if debugger:\n"
            "            debugger.debug_next()\n"
            "        yield(get_tree().create_timer(0.03), \"timeout\")\n"
            "    yield(get_tree().create_timer(0.5), \"timeout\")\n"
            "    get_tree().quit()\n",
            encoding="utf-8",
        )

        command = [
            str(godot),
            "--editor",
            "--path",
            str(project),
            "--verbose",
            "--no-window",
            "--audio-driver",
            "Dummy",
            "--video-driver",
            "GLES2",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=20.0,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as error:
            partial_stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            partial_stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            raise RuntimeError(f"custom editor timed out:\n{partial_stdout}\n{partial_stderr}") from error

    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"custom editor exited with {completed.returncode}:\n{output}")
    required = (
        "Editor Debugger v2: paired control and data channels",
        "Editor Debugger v2: stopped at stop_id",
        "Editor Debugger v2: loaded 1 stack frame(s)",
        "Editor Debugger v2: loaded variable page for Locals",
    )
    missing = [marker for marker in required if marker not in output]
    if missing:
        raise RuntimeError(f"editor smoke markers missing {missing}:\n{output}")
    stopped_count = output.count("Editor Debugger v2: stopped at stop_id")
    if stopped_count < 100:
        raise RuntimeError(f"editor completed only {stopped_count} synthetic step cycles:\n{output}")
    print(f"Godot editor protocol v2 frontend: PASS (stops={stopped_count})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.godot.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
