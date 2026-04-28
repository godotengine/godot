# Godot Game Project — Full Instructions

You are building a complete, playable Godot 4.3 game and delivering a single self-contained Windows .exe. This document covers every detail of the environment, project structure, engine conventions, and verification process. Read it before you start writing code.

## 1. Environment

Godot 4.3-stable is installed and configured. You do not need to install, compile, or set up anything.

- Editor binary: /usr/local/bin/godot — invoke as `godot` from any directory
- Windows export templates: preinstalled at ~/.local/share/godot/export_templates/4.3.stable/
- Wine + rcedit: installed and pointed at by the editor settings, used during Windows export to embed the icon and metadata into the .exe
- Class API reference: /opt/godot-docs/ (Godot 4.3 branch, reStructuredText). Consult this when uncertain about method signatures, properties, or signals. There are no tutorial docs — only the class reference.
- Display tooling: xvfb-run is available for headless runtime testing.
- Build system: GDScript only. No C#, no plugins, no addons, no external libraries.
- Renderer: Compatibility renderer only. Do not use Forward+ or Mobile features.

You run as root. Editor settings live at /root/.config/godot/editor_settings-4.tres and are already configured — do not modify them.

## 2. Project Location and Structure

Working directory: /root/godot/AI_GAME/

The following has already been scaffolded:

/root/godot/AI_GAME/
├── scenes/                  empty — put your .tscn files here
├── scripts/                 empty — put your .gd files here
├── assets/                  empty — generate any code-based assets here if needed
├── build/                   empty — your final .exe lands here
└── export_presets.cfg       preconfigured Windows Desktop preset — DO NOT EDIT

You must create:
- project.godot — the project manifest (see Section 4 for required contents)
- icon.png — 256×256 PNG at the project root, generated in code (see Section 6)
- All .tscn and .gd files for your game

Use res:// paths everywhere in your project. Never hardcode absolute filesystem paths.

## 3. Export Preset (Already Configured)

export_presets.cfg defines a "Windows Desktop" preset with:
- Architecture: x86_64
- embed_pck=true — produces a single self-contained .exe with no separate .pck file
- Icon path: res://icon.png
- Output path: build/AI_GAME.exe

You invoke this preset by name during export (Section 10). Do not edit the preset file. If your icon is missing or your project fails to import, the export will fail — fix the underlying problem rather than touching the preset.

## 4. project.godot — Required Contents

Your project.godot must include the following display configuration so the game launches fullscreen and scales correctly on any monitor (1080p, 1440p, 4K, ultrawide):

[application]
config/name="AI_GAME"
run/main_scene="res://scenes/main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")
config/icon="res://icon.png"

[display]
window/size/viewport_width=1920
window/size/viewport_height=1080
window/size/mode=3
window/stretch/mode="canvas_items"
window/stretch/aspect="expand"

[rendering]
renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"

Key points:
- mode=3 is fullscreen
- stretch/mode="canvas_items" plus aspect="expand" means the viewport is your design resolution (1920×1080) and Godot scales everything to fit the actual monitor with no black bars on any aspect ratio
- run/main_scene points to whatever you choose as your entry-point scene; the path shown is a convention, not a requirement
- The renderer must be gl_compatibility — Forward+ and Mobile will not work

You may add other sections ([input] for input map, [autoload] for singletons, [physics] for physics tweaks) as your game requires.

## 5. No Assets — Generate Everything in Code

You have no sprites, textures, audio files, or fonts. Do not reference any external asset files. Create all visuals and sounds procedurally.

Visuals:
- Use primitive nodes: ColorRect, Polygon2D, Line2D
- Or draw in code via _draw() with draw_rect(), draw_circle(), draw_polygon(), draw_line(). Call queue_redraw() to trigger redraws when state changes.
- For sprite-like textures, build an Image in code with Image.create() and set_pixel(), then convert via ImageTexture.create_from_image(image) and assign to a Sprite2D.
- For text, use Label nodes — they work without a font assigned, using Godot's built-in default font. Do not load .ttf files.

Audio:
- Use AudioStreamGenerator attached to an AudioStreamPlayer.
- Get the playback with player.get_stream_playback() (returns an AudioStreamGeneratorPlayback).
- Fill the buffer with sine, square, triangle, or noise samples computed in code. Use playback.push_buffer(frames) where frames is a PackedVector2Array of stereo samples in the range [-1.0, 1.0].
- Set mix_rate on the generator (e.g., 22050 or 44100) and compute samples accordingly.
- Do not reference .wav or .ogg files.

## 6. Icon Generation

The export preset references res://icon.png. Without this file, export will fail.

Generate a 256×256 PNG icon in code, either as a one-time setup script or in your main scene's _ready() (with a check to only generate it if missing). Example approach:

func generate_icon():
    var img = Image.create(256, 256, false, Image.FORMAT_RGBA8)
    img.fill(Color(0.1, 0.1, 0.2))
    # draw something recognizable — fill a rect, a circle, your game's motif
    for x in range(64, 192):
        for y in range(64, 192):
            img.set_pixel(x, y, Color(0.9, 0.6, 0.2))
    img.save_png("res://icon.png")

Run this once before exporting. The simplest pattern is a small standalone scene that generates the icon and quits, run via `godot --headless --script res://generate_icon.gd` — but generating in _ready() of your main scene also works as long as the file exists by the time you run the export step.

## 7. Coordinate System and UI Layout

This is the single most common source of bugs in Godot — read carefully.

Coordinate origins differ by node type:
- Sprite2D, Polygon2D, Node2D, CharacterBody2D, Area2D: the node's position IS the center of the shape (assuming default centered=true on Sprite2D). To place a sprite at screen center: position = get_viewport_rect().size / 2.
- Control and all UI nodes (Label, Button, ColorRect, Panel, Container, VBoxContainer, etc.): position is the TOP-LEFT corner of the control's rect, not the center.
- Viewport coordinates: (0,0) is the top-left of the screen. X increases to the right. Y increases DOWN, not up.

Center UI elements with anchors, never hardcoded pixel positions. Hardcoded positions like Vector2(960, 540) will break at any resolution other than 1920×1080.

To center a Control:
- In code: set anchor_left = anchor_top = anchor_right = anchor_bottom = 0.5, then offset_left = -size.x / 2 and offset_top = -size.y / 2.
- In a .tscn: use anchors_preset = 8 (Center).

To fill the entire screen with a UI container (HUD, menu background):
- Use anchors_preset = 15 (Full Rect).

UI must live inside a CanvasLayer so it stays fixed regardless of camera movement or world scrolling.

## 8. Scene Architecture

You have full freedom over scene layout. A typical structure:

Main (Node2D)
├── World (Node2D)              gameplay nodes, camera, player, enemies
│   └── Camera2D                call make_current() in _ready() if multiple cameras
├── UI (CanvasLayer)            HUD, menus — fixed to screen
│   └── HUD (Control, anchors_preset=15)
└── Audio (Node)                AudioStreamPlayers for SFX/music

Each distinct game object (player, enemy, projectile, pickup, level, menu) should be its own .tscn with a matching .gd script attached to its root. Instantiate sub-scenes from your main scene:

const EnemyScene = preload("res://scenes/enemy.tscn")

func spawn_enemy(pos: Vector2):
    var enemy = EnemyScene.instantiate()
    enemy.position = pos
    $World.add_child(enemy)

Switch between full scenes (menu → gameplay → game over) with:

get_tree().change_scene_to_file("res://scenes/game_over.tscn")

Use signals for communication between scenes — declare with `signal died`, emit with `died.emit()`, connect with `node.died.connect(callable)`. Avoid long get_node("../../X") chains across the tree.

If you override _ready() in a class that extends a parent with its own _ready(), call super._ready() first.

## 9. GDScript 4.x Syntax Reminders

Godot 4.x has different syntax from 3.x. Common patterns:

- @onready var node = $Path (not `onready var`)
- @export var speed: float = 100.0 (not `export var`)
- signal_name.emit(args) (not `emit_signal("signal_name", args)`)
- signal_name.connect(callable) (not `connect("signal_name", target, "method")`)
- super._ready() (not `._ready()`)
- Vector2i exists for integer vectors, distinct from Vector2
- await get_tree().create_timer(1.0).timeout for delays
- Type hints are encouraged: func damage(amount: int) -> void:

When unsure, consult /opt/godot-docs/ for the exact class API.

## 10. Verification Loop — Mandatory Before Finishing

Run all three steps. Fix every error and warning. Loop until the logs are clean. Do not deliver the .exe until all three steps pass without ERROR:, SCRIPT ERROR:, Parser Error:, or WARNING: lines.

Step 1 — Import check:

cd /root/godot/AI_GAME
godot --headless --import 2>&1 | tee /tmp/import.log

This parses all your scenes and scripts, generates .godot/ cache files, and reports any syntax errors, missing resources, or broken references.

Step 2 — Runtime smoke test:

timeout --kill-after=2 10 xvfb-run -a godot --path . 2>&1 | tee /tmp/runtime.log

This boots the game in a virtual display for 10 seconds and captures any runtime errors. Hitting the 10-second timeout is SUCCESS — it means the game ran without crashing. The game not quitting on its own is expected and normal. Only error/warning prefixes count as failure.

Step 3 — Export:

godot --headless --export-release "Windows Desktop" build/AI_GAME.exe 2>&1 | tee /tmp/export.log
ls -lh build/AI_GAME.exe && file build/AI_GAME.exe

The .exe must exist, be non-zero size, and report as PE32+ executable. The export log must have no error or warning lines.

Fail conditions for all three steps: any line containing ERROR:, SCRIPT ERROR:, Parser Error:, or WARNING: in the corresponding log file. Stack traces span multiple lines — read surrounding context when diagnosing. If any step fails, fix the underlying code or scene and restart from step 1.

## 11. Final Deliverable

/root/godot/AI_GAME/build/AI_GAME.exe — a single self-contained Windows executable with the .pck embedded. No separate .pck file should exist alongside it.

Report:
- Confirmation that all three verification steps passed clean (no errors, no warnings)
- The file size of the final .exe
- A brief description of what the game does and how to play it
