# Goline example projects

Additive, self-contained example projects used to exercise and validate
Goline's tooling (e.g. the game context pack in `goline/cli/context.py`)
and as reference material. Nothing here is part of the engine build.

## sample_game

A minimal Godot 4 / Goline 2D project:

- `project.godot` — project named "Goline Sample Game", main scene
  `res://scenes/Main.tscn`.
- `scripts/main.gd` — root `Node2D` that spawns a `Player`.
- `scripts/player.gd` — a `CharacterBody2D` moved with arrow keys / WASD.
- `scenes/Main.tscn` — the main scene wiring the above.

Try the context pack against it:

```
python goline/cli/goline_cli.py --print-context game \
    --project goline/examples/sample_game
```
