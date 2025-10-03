# todoG
## A “Recursive Game‑Builder” in Godot  

Below is a practical, step‑by‑step blueprint that turns the idea into a working prototype.  
Feel free to skip parts you already have, add your own twist, or adapt it to Godot 4.x if that’s your target.  

---

### 1. High‑level Concept

| Layer | Purpose | What you see | How you get to it |
|-------|---------|--------------|-------------------|
| **Editor** | The “authoring surface” (Godot editor, slightly modified) | Full‑blown Godot editor UI | Run the executable with the `--no-window` flag (or similar) and a launch script that spawns the editor. |
| **Matrix 1** | First “inside” game – a simple platformer | Standard 2‑D platformer world | Player starts in this world; on a “break‑out” event the engine loads Matrix 2. |
| **Matrix 2, 3…** | New gameplay paradigms (e.g. puzzle, top‑down shooter, text‑based, VR, etc.) | Whatever you design for that layer | Triggered by a scripted “break‑out” that calls a scene loader. |
| **Final Layer – The “Computer”** | The world’s terminal that is actually the editor UI | A terminal screen that hosts the Godot editor | When the player enters, the editor UI is instantiated inside the game. |
| **Exit** | Switch to full‑editor mode (credits, option to re‑play) | Credits text, dialogue window | When the player closes the terminal, the editor window replaces the game window. |

The whole pipeline is *live* – every time you hit **Save** in the editor, the engine automatically rebuilds (or hot‑reloads) the 
artifacts so that the next time you press “Play” you’re running the newest code.

---

### 2. Project Architecture

```
mygame/
├─ addons/
│   └─ live_rebuild/           ← Custom EditorPlugin
├─ scenes/
│   ├─ matrix1.tscn
│   ├─ matrix2.tscn
│   ├─ ...
│   └─ terminal_editor.tscn   ← Scene that hosts the editor UI
├─ scripts/
│   ├─ MatrixLoader.gd         ← Singleton that switches matrices
│   ├─ BreakOut.gd             ← Script that signals a breakout
│   └─ EditorBridge.gd         ← Handles showing the editor in‑game
├─ editor_plugins.cfg          ← Register your plugin
├─ default_env.tres
├─ main.tscn                   ← Root of the game
└─ autoload/
    └─ GameState.gd            ← Keeps track of level, has “isEditing” flag
```

**Key points**

| Component | Role |
|-----------|------|
| **GameState** | Stores whether we’re in “game” or “editor” mode, which matrix is active, etc. |
| **MatrixLoader** | Handles scene switching (`change_scene_to_file()`) and optionally fades. |
| **BreakOut** | Any node can call `MatrixLoader.break_out(target_matrix)`. |
| **EditorBridge** | Loads the Godot editor UI inside the game (see “Editor Integration” below). |
| **LiveRebuild** | An EditorPlugin that watches the filesystem and triggers re‑compile or re‑run on save. |

---

### 3. Technical Implementation

#### 3.1. Making the “Godot editor” launch automatically

1. **Launch script** (`start.sh` or `start.bat`):

   ```bash
   # Linux / macOS
   godot --no-window  # start the editor head‑less
   # or, for a minimal UI:
   # godot --editor -s addons/live_rebuild/autostart.gd
   ```

   The script will fork the editor and then start your game by launching `godot -s main.gd` once the editor finishes initialization.

2. **Editor flag** – Add a custom command‑line flag `--game-launch`.  
   In `addons/live_rebuild/autostart.gd` check for that flag and, if present, automatically switch to the game scene after the editor 
finishes loading.

#### 3.2. Live Rebuild Plugin (`addons/live_rebuild/LiveRebuild.gd`)

```gdscript
# addons/live_rebuild/LiveRebuild.gd
tool
extends EditorPlugin

var watchdog : FileSystemWatcher

func _enter_tree():
    watchdog = FileSystemWatcher.new()
    watchdog.connect("file_changed", self, "_on_file_changed")
    add_child(watchdog)
    watchdog.watch("res://")          # watch entire project

func _on_file_changed(path):
    # Simple approach: recompile scripts
    # In Godot 4.x you can call ProjectSettings.save() + ResourceCache.reload()
    # Or, if you want a full rebuild:
    var result = OS.execute("godot", ["--headless", "--export", "Standalone", "rebuild.tscn"], true)
    print("Rebuild triggered: ", result)

func _exit_tree():
    watchdog.disconnect("file_changed", self, "_on_file_changed")
```

*Tip:* For a *faster* experience you can just use Godot’s built‑in **“Reload Changed Files”** action (or hook into 
`EditorPlugin.reload_changed_scripts`). The snippet above is a minimal example.

#### 3.3. MatrixLoader (autoloaded)

```gdscript
# scripts/MatrixLoader.gd
extends Node
class_name MatrixLoader

signal matrix_changed(new_matrix_name)

func break_out(next_matrix_path : String):
    get_tree().change_scene_to_file(next_matrix_path)
    emit_signal("matrix_changed", next_matrix_path)
```

Add it to **AutoLoad** → `autoload/MatrixLoader.gd` so it’s globally available.

#### 3.4. BreakOut logic

Any node can trigger a breakout by emitting a signal or calling the loader directly:

```gdscript
# In a platformer enemy
func _on_Player_entered():
    MatrixLoader.break_out("res://scenes/matrix2.tscn")
```

You can also add a visual “breakout” effect (fade to black, screen warp, etc.) before the scene change.

#### 3.5. Terminal Editor Scene

`terminal_editor.tscn` will contain:

```
Control (root)
├─ Panel
│  └─ EditorInterface (see below)
└─ Button (Close) -> closes and returns to editor window
```

##### 3.5.1. Embedding the Editor UI

Godot 4.x exposes the `EditorInterface` API. In a script you can instantiate the editor UI like this:

```gdscript
# scripts/EditorBridge.gd
extends Node

var editor_interface : EditorInterface
var editor_control : Control

func _ready():
    # Grab the existing editor instance
    editor_interface = Engine.get_editor_interface()
    # Create a Control to host the editor
    editor_control = editor_interface.get_editor_viewport()
    add_child(editor_control)
    editor_control.set_visible(true)
    # Optionally, disable the rest of the UI or lock the input
    Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
```

When you close the terminal (e.g. click the “Close” button), simply call:

```gdscript
get_parent().queue_free()
# Switch back to the editor window:
Engine.set_editor_hint(true)   # Re‑enable editor mode
```

**Caveat:** In head‑less builds (the version you ship), the editor UI won’t be visible. You must run the *same* Godot binary in editor 
mode (i.e. start with `--editor` or `--no-window` + `--game-launch` as described above).

---

### 4. Build & Deployment Pipeline

| Step | Tool | What happens |
|------|------|--------------|
| **1. Edit** | Godot Editor (modified) | You add/modify scenes, scripts, assets. |
| **2. Save** | `LiveRebuild` plugin | Detects file change, triggers re‑compile or hot‑reload. |
| **3. Build** | `godot --headless --export` | Generates a platform‑specific executable (Windows, macOS, etc.). |
| **4. Test** | Run the built executable | Loads the editor UI (since we start with `--editor`) then automatically launches the game 
scene. |
| **5. Play** | User plays | Game runs; at the terminal it spawns the editor UI. |
| **6. Exit** | Terminal “Close” | Swaps back to editor mode, shows credits. |

*Automation tip:* Add a **Makefile** or **ninja** script that runs `godot --headless --export` whenever a `.gd` file changes. For a 
quick iteration loop you can use `godot --headless --editor` to just watch and hot‑reload.

---

### 5. Runtime Flow (Pseudo‑Code)

```text
main.tscn (root)
├─ Node (GameState)
├─ MatrixLoader (autoload)
├─ current_matrix (instance of matrix1.tscn)
```

**On startup**

1. `GameState.isEditing = false`
2. `MatrixLoader.break_out("res://scenes/matrix1.tscn")`

**When player triggers a breakout**

```gdscript
MatrixLoader.break_out("res://scenes/matrix2.tscn")
```

**When player enters terminal**

```gdscript
# In terminal node
func _on_Terminal_body_entered(body):
    var editor_scene = preload("res://scenes/terminal_editor.tscn").instantiate()
    get_parent().add_child(editor_scene)
```

**When terminal is closed**

```gdscript
func _on_Close_pressed():
    queue_free()
    # Return to editor mode
    Engine.set_editor_hint(true)
    # Show credits
    get_tree().change_scene_to_file("res://credits.tscn")
```

---

### 6. Extensibility & Gotchas

| Area | Idea | Pitfall |
|------|------|----------|
| **Adding new matrices** | Create a new `.tscn`, give it a unique name, register it in a config file or in `MatrixLoader`. | Remember 
to keep transitions smooth; avoid massive asset jumps that cause long load times. |
| **Live reload of 3D assets** | Godot can hot‑reload most resources, but large textures can take time. | Use `ResourceCache.reload()` 
for specific assets if you need instant feedback. |
| **Editor in-game on other platforms** | Windows, Linux, macOS all support editor mode. For mobile you’ll need a custom UI or skip the 
editor step. | Mobile OSes restrict “editor mode”; test on each target platform. |
| **State persistence** | Store progress (current matrix, inventory, etc.) in `GameState` or a separate `SaveGame` autoload. | Make 
sure the editor side can read/write the same files. |
| **Security** | Don’t ship the raw editor binary with your game (users could hack the editor). | Use the “export” version for 
distribution; keep the `--editor` flag for your dev builds. |

---

### 7. Minimal Working Example (Godot 4.x)

1. **Create a new project** (`mygame`).  
2. **Add the `addons/live_rebuild/LiveRebuild.gd`** (see 3.2).  
3. **Add `MatrixLoader.gd`** to `autoload`.  
4. **Create two simple scenes**: `matrix1.tscn` (a sprite + a Button that calls `break_out("res://matrix2.tscn")`) and `matrix2.tscn` 
(another simple scene).  
5. **Create `terminal_editor.tscn`** that just contains a `Button` which on press adds `EditorBridge` to the scene.  
6. **In `main.tscn`** add a `Node` called `GameState` with a script that has a single `var isEditing = false`.  
7. **Launch**: `godot --editor` → the editor opens. Press `Play` → Godot starts the game (Matrix 1).  
8. **Play**: Click the button → Matrix 2.  
9. **Trigger terminal** → you see the editor UI pop up inside the game.  
10. **Close terminal** → you’re back in the editor, and you can edit again.

---

### 8. Final Thoughts

* **“Recursive” is really a *meta‑loop* of authoring → playing → editing.**  
* Keep the editor lightweight; you’re effectively running a full Godot process twice in one runtime.  
* For production you’ll probably ship **two binaries**:  
  1. The “editor‑only” build (for dev).  
  2. The “game‑only” build (for players) that *cannot* enter the editor mode (unless you add a hidden cheat code).  
* The most elegant experience is to have the game start **inside** the editor, so the developer can “play” instantly.  
* The final “break‑out” into the editor is a powerful narrative device: the player literally “re‑enters” the code that created them.

Happy building! If you hit a snag or need sample scripts for a particular matrix (e.g. a text‑based layer or a physics‑heavy puzzle), 
let me know and I’ll dig out more code.
