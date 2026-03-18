extends Node3D
"""
Gaussian Splatting Demo Controller
Interactive demo showcasing the Gaussian Splatting implementation
"""

# Camera controls
@onready var camera_controller = $CameraController
@onready var camera = $CameraController/Camera3D

# UI elements
@onready var stats_label = $UI/Controls/InfoPanel/VBoxContainer/Stats
@onready var quality_buttons = $UI/Controls/QualityButtons
@onready var load_buttons = $UI/Controls/LoadButtons
@onready var splat_container = $SplatContainer

# Settings
var camera_speed = 10.0
var mouse_sensitivity = 0.002
var current_quality = "ultra"
var auto_rotate = false
var rotation_speed = 0.5

# Performance tracking
var frame_count = 0
var fps_update_timer = 0.0
var current_fps = 0

# Splat nodes
var active_splat_nodes = []

# Test data files
var test_files = {
    "1k": "res://test_data/small_sphere_1k.ply",
    "100k": "res://test_data/medium_sphere_100k.ply",
    "1m": "res://test_data/large_sphere_1m.ply",
    "bunny": "res://test_data/small_bunny_1k.ply",
    "cube": "res://test_data/small_cube_1k.ply",
}

## Wires UI signals, configures input capture, and loads the default splat file.
func _ready():
    # Connect button signals
    $UI/Controls/QualityButtons/LowBtn.pressed.connect(_on_quality_low)
    $UI/Controls/QualityButtons/MediumBtn.pressed.connect(_on_quality_medium)
    $UI/Controls/QualityButtons/HighBtn.pressed.connect(_on_quality_high)
    $UI/Controls/QualityButtons/UltraBtn.pressed.connect(_on_quality_ultra)

    $UI/Controls/LoadButtons/Load1K.pressed.connect(_on_load_1k)
    $UI/Controls/LoadButtons/Load100K.pressed.connect(_on_load_100k)
    $UI/Controls/LoadButtons/Load1M.pressed.connect(_on_load_1m)
    $UI/Controls/LoadButtons/LoadMulti.pressed.connect(_on_load_multiple)

    # Capture mouse for camera control
    Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

    # Load initial scene
    _load_splat_file(test_files["1k"], Vector3.ZERO)

    print("Gaussian Splatting Demo Started")
    print("Press ESC to toggle mouse capture")

## Instantiates a GaussianSplatNode3D and validates the module is available.
## @return New splat node or null when unavailable.
func _instantiate_splat_node() -> Node3D:
    var node := ClassDB.instantiate("GaussianSplatNode3D") as Node3D
    if node == null:
        push_error("GaussianSplatNode3D class is not available")
        return null
    return node

## Handles camera mouse look, hotkeys, and quality toggles.
## @param event: Input event dispatched by the scene tree.
func _input(event):
    # Handle mouse look
    if event is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
        camera_controller.rotate_y(-event.relative.x * mouse_sensitivity)
        camera.rotate_x(-event.relative.y * mouse_sensitivity)

        # Clamp vertical rotation
        var rotation = camera.rotation
        rotation.x = clamp(rotation.x, -PI/2 + 0.1, PI/2 - 0.1)
        camera.rotation = rotation

    # Toggle mouse capture
    if event.is_action_pressed("ui_cancel"):
        if Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
            Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
        else:
            Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

    # Quality hotkeys
    if event.is_action_pressed("ui_text_indent"):  # Tab cycles quality
        _cycle_quality()

    # Number keys for quality
    if event is InputEventKey and event.pressed:
        match event.keycode:
            KEY_1:
                _on_quality_low()
            KEY_2:
                _on_quality_medium()
            KEY_3:
                _on_quality_high()
            KEY_4:
                _on_quality_ultra()
            KEY_SPACE:
                auto_rotate = not auto_rotate
                print("Auto-rotate: " + str(auto_rotate))

## Updates camera movement, auto-rotation, and UI statistics per frame.
## @param delta: Frame delta in seconds.
func _process(delta):
    # Update camera movement
    _handle_camera_movement(delta)

    # Auto-rotate splats
    if auto_rotate:
        for node in active_splat_nodes:
            node.rotate_y(rotation_speed * delta)

    # Update FPS counter
    _update_fps(delta)

    # Update stats display
    _update_stats()

## Applies keyboard-driven camera translation.
## @param delta: Frame delta in seconds.
func _handle_camera_movement(delta):
    var movement = Vector3.ZERO

    # Keyboard movement
    if Input.is_action_pressed("ui_up") or Input.is_key_pressed(KEY_W):
        movement -= camera_controller.transform.basis.z
    if Input.is_action_pressed("ui_down") or Input.is_key_pressed(KEY_S):
        movement += camera_controller.transform.basis.z
    if Input.is_action_pressed("ui_left") or Input.is_key_pressed(KEY_A):
        movement -= camera_controller.transform.basis.x
    if Input.is_action_pressed("ui_right") or Input.is_key_pressed(KEY_D):
        movement += camera_controller.transform.basis.x
    if Input.is_key_pressed(KEY_Q):
        movement -= Vector3.UP
    if Input.is_key_pressed(KEY_E):
        movement += Vector3.UP

    # Speed boost with shift
    var speed = camera_speed
    if Input.is_key_pressed(KEY_SHIFT):
        speed *= 2.0

    # Apply movement
    if movement.length() > 0:
        movement = movement.normalized()
        camera_controller.position += movement * speed * delta

## Accumulates FPS samples once per second for UI display.
## @param delta: Frame delta in seconds.
func _update_fps(delta):
    frame_count += 1
    fps_update_timer += delta

    if fps_update_timer >= 1.0:
        current_fps = frame_count / fps_update_timer
        frame_count = 0
        fps_update_timer = 0.0

## Updates the on-screen stats label with splat and memory information.
func _update_stats():
    var total_splats = 0
    var total_memory = 0.0

    for node in active_splat_nodes:
        if node.has_method("get_splat_count"):
            total_splats += node.get_splat_count()

        # Estimate memory usage (68 bytes per splat)
        if node.has_method("get_splat_count"):
            total_memory += node.get_splat_count() * 68.0 / 1048576.0

    stats_label.text = "FPS: %.1f\nSplats: %s\nMemory: %.1f MB\nQuality: %s" % [
        current_fps,
        _format_number(total_splats),
        total_memory,
        current_quality.capitalize()
    ]

## Formats large numbers into human-readable K/M strings.
## @param num: Value to format.
## @return Formatted string.
func _format_number(num: int) -> String:
    if num >= 1000000:
        return "%.1fM" % (num / 1000000.0)
    elif num >= 1000:
        return "%.1fK" % (num / 1000.0)
    else:
        return str(num)

## Instantiates a splat node for the given file and places it in the scene.
## @param file_path: PLY file path to load.
## @param position: World position to place the node.
func _load_splat_file(file_path: String, position: Vector3):
    print("Loading: " + file_path)

    var splat_node := _instantiate_splat_node()
    if splat_node == null:
        return

    splat_node.ply_file_path = file_path
    splat_node.quality_preset = current_quality
    splat_node.position = position

    splat_container.add_child(splat_node)
    active_splat_nodes.append(splat_node)

## Removes and frees all currently loaded splat nodes.
func _clear_all_splats():
    for node in active_splat_nodes:
        node.queue_free()
    active_splat_nodes.clear()

## UI callback to switch to low quality preset.
func _on_quality_low():
    _set_quality("low")
    _highlight_quality_button(0)

## UI callback to switch to medium quality preset.
func _on_quality_medium():
    _set_quality("medium")
    _highlight_quality_button(1)

## UI callback to switch to high quality preset.
func _on_quality_high():
    _set_quality("high")
    _highlight_quality_button(2)

## UI callback to switch to ultra quality preset.
func _on_quality_ultra():
    _set_quality("ultra")
    _highlight_quality_button(3)

## Applies the quality preset to new and existing splat nodes.
## @param quality: Preset name string.
func _set_quality(quality: String):
    current_quality = quality
    print("Quality set to: " + quality)

    # Update existing nodes
    for node in active_splat_nodes:
        if node.has_method("set_quality_preset"):
            node.set_quality_preset(quality)

## Highlights the active quality button and resets others.
## @param index: Button index to highlight.
func _highlight_quality_button(index: int):
    # Reset all buttons
    for i in range(quality_buttons.get_child_count()):
        var btn = quality_buttons.get_child(i)
        btn.modulate = Color.WHITE

    # Highlight selected
    quality_buttons.get_child(index).modulate = Color(0.5, 1, 0.5, 1)

## Cycles through quality presets in order.
func _cycle_quality():
    match current_quality:
        "low":
            _on_quality_medium()
        "medium":
            _on_quality_high()
        "high":
            _on_quality_ultra()
        "ultra":
            _on_quality_low()

## UI callback to load the 1K splat dataset.
func _on_load_1k():
    _clear_all_splats()
    _load_splat_file(test_files["1k"], Vector3.ZERO)

## UI callback to load the 100K splat dataset.
func _on_load_100k():
    _clear_all_splats()
    _load_splat_file(test_files["100k"], Vector3.ZERO)

## UI callback to load the 1M splat dataset.
func _on_load_1m():
    _clear_all_splats()
    _load_splat_file(test_files["1m"], Vector3.ZERO)

## UI callback to load multiple splat instances at preset positions.
func _on_load_multiple():
    _clear_all_splats()

    # Load multiple different models
    var positions = [
        Vector3(-5, 0, 0),   # Sphere
        Vector3(0, 0, 0),    # Bunny
        Vector3(5, 0, 0),    # Cube
    ]

    _load_splat_file(test_files["1k"], positions[0])
    _load_splat_file(test_files["bunny"], positions[1])
    _load_splat_file(test_files["cube"], positions[2])

    print("Loaded multiple splat instances")

## Handles cleanup when the window close request is received.
## @param what: Notification identifier.
func _notification(what):
    if what == NOTIFICATION_WM_CLOSE_REQUEST:
        # Clean up on exit
        print("Demo closing, cleaning up...")
        _clear_all_splats()
        get_tree().quit()
