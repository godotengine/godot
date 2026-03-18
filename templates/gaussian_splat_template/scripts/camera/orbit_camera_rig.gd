extends Node3D
class_name OrbitCameraRig

@export var camera: Camera3D
@export var move_speed: float = 6.0
@export var boost_multiplier: float = 3.0
@export var orbit_sensitivity: float = 0.004
@export var pan_sensitivity: float = 0.01
@export var zoom_speed: float = 2.0

var _orbiting: bool = false
var _panning: bool = false
var _yaw: float = 0.0
var _pitch: float = 0.0

## Initializes the camera reference and cached orbit angles.
func _ready() -> void:
    if camera == null:
        camera = get_node_or_null("Camera3D")
    _yaw = rotation.y
    _pitch = rotation.x

## Repositions the rig to frame the provided bounds.
## @param bounds: Axis-aligned bounds to focus on.
func focus(bounds: AABB) -> void:
    if camera == null:
        return
    var center: Vector3 = bounds.position + bounds.size * 0.5
    global_transform.origin = center
    var extent: float = max(bounds.size.length(), 1.0)
    var distance: float = extent * 1.5
    camera.transform.origin = Vector3(0, extent * 0.35, distance)
    rotation = Vector3.ZERO
    camera.look_at(center, Vector3.UP)
    _yaw = rotation.y
    _pitch = rotation.x

## Handles keyboard-driven translation for the orbit rig.
## @param delta: Frame delta in seconds.
func _physics_process(delta: float) -> void:
    var velocity: Vector3 = Vector3.ZERO
    if Input.is_action_pressed("camera_move_forward"):
        velocity -= transform.basis.z
    if Input.is_action_pressed("camera_move_backward"):
        velocity += transform.basis.z
    if Input.is_action_pressed("camera_move_left"):
        velocity -= transform.basis.x
    if Input.is_action_pressed("camera_move_right"):
        velocity += transform.basis.x
    if Input.is_action_pressed("camera_move_up"):
        velocity += transform.basis.y
    if Input.is_action_pressed("camera_move_down"):
        velocity -= transform.basis.y

    if velocity != Vector3.ZERO:
        velocity = velocity.normalized()
        var speed: float = move_speed
        if Input.is_action_pressed("camera_boost_speed"):
            speed *= boost_multiplier
        global_translate(velocity * speed * delta)

## Dispatches mouse events to orbit or pan handlers.
## @param event: Input event from the scene tree.
func _unhandled_input(event: InputEvent) -> void:
    if event is InputEventMouseButton:
        _handle_mouse_button(event)
    elif event is InputEventMouseMotion:
        _handle_mouse_motion(event)

## Updates orbit/pan state and applies zoom on wheel input.
## @param event: Mouse button event.
func _handle_mouse_button(event: InputEventMouseButton) -> void:
    match event.button_index:
        MOUSE_BUTTON_RIGHT:
            _orbiting = event.pressed
            if _orbiting:
                Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
            else:
                Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
        MOUSE_BUTTON_MIDDLE:
            _panning = event.pressed
        MOUSE_BUTTON_WHEEL_UP:
            if event.pressed:
                _zoom(-zoom_speed)
        MOUSE_BUTTON_WHEEL_DOWN:
            if event.pressed:
                _zoom(zoom_speed)

## Applies orbit rotation or panning based on the current input state.
## @param event: Mouse motion event.
func _handle_mouse_motion(event: InputEventMouseMotion) -> void:
    if _orbiting:
        _yaw -= event.relative.x * orbit_sensitivity
        _pitch -= event.relative.y * orbit_sensitivity
        _pitch = clamp(_pitch, deg_to_rad(-85.0), deg_to_rad(85.0))
        rotation = Vector3(_pitch, _yaw, 0.0)
    elif _panning and camera:
        var right: Vector3 = -camera.global_transform.basis.x
        var up: Vector3 = camera.global_transform.basis.y
        global_translate((right * event.relative.x + up * event.relative.y) * pan_sensitivity)

## Moves the camera along its local forward axis.
## @param amount: Positive or negative zoom distance.
func _zoom(amount: float) -> void:
    if camera == null:
        return
    camera.translate_object_local(Vector3(0, 0, amount))
