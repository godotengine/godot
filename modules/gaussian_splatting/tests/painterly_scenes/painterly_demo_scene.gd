extends Node2D

@export_file("*.json") var definition_path: String
@export var animate_camera: bool = false
@export var animation_duration: float = 4.0
@export var output_size: Vector2i = Vector2i(256, 256)

var _scene: Dictionary = {}
var _splats: Array = []
var _elapsed: float = 0.0
var _sprite: Sprite2D

## Loads the painterly scene definition, compiles shader permutations, and renders
## the initial frame.
func _ready() -> void:
    _sprite = Sprite2D.new()
    add_child(_sprite)
    _sprite.centered = true
    _sprite.position = Vector2.ZERO

    if definition_path.is_empty():
        push_warning("Painterly demo scene requires a JSON definition path")
        return

    var definition := PainterlySceneUtil.load_definition(definition_path)
    if definition.is_empty():
        push_error("Failed to load painterly definition: %s" % definition_path)
        return

    _scene = PainterlySceneUtil.build_scene(definition, definition_path)
    _splats = PainterlySceneUtil.generate_gaussians(_scene)

    var rd := PainterlySceneUtil.ensure_rendering_device()
    if rd:
        var results := PainterlySceneUtil.compile_permutations(_scene, rd)
        for perm_name in results:
            if not results[perm_name]:
                push_warning("Shader permutation failed to compile: %s" % perm_name)

    _render_frame(0)

## Advances the camera animation (if enabled) and re-renders the current frame.
## @param delta: Frame delta in seconds.
func _process(delta: float) -> void:
    if not animate_camera:
        return
    if _scene.is_empty():
        return
    _elapsed = fmod(_elapsed + delta, max(animation_duration, 0.1))
    var camera_keys: Array = _scene.get("camera_path", [])
    if camera_keys.size() <= 1:
        return
    var progress := clamp(_elapsed / animation_duration, 0.0, 1.0)
    var frame_index := int(progress * (camera_keys.size() - 1))
    _render_frame(frame_index)

## Renders the scene from the specified camera keyframe index.
## @param camera_index: Index into the scene camera path.
func _render_frame(camera_index: int) -> void:
    if _scene.is_empty():
        return
    var image := PainterlySceneUtil.render_headless(_scene, _splats, camera_index, output_size)
    _display_image(image)

## Updates the sprite texture with the rendered image.
## @param image: Rendered image to display.
func _display_image(image: Image) -> void:
    if image == null:
        return
    var texture := ImageTexture.create_from_image(image)
    _sprite.texture = texture
    _sprite.position = Vector2(output_size.x, output_size.y) * 0.5
