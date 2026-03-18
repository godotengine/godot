extends "res://scripts/qa_test_base.gd"
## Sorting Depth Order Test: Ensures nearer splat dominates when overlapping.

@export var capture_delay_frames: int = 8
@export var capture_timeout_frames: int = 60

var splat_node: GaussianSplatNode3D
var _captured_color: Color
var _capture_ready := false
var _capture_error := ""
var _renderer: Object
var _prev_lighting_settings := {}
var _prev_animation_settings := {}
var _prev_renderer_settings := {}

func _ready():
	test_name = "Sort Depth Order"
	test_duration = 3.0
	warmup_frames = 5
	super._ready()

	splat_node = get_node_or_null("SplatNode")

func _on_test_start():
	if splat_node == null:
		return

	_prev_lighting_settings["rendering/gaussian_splatting/lighting/indirect_sh_scale"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/lighting/indirect_sh_scale"
	)
	_prev_lighting_settings["rendering/gaussian_splatting/lighting/direct_light_scale"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/lighting/direct_light_scale"
	)
	_prev_animation_settings["rendering/gaussian_splatting/animation/wind_enabled"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/animation/wind_enabled"
	)
	_prev_animation_settings["rendering/gaussian_splatting/animation/wind_strength"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/animation/wind_strength"
	)
	_prev_animation_settings["rendering/gaussian_splatting/effects/max_effectors"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/effects/max_effectors"
	)
	_prev_animation_settings["rendering/gaussian_splatting/effects/sphere_effector_enabled"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/effects/sphere_effector_enabled"
	)
	ProjectSettings.set_setting("rendering/gaussian_splatting/lighting/indirect_sh_scale", 1.0)
	ProjectSettings.set_setting("rendering/gaussian_splatting/lighting/direct_light_scale", 0.0)
	ProjectSettings.set_setting("rendering/gaussian_splatting/animation/wind_enabled", false)
	ProjectSettings.set_setting("rendering/gaussian_splatting/animation/wind_strength", 0.0)
	ProjectSettings.set_setting("rendering/gaussian_splatting/effects/max_effectors", 0)
	ProjectSettings.set_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", false)

	_renderer = get_gs_renderer("SplatNode")
	if _renderer != null:
		if _renderer.has_method("get_tiny_splat_screen_radius"):
			_prev_renderer_settings["tiny_splat_screen_radius"] = _renderer.get_tiny_splat_screen_radius()
		if _renderer.has_method("set_tiny_splat_screen_radius"):
			_renderer.set_tiny_splat_screen_radius(0.0)

		if _renderer.has_method("get_importance_cull_threshold"):
			_prev_renderer_settings["importance_cull_threshold"] = _renderer.get_importance_cull_threshold()
		if _renderer.has_method("set_importance_cull_threshold"):
			_renderer.set_importance_cull_threshold(0.0)

		if _renderer.has_method("is_static_sort_cache_enabled"):
			_prev_renderer_settings["static_sort_cache_enabled"] = _renderer.is_static_sort_cache_enabled()
		if _renderer.has_method("set_static_sort_cache_enabled"):
			_renderer.set_static_sort_cache_enabled(false)

		if _renderer.has_method("is_cached_render_reuse_enabled"):
			_prev_renderer_settings["cached_render_reuse_enabled"] = _renderer.is_cached_render_reuse_enabled()
		if _renderer.has_method("set_cached_render_reuse_enabled"):
			_renderer.set_cached_render_reuse_enabled(false)

	var asset := GaussianSplatAsset.new()
	asset.set_splat_count(2)

	var positions := PackedFloat32Array([0.0, 0.0, -2.0, 0.0, 0.0, -1.0])
	var colors := PackedColorArray([Color(0.0, 0.0, 1.0, 1.0), Color(1.0, 0.0, 0.0, 1.0)])
	var scales := PackedFloat32Array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])

	asset.set_positions(positions)
	asset.set_colors(colors)
	asset.set_scales(scales)

	splat_node.splat_asset = asset

func _on_test_frame(_delta: float):
	if _capture_ready:
		return
	if frame_count < capture_delay_frames:
		append_renderer_diagnostics("", _renderer)
		return

	append_renderer_diagnostics("", _renderer)

	if splat_node != null and splat_node.has_method("get_visible_splat_count"):
		var visible_splats = int(splat_node.get_visible_splat_count())
		result_metrics["visible_splats"] = visible_splats
		if visible_splats <= 0:
			if frame_count < capture_timeout_frames:
				return
			_capture_error = "No visible splats before capture timeout"
			_capture_ready = true
			_finish_test()
			return

	var image = capture_viewport()
	if image == null:
		return
	var center = Vector2i(image.get_width() / 2, image.get_height() / 2)
	_captured_color = _sample_center_color(image, center, 1)
	_capture_ready = true
	_finish_test()

func _on_test_complete():
	append_renderer_diagnostics("", _renderer)

	for key in _prev_lighting_settings.keys():
		ProjectSettings.set_setting(key, _prev_lighting_settings[key])
	for key in _prev_animation_settings.keys():
		ProjectSettings.set_setting(key, _prev_animation_settings[key])

	if _renderer != null:
		if _prev_renderer_settings.has("tiny_splat_screen_radius") and _renderer.has_method("set_tiny_splat_screen_radius"):
			_renderer.set_tiny_splat_screen_radius(_prev_renderer_settings["tiny_splat_screen_radius"])
		if _prev_renderer_settings.has("importance_cull_threshold") and _renderer.has_method("set_importance_cull_threshold"):
			_renderer.set_importance_cull_threshold(_prev_renderer_settings["importance_cull_threshold"])
		if _prev_renderer_settings.has("static_sort_cache_enabled") and _renderer.has_method("set_static_sort_cache_enabled"):
			_renderer.set_static_sort_cache_enabled(_prev_renderer_settings["static_sort_cache_enabled"])
		if _prev_renderer_settings.has("cached_render_reuse_enabled") and _renderer.has_method("set_cached_render_reuse_enabled"):
			_renderer.set_cached_render_reuse_enabled(_prev_renderer_settings["cached_render_reuse_enabled"])

	if not _capture_error.is_empty():
		_test_result = false
		_test_message = _capture_error
		return

	if not _capture_ready:
		_test_result = false
		_test_message = "No capture"
		return

	result_metrics["center_color"] = _captured_color
	var red_dominant = _captured_color.r > (_captured_color.b + 0.2) and _captured_color.r > 0.4

	_test_result = red_dominant
	_test_message = "center_color r=%.3f g=%.3f b=%.3f" % [_captured_color.r, _captured_color.g, _captured_color.b]

func _sample_center_color(image: Image, center: Vector2i, radius: int = 1) -> Color:
	var min_x = max(0, center.x - radius)
	var max_x = min(image.get_width() - 1, center.x + radius)
	var min_y = max(0, center.y - radius)
	var max_y = min(image.get_height() - 1, center.y + radius)
	var accum = Color(0.0, 0.0, 0.0, 0.0)
	var count = 0

	for y in range(min_y, max_y + 1):
		for x in range(min_x, max_x + 1):
			accum += image.get_pixel(x, y)
			count += 1

	if count <= 0:
		return Color(0.0, 0.0, 0.0, 1.0)
	return accum * (1.0 / float(count))
