extends "res://scripts/qa_test_base.gd"
## Sorting Multi-Instance Test: Ensures ordering across instances is correct.

@export var capture_delay_frames: int = 8
@export var capture_timeout_frames: int = 60

var near_node: GaussianSplatNode3D
var far_node: GaussianSplatNode3D
var _captured_color: Color
var _capture_ready := false
var _capture_error := ""
var _near_renderer: Object
var _far_renderer: Object
var _prev_lighting_settings := {}
var _prev_animation_settings := {}
var _prev_near_renderer_settings := {}
var _prev_far_renderer_settings := {}
var _renderer_override_snapshots := {}
var _renderer_override_refcounts := {}

func _ready():
	test_name = "Sort Multi-Instance"
	test_duration = 3.0
	warmup_frames = 5
	super._ready()

	near_node = get_node_or_null("NearInstance")
	far_node = get_node_or_null("FarInstance")

func _on_test_start():
	if near_node == null or far_node == null:
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

	_near_renderer = get_gs_renderer("NearInstance")
	_far_renderer = get_gs_renderer("FarInstance")
	_apply_renderer_overrides(_near_renderer, _prev_near_renderer_settings)
	_apply_renderer_overrides(_far_renderer, _prev_far_renderer_settings)

	var near_asset := GaussianSplatAsset.new()
	near_asset.set_splat_count(1)
	near_asset.set_positions(PackedFloat32Array([0.0, 0.0, -1.0]))
	near_asset.set_colors(PackedColorArray([Color(1.0, 0.0, 0.0, 1.0)]))
	near_asset.set_scales(PackedFloat32Array([0.7, 0.7, 0.7]))

	var far_asset := GaussianSplatAsset.new()
	far_asset.set_splat_count(1)
	far_asset.set_positions(PackedFloat32Array([0.0, 0.0, -2.0]))
	far_asset.set_colors(PackedColorArray([Color(0.0, 0.0, 1.0, 1.0)]))
	far_asset.set_scales(PackedFloat32Array([0.7, 0.7, 0.7]))

	near_node.splat_asset = near_asset
	far_node.splat_asset = far_asset

func _on_test_frame(_delta: float):
	if _capture_ready:
		return
	if frame_count < capture_delay_frames:
		append_renderer_diagnostics("near", _near_renderer)
		append_renderer_diagnostics("far", _far_renderer)
		return

	append_renderer_diagnostics("near", _near_renderer)
	append_renderer_diagnostics("far", _far_renderer)

	var near_visible := 0
	var far_visible := 0
	if near_node != null and near_node.has_method("get_visible_splat_count"):
		near_visible = int(near_node.get_visible_splat_count())
	if far_node != null and far_node.has_method("get_visible_splat_count"):
		far_visible = int(far_node.get_visible_splat_count())
	result_metrics["near_visible_splats"] = near_visible
	result_metrics["far_visible_splats"] = far_visible
	result_metrics["visible_splats"] = near_visible + far_visible

	if near_visible <= 0 or far_visible <= 0:
		if frame_count < capture_timeout_frames:
			return
		_capture_error = "Instances not both visible before capture timeout (near=%d far=%d)" % [near_visible, far_visible]
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
	append_renderer_diagnostics("near", _near_renderer)
	append_renderer_diagnostics("far", _far_renderer)

	for key in _prev_lighting_settings.keys():
		ProjectSettings.set_setting(key, _prev_lighting_settings[key])
	for key in _prev_animation_settings.keys():
		ProjectSettings.set_setting(key, _prev_animation_settings[key])

	_restore_renderer_overrides(_near_renderer, _prev_near_renderer_settings)
	_restore_renderer_overrides(_far_renderer, _prev_far_renderer_settings)

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

func _snapshot_renderer_override_settings(renderer: Object) -> Dictionary:
	var snapshot := {}
	if renderer.has_method("get_tiny_splat_screen_radius"):
		snapshot["tiny_splat_screen_radius"] = renderer.get_tiny_splat_screen_radius()
	if renderer.has_method("get_importance_cull_threshold"):
		snapshot["importance_cull_threshold"] = renderer.get_importance_cull_threshold()
	if renderer.has_method("is_static_sort_cache_enabled"):
		snapshot["static_sort_cache_enabled"] = renderer.is_static_sort_cache_enabled()
	if renderer.has_method("is_cached_render_reuse_enabled"):
		snapshot["cached_render_reuse_enabled"] = renderer.is_cached_render_reuse_enabled()
	return snapshot

func _set_renderer_override_values(renderer: Object, settings: Dictionary) -> void:
	if settings.has("tiny_splat_screen_radius") and renderer.has_method("set_tiny_splat_screen_radius"):
		renderer.set_tiny_splat_screen_radius(settings["tiny_splat_screen_radius"])
	if settings.has("importance_cull_threshold") and renderer.has_method("set_importance_cull_threshold"):
		renderer.set_importance_cull_threshold(settings["importance_cull_threshold"])
	if settings.has("static_sort_cache_enabled") and renderer.has_method("set_static_sort_cache_enabled"):
		renderer.set_static_sort_cache_enabled(settings["static_sort_cache_enabled"])
	if settings.has("cached_render_reuse_enabled") and renderer.has_method("set_cached_render_reuse_enabled"):
		renderer.set_cached_render_reuse_enabled(settings["cached_render_reuse_enabled"])

func _apply_renderer_overrides(renderer: Object, prev_settings: Dictionary) -> void:
	if renderer == null:
		return

	prev_settings.clear()
	var renderer_id := int(renderer.get_instance_id())
	var snapshot: Dictionary
	if _renderer_override_snapshots.has(renderer_id):
		snapshot = _renderer_override_snapshots[renderer_id]
		_renderer_override_refcounts[renderer_id] = int(_renderer_override_refcounts.get(renderer_id, 0)) + 1
	else:
		snapshot = _snapshot_renderer_override_settings(renderer)
		_renderer_override_snapshots[renderer_id] = snapshot
		_renderer_override_refcounts[renderer_id] = 1
	for key in snapshot.keys():
		prev_settings[key] = snapshot[key]

	var override_settings := {
		"tiny_splat_screen_radius": 0.0,
		"importance_cull_threshold": 0.0,
		"static_sort_cache_enabled": false,
		"cached_render_reuse_enabled": false,
	}
	_set_renderer_override_values(renderer, override_settings)

func _restore_renderer_overrides(renderer: Object, prev_settings: Dictionary) -> void:
	if renderer == null:
		return

	var renderer_id := int(renderer.get_instance_id())
	if _renderer_override_refcounts.has(renderer_id):
		var remaining := int(_renderer_override_refcounts[renderer_id]) - 1
		if remaining > 0:
			_renderer_override_refcounts[renderer_id] = remaining
			return
		_renderer_override_refcounts.erase(renderer_id)
		var snapshot = _renderer_override_snapshots.get(renderer_id, prev_settings)
		_renderer_override_snapshots.erase(renderer_id)
		_set_renderer_override_values(renderer, snapshot)
		return

	_set_renderer_override_values(renderer, prev_settings)

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
