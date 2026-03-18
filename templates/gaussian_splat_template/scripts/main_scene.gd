extends Node3D
class_name GaussianTemplateRoot

@export_file("*.ply") var default_ply_path: String = "res://assets/template_splats.ply"
@export var enable_demo_painterly: bool = true

@onready var gaussian_node: GaussianSplatNode3D = $GaussianSplatNode3D
@onready var performance_overlay: GaussianPerformanceOverlay = $CanvasLayer/PerformanceOverlay
@onready var camera_rig: OrbitCameraRig = $CameraRig

## Configures the template scene by wiring the node, overlay, and camera focus.
func _ready() -> void:
    _configure_gaussian_node()
    _wire_overlay()
    _focus_camera()

## Applies template defaults to the GaussianSplatNode3D instance.
func _configure_gaussian_node() -> void:
    if gaussian_node == null:
        push_error("GaussianSplatNode3D is missing from the template scene")
        return

    if default_ply_path != "" and gaussian_node.get_ply_file_path() == "":
        gaussian_node.set_ply_file_path(default_ply_path)

    gaussian_node.set_auto_load(true)
    gaussian_node.set_quality_preset(GaussianSplatNode3D.QUALITY_BALANCED)
    gaussian_node.set_lod_bias(1.0)
    gaussian_node.set_max_render_distance(150.0)
    gaussian_node.set_max_splat_count(750000)

    gaussian_node.set_enable_painterly(enable_demo_painterly)
    gaussian_node.set_edge_threshold(0.25)
    gaussian_node.set_stroke_opacity(0.85)
    gaussian_node.set_stroke_width(1.1)
    gaussian_node.set_color_variation(0.12)
    gaussian_node.set_temporal_blend(0.35)
    gaussian_node.set_painterly_seed(1337)

    gaussian_node.set_update_mode(GaussianSplatNode3D.UPDATE_MODE_WHEN_VISIBLE)
    gaussian_node.set_cast_shadow(true)
    gaussian_node.set_use_frustum_culling(true)
    gaussian_node.set_use_occlusion_culling(true)
    gaussian_node.set_opacity(1.0)

    gaussian_node.set_preview_enabled(true)
    gaussian_node.set_show_bounds(false)
    gaussian_node.set_show_statistics(false)
    gaussian_node.set_show_tile_grid(false)
    gaussian_node.set_show_density_heatmap(false)
    gaussian_node.set_show_performance_hud(false)
    gaussian_node.set_show_performance_overlay(false)
    gaussian_node.set_debug_draw_mode(GaussianSplatNode3D.DEBUG_DRAW_POINTS)
    gaussian_node.set_runtime_preview_enabled(false)
    gaussian_node.set_show_residency_hud(false)

## Binds the overlay to the gaussian node and camera rig.
func _wire_overlay() -> void:
    if performance_overlay:
        performance_overlay.set_gaussian_node(gaussian_node)
        performance_overlay.set_camera_node(camera_rig)

## Centers the orbit camera on the current Gaussian bounds.
func _focus_camera() -> void:
    if camera_rig and gaussian_node:
        var bounds: AABB = gaussian_node.get_aabb()
        if bounds.size.length() > 0.0:
            camera_rig.focus(bounds)
