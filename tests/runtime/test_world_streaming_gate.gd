extends SceneTree

const SKIP_MARKER := "[RUNTIME_SKIP]"
const FAIL_MARKER := "[RUNTIME_FAIL]"
const METRICS_MARKER := "[RUNTIME_METRICS]"

const ASSET_PATH := "res://tests/fixtures/test_splats.ply"
const MAX_TEST_FRAMES := 300
const CAMERA_MOVE_FRAME := 24
const STREAMING_ROUTE_POLICY_PATH := "rendering/gaussian_splatting/streaming/route_policy"
const STREAMING_ROUTE_RESIDENT := 0
const STREAMING_ROUTE_STREAMING := 1

var scene_root: Node3D
var world_node: GaussianSplatWorld3D
var camera: Camera3D
var manager = null

var _settings_captured := false
var _prev_streaming_route_policy := STREAMING_ROUTE_STREAMING
var _prev_instance_pipeline_enabled := false

var metrics: Dictionary = {
    "frames": 0,
    "generated_chunk_count": 0,
    "monitor_ready_seen": false,
    "total_chunks_max": 0,
    "loaded_chunks_max": 0,
    "visible_chunks_max": 0,
    "chunks_loaded_this_frame_max": 0,
    "streaming_visible_count_max": 0,
    "renderer_visible_splats_max": 0,
    "streaming_data_source_seen": false,
    "renderer_data_source": "",
    "streaming_diagnostics_category": "",
    "streaming_diagnostics_reason": "",
    "streaming_diagnostics_fingerprint": "",
    "streaming_render_readiness_state": "",
    "renderer_route_uid": "",
    "manager_total_gaussians_max": 0,
    "gate_ready": false,
    "status": "",
    "reason": ""
}

func _init() -> void:
    call_deferred("_run")


func _is_headless_runtime() -> bool:
    return OS.has_feature("headless") or DisplayServer.get_name() == "headless"


func _capture_streaming_settings() -> void:
    if _settings_captured:
        return
    _prev_streaming_route_policy = int(ProjectSettings.get_setting(STREAMING_ROUTE_POLICY_PATH, STREAMING_ROUTE_STREAMING))
    _prev_instance_pipeline_enabled = bool(ProjectSettings.get_setting("rendering/gaussian_splatting/instance_pipeline/enabled", false))
    _settings_captured = true


func _apply_streaming_settings() -> void:
    _capture_streaming_settings()
    ProjectSettings.set_setting(STREAMING_ROUTE_POLICY_PATH, STREAMING_ROUTE_STREAMING)
    ProjectSettings.set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", true)


func _restore_streaming_settings() -> void:
    if not _settings_captured:
        return
    ProjectSettings.set_setting(STREAMING_ROUTE_POLICY_PATH, _prev_streaming_route_policy)
    ProjectSettings.set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", _prev_instance_pipeline_enabled)


func _monitor_int(monitor_id: String) -> int:
    if not Performance.has_custom_monitor(monitor_id):
        return 0
    return int(Performance.get_custom_monitor(monitor_id))


func _build_world_resource() -> GaussianSplatWorld:
    var asset := GaussianSplatAsset.new()
    var load_err := asset.load_from_file(ASSET_PATH)
    if load_err != OK:
        push_error("%s Failed to load fixture asset %s (err=%d)" % [FAIL_MARKER, ASSET_PATH, load_err])
        return null

    var container := GaussianSplatContainer.new()
    container.name = "RuntimeWorldBuilder"
    container.set_merge_on_ready(false)
    container.set_chunk_size(0.75)
    scene_root.add_child(container)

    var offsets := [
        Vector3(-3.0, 0.0, -3.0),
        Vector3(3.0, 0.0, -3.0),
        Vector3(-3.0, 0.0, 3.0),
        Vector3(3.0, 0.0, 3.0)
    ]

    for offset in offsets:
        var node := GaussianSplatNode3D.new()
        node.set_splat_asset(asset)
        node.position = offset
        container.add_child(node)

    container.merge_children()
    metrics["generated_chunk_count"] = int(container.get_chunk_count())

    var world := container.export_world_resource()
    container.queue_free()
    return world


func _setup_scene() -> bool:
    scene_root = Node3D.new()
    scene_root.name = "WorldStreamingGateRoot"
    get_root().add_child(scene_root)

    camera = Camera3D.new()
    camera.name = "GateCamera"
    scene_root.add_child(camera)
    camera.position = Vector3(0.0, 4.0, 24.0)
    camera.look_at(Vector3.ZERO, Vector3.UP)
    camera.make_current()

    world_node = GaussianSplatWorld3D.new()
    world_node.name = "GateWorld"
    scene_root.add_child(world_node)

    var world := _build_world_resource()
    if world == null:
        return false
    world_node.set_world(world)
    world_node.apply_world()

    manager = Engine.get_singleton("GaussianSplatManager")
    return true


func _sample_metrics(frame_index: int) -> void:
    metrics["frames"] = frame_index + 1

    var monitor_ready := _monitor_int("gaussian_splatting/streaming_monitor_ready") > 0
    metrics["monitor_ready_seen"] = bool(metrics.get("monitor_ready_seen", false)) or monitor_ready

    var total_chunks := _monitor_int("gaussian_splatting/streaming_total_chunks")
    var loaded_chunks := _monitor_int("gaussian_splatting/streaming_loaded_chunks")
    var visible_chunks := _monitor_int("gaussian_splatting/streaming_visible_chunks")
    var loaded_this_frame := _monitor_int("gaussian_splatting/streaming_chunks_loaded_this_frame")
    var streaming_visible_count := _monitor_int("gaussian_splatting/streaming_visible_count")

    metrics["total_chunks_max"] = max(int(metrics.get("total_chunks_max", 0)), total_chunks)
    metrics["loaded_chunks_max"] = max(int(metrics.get("loaded_chunks_max", 0)), loaded_chunks)
    metrics["visible_chunks_max"] = max(int(metrics.get("visible_chunks_max", 0)), visible_chunks)
    metrics["chunks_loaded_this_frame_max"] = max(int(metrics.get("chunks_loaded_this_frame_max", 0)), loaded_this_frame)
    metrics["streaming_visible_count_max"] = max(int(metrics.get("streaming_visible_count_max", 0)), streaming_visible_count)

    if world_node != null and world_node.get_renderer() != null:
        var renderer = world_node.get_renderer()
        if renderer.has_method("get_render_stats"):
            var stats = renderer.get_render_stats()
            if stats is Dictionary:
                var renderer_visible := 0
                if stats.has("visible_splats"):
                    renderer_visible = int(stats.get("visible_splats", 0))
                elif stats.has("visible_after_culling"):
                    renderer_visible = int(stats.get("visible_after_culling", 0))
                metrics["renderer_visible_splats_max"] = max(int(metrics.get("renderer_visible_splats_max", 0)), renderer_visible)

                var data_source := str(stats.get("data_source", ""))
                if not data_source.is_empty():
                    metrics["renderer_data_source"] = data_source
                    if data_source.findn("stream") != -1:
                        metrics["streaming_data_source_seen"] = true

                var route_uid := str(stats.get("route_uid", ""))
                if not route_uid.is_empty():
                    metrics["renderer_route_uid"] = route_uid

                var diag_category := str(stats.get("streaming_diagnostics_category", ""))
                if not diag_category.is_empty():
                    metrics["streaming_diagnostics_category"] = diag_category
                var diag_fingerprint := str(stats.get("streaming_diagnostics_fingerprint", ""))
                if not diag_fingerprint.is_empty():
                    metrics["streaming_diagnostics_fingerprint"] = diag_fingerprint
                if stats.has("streaming_state"):
                    var stream_state = stats.get("streaming_state", {})
                    if stream_state is Dictionary:
                        var readiness_state := str(stream_state.get("render_readiness_state", ""))
                        if not readiness_state.is_empty():
                            metrics["streaming_render_readiness_state"] = readiness_state
                if stats.has("streaming_diagnostics"):
                    var streaming_diag = stats.get("streaming_diagnostics", {})
                    if streaming_diag is Dictionary:
                        metrics["streaming_diagnostics_reason"] = str(streaming_diag.get("reason", ""))

    if manager != null and manager.has_method("get_global_stats"):
        var global_stats = manager.get_global_stats()
        if global_stats is Dictionary:
            var total_gaussians: int = maxi(
                int(global_stats.get("total_gaussians", 0)),
                int(global_stats.get("reported_gaussians", 0))
            )
            metrics["manager_total_gaussians_max"] = max(int(metrics.get("manager_total_gaussians_max", 0)), total_gaussians)


func _is_gate_ready() -> bool:
    var monitor_ready := bool(metrics.get("monitor_ready_seen", false))
    var total_chunks := int(metrics.get("total_chunks_max", 0))
    var loaded_chunks := int(metrics.get("loaded_chunks_max", 0))
    var visible_chunks := int(metrics.get("visible_chunks_max", 0))
    var visible_count := int(metrics.get("streaming_visible_count_max", 0))
    var renderer_visible := int(metrics.get("renderer_visible_splats_max", 0))
    var streaming_source := bool(metrics.get("streaming_data_source_seen", false))

    var chunk_signal := total_chunks > 0 and loaded_chunks > 0
    var visibility_signal := visible_chunks > 0 or visible_count > 0 or renderer_visible > 0
    return monitor_ready and chunk_signal and visibility_signal and streaming_source


func _emit_metrics(status: String, reason: String) -> void:
    metrics["gate_ready"] = _is_gate_ready()
    metrics["status"] = status
    metrics["reason"] = reason
    print("%s %s" % [METRICS_MARKER, JSON.stringify(metrics)])


func _cleanup_scene() -> void:
    if scene_root != null:
        scene_root.queue_free()
        scene_root = null
    world_node = null
    camera = null


func _run() -> void:
    if _is_headless_runtime():
        var skip_reason := "World streaming gate requires non-headless execution."
        _emit_metrics("skipped", skip_reason)
        print("%s %s" % [SKIP_MARKER, skip_reason])
        quit(0)
        return

    _apply_streaming_settings()

    if not _setup_scene():
        var setup_reason := "Failed to construct runtime world streaming scene."
        _emit_metrics("failed", setup_reason)
        push_error("%s %s" % [FAIL_MARKER, setup_reason])
        _cleanup_scene()
        _restore_streaming_settings()
        quit(1)
        return

    var passed := false
    var failure_reason := "World streaming gate did not reach chunk/residency readiness within frame budget."

    for frame_idx in range(MAX_TEST_FRAMES):
        await process_frame

        if frame_idx == CAMERA_MOVE_FRAME and camera != null:
            camera.position = Vector3(0.0, 2.5, 10.0)
            camera.look_at(Vector3.ZERO, Vector3.UP)

        _sample_metrics(frame_idx)
        if _is_gate_ready():
            passed = true
            break

    if passed:
        _emit_metrics("passed", "World streaming monitors and residency signals progressed.")
    else:
        _emit_metrics("failed", failure_reason)
        push_error("%s %s" % [FAIL_MARKER, failure_reason])

    _cleanup_scene()
    _restore_streaming_settings()
    quit(0 if passed else 1)
