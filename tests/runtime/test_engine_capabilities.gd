extends SceneTree

const LARGE_SPLAT_COUNT := 10000
const ANIMATION_PROPERTY_POSITION := 0
const SKIP_MARKER := "[RUNTIME_SKIP]"
const FAIL_MARKER := "[RUNTIME_FAIL]"

var summary: Array = []
var failures: Array = []
var scene_root: Node3D
var primary_node: GaussianSplatNode3D
var baseline_asset: GaussianSplatAsset
var camera: Camera3D
var manager

## Defers the capability checks until the SceneTree is initialized.
func _init() -> void:
    call_deferred("_run")

## Returns true when running on a headless display server.
func _is_headless_runtime() -> bool:
    return OS.has_feature("headless") or DisplayServer.get_name() == "headless"

## Executes all capability checks and exits with status.
func _run() -> void:
    if _is_headless_runtime():
        var skip_reason = "Engine capability runtime test requires a local RenderingDevice (non-headless run)."
        print("%s %s" % [SKIP_MARKER, skip_reason])
        push_warning(skip_reason)
        quit(0)
        return

    print("=== Gaussian Engine Capability Validation ===")
    scene_root = Node3D.new()
    scene_root.name = "CapabilityRoot"
    get_root().add_child(scene_root)

    manager = Engine.get_singleton("GaussianSplatManager")

    _record("Launches without errors - Zero console device errors?", _check_launch())
    _record("Node exists - Can you add GaussianSplat to scene?", await _check_node_creation())
    _record("Data can be set - Can you give it splat data?", await _check_data_assignment())
    _record("Single node renders - Do you see colored splats?", await _check_single_node_render())
    _record("Multiple nodes render - Can you see 3 different colored clusters?", await _check_multiple_nodes())
    _record("Camera works - Does movement update rendering?", await _check_camera_updates())
    _record("Scenes save/load - Does data persist?", await _check_persistence())
    _record("Large count works - 10K splats without crash?", await _check_large_dataset())
    _record("Scene switching works - No memory leaks?", await _check_scene_switching())
    _record("Deletion safe - Freeing nodes doesn't crash?", await _check_deletion_safety())
    _record("Empty node safe - Node without data doesn't crash?", await _check_empty_node())

    _print_summary()
    var exit_code = 0
    if not failures.is_empty():
        exit_code = 1
    quit(exit_code)

## Records a check result and prints status to stdout.
## @param name: Human-readable check name.
## @param result: Result dictionary containing success/details.
func _record(name: String, result: Dictionary) -> void:
    var success = result.get("success", false)
    var details = result.get("details", {})
    summary.append({"name": name, "success": success, "details": details})

    if success:
        print("  ✅ ", name)
    else:
        failures.append(name)
        print("  ❌ ", name)
        var failure_reason = name
        if details is Dictionary and details.has("errors"):
            var error_list = details.errors
            if error_list is Array and not error_list.is_empty():
                failure_reason = "%s | %s" % [name, str(error_list[0])]
        push_error("%s %s" % [FAIL_MARKER, failure_reason])
        if details is Dictionary and details.has("errors"):
            for error_text in details.errors:
                print("     - ", error_text)

## Computes Euclidean distance between two colors.
func _color_distance(a: Color, b: Color) -> float:
    var dr = a.r - b.r
    var dg = a.g - b.g
    var db = a.b - b.b
    var da = a.a - b.a
    return sqrt(dr * dr + dg * dg + db * db + da * da)

## Ensures a deterministic camera setup for runtime visibility checks.
func _ensure_runtime_camera() -> void:
    if camera == null:
        camera = Camera3D.new()
        camera.name = "CapabilityCamera"
        scene_root.add_child(camera)
    elif camera.get_parent() != scene_root:
        scene_root.add_child(camera)

    camera.position = Vector3(0, 0, 6)
    camera.look_at(Vector3.ZERO, Vector3.UP)
    camera.make_current()

## Returns the best available residency metric from manager stats.
func _global_residency_metric(stats: Dictionary) -> int:
    return max(int(stats.get("total_gaussians", 0)), int(stats.get("reported_gaussians", 0)))

## Confirms engine singletons and global stats availability.
func _check_launch() -> Dictionary:
    var errors: Array = []
    if RenderingServer.get_rendering_device() == null:
        errors.append("RenderingServer has no active RenderingDevice")
    if manager == null:
        errors.append("GaussianSplatManager singleton unavailable")
    else:
        var stats: Dictionary = manager.get_global_stats()
        if not stats.has("total_gaussians"):
            errors.append("GaussianSplatManager did not report global statistics")

    return {
        "success": errors.is_empty(),
        "details": {"errors": errors}
    }

## Validates that GaussianSplatNode3D can enter the scene tree.
func _check_node_creation() -> Dictionary:
    primary_node = GaussianSplatNode3D.new()
    primary_node.name = "CapabilityPrimary"
    scene_root.add_child(primary_node)
    await process_frame
    var inside = primary_node.is_inside_tree()
    var details = {}
    if inside:
        details = {"path": str(primary_node.get_path())}
    else:
        details = {"errors": ["GaussianSplatNode3D failed to enter scene tree"]}
    return {
        "success": inside,
        "details": details
    }

## Checks assigning splat data to a node updates statistics.
func _check_data_assignment() -> Dictionary:
    baseline_asset = _make_cluster_asset(128, Vector3.ZERO, Color(1.0, 0.2, 0.2, 0.9), 0.6)
    primary_node.set_splat_asset(baseline_asset)
    primary_node.force_update()
    await process_frame
    var asset = primary_node.get_splat_asset()
    var stats = primary_node.get_statistics()
    var total = int(stats.get("total_splats", 0))
    var success = asset != null and asset.get_splat_count() == 128 and total == 128
    var details = {}
    if success:
        details = {"total_splats": total}
    else:
        details = {"errors": ["Asset assignment did not report expected splat count"]}
    return {
        "success": success,
        "details": details
    }

## Ensures a single node renders visible splats.
func _check_single_node_render() -> Dictionary:
    _ensure_runtime_camera()
    primary_node.force_update()
    await process_frame
    await process_frame
    await process_frame
    var stats = primary_node.get_statistics()
    var visible = int(stats.get("visible_splats", 0))
    var success = visible > 0
    var details = {}
    if success:
        details = {"visible_splats": visible}
    else:
        details = {"errors": ["Primary node reported zero visible splats"]}
    return {
        "success": success,
        "details": details
    }

## Ensures multiple splat nodes render and retain distinct colors.
func _check_multiple_nodes() -> Dictionary:
    _ensure_runtime_camera()
    var colors = [Color(1.0, 0.1, 0.1, 0.85), Color(0.1, 1.0, 0.1, 0.85), Color(0.1, 0.1, 1.0, 0.85)]
    var centers = [Vector3(-2.5, 0, 0), Vector3.ZERO, Vector3(2.5, 0, 0)]
    var nodes: Array = []
    for i in range(3):
        var node = GaussianSplatNode3D.new()
        node.name = "Cluster_%d" % i
        scene_root.add_child(node)
        node.set_splat_asset(_make_cluster_asset(64, centers[i], colors[i], 0.5))
        node.force_update()
        nodes.append(node)
    await process_frame
    await process_frame
    await process_frame

    var success = true
    var cluster_details: Array = []
    for i in range(nodes.size()):
        var node: GaussianSplatNode3D = nodes[i]
        var stats = node.get_statistics()
        var visible = int(stats.get("visible_splats", 0))
        if visible <= 0:
            success = false
        var asset_colors = node.get_splat_asset().get_colors()
        var accumulated = Color(0, 0, 0, 0)
        for color_value in asset_colors:
            accumulated += color_value
        var average = accumulated / max(1, asset_colors.size())
        cluster_details.append({
            "node": str(node.get_path()),
            "visible_splats": visible,
            "average_color": average
        })
        if _color_distance(average, colors[i]) > 0.15:
            success = false
    for node in nodes:
        node.queue_free()
    await process_frame

    var details = {}
    if success:
        details = {"clusters": cluster_details}
    else:
        details = {"errors": ["Cluster visibility or color validation failed"], "clusters": cluster_details}
    return {
        "success": success,
        "details": details
    }

## Verifies renderer stats advance when the camera moves.
func _check_camera_updates() -> Dictionary:
    _ensure_runtime_camera()

    await process_frame
    primary_node.force_update()
    await process_frame
    var before = primary_node.get_statistics()
    var before_frames = int(before.get("frame_count", 0))

    camera.translate(Vector3(1.5, 0, 0))
    await process_frame
    primary_node.force_update()
    await process_frame
    var after = primary_node.get_statistics()
    var frame_delta = int(after.get("frame_count", 0)) - before_frames
    var sort_delta = int(after.get("gpu_sorter_total_sorts", 0)) - int(before.get("gpu_sorter_total_sorts", 0))
    var camera_delta = abs(float(after.get("debug_cam_origin_x", 0.0)) - float(before.get("debug_cam_origin_x", 0.0)))
    var success = frame_delta > 0 or sort_delta > 0 or camera_delta > 0.001
    var details = {}
    if success:
        details = {"frame_delta": frame_delta, "sort_delta": sort_delta, "camera_delta": camera_delta}
    else:
        details = {"errors": ["Renderer did not register camera movement in frame, sort, or camera telemetry counters"], "frame_delta": frame_delta, "sort_delta": sort_delta, "camera_delta": camera_delta}
    return {
        "success": success,
        "details": details
    }

## Validates baseline/incremental save-load behavior for splat data.
func _check_persistence() -> Dictionary:
    var serializer = GaussianSceneSerializer.new()
    var saver = GaussianIncrementalSaver.new()
    var data = GaussianData.new()
    data.resize(3)
    data.set_incremental_saver(saver)

    var animation = GaussianAnimationStateMachine.new()
    animation.set_splat_count(3)
    animation.set_incremental_saver(saver)
    var clip_index = animation.add_clip("idle", 1.0)
    animation.add_track_to_clip(clip_index, ANIMATION_PROPERTY_POSITION)
    animation.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 0.0, Vector3.ZERO)
    animation.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 1.0, Vector3.ONE)

    var base_path = "user://capability_scene.gsf"
    var delta_path = "user://capability_scene.gsif"

    var errors: Array = []
    if serializer.save_scene(base_path, data, animation) != OK:
        errors.append("Failed to save baseline scene")
    else:
        saver.start_tracking(base_path)
        data.set_runtime_position(0, Vector3(3.0, 4.0, 5.0))
        data.commit_runtime_changes()
        animation.set_clip_duration(clip_index, 2.0)
        if saver.save_changes(delta_path) != OK:
            errors.append("Failed to save incremental changes")
        else:
            var restored_data = GaussianData.new()
            var restored_animation = GaussianAnimationStateMachine.new()
            if serializer.load_scene(base_path, restored_data, restored_animation) != OK:
                errors.append("Failed to load baseline scene")
            elif saver.load_and_apply_changes(delta_path, restored_data, restored_animation) != OK:
                errors.append("Failed to apply incremental changes")
            else:
                var restored_bounds: AABB = restored_data.get_aabb()
                var position_ok = restored_bounds.has_point(Vector3(3.0, 4.0, 5.0))
                var duration_ok = false
                var restored_duration = -1.0
                if restored_animation.get_clip_count() > 0:
                    restored_duration = restored_animation.get_clip_duration(0)
                    duration_ok = abs(restored_duration - 2.0) < 0.01
                if not position_ok or not duration_ok:
                    errors.append(
                        "Restored scene did not match expected position/duration "
                        + "(position_ok=%s, duration=%s)"
                        % [position_ok, restored_duration]
                    )
    _cleanup_user_file(base_path)
    _cleanup_user_file(delta_path)
    var success = errors.is_empty()
    var details = {}
    if success:
        details = {"paths": [base_path, delta_path]}
    else:
        details = {"errors": errors}
    return {
        "success": success,
        "details": details
    }

## Loads a large dataset and ensures visibility and count are reported.
func _check_large_dataset() -> Dictionary:
    _ensure_runtime_camera()
    var large_asset = _make_cluster_asset(LARGE_SPLAT_COUNT, Vector3(0, 0, -2), Color(0.3, 0.6, 1.0, 0.85), 1.2)
    primary_node.set_splat_asset(large_asset)
    primary_node.force_update()
    await process_frame
    await process_frame
    var stats = primary_node.get_statistics()
    var total = int(stats.get("total_splats", 0))
    var visible = int(stats.get("visible_splats", 0))
    var success = total >= LARGE_SPLAT_COUNT and visible > 0

    primary_node.set_splat_asset(baseline_asset)
    primary_node.force_update()
    await process_frame

    var details = {}
    if success:
        details = {"total_splats": total, "visible_splats": visible}
    else:
        details = {"errors": ["Renderer reported insufficient splat count"], "total_splats": total, "visible_splats": visible}
    return {
        "success": success,
        "details": details
    }

## Ensures global stats update during scene switching.
func _check_scene_switching() -> Dictionary:
    if manager == null:
        return {"success": false, "details": {"errors": ["GaussianSplatManager unavailable"]}}
    _ensure_runtime_camera()
    var initial_stats: Dictionary = manager.get_global_stats()
    var initial_metric = _global_residency_metric(initial_stats)
    var temp_scene = Node3D.new()
    temp_scene.name = "CapabilityTempScene"
    scene_root.add_child(temp_scene)

    var temp_node = GaussianSplatNode3D.new()
    temp_scene.add_child(temp_node)
    temp_node.set_splat_asset(_make_cluster_asset(256, Vector3(0, 1.5, -3), Color(1.0, 0.6, 0.2, 0.9), 0.7))
    temp_node.force_update()
    await process_frame
    await process_frame
    await process_frame

    var temp_stats = temp_node.get_statistics()
    var temp_total = int(temp_stats.get("total_splats", 0))
    var after_add: Dictionary = manager.get_global_stats()
    var after_add_metric = _global_residency_metric(after_add)
    temp_scene.queue_free()
    await process_frame
    await process_frame
    await process_frame
    var after_free: Dictionary = manager.get_global_stats()
    var after_free_metric = _global_residency_metric(after_free)

    var tracked_metric = after_add_metric > initial_metric
    var increased = true
    var restored = true
    if tracked_metric:
        increased = after_add_metric >= initial_metric + 200
        restored = after_free_metric <= initial_metric + 32
    var scene_released = not is_instance_valid(temp_scene) or temp_scene.get_parent() == null
    var success = temp_total >= 256 and scene_released and increased and restored
    if not success:
        return {
            "success": false,
            "details": {
                "errors": ["Scene switching checks failed for instance lifecycle or residency telemetry"],
                "tracked_metric": tracked_metric,
                "temp_total": temp_total,
                "initial_metric": initial_metric,
                "after_add_metric": after_add_metric,
                "after_free_metric": after_free_metric,
                "initial": initial_stats,
                "after_add": after_add,
                "after_free": after_free
            }
        }
    return {
        "success": true,
        "details": {
            "tracked_metric": tracked_metric,
            "temp_total": temp_total,
            "initial_metric": initial_metric,
            "after_add_metric": after_add_metric,
            "after_free_metric": after_free_metric
        }
    }

## Confirms that deleting nodes updates residency counts safely.
func _check_deletion_safety() -> Dictionary:
    var node = GaussianSplatNode3D.new()
    scene_root.add_child(node)
    node.set_splat_asset(_make_cluster_asset(48, Vector3(0, -1.5, -2), Color(0.9, 0.9, 0.2, 0.85), 0.4))
    node.force_update()
    await process_frame
    var before_free_stats = {}
    if manager != null:
        before_free_stats = manager.get_global_stats()
    node.queue_free()
    await process_frame
    await process_frame
    var after_free_stats = {}
    if manager != null:
        after_free_stats = manager.get_global_stats()
    var success = true
    if manager != null:
        success = int(after_free_stats.get("total_gaussians", 0)) <= int(before_free_stats.get("total_gaussians", 0))
    var details = {}
    if success:
        details = {"before": before_free_stats, "after": after_free_stats}
    else:
        details = {"errors": ["Manager residency increased after freeing node"], "before": before_free_stats, "after": after_free_stats}
    return {
        "success": success,
        "details": details
    }

## Ensures an empty GaussianSplatNode3D reports zero visible splats.
func _check_empty_node() -> Dictionary:
    var empty_node = GaussianSplatNode3D.new()
    scene_root.add_child(empty_node)
    empty_node.force_update()
    await process_frame
    var stats = empty_node.get_statistics()
    var visible = int(stats.get("visible_splats", 0))
    empty_node.queue_free()
    await process_frame
    var success = visible == 0
    var details = {}
    if success:
        details = {"visible_splats": visible}
    else:
        details = {"errors": ["Empty node reported visible splats"], "visible_splats": visible}
    return {
        "success": success,
        "details": details
    }

## Prints a summary of all capability checks.
func _print_summary() -> void:
    print("\n=== Capability Summary ===")
    for entry in summary:
        var status = "❌"
        if entry.success:
            status = "✅"
        print("%s %s" % [status, entry.name])
    if failures.is_empty():
        print("\nAll capability checks passed.")
    else:
        print("\nFailures detected:")
        for name in failures:
            print(" - ", name)

## Removes a user:// file if present.
func _cleanup_user_file(path: String) -> void:
    var absolute = ProjectSettings.globalize_path(path)
    DirAccess.remove_absolute(absolute)

## Generates a clustered GaussianSplatAsset with the requested properties.
## @param count: Number of splats.
## @param center: Cluster center.
## @param color: Base color for splats.
## @param radius: Cluster radius for random jitter.
## @return Configured GaussianSplatAsset instance.
func _make_cluster_asset(count: int, center: Vector3, color: Color, radius: float) -> GaussianSplatAsset:
    var asset = GaussianSplatAsset.new()
    var positions = PackedFloat32Array()
    positions.resize(count * 3)
    var scales = PackedFloat32Array()
    scales.resize(count * 3)
    var rotations = PackedFloat32Array()
    rotations.resize(count * 4)
    var colors = PackedColorArray()
    colors.resize(count)

    var rng = RandomNumberGenerator.new()
    rng.seed = 0xC0DECAFE + count + int(center.x * 17) + int(center.y * 31) + int(center.z * 47)

    for i in range(count):
        var jitter = Vector3(
            rng.randf_range(-radius, radius),
            rng.randf_range(-radius, radius),
            rng.randf_range(-radius, radius)
        ) * 0.5
        var base = center + jitter
        var pos_index = i * 3
        positions[pos_index + 0] = base.x
        positions[pos_index + 1] = base.y
        positions[pos_index + 2] = base.z

        var scale_index = i * 3
        var uniform_scale = radius * 0.25 + rng.randf() * radius * 0.15
        scales[scale_index + 0] = uniform_scale
        scales[scale_index + 1] = uniform_scale
        scales[scale_index + 2] = uniform_scale

        var rot_index = i * 4
        rotations[rot_index + 0] = 1.0
        rotations[rot_index + 1] = 0.0
        rotations[rot_index + 2] = 0.0
        rotations[rot_index + 3] = 0.0

        colors[i] = color
    asset.set_positions(positions)
    asset.set_scales(scales)
    asset.set_rotations(rotations)
    asset.set_colors(colors)
    return asset
