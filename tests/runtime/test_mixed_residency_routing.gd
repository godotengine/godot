extends SceneTree

# Exploratory sequential route-phase script only.
# This is intentionally excluded from blocking runtime profiles until the
# runtime surface can prove true resident/streaming coexistence.

const SKIP_MARKER := "[RUNTIME_SKIP]"
const FAIL_MARKER := "[RUNTIME_FAIL]"
const METRICS_MARKER := "[RUNTIME_METRICS]"

const WORLD_PATH := "res://tests/fixtures/test_splats.gsplatworld"
const STREAMING_ROUTE_POLICY_PATH := "rendering/gaussian_splatting/streaming/route_policy"
const INSTANCE_PIPELINE_ENABLED_PATH := "rendering/gaussian_splatting/instance_pipeline/enabled"
const ROUTE_RESIDENT := 0
const ROUTE_STREAMING := 1
const PHASE_TIMEOUT_FRAMES := 180

var scene_root: Node3D
var camera: Camera3D
var world_a: GaussianSplatWorld3D
var world_b: GaussianSplatWorld3D
var manager := null

var _settings_captured := false
var _prev_route_policy := ROUTE_STREAMING
var _prev_instance_pipeline_enabled := false

var metrics: Dictionary = {
	"resident": {},
	"streaming": {},
	"streaming_total_chunks": 0,
	"manager_total_gaussians": 0,
	"status": "failed",
	"reason": ""
}

func _init() -> void:
	call_deferred("_run")

func _is_headless_runtime() -> bool:
	return OS.has_feature("headless") or DisplayServer.get_name() == "headless"

func _record_failure(reason: String, context: Dictionary = {}) -> void:
	var message := reason
	if not context.is_empty():
		message = "%s | context=%s" % [reason, str(context)]
	metrics["status"] = "failed"
	metrics["reason"] = message
	print("%s %s" % [METRICS_MARKER, JSON.stringify(metrics)])
	push_error("%s %s" % [FAIL_MARKER, message])
	_cleanup_scene()
	_restore_settings()
	quit(1)

func _capture_settings() -> void:
	if _settings_captured:
		return
	_prev_route_policy = int(ProjectSettings.get_setting(STREAMING_ROUTE_POLICY_PATH, ROUTE_STREAMING))
	_prev_instance_pipeline_enabled = bool(ProjectSettings.get_setting(INSTANCE_PIPELINE_ENABLED_PATH, false))
	_settings_captured = true

func _restore_settings() -> void:
	if not _settings_captured:
		return
	ProjectSettings.set_setting(STREAMING_ROUTE_POLICY_PATH, _prev_route_policy)
	ProjectSettings.set_setting(INSTANCE_PIPELINE_ENABLED_PATH, _prev_instance_pipeline_enabled)

func _setup_scene() -> bool:
	scene_root = Node3D.new()
	scene_root.name = "MixedResidencyRuntimeRoot"
	get_root().add_child(scene_root)

	camera = Camera3D.new()
	camera.name = "MixedResidencyCamera"
	camera.position = Vector3(0.0, 1.5, 12.0)
	camera.look_at(Vector3.ZERO, Vector3.UP)
	camera.current = true
	scene_root.add_child(camera)

	var world_resource_a: GaussianSplatWorld = load(WORLD_PATH)
	var world_resource_b: GaussianSplatWorld = load(WORLD_PATH)
	if world_resource_a == null or world_resource_b == null:
		return false

	world_a = GaussianSplatWorld3D.new()
	world_a.name = "WorldA"
	world_a.set_world(world_resource_a)
	scene_root.add_child(world_a)

	world_b = GaussianSplatWorld3D.new()
	world_b.name = "WorldB"
	world_b.position = Vector3(8.0, 0.0, 0.0)
	world_b.set_world(world_resource_b)
	scene_root.add_child(world_b)

	manager = Engine.get_singleton("GaussianSplatManager")
	return true

func _cleanup_scene() -> void:
	if scene_root != null:
		scene_root.queue_free()
	scene_root = null
	camera = null
	world_a = null
	world_b = null

func _capture_world_stats(world: GaussianSplatWorld3D) -> Dictionary:
	var snapshot := {
		"chunk_count": 0,
		"gaussian_count": 0,
		"route_uid": "",
		"requested_route_policy": "",
		"instance_backend_policy": "",
		"backend_selection_reason": "",
		"data_source": "",
		"instance_contract_ready": false
	}
	if world == null or world.world == null:
		return snapshot
	snapshot["chunk_count"] = int(world.world.get_chunk_count())
	var data = world.world.get_gaussian_data()
	if data != null:
		snapshot["gaussian_count"] = int(data.get_count())
	if world.has_method("get_renderer"):
		var renderer = world.get_renderer()
		if renderer != null and renderer.has_method("get_render_stats"):
			var stats = renderer.get_render_stats()
			if stats is Dictionary:
				snapshot["route_uid"] = str(stats.get("route_uid", ""))
				snapshot["requested_route_policy"] = str(stats.get("requested_route_policy", ""))
				snapshot["instance_backend_policy"] = str(stats.get("instance_backend_policy", ""))
				snapshot["backend_selection_reason"] = str(stats.get("backend_selection_reason", ""))
				snapshot["data_source"] = str(stats.get("data_source", ""))
				snapshot["instance_contract_ready"] = bool(stats.get("instance_contract_ready", false))
	return snapshot

func _phase_ready(snapshot: Dictionary, expected_policy: String) -> bool:
	var route_uid := str(snapshot.get("route_uid", ""))
	var requested_policy := str(snapshot.get("requested_route_policy", ""))
	var backend_policy := str(snapshot.get("instance_backend_policy", ""))
	var data_source := str(snapshot.get("data_source", ""))
	var contract_ready := bool(snapshot.get("instance_contract_ready", false))
	var chunk_count := int(snapshot.get("chunk_count", 0))
	var gaussian_count := int(snapshot.get("gaussian_count", 0))

	if requested_policy != expected_policy or backend_policy != expected_policy or not contract_ready:
		return false
	if chunk_count <= 0 or gaussian_count <= 0:
		return false

	if expected_policy == "resident":
		return route_uid.begins_with("INSTANCE.RESIDENT") and (
			data_source == "ResidentInstanceAtlas" or data_source.findn("resident") != -1
		)

	return route_uid.begins_with("INSTANCE.STREAMING") and (
		data_source == "StreamingGPU" or data_source == "GPUBufferManager" or data_source.findn("stream") != -1
	)

func _wait_for_phase(world: GaussianSplatWorld3D, phase_name: String, expected_policy: String) -> Dictionary:
	var last_snapshot := {}
	for _frame in range(PHASE_TIMEOUT_FRAMES):
		await process_frame
		last_snapshot = _capture_world_stats(world)
		if _phase_ready(last_snapshot, expected_policy):
			metrics[phase_name] = last_snapshot
			return last_snapshot
	metrics[phase_name] = last_snapshot
	return {}

func _run() -> void:
	if _is_headless_runtime():
		var skip_reason := "Mixed residency routing runtime test requires non-headless execution."
		metrics["status"] = "skipped"
		metrics["reason"] = skip_reason
		print("%s %s" % [METRICS_MARKER, JSON.stringify(metrics)])
		print("%s %s" % [SKIP_MARKER, skip_reason])
		quit(0)
		return

	_capture_settings()
	ProjectSettings.set_setting(INSTANCE_PIPELINE_ENABLED_PATH, true)

	if not _setup_scene():
		_record_failure("Failed to construct mixed-residency runtime scene.")
		return

	ProjectSettings.set_setting(STREAMING_ROUTE_POLICY_PATH, ROUTE_RESIDENT)
	world_a.apply_world()
	world_b.clear_world()
	var resident_snapshot := await _wait_for_phase(world_a, "resident", "resident")
	if resident_snapshot.is_empty():
		_record_failure("Resident mixed-residency phase did not stabilize.", metrics["resident"])
		return

	ProjectSettings.set_setting(STREAMING_ROUTE_POLICY_PATH, ROUTE_STREAMING)
	world_a.clear_world()
	world_b.apply_world()
	var streaming_snapshot := await _wait_for_phase(world_b, "streaming", "streaming")
	if streaming_snapshot.is_empty():
		_record_failure("Streaming mixed-residency phase did not stabilize.", metrics["streaming"])
		return

	if manager != null and manager.has_method("get_global_stats"):
		var stats: Dictionary = manager.get_global_stats()
		metrics["manager_total_gaussians"] = int(stats.get("total_gaussians", 0))

	if Performance.has_custom_monitor("gaussian_splatting/streaming_total_chunks"):
		metrics["streaming_total_chunks"] = int(Performance.get_custom_monitor("gaussian_splatting/streaming_total_chunks"))

	metrics["status"] = "passed"
	metrics["reason"] = "Resident and streaming instance routes both stabilized."
	print("%s %s" % [METRICS_MARKER, JSON.stringify(metrics)])
	_cleanup_scene()
	_restore_settings()
	quit(0)
