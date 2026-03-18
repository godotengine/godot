extends SceneTree

const SKIP_MARKER := "[RUNTIME_SKIP]"
const FAIL_MARKER := "[RUNTIME_FAIL]"
const METRICS_MARKER := "[RUNTIME_METRICS]"

const ASSET_PATH := "res://tests/fixtures/test_splats.ply"
const MAX_TEST_FRAMES := 240
const TRACE_DISABLE_FRAME := 140
const MIN_FRESH_EVENT_FRAMES := 4

var scene_root: Node3D
var splat_node: GaussianSplatNode3D
var camera: Camera3D
var renderer = null

var metrics: Dictionary = {
	"frames": 0,
	"fresh_event_frames_before_disable": 0,
	"max_events_before_disable": 0,
	"trace_disable_frame": -1,
	"trace_generation_start": -1,
	"trace_generation_at_disable": -1,
	"trace_generation_end": -1,
	"trace_generation_monotonic": true,
	"stale_events_after_disable": false,
	"stale_fresh_flag_after_disable": false,
	"status": "",
	"reason": "",
}

func _init() -> void:
	call_deferred("_run")

func _is_headless_runtime() -> bool:
	return OS.has_feature("headless") or DisplayServer.get_name() == "headless"

func _record_failure(reason: String) -> void:
	push_error("%s %s" % [FAIL_MARKER, reason])

func _emit_metrics(status: String, reason: String) -> void:
	metrics["status"] = status
	metrics["reason"] = reason
	print("%s %s" % [METRICS_MARKER, JSON.stringify(metrics)])

func _cleanup() -> void:
	if scene_root != null:
		scene_root.queue_free()
	scene_root = null
	splat_node = null
	camera = null
	renderer = null

func _setup_scene() -> bool:
	scene_root = Node3D.new()
	scene_root.name = "PipelineTraceFreshnessRoot"
	get_root().add_child(scene_root)

	camera = Camera3D.new()
	camera.name = "TraceCamera"
	camera.position = Vector3(0.0, 1.8, 6.0)
	camera.look_at(Vector3.ZERO, Vector3.UP)
	camera.make_current()
	scene_root.add_child(camera)

	splat_node = GaussianSplatNode3D.new()
	splat_node.name = "TraceSplat"
	scene_root.add_child(splat_node)

	var asset := GaussianSplatAsset.new()
	var load_err := asset.load_from_file(ASSET_PATH)
	if load_err != OK:
		_record_failure("Failed to load fixture asset %s (err=%d)" % [ASSET_PATH, load_err])
		return false
	splat_node.set_splat_asset(asset)
	return true

func _run() -> void:
	if _is_headless_runtime():
		var skip_reason := "Pipeline trace freshness requires non-headless execution."
		_emit_metrics("skipped", skip_reason)
		print("%s %s" % [SKIP_MARKER, skip_reason])
		quit(0)
		return

	if not _setup_scene():
		_emit_metrics("failed", "scene_setup_failed")
		_cleanup()
		quit(1)
		return

	ProjectSettings.set_setting("rendering/gaussian_splatting/debug/enable_pipeline_trace", true)

	var max_renderer_wait_frames := 90
	for i in range(max_renderer_wait_frames):
		await process_frame
		metrics["frames"] = i + 1
		if splat_node != null:
			renderer = splat_node.get_renderer()
		if renderer != null:
			break

	if renderer == null:
		var skip_reason := "Renderer unavailable (local RenderingDevice required)."
		_emit_metrics("skipped", skip_reason)
		print("%s %s" % [SKIP_MARKER, skip_reason])
		_cleanup()
		quit(0)
		return

	if renderer.has_method("set_debug_pipeline_trace_enabled"):
		renderer.set_debug_pipeline_trace_enabled(true)

	var trace_disable_armed := false
	var last_generation := -1
	for frame in range(MAX_TEST_FRAMES):
		await process_frame
		metrics["frames"] = max(int(metrics.get("frames", 0)), max_renderer_wait_frames + frame + 1)
		var snapshot: Dictionary = renderer.get_pipeline_trace_snapshot()
		var trace_generation := int(snapshot.get("trace_generation", -1))
		var trace_fresh := bool(snapshot.get("trace_fresh", snapshot.get("events_valid", false)))
		var events: Array = snapshot.get("events", [])

		if trace_generation >= 0 and int(metrics.get("trace_generation_start", -1)) < 0:
			metrics["trace_generation_start"] = trace_generation
		if trace_generation >= 0:
			metrics["trace_generation_end"] = trace_generation
		if last_generation >= 0 and trace_generation >= 0 and trace_generation < last_generation:
			metrics["trace_generation_monotonic"] = false
		last_generation = trace_generation

		if frame < TRACE_DISABLE_FRAME:
			if trace_fresh and not events.is_empty():
				metrics["fresh_event_frames_before_disable"] = int(metrics.get("fresh_event_frames_before_disable", 0)) + 1
				metrics["max_events_before_disable"] = max(int(metrics.get("max_events_before_disable", 0)), events.size())
			if int(metrics.get("fresh_event_frames_before_disable", 0)) >= MIN_FRESH_EVENT_FRAMES:
				trace_disable_armed = true
		elif frame == TRACE_DISABLE_FRAME:
			metrics["trace_disable_frame"] = frame
			metrics["trace_generation_at_disable"] = trace_generation
			if renderer.has_method("set_debug_pipeline_trace_enabled"):
				renderer.set_debug_pipeline_trace_enabled(false)
		else:
			if trace_fresh:
				metrics["stale_fresh_flag_after_disable"] = true
			if not events.is_empty():
				metrics["stale_events_after_disable"] = true
			if frame > TRACE_DISABLE_FRAME + 40:
				break

	if not trace_disable_armed:
		_record_failure("Unable to capture fresh trace events before disabling trace")
		_emit_metrics("failed", "insufficient_pre_disable_trace_events")
		_cleanup()
		quit(1)
		return

	if not bool(metrics.get("trace_generation_monotonic", false)):
		_record_failure("trace_generation was not monotonic")
		_emit_metrics("failed", "trace_generation_not_monotonic")
		_cleanup()
		quit(1)
		return

	var generation_at_disable := int(metrics.get("trace_generation_at_disable", -1))
	var generation_end := int(metrics.get("trace_generation_end", -1))
	if generation_at_disable >= 0 and generation_end <= generation_at_disable:
		_record_failure("trace_generation did not advance after disabling trace")
		_emit_metrics("failed", "trace_generation_not_advancing_after_disable")
		_cleanup()
		quit(1)
		return

	if bool(metrics.get("stale_fresh_flag_after_disable", false)):
		_record_failure("trace_fresh remained true after trace disable")
		_emit_metrics("failed", "trace_fresh_not_cleared")
		_cleanup()
		quit(1)
		return

	if bool(metrics.get("stale_events_after_disable", false)):
		_record_failure("Snapshot events were not cleared after trace disable")
		_emit_metrics("failed", "stale_events_detected")
		_cleanup()
		quit(1)
		return

	_emit_metrics("passed", "pipeline trace freshness semantics validated")
	_cleanup()
	quit(0)
