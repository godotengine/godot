extends "res://scripts/qa_test_base.gd"
## Performance Budget Test: Verifies rendering stays within timing budgets.
## Tracks FPS, frame time, and Gaussian Splatting specific metrics.

@export var min_fps: float = 30.0
@export var max_frame_time_ms: float = 33.3  # 30 FPS target

var fps_samples: Array[float] = []
var frame_time_samples: Array[float] = []
var splat_node: Node3D

func _ready():
	test_name = "Performance Budget"
	test_duration = 15.0
	warmup_frames = 60
	super._ready()

	splat_node = get_node_or_null("SplatNode")

func _on_test_start():
	fps_samples.clear()
	frame_time_samples.clear()

	# Verify splat node is rendering
	if splat_node and splat_node.has_method("get_visible_splat_count"):
		var count = splat_node.get_visible_splat_count()
		print("[QA:%s] Visible splats: %d" % [test_name, count])
	else:
		print("[QA:%s] WARNING: Cannot verify visible splat count" % test_name)

func _on_test_frame(delta: float):
	var fps = 1.0 / delta if delta > 0 else 0.0
	var frame_time_ms = delta * 1000.0

	fps_samples.append(fps)
	frame_time_samples.append(frame_time_ms)

func _on_test_complete():
	if fps_samples.is_empty():
		_test_result = false
		_test_message = "No samples collected"
		return

	var avg_fps = _calculate_avg(fps_samples)
	var min_fps_observed = fps_samples.min()
	var max_frame_time = frame_time_samples.max()
	var p1_fps = percentile(fps_samples, 1)
	var p99_frame_time = percentile(frame_time_samples, 99)
	result_metrics["avg_fps"] = avg_fps
	result_metrics["min_fps"] = min_fps_observed
	result_metrics["p1_fps"] = p1_fps
	result_metrics["max_frame_time_ms"] = max_frame_time
	result_metrics["p99_frame_time_ms"] = p99_frame_time

	var budget_issues: Array[String] = []

	if avg_fps < min_fps:
		budget_issues.append("Avg FPS %.1f < %.1f" % [avg_fps, min_fps])

	if p1_fps < min_fps * 0.5:
		budget_issues.append("P1 FPS %.1f < %.1f (50%% of target)" % [p1_fps, min_fps * 0.5])

	if p99_frame_time > max_frame_time_ms * 1.5:
		budget_issues.append("P99 frame time %.1fms > %.1fms (150%% budget)" % [
			p99_frame_time, max_frame_time_ms * 1.5
		])

	if budget_issues.is_empty():
		_test_result = true
		_test_message = "FPS avg=%.1f p1=%.1f | Frame p99=%.1fms" % [
			avg_fps, p1_fps, p99_frame_time
		]
	else:
		_test_result = false
		_test_message = "; ".join(budget_issues)

func _calculate_avg(samples: Array[float]) -> float:
	if samples.is_empty():
		return 0.0
	var sum = 0.0
	for s in samples:
		sum += s
	return sum / samples.size()
