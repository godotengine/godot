extends Node

@export_range(0.25, 10.0, 0.25) var sample_interval_seconds: float = 1.0
@export_range(16.0, 100.0, 1.0) var stutter_threshold_ms: float = 33.3

var elapsed := 0.0
var frame_count := 0
var frame_times_ms: Array[float] = []
var total_frame_time_ms := 0.0
var stutter_frames := 0
var severe_stutter_frames := 0

@onready var status_label: Label = $CanvasLayer/Status

func _ready() -> void:
	status_label.text = "Nexo-3D Forward+\nWarming up frame-time sampler..."
	print("[MobileForwardPlusTest] renderer=%s" % RenderingServer.get_current_rendering_method())

func _process(delta: float) -> void:
	var frame_ms := delta * 1000.0
	elapsed += delta
	frame_count += 1
	frame_times_ms.append(frame_ms)
	total_frame_time_ms += frame_ms
	if frame_ms >= stutter_threshold_ms:
		stutter_frames += 1
	if frame_ms >= 50.0:
		severe_stutter_frames += 1

	if elapsed < maxf(sample_interval_seconds, 0.25):
		return

	var fps := float(frame_count) / maxf(elapsed, 0.001)
	var sorted_times := frame_times_ms.duplicate()
	sorted_times.sort()
	var p95_index := clampi(ceili(float(sorted_times.size()) * 0.95) - 1, 0, sorted_times.size() - 1)
	var p95_ms := sorted_times[p95_index]
	var max_ms := sorted_times[sorted_times.size() - 1]
	var avg_ms := total_frame_time_ms / maxf(float(frame_count), 1.0)
	var renderer := RenderingServer.get_current_rendering_method()
	var report := "Nexo-3D Mobile Renderer Test\nRenderer: %s\nFPS: %.1f\nAvg frame: %.2f ms\nP95 frame: %.2f ms\nWorst frame: %.2f ms\nStutter frames: %d (%.1f%%)\nSevere frames (>50 ms): %d\nElapsed: %.0f s" % [
		renderer,
		fps,
		avg_ms,
		p95_ms,
		max_ms,
		stutter_frames,
		100.0 * float(stutter_frames) / maxf(float(frame_count), 1.0),
		severe_stutter_frames,
		Time.get_ticks_msec() / 1000.0
	]
	status_label.text = report
	print("[MobileForwardPlusTest] renderer=%s fps=%.1f avg_ms=%.2f p95_ms=%.2f worst_ms=%.2f stutters=%d severe=%d elapsed_s=%.0f" % [
		renderer,
		fps,
		avg_ms,
		p95_ms,
		max_ms,
		stutter_frames,
		severe_stutter_frames,
		Time.get_ticks_msec() / 1000.0
	])

	elapsed = 0.0
	frame_count = 0
	frame_times_ms.clear()
	total_frame_time_ms = 0.0
	stutter_frames = 0
	severe_stutter_frames = 0
