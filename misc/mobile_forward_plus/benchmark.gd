extends Node

@export var sample_interval_seconds: float = 1.0

var elapsed := 0.0
var frame_count := 0
var last_fps := 0.0

@onready var status_label: Label = $CanvasLayer/Status

func _ready() -> void:
	status_label.text = "Nexo-3D Forward+\nCollecting frame-time samples..."
	print("[MobileForwardPlusTest] Rendering method: ", RenderingServer.get_current_rendering_method())

func _process(delta: float) -> void:
	elapsed += delta
	frame_count += 1
	if elapsed >= maxf(sample_interval_seconds, 0.25):
		last_fps = float(frame_count) / elapsed
		status_label.text = "Nexo-3D Mobile Renderer Test\nRenderer: %s\nFPS: %.1f\nFrame time: %.2f ms\nElapsed: %.0f s" % [
			RenderingServer.get_current_rendering_method(),
			last_fps,
			1000.0 / maxf(last_fps, 0.001),
			Time.get_ticks_msec() / 1000.0
		]
		print("[MobileForwardPlusTest] renderer=%s fps=%.1f frame_ms=%.2f elapsed_s=%.0f" % [
			RenderingServer.get_current_rendering_method(),
			last_fps,
			1000.0 / maxf(last_fps, 0.001),
			Time.get_ticks_msec() / 1000.0
		])
		elapsed = 0.0
		frame_count = 0
