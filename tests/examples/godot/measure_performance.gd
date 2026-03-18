extends Node

var frame_count = 0
var time_accumulator = 0.0
var fps_samples = []
var splat_renderer

## Initializes the measurement run by locating the GaussianSplatRenderer node.
func _ready():
	print("=== PERFORMANCE MEASUREMENT START ===")
	print("Measuring Gaussian Splatting Performance...")

	# Find the GaussianSplatRenderer
	splat_renderer = get_node_or_null("/root/Node3D/GaussianSplatRenderer")
	if splat_renderer:
		print("Found GaussianSplatRenderer")
		var children = splat_renderer.get_children()
		print("Number of splat meshes: ", children.size())
	else:
		print("ERROR: GaussianSplatRenderer not found!")
		get_tree().quit()

## Samples per-frame FPS and prints rolling stats once per second.
## @param delta: Frame delta in seconds.
func _process(delta):
	frame_count += 1
	time_accumulator += delta

	var current_fps = 1.0 / delta if delta > 0 else 0
	fps_samples.append(current_fps)

	# Every second, report FPS
	if time_accumulator >= 1.0:
		var avg_fps = 0.0
		for fps in fps_samples:
			avg_fps += fps
		avg_fps /= fps_samples.size()

		print("[PERFORMANCE] Frame ", frame_count, " | Avg FPS: ", "%.1f" % avg_fps,
			  " | Min FPS: ", "%.1f" % fps_samples.min(),
			  " | Max FPS: ", "%.1f" % fps_samples.max())

		time_accumulator = 0.0
		fps_samples.clear()

	# Run for 10 seconds then quit
	if frame_count > 600:
		print("=== PERFORMANCE MEASUREMENT COMPLETE ===")
		print("Total frames rendered: ", frame_count)
		print("Average frame time: ", "%.2f" % (Engine.get_frames_drawn() / Engine.get_frames_per_second()), "ms")
		get_tree().quit()
