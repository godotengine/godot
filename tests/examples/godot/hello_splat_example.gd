extends Node3D

## Hello Splat Example
## Demonstrates basic usage of the GaussianSplatRenderer
##
## This example shows how to:
## - Create a GaussianSplatRenderer node
## - Configure basic rendering settings
## - Monitor performance statistics

var gaussian_renderer: GaussianSplatRenderer

## Reads a renderer stat, preferring the telemetry snapshot when available.
func _get_metric(stats: Dictionary, key: String, default_value):
	var telemetry = stats.get("telemetry", {})
	if telemetry is Dictionary and telemetry.has(key):
		return telemetry[key]
	return stats.get(key, default_value)

## Builds the demo renderer, configures defaults, and starts periodic stats output.
func _ready():
	print("[Hello Splat Example] Starting...")

	# Create and configure the Gaussian Splat Renderer
	gaussian_renderer = GaussianSplatRenderer.new()
	add_child(gaussian_renderer)

	# Configure basic settings for the micro-MVP
	gaussian_renderer.set_render_mode(GaussianSplatRenderer.MODE_3D)
	gaussian_renderer.set_max_splats(1000)  # Low for testing
	gaussian_renderer.set_frustum_culling(false)  # Disabled for MVP
	gaussian_renderer.set_lod_enabled(false)  # Disabled for MVP

	print("[Hello Splat Example] GaussianSplatRenderer created with 100 test splats")

	# Set up camera position to view the test splats
	_setup_camera()

	# Print performance stats every 2 seconds
	var timer = Timer.new()
	timer.wait_time = 2.0
	timer.timeout.connect(_print_performance_stats)
	timer.autostart = true
	add_child(timer)

## Positions the active 3D camera to frame the default splat cube.
func _setup_camera():
	# Position camera to view the test splats (they spawn in a -5 to 5 cube)
	var camera = get_viewport().get_camera_3d()
	if camera:
		# Position camera outside the splat cube
		camera.position = Vector3(8, 3, 8)
		camera.look_at(Vector3.ZERO, Vector3.UP)
		print("[Hello Splat Example] Camera positioned to view test splats")

## Prints the current renderer statistics to the console.
func _print_performance_stats():
	if gaussian_renderer:
		var stats = gaussian_renderer.get_render_stats()
		print("[Hello Splat Example] Performance Stats:")
		print("  Visible splats: ", _get_metric(stats, "visible_splats", 0))
		print("  Total splats: ", _get_metric(stats, "total_splats", 0))
		print("  Render time: %.2fms" % _get_metric(stats, "render_time_ms", 0.0))
		print("  Sort time: %.2fms" % _get_metric(stats, "sort_time_ms", 0.0))
		print("  Frame count: ", _get_metric(stats, "frame_count", 0))

## Handles simple keyboard shortcuts for toggling renderer options.
## @param event: Input event dispatched by the scene tree.
func _input(event):
	# Simple controls for testing
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_1:
				# Toggle render mode
				if gaussian_renderer.get_render_mode() == GaussianSplatRenderer.MODE_3D:
					gaussian_renderer.set_render_mode(GaussianSplatRenderer.MODE_2D)
					print("[Hello Splat Example] Switched to 2D mode")
				else:
					gaussian_renderer.set_render_mode(GaussianSplatRenderer.MODE_3D)
					print("[Hello Splat Example] Switched to 3D mode")

			KEY_2:
				# Toggle frustum culling
				var culling = not gaussian_renderer.get_frustum_culling()
				gaussian_renderer.set_frustum_culling(culling)
				print("[Hello Splat Example] Frustum culling: ", culling)

			KEY_3:
				# Toggle LOD
				var lod = not gaussian_renderer.get_lod_enabled()
				gaussian_renderer.set_lod_enabled(lod)
				print("[Hello Splat Example] LOD enabled: ", lod)

			KEY_P:
				# Print current stats
				_print_performance_stats()

## Quits the demo when the window close request is received.
## @param what: Notification identifier.
func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		print("[Hello Splat Example] Shutting down...")
		get_tree().quit()
