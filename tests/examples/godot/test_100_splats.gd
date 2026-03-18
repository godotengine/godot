extends Node3D

var renderer: GaussianSplatRenderer
var camera: Camera3D

## Creates a 100-splat test scene and configures the renderer for inspection.
func _ready():
	print("=== Testing 100 Gaussian Splats Rendering ===")

	# Create camera
	camera = Camera3D.new()
	camera.position = Vector3(0, 0, 5)
	camera.look_at(Vector3.ZERO, Vector3.UP)
	add_child(camera)

	# Create GaussianSplatRenderer
	renderer = GaussianSplatRenderer.new()
	renderer.name = "GaussianRenderer"
	add_child(renderer)

	# Generate test data with 100 splats
	var test_data = GaussianData.new()

	# Create a grid of splats
	var grid_size = 10  # 10x10 = 100 splats
	var spacing = 0.5

	for x in range(grid_size):
		for z in range(grid_size):
			var pos = Vector3(
				(x - grid_size/2.0) * spacing,
				randf_range(-0.2, 0.2),  # Small height variation
				(z - grid_size/2.0) * spacing
			)

			# Random colors
			var color = Color(randf(), randf(), randf(), 0.9)

			# Random scale
			var scale = randf_range(0.1, 0.3)
			var scale_vec = Vector3(scale, scale, scale)

			# No rotation for simplicity
			var rotation = Quaternion()

			test_data.add_splat(pos, color, scale_vec, rotation)

	print("Created %d test splats" % test_data.get_splat_count())

	# Set the data on the renderer
	renderer.set_gaussian_data(test_data)

	# Configure rendering
	renderer.set_render_mode(GaussianSplatRenderer.MODE_3D)
	renderer.set_lod_enabled(false)

	print("Renderer configured")
	print("- Render mode: 3D")
	print("- Sorting: GPU Radix")
	print("- LOD: Disabled")

	# Test GPU sort
	renderer.test_gpu_sort()

## Rotates the camera and prints periodic FPS measurements.
## @param delta: Frame delta in seconds.
func _process(delta):
	# Rotate camera around the splats
	if camera:
		var angle = Time.get_ticks_msec() * 0.001
		camera.position = Vector3(sin(angle) * 5, 2, cos(angle) * 5)
		camera.look_at(Vector3.ZERO, Vector3.UP)

	# Print performance every second
	if Engine.get_frames_drawn() % 60 == 0:
		print("FPS: %.1f" % Engine.get_frames_per_second())

## Handles hotkeys for quitting or switching sorting methods.
## @param event: Input event dispatched by the scene tree.
func _input(event):
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_ESCAPE:
			print("=== Test Complete ===")
			get_tree().quit()
