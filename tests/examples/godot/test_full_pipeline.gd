extends Node3D

var renderer: GaussianSplatRenderer
var camera: Camera3D
var ply_loader: PLYLoader
var ply_path: String = ""

# Performance tracking
var frame_times: Array = []
var last_fps_update: float = 0.0

## Initializes the full pipeline test scene and prints controls.
func _ready():
	print("=== Gaussian Splatting Full Pipeline Test ===")
	print("Features to test:")
	print("- PLY file loading")
	print("- GPU memory streaming")
	print("- Tile-based rasterization")
	print("- Bitonic GPU sorting")
	print("")

	# Setup scene
	_setup_camera()
	_setup_renderer()

	# Try to load PLY file
	_load_ply_data()

	print("\nControls:")
	print("- Mouse: Rotate camera")
	print("- W/A/S/D: Move camera")
	print("- Space: Toggle sorting method")
	print("- R: Reload PLY file")
	print("- T: Test tile renderer")
	print("- L: Toggle LOD")
	print("- ESC: Exit")

## Creates the test camera with a default view of the scene.
func _setup_camera():
	camera = Camera3D.new()
	camera.position = Vector3(0, 0, 5)
	camera.look_at(Vector3.ZERO, Vector3.UP)
	camera.fov = 60
	add_child(camera)
	print("Camera created")

## Instantiates and configures the GaussianSplatRenderer for the test.
func _setup_renderer():
	renderer = GaussianSplatRenderer.new()
	renderer.name = "GaussianRenderer"
	add_child(renderer)

	# Configure renderer
	renderer.set_render_mode(GaussianSplatRenderer.MODE_3D)
	renderer.set_max_splats(2000000)  # Support up to 2M splats
	renderer.set_lod_enabled(true)
	renderer.set_frustum_culling(true)

	print("Renderer configured:")
	print("- Max splats: 2,000,000")
	print("- Sorting: GPU Radix")
	print("- LOD: Enabled")
	print("- Frustum culling: Enabled")

## Loads a PLY file specified on the command line or falls back to synthetic data.
func _load_ply_data():
	# Check if PLY file path is provided
	var args = OS.get_cmdline_args()
	for arg in args:
		if arg.ends_with(".ply"):
			ply_path = arg
			break

	if ply_path.is_empty():
		print("\nNo PLY file specified. Using test data.")
		print("To load a PLY file, run with: --path your_file.ply")
		_create_test_data()
		return

	print("\nLoading PLY file: " + ply_path)

	# Load PLY file
	ply_loader = PLYLoader.new()
	var error = ply_loader.load_file(ply_path)

	if error != OK:
		print("Failed to load PLY file: " + str(error))
		_create_test_data()
		return

	# Get loaded data
	var gaussian_data = ply_loader.get_gaussian_data()
	if gaussian_data:
		renderer.set_gaussian_data(gaussian_data)
		var stats = ply_loader.get_load_statistics()
		print("PLY loaded successfully!")
		print("- Splat count: " + str(ply_loader.get_splat_count()))
		print("- Load time: " + str(stats.get("load_time_ms", 0)) + " ms")
		print("- Memory usage: " + str(stats.get("memory_mb", 0)) + " MB")
	else:
		print("No data in PLY file")
		_create_test_data()

## Generates synthetic splats to validate pipeline behavior without external data.
func _create_test_data():
	print("Creating synthetic test data...")

	var test_data = GaussianData.new()

	# Create a larger test dataset
	var grid_size = 20  # 20x20x5 = 2000 splats
	var spacing = 0.3

	for x in range(grid_size):
		for y in range(5):  # Fewer layers vertically
			for z in range(grid_size):
				var pos = Vector3(
					(x - grid_size/2.0) * spacing,
					y * spacing - 1.0,
					(z - grid_size/2.0) * spacing
				)

				# Create color gradient
				var color = Color(
					float(x) / grid_size,
					float(y) / 5.0,
					float(z) / grid_size,
					0.9
				)

				# Varying scales for visual interest
				var scale = randf_range(0.05, 0.15)
				var scale_vec = Vector3(scale, scale, scale)

				# Random rotation
				var rotation = Quaternion(Vector3.UP, randf() * TAU)

				test_data.add_splat(pos, color, scale_vec, rotation)

	print("Created %d test splats" % test_data.get_splat_count())
	renderer.set_gaussian_data(test_data)

	# Test GPU systems
	renderer.test_gpu_sort()

## Updates camera movement and prints periodic performance stats.
## @param delta: Frame delta in seconds.
func _process(delta):
	# Camera movement
	_handle_camera_movement(delta)

	# Update FPS display
	_update_performance_stats(delta)

## Moves the camera in response to keyboard and mouse input.
## @param delta: Frame delta in seconds.
func _handle_camera_movement(delta):
	if !camera:
		return

	var speed = 5.0 * delta
	var rotation_speed = 2.0 * delta

	# Keyboard movement
	if Input.is_key_pressed(KEY_W):
		camera.position -= camera.transform.basis.z * speed
	if Input.is_key_pressed(KEY_S):
		camera.position += camera.transform.basis.z * speed
	if Input.is_key_pressed(KEY_A):
		camera.position -= camera.transform.basis.x * speed
	if Input.is_key_pressed(KEY_D):
		camera.position += camera.transform.basis.x * speed

	# Mouse rotation (when button pressed)
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		var mouse_delta = Input.get_last_mouse_velocity() * 0.001
		camera.rotate_y(-mouse_delta.x)
		camera.rotate_object_local(Vector3.RIGHT, -mouse_delta.y)

## Prints renderer and FPS stats once per second.
## @param delta: Frame delta in seconds.
func _update_performance_stats(delta):
	last_fps_update += delta

	if last_fps_update >= 1.0:
		var fps = Engine.get_frames_per_second()
		var frame_time = 1000.0 / max(fps, 1.0)

		print("Performance: %.1f FPS (%.2f ms)" % [fps, frame_time])

		# Get renderer stats
		var stats = renderer.get_performance_stats()
		if stats:
			print("  Sort: %.2f ms" % stats.get("sort_time", 0))
			print("  Render: %.2f ms" % stats.get("render_time", 0))
			print("  Visible: %d splats" % stats.get("visible_count", 0))

		last_fps_update = 0.0

## Handles hotkeys for renderer toggles and diagnostics.
## @param event: Input event dispatched by the scene tree.
func _input(event):
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_ESCAPE:
				print("\n=== Test Complete ===")
				get_tree().quit()

			KEY_R:
				# Reload PLY
				if !ply_path.is_empty():
					_load_ply_data()

			KEY_T:
				# Test tile renderer
				print("Testing tile-based renderer...")
				renderer.test_tile_rendering()

			KEY_L:
				# Toggle LOD
				var lod_enabled = !renderer.get_lod_enabled()
				renderer.set_lod_enabled(lod_enabled)
				print("LOD: " + ("Enabled" if lod_enabled else "Disabled"))

			KEY_F:
				# Toggle frustum culling
				var culling = !renderer.get_frustum_culling()
				renderer.set_frustum_culling(culling)
				print("Frustum culling: " + ("Enabled" if culling else "Disabled"))

			KEY_M:
				# Memory stats
				var mem_stats = renderer.get_memory_stats()
				print("\nMemory Statistics:")
				print("  Allocated: %.2f MB" % mem_stats.get("allocated_mb", 0))
				print("  Used: %.2f MB" % mem_stats.get("used_mb", 0))
				print("  Efficiency: %.1f%%" % (mem_stats.get("efficiency", 0) * 100))
