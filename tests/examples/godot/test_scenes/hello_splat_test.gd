extends Node3D

# Hello Splat Test Scene
# This script demonstrates the Micro-MVP Gaussian Splat renderer
# Attach this to a Node3D and run to see 100 colorful splats

var gaussian_renderer: GaussianSplatRenderer

## Instantiates the Hello Splat renderer, camera, and lighting for the demo scene.
func _ready():
	print("[Test Scene] Setting up Hello Splat test...")

	# Create the Gaussian Splat Renderer node
	gaussian_renderer = GaussianSplatRenderer.new()
	gaussian_renderer.name = "HelloSplatRenderer"
	add_child(gaussian_renderer)

	# The renderer will automatically generate 100 test splats on _ready

	# Add a simple camera if one doesn't exist
	if not has_node("Camera3D"):
		var camera = Camera3D.new()
		camera.name = "Camera3D"
		camera.position = Vector3(0, 0, 10)
		camera.look_at(Vector3.ZERO, Vector3.UP)
		add_child(camera)
		print("[Test Scene] Added camera at position (0, 0, 10)")

	# Add some basic lighting
	if not has_node("DirectionalLight3D"):
		var light = DirectionalLight3D.new()
		light.name = "DirectionalLight3D"
		light.rotation_degrees = Vector3(-45, -45, 0)
		add_child(light)
		print("[Test Scene] Added directional light")

	print("[Test Scene] Hello Splat test scene ready!")
	print("[Test Scene] You should see 100 colored splats in a 10x10x10 cube")

	# Print render stats after a short delay
	await get_tree().create_timer(1.0).timeout
	_print_stats()

## Prints current renderer statistics to the console.
func _print_stats():
	if gaussian_renderer:
		var stats = gaussian_renderer.get_render_stats()
		print("\n=== Gaussian Splat Renderer Stats ===")
		print("Visible Splats: ", stats.visible_splats)
		print("Total Splats: ", stats.total_splats)
		print("Frame Count: ", stats.frame_count)
		print("Render Mode: ", stats.render_mode)
		print("Sorting Method: ", stats.sorting_method)
		print("=====================================\n")

## Handles input shortcuts for stats output and renderer visibility.
## @param event: Input event dispatched by the scene tree.
func _input(event):
	# Press SPACE to print current stats
	if event.is_action_pressed("ui_select"):
		_print_stats()

	# Press R to toggle renderer on/off for debugging
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_R:
			if gaussian_renderer:
				gaussian_renderer.visible = !gaussian_renderer.visible
				print("[Test Scene] Renderer visibility: ", gaussian_renderer.visible)
