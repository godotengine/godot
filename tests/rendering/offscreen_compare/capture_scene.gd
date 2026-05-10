extends Node3D

const DEFAULT_OUTPUT := "user://offscreen_compare_capture.png"

func _ready() -> void:
	_build_scene()

	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw

	var image := get_viewport().get_texture().get_image()
	var output_path := _get_user_arg_value("--capture-output", DEFAULT_OUTPUT)
	var err := image.save_png(output_path)
	if image.is_empty() or err != OK:
		push_error("Failed to save capture to %s, image empty: %s, error: %s" % [output_path, image.is_empty(), error_string(err)])
		get_tree().quit(1)
	else:
		print("CAPTURE_PATH=", output_path)
		print("CAPTURE_SIZE=", image.get_width(), "x", image.get_height())

	get_tree().quit()

func _build_scene() -> void:
	var world := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.06, 0.08, 0.11)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.24, 0.28, 0.34)
	env.ambient_light_energy = 0.7
	world.environment = env
	add_child(world)

	var camera := Camera3D.new()
	camera.look_at_from_position(Vector3(0.0, 1.25, 5.0), Vector3(0.0, 0.5, 0.0))
	camera.current = true
	add_child(camera)

	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-45.0, -35.0, 0.0)
	light.light_energy = 2.5
	add_child(light)

	var cube := MeshInstance3D.new()
	cube.mesh = BoxMesh.new()
	cube.position = Vector3(-0.95, 0.6, 0.0)
	cube.rotation_degrees = Vector3(22.0, 35.0, 0.0)
	cube.material_override = _make_material(Color(0.95, 0.22, 0.16))
	add_child(cube)

	var sphere := MeshInstance3D.new()
	sphere.mesh = SphereMesh.new()
	sphere.position = Vector3(0.9, 0.55, 0.0)
	sphere.material_override = _make_material(Color(0.1, 0.65, 1.0))
	add_child(sphere)

	var plane := MeshInstance3D.new()
	var plane_mesh := PlaneMesh.new()
	plane_mesh.size = Vector2(5.0, 3.0)
	plane.mesh = plane_mesh
	plane.position = Vector3(0.0, -0.05, 0.0)
	plane.material_override = _make_material(Color(0.18, 0.22, 0.18))
	add_child(plane)

	var overlay := Control.new()
	overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(overlay)

	var banner := ColorRect.new()
	banner.color = Color(0.0, 0.0, 0.0, 0.45)
	banner.position = Vector2(18.0, 18.0)
	banner.size = Vector2(282.0, 58.0)
	overlay.add_child(banner)

	var label := Label.new()
	label.text = "OFFSCREEN RD VISUAL"
	label.position = Vector2(32.0, 34.0)
	overlay.add_child(label)

func _make_material(color: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 0.55
	mat.metallic = 0.0
	return mat

func _get_user_arg_value(name: String, default_value: String) -> String:
	var args := OS.get_cmdline_user_args()
	for i in range(args.size()):
		if args[i] == name and i + 1 < args.size():
			return args[i + 1]
	return default_value
