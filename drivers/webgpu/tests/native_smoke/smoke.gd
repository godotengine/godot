extends Node3D

var frames := 0
var light: DirectionalLight3D
var lit := 0.0
var started := Time.get_ticks_msec()


func _ready() -> void:
	if RenderingServer.get_current_rendering_driver_name() != "webgpu":
		push_error("Native WebGPU fell back to another renderer")
		get_tree().quit(1)
		return
	var box := MeshInstance3D.new()
	box.mesh = BoxMesh.new()
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.15, 0.35, 1)
	box.material_override = material
	add_child(box)
	var camera := Camera3D.new()
	camera.position = Vector3(0, 0, 3)
	camera.current = true
	add_child(camera)
	light = DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-30, -30, 0)
	add_child(light)
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color(0.118, 0.149, 0.196)
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color(0.62, 0.7, 0.85)
	environment.ambient_light_energy = 0.55
	var world := WorldEnvironment.new()
	world.environment = environment
	add_child(world)


func _process(_delta: float) -> void:
	frames += 1
	if frames == 8 or frames == 16:
		_check_pixels()


func _check_pixels() -> void:
	await RenderingServer.frame_post_draw
	var image := get_viewport().get_texture().get_image()
	if image == null or image.is_empty():
		push_error("Native WebGPU returned no viewport image")
		get_tree().quit(1)
		return
	var center := image.get_pixel(image.get_width() / 2, image.get_height() / 2)
	if center.b < 0.2 or center.b < center.r * 2.0:
		push_error("Native WebGPU lost the blue material color: %s" % center)
		get_tree().quit(1)
		return
	var brightness := (center.r + center.g + center.b) / 3.0
	print("NATIVE_WEBGPU_LIGHTING_PIXEL: light=", light.visible, " center=", center)
	if light.visible:
		lit = brightness
		light.visible = false
		return
	if brightness < 0.05 or lit < brightness + 0.05:
		push_error("Native WebGPU lost ambient or directional lighting: lit=%f ambient=%f" % [lit, brightness])
		get_tree().quit(1)
		return
	print("NATIVE_WEBGPU_SMOKE_PASSED: lit=", lit, " ambient=", brightness, " elapsed_ms=", Time.get_ticks_msec() - started)
	get_tree().quit()
