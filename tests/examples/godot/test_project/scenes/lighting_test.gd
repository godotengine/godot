extends Node3D

## Lighting test scene controller
## Tests directional, omni, and spot lights on Gaussian splats
## Controls: WASD=move, mouse=look, Shift=fast, ESC=quit
## F1-F4: Toggle lights, F5-F9: Debug modes

@onready var directional_light: DirectionalLight3D = $DirectionalLight3D
@onready var omni_red: OmniLight3D = $OmniRed
@onready var omni_green: OmniLight3D = $OmniGreen
@onready var omni_blue: OmniLight3D = $OmniBlue
@onready var spot_light: SpotLight3D = $SpotLight3D
@onready var white_spots: Node3D = $WhiteSpots
@onready var cabin: GaussianSplatNode3D = $Cabin
@onready var label: Label = $UI/Label

func _ready() -> void:
	print("[LightingTest] Scene ready - FPS camera mode")
	print("[LightingTest] Controls: WASD=move, mouse=look, Shift=fast")
	print("[LightingTest] F1=Dir, F2=Omni, F3=Spot, F4=All, F5=Normal, F6=Stats")
	print("[LightingTest] F7=ProjHeat, F8=Unclustered, F9=WhiteAlbedo, ESC=Quit")
	print("[LightingTest] F10=Shadow Opacity Debug, F11=White Spots (5x neutral)")
	if white_spots:
		white_spots.visible = false  # Start with white spots off
	print("[LightingTest] Directional: ", directional_light != null)
	print("[LightingTest] Omni lights: ", omni_red != null, omni_green != null, omni_blue != null)
	print("[LightingTest] Spot light: ", spot_light != null)
	print("[LightingTest] Cabin splat: ", cabin != null)
	_update_label()

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_F1:
				directional_light.visible = not directional_light.visible
				print("[LightingTest] Directional: ", "ON" if directional_light.visible else "OFF")
			KEY_F2:
				var on = not omni_red.visible
				omni_red.visible = on
				omni_green.visible = on
				omni_blue.visible = on
				print("[LightingTest] Omni lights: ", "ON" if on else "OFF")
			KEY_F3:
				spot_light.visible = not spot_light.visible
				print("[LightingTest] Spot light: ", "ON" if spot_light.visible else "OFF")
			KEY_F4:
				# Toggle all lights
				var all_on = directional_light.visible and omni_red.visible and spot_light.visible
				directional_light.visible = not all_on
				omni_red.visible = not all_on
				omni_green.visible = not all_on
				omni_blue.visible = not all_on
				spot_light.visible = not all_on
				print("[LightingTest] All lights: ", "ON" if not all_on else "OFF")
			KEY_F5:
				# Cycle normal mode
				if cabin:
					var current = cabin.get("rendering/normal_mode")
					if current == null:
						current = 2
					var next_mode = (current + 1) % 3
					cabin.set("rendering/normal_mode", next_mode)
					print("[LightingTest] Normal mode: ", next_mode)
			KEY_F6:
				# Print overflow stats
				_print_overflow_stats()
			KEY_F7:
				# Toggle projection mismatch heatmap
				_toggle_projection_debug()
			KEY_F8:
				# Toggle unclustered lighting
				_toggle_unclustered_lights()
			KEY_F9:
				# Toggle white albedo debug mode
				_toggle_white_albedo()
			KEY_F10:
				# Toggle shadow opacity debug mode
				_toggle_shadow_opacity_debug()
			KEY_F11:
				# Toggle 5 white spot lights
				if white_spots:
					white_spots.visible = not white_spots.visible
					print("[LightingTest] White spots (5x neutral): ", "ON" if white_spots.visible else "OFF")
		_update_label()

func _update_label() -> void:
	if label:
		var text = "Lighting Test (FPS Camera)\n"
		text += "WASD: Move, Mouse: Look\n"
		text += "Shift: Fast, ESC: Quit\n\n"
		text += "F1: Directional [%s]\n" % ("ON" if directional_light.visible else "OFF")
		text += "F2: Omni RGB [%s]\n" % ("ON" if omni_red.visible else "OFF")
		text += "F3: Spot [%s]\n" % ("ON" if spot_light.visible else "OFF")
		text += "F4: Toggle All\n"
		text += "F5: Cycle Normal Mode\n"
		text += "F6: Print Overflow Stats\n"
		text += "F7: Proj Heatmap [%s]\n" % ("ON" if projection_debug_enabled else "OFF")
		text += "F8: Unclustered [%s]\n" % ("ON" if force_unclustered else "OFF")
		text += "F9: White Albedo [%s]\n" % ("ON" if white_albedo_enabled else "OFF")
		text += "F10: Shadow Opacity [%s]\n" % ("ON" if shadow_opacity_debug_enabled else "OFF")
		text += "F11: White Spots [%s]" % ("ON" if white_spots and white_spots.visible else "OFF")
		label.text = text

var projection_debug_enabled: bool = false
var force_unclustered: bool = false
var white_albedo_enabled: bool = false
var shadow_opacity_debug_enabled: bool = false

func _toggle_projection_debug() -> void:
	projection_debug_enabled = not projection_debug_enabled
	if cabin and cabin.has_method("get_renderer"):
		var renderer = cabin.get_renderer()
		if renderer:
			renderer.set_debug_show_projection_issues(projection_debug_enabled)
			print("[LightingTest] Projection mismatch heatmap: ", "ON" if projection_debug_enabled else "OFF")
			return
	print("[LightingTest] Could not toggle heatmap - no renderer")

func _toggle_unclustered_lights() -> void:
	force_unclustered = not force_unclustered
	ProjectSettings.set_setting("rendering/gaussian_splatting/debug/force_unclustered_lights", force_unclustered)
	print("[LightingTest] Force unclustered lights: ", "ON" if force_unclustered else "OFF")
	print("[LightingTest] NOTE: May need scene reload to take effect")

func _toggle_white_albedo() -> void:
	white_albedo_enabled = not white_albedo_enabled
	if cabin and cabin.has_method("get_renderer"):
		var renderer = cabin.get_renderer()
		if renderer:
			renderer.set_debug_show_white_albedo(white_albedo_enabled)
			print("[LightingTest] White albedo: ", "ON" if white_albedo_enabled else "OFF")
			return
	print("[LightingTest] Could not toggle white albedo - no renderer")

func _toggle_shadow_opacity_debug() -> void:
	shadow_opacity_debug_enabled = not shadow_opacity_debug_enabled
	if cabin and cabin.has_method("get_renderer"):
		var renderer = cabin.get_renderer()
		if renderer and renderer.has_method("set_debug_show_shadow_opacity"):
			renderer.set_debug_show_shadow_opacity(shadow_opacity_debug_enabled)
			print("[LightingTest] Shadow opacity debug: ", "ON" if shadow_opacity_debug_enabled else "OFF")
			return
	print("[LightingTest] Could not toggle shadow opacity debug - no renderer")

func _print_overflow_stats() -> void:
	print("[LightingTest] === Overflow Statistics ===")
	# Try to find a GaussianSplatWorld3D or get renderer from cabin
	var world = get_node_or_null("GaussianSplatWorld3D")
	if world and world.has_method("get_renderer"):
		var renderer = world.get_renderer()
		if renderer and renderer.has_method("get_overflow_stats"):
			var overflow = renderer.get_overflow_stats()
			print("[LightingTest] Overflow clamped: ", overflow.get("clamped_records", 0))
			print("[LightingTest] Overflow tiles: ", overflow.get("overflow_tile_count", 0))
			print("[LightingTest] Overflow aggregated: ", overflow.get("overflow_splats_aggregated", 0))
			print("[LightingTest] Full stats: ", overflow)
			return
	# Fallback: try cabin node directly
	if cabin and cabin.has_method("get_renderer"):
		var renderer = cabin.get_renderer()
		if renderer and renderer.has_method("get_overflow_stats"):
			var overflow = renderer.get_overflow_stats()
			print("[LightingTest] Overflow clamped: ", overflow.get("clamped_records", 0))
			print("[LightingTest] Overflow tiles: ", overflow.get("overflow_tile_count", 0))
			print("[LightingTest] Overflow aggregated: ", overflow.get("overflow_splats_aggregated", 0))
			print("[LightingTest] Full stats: ", overflow)
			return
	print("[LightingTest] Could not get overflow stats - no renderer found")
