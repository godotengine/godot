
extends Spatial


func _ready():
	# Initalization here
	var tex = get_node("Viewport").get_render_target_texture()
	get_node("Quad").get_material_override().set_texture(FixedMaterial.PARAM_DIFFUSE, tex)
