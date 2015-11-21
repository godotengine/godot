
extends Spatial

# Member variables
var prev_pos = null


func _input(event):
	# All other (non-mouse) events
	if (not event.type in [InputEvent.MOUSE_BUTTON, InputEvent.MOUSE_MOTION, InputEvent.SCREEN_DRAG, InputEvent.SCREEN_TOUCH]):
		get_node("viewport").input(event)


# Mouse events for Area
func _on_area_input_event(camera, event, click_pos, click_normal, shape_idx):
	# Use click pos (click in 3d space, convert to area space)
	var pos = get_node("area").get_global_transform().affine_inverse()*click_pos
	# Convert to 2D
	pos = Vector2(pos.x, pos.y)
	# Convert to viewport coordinate system
	pos.x = (pos.x + 1.5)*100
	pos.y = (-pos.y + 0.75)*100
	# Set to event
	event.pos = pos
	event.global_pos = pos
	if (prev_pos == null):
		prev_pos = pos
	if (event.type == InputEvent.MOUSE_MOTION):
		event.relative_pos = pos - prev_pos
	prev_pos = pos
	# Send the event to the viewport
	get_node("viewport").input(event)


func _ready():
	# Initalization here
	get_node("area/quad").get_material_override().set_texture(FixedMaterial.PARAM_DIFFUSE, get_node("viewport").get_render_target_texture())
	set_process_input(true)
