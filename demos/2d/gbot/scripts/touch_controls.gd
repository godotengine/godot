
extends CanvasLayer

# member variables here, example:
# var a=2
# var b="textvar"

func _ready():
	# Initalization here
	#var viewport = get_scene().get_root().get_node("world").get_viewport_rect().size
	var viewport = get_node("Left_Input/Touch_BG").get_viewport_rect().size
	var controls_left = get_node("Left_Input")
	var controls_right = get_node("Right_Input")
	
	var scale = viewport.y/768

	controls_left.set_pos(Vector2(0,viewport.y))
	controls_right.set_pos(Vector2(viewport.x,viewport.y))
	
	controls_left.set_scale(Vector2(scale,scale))
	controls_right.set_scale(Vector2(scale,scale))


