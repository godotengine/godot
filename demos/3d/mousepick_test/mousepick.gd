
extends RigidBody

# member variables here, example:
# var a=2
# var b="textvar"

var gray_mat = FixedMaterial.new()

var selected=false

func _input_event(camera,event,pos,normal,shape):
	if (event.type==InputEvent.MOUSE_BUTTON and event.pressed):
		if (not selected):
			get_node("mesh").set_material_override(gray_mat)
		else:
			get_node("mesh").set_material_override(null)
		
		selected = not selected
		

func _mouse_enter():
	get_node("mesh").set_scale( Vector3(1.1,1.1,1.1) )

func _mouse_exit():
	get_node("mesh").set_scale( Vector3(1,1,1) )

func _ready():
	# Initalization here
	pass


