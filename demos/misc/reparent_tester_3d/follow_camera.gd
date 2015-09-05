
extends Camera

# member variables here, example:
# var a=2
# var b="textvar"

var collision_exception=[]
export var min_distance=0.5
export var max_distance=4.0
export var angle_v_adjust=0.0
export var autoturn_ray_aperture=25
export var autoturn_speed=50
var max_height = 2.0
var min_height = 0


func _ready():

#find collision exceptions for ray
	var node = self
	while(node):
		if (node extends RigidBody):
			collision_exception.append(node.get_rid())
			break
		else:
			node=node.get_parent()
	# Initalization here
	set_fixed_process(true)
	#this detaches the camera transform from the parent spatial node
	set_as_toplevel(true)

	
	



