
extends Spatial

var raycast
func _ready():
	raycast = get_node("RayCast")
	set_fixed_process(true)

func _fixed_process(delta):
	print(raycast.get_collision_point())

