
extends RayCast2D

# member variables here, example:
# var a=2
# var b="textvar"
var sprite
func _ready():
	sprite = get_node("Sprite")
	set_fixed_process(true)


func _fixed_process(delta):
	sprite.set_pos(get_collision_point() - get_parent().get_parent().get_parent().get_pos())

