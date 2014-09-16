
extends Node2D

# class for a visual effect when the player is hit

func _ready():
	pass

func _die():
	queue_free()

# flip the node's direction
# boolean siding_left : true if siding to the left
func set_direction(siding_left):
	if(siding_left):
		set_scale( Vector2(-1,1))
	else:
		set_scale( Vector2(1,1))
