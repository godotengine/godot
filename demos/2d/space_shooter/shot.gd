
extends Area2D

# member variables here, example:
# var a=2
# var b="textvar"

const SPEED = 800

func _process(delta):
	translate(Vector2(delta*SPEED,0))

func _ready():
	# Initialization here
	set_process(true)
	pass

var hit=false

func _hit_something():
	if (hit):
		return
	hit=true
	set_process(false)
	get_node("anim").play("splash")

func _on_visibility_exit_screen():
	queue_free()
	pass # replace with function body



func _on_shot_area_enter( area ):
	#hit an enemy or asteroid
	if (area.has_method("destroy")):
		#duck typing at it's best
		area.destroy()
		_hit_something()
	
	
	pass 


func _on_shot_body_enter( body ):
	#hit the tilemap
	_hit_something()
	pass # replace with function body

