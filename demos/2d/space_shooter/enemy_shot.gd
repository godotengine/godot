
extends Area2D

# Member variables
const SPEED = -800

var hit = false


func _process(delta):
	translate(Vector2(delta*SPEED, 0))


func _ready():
	set_process(true)


func is_enemy():
	return true


func _hit_something():
	if (hit):
		return
	hit = true
	set_process(false)
	get_node("anim").play("splash")


func _on_visibility_exit_screen():
	queue_free()
