extends "res://enemies/enemy.gd"

# obsolete. does the same thing as the default behavior of enemies

const WALK_FORCE = 200
const WALK_MAX_SPEED = 100
export (float) var loop_duration=1
var timer=loop_duration

func move_process(delta):
	var force = Vector2(0,GRAVITY)
	
	if(abs(velocity.x)<WALK_MAX_SPEED):
		force.x+=direction*WALK_FORCE
	
	.slide(delta,force)

	timer-=delta
	if(timer<=0):
		timer=loop_duration
		_flip_direction()
		velocity.x=velocity.x/2
