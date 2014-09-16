extends "res://enemies/enemy.gd"

# Class for enemy who jumps in circles, meaning he moves with jumps in a short distance and go back to starting point.

export (float) var loop_duration=1.2 # maximum time before he goes back, in [s]
const WALK_MAX_SPEED = 120           # max speed
var timer=loop_duration              # current timer before going back, in [s]

func move_process(delta):
	var force = Vector2(0,GRAVITY)
	
	# accelerate horizontally, if possible
	if(abs(velocity.x)<WALK_MAX_SPEED):
		force.x+=direction*WALK_FORCE
	
	.slide(delta,force)

	timer-=delta
	if(timer<=0):
		# time to go back
		timer=loop_duration # reset timer
		_flip_direction()   # update direction, including sprite
		velocity.x=velocity.x/2 # give a little inertia by reducing current speed to its half
	
	# jump again if touching floor
	if(is_on_floor):
		velocity.y=-80
