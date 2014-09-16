extends "res://enemies/enemy.gd"

# Class for enemy birds that jump from the ground but doesn't move.

const TIMEOUT_TURN=1 # maximum time to wait before jumping again, in [s]
var timer=TIMEOUT_TURN # time before next jump, in [s]

func move_process(delta):
	# apply gravity to it
	var force = Vector2(0,GRAVITY)
	.slide(delta,force)

	# jump whenever it's touching the ground, but only after a little time
	if(is_on_floor):
		if(timer==TIMEOUT_TURN): # if just landed, change the animation to sit
			set_sub_animation(0)
		timer-=delta
		if(timer<=0): # if time is out
			timer=TIMEOUT_TURN        # prepare timer for next landing
			velocity=Vector2(0,-300)  # make it jump
			set_sub_animation(1)      # set animation to flapping
