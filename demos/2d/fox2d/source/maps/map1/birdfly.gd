extends "res://enemies/enemy.gd"

# Class for enemies that fly like a sinus function in air. It is not affected by gravity.

# Parameters -----------------------------------------------
export (float) var amplitude=2 # amplitude of the wave
export (float) var frequency=2 # frequency of the wave (bigger freq. is smaller wave)

# Functions ---------------------------------------------------
func move_process(delta):
	var force = Vector2(0,0)
	
	#if not at full speed, continue accelerate in the same direction
	if(abs(velocity.x)<WALK_MAX_SPEED):
		force.x+=direction*WALK_FORCE
	
	velocity+=force*delta # add current horizontal force to velocity, so it moves anyway in x axe
	
	velocity.y=sin(OS.get_ticks_msec()/(100*frequency))*100*amplitude # set the vertical velocity as a sinus function. Note that for better control, it doesn't add force but set directly the velocity.
	
	# make the node move
	var motion = velocity * delta
	motion = move(motion)
	
	if (is_colliding()):
		var n = get_collision_normal()
		
		motion = n.slide(motion)
		velocity = n.slide(velocity)
		
		move(motion)
