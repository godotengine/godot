extends KinematicBody2D

# Class for a breath object. It is a kinematic object with an animation 
# to manage flipping to the left side.
# The breath won't go through the map, so the movement cannot be in an animation.
# Note: It was a first attempt, very soon replaced by a Flip H of the sprite.

# Constants ---------------------------------------------------------------
const TIMEOUT_MOVING_BREATH=0.1 # max time left before it stops to move.

# Variables ---------------------------------------------------------------
var age=0 # timer before it stops to move
var velocity=0 # velocity of the node

# Functions ---------------------------------------------------------------

# called by the animation to kill the object
func _die():
	queue_free()

# initializer
func _ready():
	age=0
	set_fixed_process(true)
	get_node("anim").play("main")

# moves the breath, but only if not colliding and if still within the timeout.
# float delta : passed time since last update, in [s]
func _fixed_process(delta):
	age=age+delta
	if(age>TIMEOUT_MOVING_BREATH):
		velocity=Vector2(0,0)
		
	var motion = velocity * delta
	motion = move(motion)
	if (is_colliding()):
		queue_free()

# set the original speed, and switch the animation if the object must side to the left.
# Vector2 vel : original speed of the object
func set_linear_velocity(vel):
	velocity=vel
	if(vel.x<0):
		get_node("anim").play("main_left")
