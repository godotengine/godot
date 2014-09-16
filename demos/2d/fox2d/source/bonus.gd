extends "Edible.gd"

# class for the bonus. Its effect to the player changes depending on its type.
# the bonus falls but doesn't move by itself.

# Parameters --------------------------------------------------------------
# type of bonus. By default it's health
export(int,"Health","Full health","1up","Mik","Curry","Lemon") var bonus_type=0

# Constants ---------------------------------------------------------------
const GRAVITY = 700.0

# Variables ---------------------------------------------------------------
var velocity= Vector2()

# Functions ---------------------------------------------------------------

# Initializer
func _ready():
	# update the sprite depends on the bonus type
	get_node("sprite").set_frame(bonus_type)
	set_fixed_process(true)
	
# make the object fall if possible
func _fixed_process(delta):
	fall(delta)

# generic function to make a KinematicBody2D slide.
# float delta : passed time since last update, in [s]
# Vector2 force : force applied to the object, to make it move
func slide(delta,force):
	velocity+=force*delta
	var motion = velocity * delta
	motion = move(motion)
	
	var floor_velocity=Vector2()
	if (is_colliding()):
		var n = get_collision_normal()
		
		if ( rad2deg(acos(n.dot( Vector2(0,-1)))) < 40 ):
			floor_velocity=get_collider_velocity()
		
		motion = n.slide(motion)
		velocity = n.slide(velocity)
		
		move(motion)
	
	if (floor_velocity!=Vector2()):
		# if floor moves, move the object with the floor
		move(floor_velocity*delta)

# generic function to make a KinematicBody2D fall
# float delta : passed time since last update, in [s]
func fall(delta):
	var force = Vector2(0,GRAVITY)
	
	slide(delta,force)
