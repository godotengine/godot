extends "res://Edible.gd"

# class of the bombs in the first boss.
# it flies to its original direction, without falling. It slides agains the terrain, thought.
# It can be eaten by the player, as a normal enemy, but it explodes if it touches the player without being eaten.
# It can also be destroyed by a star.
#
# Note: because it doesn't work exactly like an enemy, it doesn't extends the class Enemy. 
# In this case, duck typing is used to qualify bombs like enemy.

# Constants ---------------------------------------------------------------
const is_enemy=true      # used for duck typing

# Variables ---------------------------------------------------------------
var velocity= Vector2()  
var lifespan=1000        # maximum time before the bomb dies by itself, in case it get out of screen.

# Functions ---------------------------------------------------------------

# override startMove from Edible. Stops what it was doing.
func startMove(target):
	set_fixed_process(false)
	.startMove(target)

# override moveTimeOut from Edible. Restart what it was doing.
func moveTimeOut():
	.moveTimeOut()
	set_fixed_process(true)

# Initializer
func _ready():
	set_fixed_process(true)

# when hit by star, die
func hit():
	queue_free()

# move and check if lifespan is not depleted
func _fixed_process(delta):
	lifespan-=delta
	if lifespan<=0:
		queue_free()
	else:
		slide(delta)

# slide with physics
func slide(delta):
	var motion = velocity * delta
	motion = move(motion)
	
	if (is_colliding()):
		var n = get_collision_normal()
		
		motion = n.slide(motion)
		velocity = n.slide(velocity)
		
		move(motion)

# setter for default velocity
func set_velocity(new_velocity):
	velocity=new_velocity