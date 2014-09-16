extends "res://enemies/enemy.gd"

# Class for enemies that move very little but that are trembling constantly. The trick is to make it always jump but very low, and make it move horizontally in a random way. 

# Constants ---------------------------------------------------
const MUSHROOM_SPEED=10 # speed of the enemy, in [px/s]

# Functions ---------------------------------------------------
func move_process(delta):
	
	slide(delta,Vector2(0,GRAVITY)) # make it fall
	
	if(is_on_floor): # jump when possible
		velocity.x=randf()*MUSHROOM_SPEED*2-MUSHROOM_SPEED
		velocity.y=-60
