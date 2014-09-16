extends "res://enemies/enemy.gd"

# Class for a flying enemy that goes up and down (sinus function) without moving horizontally.

func move_process(delta):
	velocity.y=sin(OS.get_ticks_msec()/400)*150
	
	var motion = velocity * delta
	motion = move(motion)
