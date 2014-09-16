extends Node2D

# class for the travelStar. It's a trigger to start a cinematic that brings the player to a new level.
# The animation is managed in the level itself with another object, because it's very specific to each level.

const PlayerClass = preload("res://player.gd")

# trigger event when the player jump on the star
func _on_Area2D_body_enter( body ):
	if(body extends PlayerClass):
		# start animation
		start_travel()

# abstract function that contains the start of the animation
func start_travel():
	pass
