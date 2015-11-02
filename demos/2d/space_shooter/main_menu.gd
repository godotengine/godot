
extends Control

# member variables here, example:
# var a=2
# var b="textvar"

func _ready():

	get_node("score").set_text( "HIGH SCORE: "+str( get_node("/root/game_state").max_points ) )
	# Initialization here
	pass




func _on_play_pressed():
	get_node("/root/game_state").points=0
	get_tree().change_scene("res://level.scn")
	pass # replace with function body
