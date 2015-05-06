
extends Area2D

# member variables here, example:
# var a=2
# var b="textvar"

var taken=false


func _on_body_enter( body ):
	if (not taken and body extends preload("res://player.gd")):
		get_node("anim").play("taken")
		taken=true


func _ready():
	# Initalization here
	pass

