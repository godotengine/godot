
extends Control

# member variables here, example:
# var a=2
# var b="textvar"

const MAX_BUBBLES=10

func _ready():
	# Initialization here
	for i in range(MAX_BUBBLES):
		var bubble = preload("res://lens.scn").instance()
		add_child(bubble)
	pass


