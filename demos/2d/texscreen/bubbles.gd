
extends Control

# Member variables
const MAX_BUBBLES = 10


func _ready():
	for i in range(MAX_BUBBLES):
		var bubble = preload("res://lens.tscn").instance()
		add_child(bubble)
