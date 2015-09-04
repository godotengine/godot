
extends Node2D

# member variables here, example:
# var a=2
# var b="textvar"
var firstTime = true
func _ready():
	# Initialization here
	if(firstTime):
		get_node("AcceptDialog").popup()
		firstTime = false


func _on_Button_pressed():
	print("button default pressed")


func _on_GraphNode_close_request():
	print("graph close")