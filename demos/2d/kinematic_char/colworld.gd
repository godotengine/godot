
extends Node2D

#member variables here, example:
#var a=2
#var b="textvar"

func _ready():
	#Initalization here
	pass




func _on_princess_body_enter( body ):
	#the name of this editor-generated callback is unfortunate
	get_node("youwin").show()
