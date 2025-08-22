extends Node2D

@onready var label = get_node('Label')

func return_hello():
	return 'hello'

func set_label_text(text):
	$Label.set_text(text)

func get_button():
	return $MyPanel/MyButton
