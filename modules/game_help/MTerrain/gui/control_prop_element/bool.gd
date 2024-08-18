@tool
extends BoxContainer

signal prop_changed

var prop_name:String
var value:bool


func set_name(input:String):
	prop_name = input
	$label.text = input

func set_value(input:bool):
	value = input
	$Control/CheckButton.button_pressed = input


func _on_check_button_toggled(button_pressed):
	value = button_pressed
	emit_signal("prop_changed",prop_name,value)
