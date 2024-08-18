@tool
extends BoxContainer

signal prop_changed

var prop_name:String
var value:float
var max:float = 1000000000000000000
var min:float = -100000000000000000

func set_max(input):
	max=input

func set_min(input):
	max=input


func set_name(input:String):
	prop_name = input
	$label.text = input

func set_value(input:float):
	value = input
	$value.text = str(input)

func _on_value_text_submitted(new_text:String):
	if new_text.is_valid_float():
		value = new_text.to_float()
		if value>max:
			value = max
		if value<min:
			value = min
		emit_signal("prop_changed",prop_name,value)
	$value.text = str(value)
	


func _on_value_focus_exited():
	_on_value_text_submitted($value.text)
