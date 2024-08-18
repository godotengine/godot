@tool
extends BoxContainer

signal prop_changed

var prop_name:String
var value:int
var max:float = 1000000000000000000
var min:float = -100000000000000000

func set_max(input):
	$value.max_value = input

func set_min(input):
	$value.min_value = input


func set_name(input:String):
	prop_name = input
	$label.text = input

func set_value(input:float):
	value = input
	$value.value = input
	

func _on_value_value_changed(v):
	value = int(v)
	emit_signal("prop_changed",prop_name,value)
	
