@tool
extends HBoxContainer

signal prop_changed

var prop_name:String
var value:int

func set_options(input:String):
	var options = input.split(",")
	for o in options:
		$values.add_item(o)

func set_name(input:String):
	prop_name = input
	$lable.text = input

func set_value(input:int):
	print("set select ", input)
	value = input
	$values.selected = input

func _on_values_item_selected(index):
	value = index
	emit_signal("prop_changed",prop_name,value)





