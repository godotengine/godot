@tool
extends Control

signal prop_changed

var prop_name:String
var value:int

@onready var item_list = find_child("ItemList")
func set_options(input:String):
	if not item_list:
		await ready
	var options = input.split(",")
	for o in options:
		item_list.add_item(o)	

func set_name(input:String):
	prop_name = input
	for i in item_list.item_count:
		if item_list.get_item_text(i) == input:
			item_list.select(i)
	#$Button.text = input

func set_value(input:int):
	#print("set select ", input)
	value = input
	item_list.select(input)

func _on_values_item_selected(index):
	value = index
	emit_signal("prop_changed",prop_name,value)
