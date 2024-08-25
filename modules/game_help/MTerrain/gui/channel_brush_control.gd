@tool
extends HBoxContainer

@export var is_bit = false
@export var force_int = false
var slider:Slider
var button:CheckBox
var text:LineEdit
func _ready():
	slider = find_child("HSlider")
	button = find_child("CheckBox")
	text = find_child("LineEdit")
	
	if not is_bit:
		button.toggled.connect(toggle)
		slider.editable = false
		text.editable = false
		
	slider.value_changed.connect(value_changed)	
	text.text = "0"
	text.text_changed.connect(text_changed)	
	
	
func value_changed(new_value):	
	text.text = str(new_value)
	if slider.value != new_value:
		slider.value = new_value

func text_changed(new_text):	
	var caret_position = text.caret_column
	if not force_int:
		if new_text.to_float() > slider.min_value and new_text.to_float() < slider.max_value: 			
			slider.value = new_text.to_float()
			if text.text != str(slider.value):
				text.text = str(slider.value)
		else:			
			slider.value = clamp(new_text.to_float(), slider.min_value, slider.max_value)
			text.text = str(slider.value)
		
	else:		
		if text.text.to_int() == new_text.to_int() and slider.value == new_text.to_int(): return	
		slider.value = new_text.to_int()
		text.text = new_text.to_int()
	text.caret_column = caret_position

func toggle(toggle_on):	
	slider.editable = toggle_on
	text.editable = toggle_on
	if not toggle_on:
		text.text = "0"
		slider.value = 0
		
