@tool
extends BoxContainer
signal value_changed
@onready var slider = $VSlider
@onready var textbox = $LineEdit


func _ready():
	slider.value_changed.connect(update_value)
	textbox.text_submitted.connect(update_value)
	
func update_value(new_value):
	var changed = false
	new_value = clamp(float(new_value), slider.min_value, slider.max_value)
	if slider.value != float(new_value): 		
		textbox.text = str(new_value) #to ensure that it is clamped
		slider.value = new_value
		changed = true
	if textbox.text != str(new_value): 
		textbox.text = str(new_value)
		changed = true
	if changed:
		value_changed.emit(new_value)
