@tool 
extends Button

signal brush_size_changed

@export var slider: VSlider
@export var textbox: LineEdit

func _ready():
	slider.value_changed.connect(update_value)
	textbox.text_submitted.connect(update_value)
	
	var panel = get_child(0)
	panel.visible = false
	panel.position.y = -panel.size.y	

func update_value(new_value):
	var changed = false
	if slider.value != float(new_value): 
		slider.value = float(new_value)
		changed = true
	if textbox.text != str(new_value): 
		textbox.text = str(new_value)	
		changed = true
	if changed:
		brush_size_changed.emit(float(new_value))
