@tool
extends HBoxContainer

signal keymap_changed

@onready var label = $Label	
@onready var value = $Button

var waiting_for_input = false

func _ready():
	value.pressed.connect(begin_remap)
	
func begin_remap():
	value.text = "...waiting for input..."
	waiting_for_input = true
	
func _input(event):
	if waiting_for_input and event is InputEventKey and event.pressed:		
		if event.keycode in [KEY_SHIFT, KEY_CTRL, KEY_ALT]: 
			return			
		var new_text = ""
		if Input.is_key_pressed(KEY_CTRL):
			new_text += "ctrl "
		if Input.is_key_pressed(KEY_ALT):
			if new_text != "":
				new_text += "+ "				
			new_text += "alt "
		if Input.is_key_pressed(KEY_SHIFT):
			if new_text != "":
				new_text += "+ "							
			new_text += "shift "
		value.text = OS.get_keycode_string(event.keycode)
		keymap_changed.emit(name, event.keycode, Input.is_key_pressed(KEY_CTRL), Input.is_key_pressed(KEY_ALT), Input.is_key_pressed(KEY_SHIFT) )
