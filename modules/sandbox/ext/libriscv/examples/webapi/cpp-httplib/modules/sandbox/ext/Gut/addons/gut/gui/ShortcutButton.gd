@tool
extends Control


@onready var _ctrls = {
	shortcut_label = $Layout/lblShortcut,
	set_button = $Layout/SetButton,
	save_button = $Layout/SaveButton,
	cancel_button = $Layout/CancelButton,
	clear_button = $Layout/ClearButton
}

signal changed
signal start_edit
signal end_edit

const NO_SHORTCUT = '<None>'

var _source_event = InputEventKey.new()
var _pre_edit_event = null
var _key_disp = NO_SHORTCUT
var _editing = false

var _modifier_keys = [KEY_ALT, KEY_CTRL, KEY_META, KEY_SHIFT]

# Called when the node enters the scene tree for the first time.
func _ready():
	set_process_unhandled_key_input(false)


func _display_shortcut():
	if(_key_disp == ''):
		_key_disp = NO_SHORTCUT
	_ctrls.shortcut_label.text = _key_disp


func _is_shift_only_modifier():
	return _source_event.shift_pressed and \
		!(_source_event.alt_pressed or \
			_source_event.ctrl_pressed or \
			_source_event.meta_pressed) \
		and !_is_modifier(_source_event.keycode)


func _has_modifier(event):
	return event.alt_pressed or event.ctrl_pressed or \
		event.meta_pressed or event.shift_pressed


func _is_modifier(keycode):
	return _modifier_keys.has(keycode)


func _edit_mode(should):
	_editing = should
	set_process_unhandled_key_input(should)
	_ctrls.set_button.visible = !should
	_ctrls.save_button.visible = should
	_ctrls.save_button.disabled = should
	_ctrls.cancel_button.visible = should
	_ctrls.clear_button.visible = !should

	if(should and to_s() == ''):
		_ctrls.shortcut_label.text = 'press buttons'
	else:
		_ctrls.shortcut_label.text = to_s()

	if(should):
		emit_signal("start_edit")
	else:
		emit_signal("end_edit")

# ---------------
# Events
# ---------------
func _unhandled_key_input(event):
	if(event is InputEventKey):
		if(event.pressed):
			if(_has_modifier(event) and !_is_modifier(event.get_keycode_with_modifiers())):
				_source_event = event
				_key_disp = OS.get_keycode_string(event.get_keycode_with_modifiers())
			else:
				_source_event = InputEventKey.new()
				_key_disp = NO_SHORTCUT
			_display_shortcut()
			_ctrls.save_button.disabled = !is_valid()


func _on_SetButton_pressed():
	_pre_edit_event = _source_event.duplicate(true)
	_edit_mode(true)


func _on_SaveButton_pressed():
	_edit_mode(false)
	_pre_edit_event = null
	emit_signal('changed')


func _on_CancelButton_pressed():
	cancel()


func _on_ClearButton_pressed():
	clear_shortcut()

# ---------------
# Public
# ---------------
func to_s():
	return OS.get_keycode_string(_source_event.get_keycode_with_modifiers())


func is_valid():
	return _has_modifier(_source_event) and !_is_shift_only_modifier()


func get_shortcut():
	var to_return = Shortcut.new()
	to_return.events.append(_source_event)
	return to_return


func set_shortcut(sc):
	if(sc == null or sc.events == null || sc.events.size() <= 0):
		clear_shortcut()
	else:
		_source_event = sc.events[0]
		_key_disp = to_s()
		_display_shortcut()


func clear_shortcut():
	_source_event = InputEventKey.new()
	_key_disp = NO_SHORTCUT
	_display_shortcut()


func disable_set(should):
	_ctrls.set_button.disabled = should


func disable_clear(should):
	_ctrls.clear_button.disabled = should
	
	
func cancel():
	if(_editing):
		_edit_mode(false)
		_source_event = _pre_edit_event
		_key_disp = to_s()
		_display_shortcut()
