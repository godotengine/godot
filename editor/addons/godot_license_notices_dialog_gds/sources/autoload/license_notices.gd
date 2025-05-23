extends CanvasLayer

var shortcut_action_name = "ui_toggle_licenses_dialog";

func _ready() -> void:
	visible = false
	if !InputMap.has_action(shortcut_action_name):
		push_error("Shortcut action \"", shortcut_action_name, "\" for license notices dialog is not defined in project settings!")
		set_process_input(false)


func _on_close() -> void:
	visible = false


func _input(event: InputEvent) -> void:
	if shortcut_action_name.is_empty():
		return

	if event.is_action_pressed(shortcut_action_name):
		visible = true


func _unhandled_key_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel", false, true):
		_on_close()
		get_viewport().set_input_as_handled()
