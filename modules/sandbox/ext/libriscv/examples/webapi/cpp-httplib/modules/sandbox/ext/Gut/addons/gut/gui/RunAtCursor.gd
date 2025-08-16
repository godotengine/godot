@tool
extends Control


var ScriptTextEditors = load('res://addons/gut/gui/script_text_editor_controls.gd')

@onready var _ctrls = {
	btn_script = $HBox/BtnRunScript,
	btn_inner = $HBox/BtnRunInnerClass,
	btn_method = $HBox/BtnRunMethod,
	lbl_none = $HBox/LblNoneSelected,
	arrow_1 = $HBox/Arrow1,
	arrow_2 = $HBox/Arrow2
}

var _editors = null
var _cur_editor = null
var _last_line = -1
var _cur_script_path = null
var _last_info = {
	script = null,
	inner_class = null,
	test_method = null
}


signal run_tests(what)


func _ready():
	_ctrls.lbl_none.visible = true
	_ctrls.btn_script.visible = false
	_ctrls.btn_inner.visible = false
	_ctrls.btn_method.visible = false
	_ctrls.arrow_1.visible = false
	_ctrls.arrow_2.visible = false

# ----------------
# Private
# ----------------
func _set_editor(which):
	_last_line = -1
	if(_cur_editor != null and _cur_editor.get_ref()):
		# _cur_editor.get_ref().disconnect('cursor_changed',Callable(self,'_on_cursor_changed'))
		_cur_editor.get_ref().caret_changed.disconnect(_on_cursor_changed)

	if(which != null):
		_cur_editor = weakref(which)
		which.caret_changed.connect(_on_cursor_changed.bind(which))
		# which.connect('cursor_changed',Callable(self,'_on_cursor_changed'),[which])

		_last_line = which.get_caret_line()
		_last_info = _editors.get_line_info()
		_update_buttons(_last_info)


func _update_buttons(info):
	_ctrls.lbl_none.visible = _cur_script_path == null
	_ctrls.btn_script.visible = _cur_script_path != null

	_ctrls.btn_inner.visible = info.inner_class != null
	_ctrls.arrow_1.visible = info.inner_class != null
	_ctrls.btn_inner.text = str(info.inner_class)
	_ctrls.btn_inner.tooltip_text = str("Run all tests in Inner-Test-Class ", info.inner_class)

	_ctrls.btn_method.visible = info.test_method != null
	_ctrls.arrow_2.visible = info.test_method != null
	_ctrls.btn_method.text = str(info.test_method)
	_ctrls.btn_method.tooltip_text = str("Run test ", info.test_method)

	# The button's new size won't take effect until the next frame.
	# This appears to be what was causing the button to not be clickable the
	# first time.
	call_deferred("_update_size")

func _update_size():
	custom_minimum_size.x = _ctrls.btn_method.size.x + _ctrls.btn_method.position.x

# ----------------
# Events
# ----------------
func _on_cursor_changed(which):
	if(which.get_caret_line() != _last_line):
		_last_line = which.get_caret_line()
		_last_info = _editors.get_line_info()
		_update_buttons(_last_info)


func _on_BtnRunScript_pressed():
	var info = _last_info.duplicate()
	info.script = _cur_script_path.get_file()
	info.inner_class = null
	info.test_method = null
	emit_signal("run_tests", info)


func _on_BtnRunInnerClass_pressed():
	var info = _last_info.duplicate()
	info.script = _cur_script_path.get_file()
	info.test_method = null
	emit_signal("run_tests", info)


func _on_BtnRunMethod_pressed():
	var info = _last_info.duplicate()
	info.script = _cur_script_path.get_file()
	emit_signal("run_tests", info)


# ----------------
# Public
# ----------------
func set_script_text_editors(value):
	_editors = value


func activate_for_script(path):
	_ctrls.btn_script.visible = true
	_ctrls.btn_script.text = path.get_file()
	_ctrls.btn_script.tooltip_text = str("Run all tests in script ", path)
	_cur_script_path = path
	_editors.refresh()
	# We have to wait a beat for the visibility to change on
	# the editors, otherwise we always get the first one.
	await get_tree().process_frame
	_set_editor(_editors.get_current_text_edit())


func get_script_button():
	return _ctrls.btn_script


func get_inner_button():
	return _ctrls.btn_inner


func get_test_button():
	return _ctrls.btn_method


# not used, thought was configurable but it's just the script prefix
func set_method_prefix(value):
	_editors.set_method_prefix(value)


# not used, thought was configurable but it's just the script prefix
func set_inner_class_prefix(value):
	_editors.set_inner_class_prefix(value)


# Mashed this function in here b/c it has _editors.  Probably should be
# somewhere else (possibly in script_text_editor_controls).
func search_current_editor_for_text(txt):
	var te = _editors.get_current_text_edit()
	var result = te.search(txt, 0, 0, 0)
	var to_return = -1

	return to_return
