extends Control
# ##############################################################################
# This is the decoupled GUI for gut.gd
#
# This is a "generic" interface between a GUI and gut.gd.  It assumes there are
# certain controls with specific names.  It will then interact with those
# controls based on signals emitted from gut.gd in order to give the user
# feedback about the progress of the test run and the results.
#
# Optional controls are marked as such in the _ctrls dictionary.  The names
# of the controls can be found in _populate_ctrls.
# ##############################################################################
var _gut = null

var _ctrls = {
	btn_continue = null,
	path_dir = null,
	path_file = null,
	prog_script = null,
	prog_test = null,
	rtl = null,                 # optional
	rtl_bg = null,              # required if rtl exists
	switch_modes = null,
	time_label = null,
	title = null,
	title_bar = null,
}

var _title_mouse = {
	down = false
}


signal switch_modes()

var _max_position = Vector2(100, 100)

func _ready():
	_populate_ctrls()

	_ctrls.btn_continue.visible = false
	_ctrls.btn_continue.pressed.connect(_on_continue_pressed)
	_ctrls.switch_modes.pressed.connect(_on_switch_modes_pressed)
	_ctrls.title_bar.gui_input.connect(_on_title_bar_input)

	_ctrls.prog_script.value = 0
	_ctrls.prog_test.value = 0
	_ctrls.path_dir.text = ''
	_ctrls.path_file.text = ''
	_ctrls.time_label.text = ''

	_max_position = get_display_size() - Vector2(30, _ctrls.title_bar.size.y)


func _process(_delta):
	if(_gut != null and _gut.is_running()):
		set_elapsed_time(_gut.get_elapsed_time())


# ------------------
# Private
# ------------------
func get_display_size():
	return get_viewport().get_visible_rect().size


func _populate_ctrls():
	# Brute force, but flexible.  This allows for all the controls to exist
	# anywhere, and as long as they all have the right name, they will be
	# found.
	_ctrls.btn_continue = _get_first_child_named('Continue', self)
	_ctrls.path_dir = _get_first_child_named('Path', self)
	_ctrls.path_file = _get_first_child_named('File', self)
	_ctrls.prog_script = _get_first_child_named('ProgressScript', self)
	_ctrls.prog_test = _get_first_child_named('ProgressTest', self)
	_ctrls.rtl = _get_first_child_named('TestOutput', self)
	_ctrls.rtl_bg = _get_first_child_named('OutputBG', self)
	_ctrls.switch_modes = _get_first_child_named("SwitchModes", self)
	_ctrls.time_label = _get_first_child_named('TimeLabel', self)
	_ctrls.title = _get_first_child_named("Title", self)
	_ctrls.title_bar = _get_first_child_named("TitleBar", self)


func _get_first_child_named(obj_name, parent_obj):
	if(parent_obj == null):
		return null

	var kids = parent_obj.get_children()
	var index = 0
	var to_return = null

	while(index < kids.size() and to_return == null):
		if(str(kids[index]).find(str(obj_name, ':')) != -1):
			to_return = kids[index]
		else:
			to_return = _get_first_child_named(obj_name, kids[index])
			if(to_return == null):
				index += 1

	return to_return



# ------------------
# Events
# ------------------
func _on_title_bar_input(event : InputEvent):
	if(event is InputEventMouseMotion):
		if(_title_mouse.down):
			position += event.relative
			position.x = clamp(position.x, 0, _max_position.x)
			position.y = clamp(position.y, 0, _max_position.y)
	elif(event is InputEventMouseButton):
		if(event.button_index == MOUSE_BUTTON_LEFT):
			_title_mouse.down = event.pressed


func _on_continue_pressed():
	_gut.end_teardown_pause()


func _on_gut_start_run():
	if(_ctrls.rtl != null):
		_ctrls.rtl.clear()
	set_num_scripts(_gut.get_test_collector().scripts.size())


func _on_gut_end_run():
	_ctrls.prog_test.value = _ctrls.prog_test.max_value
	_ctrls.prog_script.value = _ctrls.prog_script.max_value


func _on_gut_start_script(script_obj):
	next_script(script_obj.get_full_name(), script_obj.tests.size())


func _on_gut_end_script():
	pass


func _on_gut_start_test(test_name):
	next_test(test_name)


func _on_gut_end_test():
	pass


func _on_gut_start_pause():
	pause_before_teardown()


func _on_gut_end_pause():
	_ctrls.btn_continue.visible = false


func _on_switch_modes_pressed():
	switch_modes.emit()

# ------------------
# Public
# ------------------
func set_num_scripts(val):
	_ctrls.prog_script.value = 0
	_ctrls.prog_script.max_value = val


func next_script(path, num_tests):
	_ctrls.prog_script.value += 1
	_ctrls.prog_test.value = 0
	_ctrls.prog_test.max_value = num_tests

	_ctrls.path_dir.text = path.get_base_dir()
	_ctrls.path_file.text = path.get_file()


func next_test(__test_name):
	_ctrls.prog_test.value += 1


func pause_before_teardown():
	_ctrls.btn_continue.visible = true


func set_gut(g):
	if(_gut == g):
		return
	_gut = g
	g.start_run.connect(_on_gut_start_run)
	g.end_run.connect(_on_gut_end_run)

	g.start_script.connect(_on_gut_start_script)
	g.end_script.connect(_on_gut_end_script)

	g.start_test.connect(_on_gut_start_test)
	g.end_test.connect(_on_gut_end_test)

	g.start_pause_before_teardown.connect(_on_gut_start_pause)
	g.end_pause_before_teardown.connect(_on_gut_end_pause)

func get_gut():
	return _gut

func get_textbox():
	return _ctrls.rtl

func set_elapsed_time(t):
	_ctrls.time_label.text = str("%6.1f" % t, 's')


func set_bg_color(c):
	_ctrls.rtl_bg.color = c


func set_title(text):
	_ctrls.title.text = text


func to_top_left():
	self.position = Vector2(5, 5)


func to_bottom_right():
	var win_size = get_display_size()
	self.position = win_size - Vector2(self.size) - Vector2(5, 5)


func align_right():
	var win_size = get_display_size()
	self.position.x = win_size.x - self.size.x -5
	self.position.y = 5
	self.size.y = win_size.y - 10
