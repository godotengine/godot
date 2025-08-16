var PanelControls = load("res://addons/gut/gui/panel_controls.gd")
var GutConfig = load('res://addons/gut/gut_config.gd')

const DIRS_TO_LIST = 6

var _base_container = null
# All the various PanelControls indexed by thier gut_config keys.
var _cfg_ctrls = {}

# specific titles that we need to do stuff with
var _titles = {
	dirs = null
}
# All titles so we can free them when we want.
var _all_titles = []


func _init(cont):
	_base_container = cont


func _add_title(text):
	var row = PanelControls.BaseGutPanelControl.new(text, text)
	_base_container.add_child(row)
	row.connect('draw', _on_title_cell_draw.bind(row))
	_all_titles.append(row)
	return row

func _add_ctrl(key, ctrl):
	_cfg_ctrls[key] = ctrl
	_base_container.add_child(ctrl)


func _add_number(key, value, disp_text, v_min, v_max, hint=''):
	var ctrl = PanelControls.NumberControl.new(disp_text, value, v_min, v_max, hint)
	_add_ctrl(key, ctrl)
	return ctrl


func _add_select(key, value, values, disp_text, hint=''):
	var ctrl = PanelControls.SelectControl.new(disp_text, value, values, hint)
	_add_ctrl(key, ctrl)
	return ctrl


func _add_value(key, value, disp_text, hint=''):
	var ctrl = PanelControls.StringControl.new(disp_text, value, hint)
	_add_ctrl(key, ctrl)
	return ctrl


func _add_boolean(key, value, disp_text, hint=''):
	var ctrl = PanelControls.BooleanControl.new(disp_text, value, hint)
	_add_ctrl(key, ctrl)
	return ctrl


func _add_directory(key, value, disp_text, hint=''):
	var ctrl = PanelControls.DirectoryControl.new(disp_text, value, hint)
	_add_ctrl(key, ctrl)
	ctrl.dialog.title = disp_text
	return ctrl


func _add_file(key, value, disp_text, hint=''):
	var ctrl = PanelControls.DirectoryControl.new(disp_text, value, hint)
	_add_ctrl(key, ctrl)
	ctrl.dialog.file_mode = ctrl.dialog.FILE_MODE_OPEN_FILE
	ctrl.dialog.title = disp_text
	return ctrl


func _add_save_file_anywhere(key, value, disp_text, hint=''):
	var ctrl = PanelControls.DirectoryControl.new(disp_text, value, hint)
	_add_ctrl(key, ctrl)
	ctrl.dialog.file_mode = ctrl.dialog.FILE_MODE_SAVE_FILE
	ctrl.dialog.access = ctrl.dialog.ACCESS_FILESYSTEM
	ctrl.dialog.title = disp_text
	return ctrl


func _add_color(key, value, disp_text, hint=''):
	var ctrl = PanelControls.ColorControl.new(disp_text, value, hint)
	_add_ctrl(key, ctrl)
	return ctrl


func _add_save_load():
	var ctrl = PanelControls.SaveLoadControl.new('Config', '', '')
	_base_container.add_child(ctrl)

	ctrl.save_path_chosen.connect(_on_save_path_chosen)
	ctrl.load_path_chosen.connect(_on_load_path_chosen)

	_cfg_ctrls['save_load'] = ctrl
	return ctrl

# ------------------
# Events
# ------------------
func _on_title_cell_draw(which):
	which.draw_rect(Rect2(Vector2(0, 0), which.size), Color(0, 0, 0, .15))


func _on_save_path_chosen(path):
	save_file(path)


func _on_load_path_chosen(path):
	load_file.bind(path).call_deferred()

# ------------------
# Public
# ------------------
func get_config_issues():
	var to_return = []
	var has_directory = false

	for i in range(DIRS_TO_LIST):
		var key = str('directory_', i)
		var path = _cfg_ctrls[key].value
		if(path != null and path != ''):
			has_directory = true
			if(!DirAccess.dir_exists_absolute(path)):
				_cfg_ctrls[key].mark_invalid(true)
				to_return.append(str('Test directory ', path, ' does not exist.'))
			else:
				_cfg_ctrls[key].mark_invalid(false)
		else:
			_cfg_ctrls[key].mark_invalid(false)

	if(!has_directory):
		to_return.append('You do not have any directories set.')
		_titles.dirs.mark_invalid(true)
	else:
		_titles.dirs.mark_invalid(false)

	if(!_cfg_ctrls.suffix.value.ends_with('.gd')):
		_cfg_ctrls.suffix.mark_invalid(true)
		to_return.append("Script suffix must end in '.gd'")
	else:
		_cfg_ctrls.suffix.mark_invalid(false)

	return to_return


func clear():
	for key in _cfg_ctrls:
		_cfg_ctrls[key].free()

	_cfg_ctrls.clear()

	for entry in _all_titles:
		entry.free()

	_all_titles.clear()


func save_file(path):
	var gcfg = GutConfig.new()
	gcfg.options = get_options({})
	gcfg.save_file(path)



func load_file(path):
	var gcfg = GutConfig.new()
	gcfg.load_options(path)
	clear()
	set_options(gcfg.options)


# --------------
# SUPER dumb but VERY fun hack to hide settings.  The various _add methods will
# return what they add.  If you want to hide it, just assign the result to this.
# YES, I could have just put .visible at the end, but I didn't think of that
# until just now, and this was fun, non-permanent and the .visible at the end
# isn't as obvious as hide_this =
#
# Also, we can't just skip adding the controls because other things are looking
# for them and things start to blow up if you don't add them.
var hide_this = null :
	set(val):
		val.visible = false

# --------------

func set_options(opts):
	var options = opts.duplicate()

	# _add_title('Save/Load')
	_add_save_load()

	_add_title("Settings")
	_add_number("log_level", options.log_level, "Log Level", 0, 3,
		"Detail level for log messages.\n" + \
		"\t0: Errors and failures only.\n" + \
		"\t1: Adds all test names + warnings + info\n" + \
		"\t2: Shows all asserts\n" + \
		"\t3: Adds more stuff probably, maybe not.")
	_add_boolean('ignore_pause', options.ignore_pause, 'Ignore Pause',
		"Ignore calls to pause_before_teardown")
	_add_boolean('hide_orphans', options.hide_orphans, 'Hide Orphans',
		'Do not display orphan counts in output.')
	_add_boolean('should_exit', options.should_exit, 'Exit on Finish',
		"Exit when tests finished.")
	_add_boolean('should_exit_on_success', options.should_exit_on_success, 'Exit on Success',
		"Exit if there are no failures.  Does nothing if 'Exit on Finish' is enabled.")
	_add_select('double_strategy', 'Script Only', ['Include Native', 'Script Only'], 'Double Strategy',
		'"Include Native" will include native methods in Doubles.  "Script Only" will not.  ' + "\n" + \
		'The native method override warning is disabled when creating Doubles.' + "\n" + \
		'This is the default, you can override this at the script level or when creating doubles.')
	_cfg_ctrls.double_strategy.value = GutUtils.get_enum_value(
		options.double_strategy, GutUtils.DOUBLE_STRATEGY, GutUtils.DOUBLE_STRATEGY.SCRIPT_ONLY)
	_add_boolean('errors_cause_failure', !options.errors_do_not_cause_failure, 'Errors cause failures.',
		"When GUT generates an error (not an engine error) it causes tests to fail.")


	_add_title('Runner Appearance')
	hide_this = _add_boolean("gut_on_top", options.gut_on_top, "On Top",
		"The GUT Runner appears above children added during tests.")
	_add_number('opacity', options.opacity, 'Opacity', 0, 100,
		"The opacity of GUT when tests are running.")
	hide_this = _add_boolean('should_maximize', options.should_maximize, 'Maximize',
		"Maximize GUT when tests are being run.")
	_add_boolean('compact_mode', options.compact_mode, 'Compact Mode',
		'The runner will be in compact mode.  This overrides Maximize.')
	_add_select('font_name', options.font_name, GutUtils.avail_fonts, 'Font',
		"The font to use for text output in the Gut Runner.")
	_add_number('font_size', options.font_size, 'Font Size', 5, 100,
		"The font size for text output in the Gut Runner.")
	hide_this = _add_color('font_color', options.font_color, 'Font Color',
		"The font color for text output in the Gut Runner.")
	_add_color('background_color', options.background_color, 'Background Color',
		"The background color for text output in the Gut Runner.")
	_add_boolean('disable_colors', options.disable_colors, 'Disable Formatting',
		'Disable formatting and colors used in the Runner.  Does not affect panel output.')


	_titles.dirs = _add_title('Test Directories')
	_add_boolean('include_subdirs', options.include_subdirs, 'Include Subdirs',
		"Include subdirectories of the directories configured below.")

	var dirs_to_load = options.configured_dirs
	if(options.dirs.size() > dirs_to_load.size()):
		dirs_to_load = options.dirs

	for i in range(DIRS_TO_LIST):
		var value = ''
		if(dirs_to_load.size() > i):
			value = dirs_to_load[i]

		var test_dir = _add_directory(str('directory_', i), value, str(i))
		test_dir.enabled_button.visible = true
		test_dir.enabled_button.button_pressed = options.dirs.has(value)


	_add_title("XML Output")
	_add_save_file_anywhere("junit_xml_file", options.junit_xml_file, "Output Path",
		"Path3D and filename where GUT should create a JUnit compliant XML file.  " +
		"This file will contain the results of the last test run.  To avoid " +
		"overriding the file use Include Timestamp.")
	_add_boolean("junit_xml_timestamp", options.junit_xml_timestamp, "Include Timestamp",
		"Include a timestamp in the filename so that each run gets its own xml file.")


	_add_title('Hooks')
	_add_file('pre_run_script', options.pre_run_script, 'Pre-Run Hook',
		'This script will be run by GUT before any tests are run.')
	_add_file('post_run_script', options.post_run_script, 'Post-Run Hook',
		'This script will be run by GUT after all tests are run.')


	_add_title('Misc')
	_add_value('prefix', options.prefix, 'Script Prefix',
		"The filename prefix for all test scripts.")
	_add_value('suffix', options.suffix, 'Script Suffix',
		"Script suffix, including .gd extension.  For example '_foo.gd'.")
	_add_number('paint_after', options.paint_after, 'Paint After', 0.0, 1.0,
		"How long GUT will wait before pausing for 1 frame to paint the screen.  0 is never.")

	# since _add_number doesn't set step property, it will default to a step of
	# 1 and cast values to int.  Give it a .5 step and re-set the value.
	_cfg_ctrls.paint_after.value_ctrl.step = .05
	_cfg_ctrls.paint_after.value = options.paint_after



func get_options(base_opts):
	var to_return = base_opts.duplicate()

	# Settings
	to_return.log_level = _cfg_ctrls.log_level.value
	to_return.ignore_pause = _cfg_ctrls.ignore_pause.value
	to_return.hide_orphans = _cfg_ctrls.hide_orphans.value
	to_return.should_exit = _cfg_ctrls.should_exit.value
	to_return.should_exit_on_success = _cfg_ctrls.should_exit_on_success.value
	to_return.double_strategy = _cfg_ctrls.double_strategy.value
	to_return.errors_do_not_cause_failure = !_cfg_ctrls.errors_cause_failure.value


	# Runner Appearance
	to_return.font_name = _cfg_ctrls.font_name.text
	to_return.font_size = _cfg_ctrls.font_size.value
	to_return.should_maximize = _cfg_ctrls.should_maximize.value
	to_return.compact_mode = _cfg_ctrls.compact_mode.value
	to_return.opacity = _cfg_ctrls.opacity.value
	to_return.background_color = _cfg_ctrls.background_color.value.to_html()
	to_return.font_color = _cfg_ctrls.font_color.value.to_html()
	to_return.disable_colors = _cfg_ctrls.disable_colors.value
	to_return.gut_on_top = _cfg_ctrls.gut_on_top.value
	to_return.paint_after = _cfg_ctrls.paint_after.value


	# Directories
	to_return.include_subdirs = _cfg_ctrls.include_subdirs.value
	var dirs = []
	var configured_dirs = []
	for i in range(DIRS_TO_LIST):
		var key = str('directory_', i)
		var ctrl = _cfg_ctrls[key]
		if(ctrl.value != '' and ctrl.value != null):
			configured_dirs.append(ctrl.value)
			if(ctrl.enabled_button.button_pressed):
				dirs.append(ctrl.value)
	to_return.dirs = dirs
	to_return.configured_dirs = configured_dirs

	# XML Output
	to_return.junit_xml_file = _cfg_ctrls.junit_xml_file.value
	to_return.junit_xml_timestamp = _cfg_ctrls.junit_xml_timestamp.value

	# Hooks
	to_return.pre_run_script = _cfg_ctrls.pre_run_script.value
	to_return.post_run_script = _cfg_ctrls.post_run_script.value

	# Misc
	to_return.prefix = _cfg_ctrls.prefix.value
	to_return.suffix = _cfg_ctrls.suffix.value

	return to_return


func mark_saved():
	for key in _cfg_ctrls:
		_cfg_ctrls[key].mark_unsaved(false)
