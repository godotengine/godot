# ##############################################################################
# This class joins together GUT, GUT Gui, GutConfig and is THE way to kick off a
# run of a test suite.
#
# This creates its own instance of gut.gd that it manages.  You can set the
# gut.gd instance if you need to for testing.
#
# Set gut_config to an instance of a configured gut_config.gd instance prior to
# running tests.
#
# This will create a GUI and wire it up and apply gut_config.gd options.
#
# Running tests:  Call run_tests
# ##############################################################################
extends Node2D

const EXIT_OK = 0
const EXIT_ERROR = 1

var Gut = load('res://addons/gut/gut.gd')
var ResultExporter = load('res://addons/gut/result_exporter.gd')
var GutConfig = load('res://addons/gut/gut_config.gd')

var runner_json_path = null
var result_bbcode_path = null
var result_json_path = null

var lgr = GutUtils.get_logger()
var gut_config = null

var _hid_gut = null;
# Lazy loaded gut instance.  Settable for testing purposes.
var gut = _hid_gut :
	get:
		if(_hid_gut == null):
			_hid_gut = Gut.new()
		return _hid_gut
	set(val):
		_hid_gut = val

var _wrote_results = false
var _ran_from_editor = false

@onready var _gut_layer = $GutLayer
@onready var _gui = $GutLayer/GutScene


func _ready():
	GutUtils.WarningsManager.apply_warnings_dictionary(
		GutUtils.warnings_at_start)


func _exit_tree():
	if(!_wrote_results and _ran_from_editor):
		_write_results_for_gut_panel()


func _setup_gui(show_gui):
	if(show_gui):
		_gui.gut = gut
		var printer = gut.logger.get_printer('gui')
		printer.set_textbox(_gui.get_textbox())
	else:
		gut.logger.disable_printer('gui', true)
		_gui.visible = false

	var opts = gut_config.options
	_gui.set_font_size(opts.font_size)
	_gui.set_font(opts.font_name)
	if(opts.font_color != null and opts.font_color.is_valid_html_color()):
		_gui.set_default_font_color(Color(opts.font_color))
	if(opts.background_color != null and opts.background_color.is_valid_html_color()):
		_gui.set_background_color(Color(opts.background_color))

	_gui.set_opacity(min(1.0, float(opts.opacity) / 100))
	_gui.use_compact_mode(opts.compact_mode)


func _write_results_for_gut_panel():
	var content = _gui.get_textbox().get_parsed_text() #_gut.logger.get_gui_bbcode()
	var f = FileAccess.open(result_bbcode_path, FileAccess.WRITE)
	if(f != null):
		f.store_string(content)
		f = null # closes file
	else:
		push_error('Could not save bbcode, result = ', FileAccess.get_open_error())

	var exporter = ResultExporter.new()
	# TODO this should be checked and _wrote_results should maybe not be set, or
	# maybe we do not care.  Whichever, it should be clear.
	var _f_result = exporter.write_json_file(gut, result_json_path)
	_wrote_results = true


func _handle_quit(should_exit, should_exit_on_success, override_exit_code=EXIT_OK):
	var quitting_time = should_exit or \
		(should_exit_on_success and gut.get_fail_count() == 0)

	if(!quitting_time):
		if(should_exit_on_success):
			lgr.log("There are failing tests, exit manually.")
		_gui.use_compact_mode(false)
		return

	# For some reason, tests fail asserting that quit was called with 0 if we
	# do not do this, but everything is defaulted so I don't know why it gets
	# null.
	var exit_code = GutUtils.nvl(override_exit_code, EXIT_OK)

	if(gut.get_fail_count() > 0):
		exit_code = EXIT_ERROR

	# Overwrite the exit code with the post_script's exit code if it is set
	var post_hook_inst = gut.get_post_run_script_instance()
	if(post_hook_inst != null and post_hook_inst.get_exit_code() != null):
		exit_code = post_hook_inst.get_exit_code()

	quit(exit_code)


func _end_run(override_exit_code=EXIT_OK):
	if(_ran_from_editor):
		_write_results_for_gut_panel()

	_handle_quit(gut_config.options.should_exit,
		gut_config.options.should_exit_on_success,
		override_exit_code)


# -------------
# Events
# -------------
func _on_tests_finished():
	_end_run()


# -------------
# Public
# -------------
# For internal use only, but still public.  Consider it "protected" and you
# don't have my permission to call this, unless "you" is "me".
func run_from_editor():
	_ran_from_editor = true
	var GutEditorGlobals = load('res://addons/gut/gui/editor_globals.gd')
	runner_json_path = GutUtils.nvl(runner_json_path, GutEditorGlobals.editor_run_gut_config_path)
	result_bbcode_path = GutUtils.nvl(result_bbcode_path, GutEditorGlobals.editor_run_bbcode_results_path)
	result_json_path = GutUtils.nvl(result_json_path, GutEditorGlobals.editor_run_json_results_path)

	if(gut_config == null):
		gut_config = GutConfig.new()
		gut_config.load_options(runner_json_path)

	call_deferred('run_tests')


func run_tests(show_gui=true):
	_setup_gui(show_gui)

	if(gut_config.options.dirs.size() + gut_config.options.tests.size() == 0):
		var err_text = "You do not have any directories configured, so GUT doesn't know where to find the tests.  Tell GUT where to find the tests and GUT shall run the tests."
		lgr.error(err_text)
		push_error(err_text)
		_end_run(EXIT_ERROR)
		return

	var install_check_text = GutUtils.make_install_check_text()
	if(install_check_text != GutUtils.INSTALL_OK_TEXT):
		print("\n\n", GutUtils.version_numbers.get_version_text())
		lgr.error(install_check_text)
		push_error(install_check_text)
		_end_run(EXIT_ERROR)
		return

	gut.add_children_to = self
	if(gut.get_parent() == null):
		if(gut_config.options.gut_on_top):
			_gut_layer.add_child(gut)
		else:
			add_child(gut)

	gut.end_run.connect(_on_tests_finished)

	gut_config.apply_options(gut)
	var run_rest_of_scripts = gut_config.options.unit_test_name == ''

	gut.test_scripts(run_rest_of_scripts)


func set_gut_config(which):
	gut_config = which


# for backwards compatibility
func get_gut():
	return gut


func quit(exit_code):
	# Sometimes quitting takes a few seconds.  This gives some indicator
	# of what is going on.
	_gui.set_title("Exiting")
	await get_tree().process_frame

	lgr.info(str('Exiting with code ', exit_code))
	get_tree().quit(exit_code)




# ##############################################################################
# The MIT License (MIT)
# =====================
#
# Copyright (c) 2025 Tom "Butch" Wesley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ##############################################################################
