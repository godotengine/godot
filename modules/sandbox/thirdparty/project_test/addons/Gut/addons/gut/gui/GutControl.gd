@tool
extends Control

const RUNNER_JSON_PATH = 'res://.gut_editor_config.json'

var GutConfig = load('res://addons/gut/gut_config.gd')
var GutRunnerScene = load('res://addons/gut/gui/GutRunner.tscn')
var GutConfigGui = load('res://addons/gut/gui/gut_config_gui.gd')

var _config = GutConfig.new()
var _config_gui = null
var _gut_runner = GutRunnerScene.instantiate()
var _has_connected = false
var _tree_root : TreeItem = null

var _script_icon = load('res://addons/gut/images/Script.svg')
var _folder_icon = load('res://addons/gut/images/Folder.svg')

var _tree_scripts = {}
var _tree_directories = {}

const TREE_SCRIPT = 'Script'
const TREE_DIR = 'Directory'

@onready var _ctrls = {
	run_tests_button = $VBox/Buttons/RunTests,
	run_selected = $VBox/Buttons/RunSelected,
	test_tree = $VBox/Tabs/Tests,
	settings_vbox = $VBox/Tabs/SettingsScroll/Settings,
	tabs = $VBox/Tabs,
	bg = $Bg
}

@export var bg_color : Color = Color(.36, .36, .36) :
	get: return bg_color
	set(val):
		bg_color = val
		if(is_inside_tree()):
			$Bg.color = bg_color


func _ready():
	if Engine.is_editor_hint():
		return

	$Bg.color = bg_color
	_ctrls.tabs.set_tab_title(0, 'Tests')
	_ctrls.tabs.set_tab_title(1, 'Settings')

	_config_gui = GutConfigGui.new(_ctrls.settings_vbox)

	_ctrls.test_tree.hide_root = true
	add_child(_gut_runner)

	# TODO This might not need to be called deferred after changing GutUtils to
	# an all static class.
	call_deferred('_post_ready')


func _draw():
	if Engine.is_editor_hint():
		return

	var gut = _gut_runner.get_gut()
	if(!gut.is_running()):
		var r = Rect2(Vector2(0, 0), get_rect().size)
		draw_rect(r, Color.BLACK, false, 2)


func _post_ready():
	var gut = _gut_runner.get_gut()
	gut.start_run.connect(_on_gut_run_started)
	gut.end_run.connect(_on_gut_run_ended)
	_refresh_tree_and_settings()


func _set_meta_for_script_tree_item(item, script, test=null):
	var meta = {
		type = TREE_SCRIPT,
		script = script.path,
		inner_class = script.inner_class_name,
		test = ''
	}

	if(test != null):
		meta.test = test.name

	item.set_metadata(0, meta)


func _set_meta_for_directory_tree_item(item, path, temp_item):
	var meta = {
		type = TREE_DIR,
		path = path,
		temp_item = temp_item
	}
	item.set_metadata(0, meta)


func _get_script_tree_item(script, parent_item):
	if(!_tree_scripts.has(script.path)):
		var item = _ctrls.test_tree.create_item(parent_item)
		item.set_text(0, script.path.get_file())
		item.set_icon(0, _script_icon)
		_tree_scripts[script.path] = item
		_set_meta_for_script_tree_item(item, script)

	return _tree_scripts[script.path]


func _get_directory_tree_item(path):
	var parent = _tree_root
	if(!_tree_directories.has(path)):

		var item : TreeItem = null
		if(parent != _tree_root):
			item = parent.create_child(0)
		else:
			item = parent.create_child()

		_tree_directories[path] = item
		item.collapsed = false
		item.set_text(0, path)
		item.set_icon(0, _folder_icon)
		item.set_icon_modulate(0, Color.ROYAL_BLUE)
		# temp_item is used in calls with move_before since you must use
		# move_before or move_after to reparent tree items. This ensures that
		# there is an item on all directories.  These are deleted later.
		var temp_item = item.create_child()
		temp_item.set_text(0, '<temp>')

		_set_meta_for_directory_tree_item(item, path, temp_item)

	return _tree_directories[path]


func _find_dir_item_to_move_before(path):
	var max_matching_len = 0
	var best_parent = null

	# Go through all the directory items finding the one that has the longest
	# path that contains our path.
	for key in _tree_directories.keys():
		if(path != key and path.begins_with(key) and key.length() > max_matching_len):
				max_matching_len = key.length()
				best_parent = _tree_directories[key]

	var to_return = null
	if(best_parent != null):
		to_return = best_parent.get_metadata(0).temp_item
	return to_return


func _reorder_dir_items():
	var the_keys = _tree_directories.keys()
	the_keys.sort()
	for key in _tree_directories.keys():
		var to_move = _tree_directories[key]
		to_move.collapsed = false
		var move_before = _find_dir_item_to_move_before(key)
		if(move_before != null):
			to_move.move_before(move_before)
			var new_text = key.substr(move_before.get_parent().get_metadata(0).path.length())
			to_move.set_text(0, new_text)


func _remove_dir_temp_items():
	for key in _tree_directories.keys():
		var item = _tree_directories[key].get_metadata(0).temp_item
		_tree_directories[key].remove_child(item)


func _add_dir_and_script_tree_items():
	var tree : Tree = _ctrls.test_tree
	tree.clear()
	_tree_root = _ctrls.test_tree.create_item()

	var scripts = _gut_runner.get_gut().get_test_collector().scripts
	for script in scripts:
		var dir_item = _get_directory_tree_item(script.path.get_base_dir())
		var item = _get_script_tree_item(script, dir_item)

		if(script.inner_class_name != ''):
			var inner_item = tree.create_item(item)
			inner_item.set_text(0, script.inner_class_name)
			_set_meta_for_script_tree_item(inner_item, script)
			item = inner_item

		for test in script.tests:
			var test_item = tree.create_item(item)
			test_item.set_text(0, test.name)
			_set_meta_for_script_tree_item(test_item, script, test)


func _populate_tree():
	_add_dir_and_script_tree_items()
	_tree_root.set_collapsed_recursive(true)
	_tree_root.set_collapsed(false)
	_reorder_dir_items()
	_remove_dir_temp_items()


func _refresh_tree_and_settings():
	_config.apply_options(_gut_runner.get_gut())
	_gut_runner.set_gut_config(_config)
	_populate_tree()

# ---------------------------
# Events
# ---------------------------
func _on_gut_run_started():
	_ctrls.run_tests_button.disabled = true
	_ctrls.run_selected.visible = false
	_ctrls.tabs.visible = false
	_ctrls.bg.visible = false
	_ctrls.run_tests_button.text = 'Running'
	queue_redraw()


func _on_gut_run_ended():
	_ctrls.run_tests_button.disabled = false
	_ctrls.run_selected.visible = true
	_ctrls.tabs.visible = true
	_ctrls.bg.visible = true
	_ctrls.run_tests_button.text = 'Run All'
	queue_redraw()


func _on_run_tests_pressed():
	run_all()


func _on_run_selected_pressed():
	run_selected()


func _on_tests_item_activated():
	run_selected()

# ---------------------------
# Public
# ---------------------------
func get_gut():
	return _gut_runner.get_gut()


func get_config():
	return _config


func run_all():
	_config.options.selected = ''
	_config.options.inner_class_name = ''
	_config.options.unit_test_name = ''
	run_tests()


func run_tests(options = null):
	if(options == null):
		_config.options = _config_gui.get_options(_config.options)
	else:
		_config.options = options

	_gut_runner.get_gut().get_test_collector().clear()
	_gut_runner.set_gut_config(_config)
	_gut_runner.run_tests()


func run_selected():
	var sel_item = _ctrls.test_tree.get_selected()
	if(sel_item == null):
		return

	var options = _config_gui.get_options(_config.options)
	var meta = sel_item.get_metadata(0)
	if(meta.type == TREE_SCRIPT):
		options.selected = meta.script.get_file()
		options.inner_class_name = meta.inner_class
		options.unit_test_name = meta.test
	elif(meta.type == TREE_DIR):
		options.dirs = [meta.path]
		options.include_subdirectories = true
		options.selected = ''
		options.inner_class_name = ''
		options.unit_test_name = ''

	run_tests(options)


func load_config_file(path):
	_config.load_options(path)
	_config.options.selected = ''
	_config.options.inner_class_name = ''
	_config.options.unit_test_name = ''
	_config_gui.load_file(path)


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
