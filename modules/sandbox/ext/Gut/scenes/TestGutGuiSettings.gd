extends Node2D
const RUNNER_JSON_PATH = 'user://test_gut_editor_config.json'
var GutConfigGui = load('res://addons/gut/gui/gut_config_gui.gd')

@onready var _ctrls = {
	settings = $ColorRect/ScrollContainer/Settings,
	issues = $Controls/VBox/Issues
}

var _gut_config = load('res://addons/gut/gut_config.gd').new()
var _gut_config_gui = null
var _settings_vbox = null

# Called when the node enters the scene tree for the first time.
func _ready():
	_settings_vbox = _ctrls.settings.duplicate()
	_gut_config.load_options(RUNNER_JSON_PATH)
	_create_options()


func _clear_options():
	if(_gut_config_gui != null):
		_gut_config_gui.clear()


func _display_issues():
	var issues : Array = _gut_config_gui.get_config_issues()
	if(issues.size() > 0):
		_ctrls.issues.text = "\n".join(issues)
	else:
		_ctrls.issues.text = "-- No Issues --"


func _create_options():
	_gut_config_gui = GutConfigGui.new(_ctrls.settings)
	_gut_config_gui.set_options(_gut_config.options)

func save_options():
	_gut_config.options = _gut_config_gui.get_options(_gut_config.options)
	var w_result = _gut_config.write_options(RUNNER_JSON_PATH)
	if(w_result != OK):
		push_error(str('Could not write options to ', RUNNER_JSON_PATH, ': ', w_result))
	else:
		_gut_config_gui.mark_saved()


func _on_save_pressed():
	save_options()
	_display_issues()
	print('saved')


func _on_load_pressed():
	_clear_options()
	await get_tree().create_timer(.5).timeout
	_create_options()
	_display_issues()
	print('loaded')


func _on_get_issues_pressed():
	_display_issues()
