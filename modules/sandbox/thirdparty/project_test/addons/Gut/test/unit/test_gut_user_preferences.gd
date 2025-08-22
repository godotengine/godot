extends GutTest

var Gup = load("res://addons/gut/gui/gut_user_preferences.gd")
var pref_prefix = 'gut/'

class MockEditorSettings:
	var settings = {}

	func set_setting(n, v):
		settings[n] = v

	func get_setting(n):
		return settings[n]

	func has_setting(n):
		return settings.has(n)

	func erase(n):
		settings.erase(n)


func before_all():
	register_inner_classes(self.get_script())


func test_can_make_one():
	var es = MockEditorSettings.new()
	var gup = Gup.new(es)
	assert_not_null(gup)


var default_settings = ParameterFactory.named_parameters(
	['name', 'value'],[
		['output_font_name', 'CourierPrime'],
		['output_font_size', 30],
		['hide_result_tree', false],
		['hide_output_text', false],
		['hide_settings', false],
		['use_colors', true]
	])
func test_save_sets_values_to_default_when_not_set(p = use_parameters(default_settings)):
	var es = MockEditorSettings.new()
	var gup = Gup.new(es)
	gup.save_it()
	assert_true(es.has_setting(pref_prefix + p.name), 'has ' + p.name)
	if(is_passing()):
		assert_eq(es.get_setting(pref_prefix + p.name), p.value, p.name + ' default')


var non_default_settings = ParameterFactory.named_parameters(
	['name', 'value'],[
		['output_font_name', 'test'],
		['output_font_size', 99],
		['hide_result_tree', true],
		['hide_output_text', true],
		['hide_settings', true],
		['use_colors', false]
	])
func test_all_values_are_loaded_from_settings(p = use_parameters(non_default_settings)):
	var es = MockEditorSettings.new()
	var to_save = Gup.new(es)
	to_save.get(p.name).value = p.value
	to_save.save_it()
	var loaded = Gup.new(es)
	assert_eq(loaded.get(p.name).value, p.value)

var pref_list = [
	'output_font_name',
	'output_font_size',
	'hide_result_tree',
	'hide_output_text',
	'hide_settings',
	'use_colors',
]
func test_all_values_are_erased_by_erase_all(p = use_parameters(pref_list)):
	var es = partial_double(MockEditorSettings).new()
	var prefs = Gup.new(es)
	var erased = prefs.get(p)
	prefs.erase_all()
	assert_called(es, 'erase', [erased._prefstr()])
