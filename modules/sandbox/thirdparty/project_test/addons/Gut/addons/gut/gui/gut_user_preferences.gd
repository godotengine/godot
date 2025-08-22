class GutEditorPref:
	var gut_pref_prefix = 'gut/'
	var pname = '__not_set__'
	var default = null
	var value = '__not_set__'
	var _settings = null

	func _init(n, d, s):
		pname = n
		default = d
		_settings = s
		load_it()

	func _prefstr():
		var to_return = str(gut_pref_prefix, pname)
		return to_return

	func save_it():
		_settings.set_setting(_prefstr(), value)

	func load_it():
		if(_settings.has_setting(_prefstr())):
			value = _settings.get_setting(_prefstr())
		else:
			value = default

	func erase():
		_settings.erase(_prefstr())


const EMPTY = '-- NOT_SET --'

# -- Editor ONLY Settings --
var output_font_name = null
var output_font_size = null
var hide_result_tree = null
var hide_output_text = null
var hide_settings = null
var use_colors = null	# ? might be output panel

# var shortcut_run_all = null
# var shortcut_run_current_script = null
# var shortcut_run_current_inner = null
# var shortcut_run_current_test = null
# var shortcut_panel_button = null


func _init(editor_settings):
	output_font_name = GutEditorPref.new('output_font_name', 'CourierPrime', editor_settings)
	output_font_size = GutEditorPref.new('output_font_size', 30, editor_settings)
	hide_result_tree = GutEditorPref.new('hide_result_tree', false, editor_settings)
	hide_output_text = GutEditorPref.new('hide_output_text', false, editor_settings)
	hide_settings = GutEditorPref.new('hide_settings', false, editor_settings)
	use_colors = GutEditorPref.new('use_colors', true, editor_settings)

	# shortcut_run_all = GutEditorPref.new('shortcut_run_all', EMPTY, editor_settings)
	# shortcut_run_current_script = GutEditorPref.new('shortcut_run_current_script', EMPTY, editor_settings)
	# shortcut_run_current_inner = GutEditorPref.new('shortcut_run_current_inner', EMPTY, editor_settings)
	# shortcut_run_current_test = GutEditorPref.new('shortcut_run_current_test', EMPTY, editor_settings)
	# shortcut_panel_button = GutEditorPref.new('shortcut_panel_button', EMPTY, editor_settings)

func save_it():
	for prop in get_property_list():
		var val = get(prop.name)
		if(val is GutEditorPref):
			val.save_it()


func load_it():
	for prop in get_property_list():
		var val = get(prop.name)
		if(val is GutEditorPref):
			val.load_it()


func erase_all():
	for prop in get_property_list():
		var val = get(prop.name)
		if(val is GutEditorPref):
			val.erase()