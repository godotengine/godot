extends Node2D
# ##############################################################################
# This is a wrapper around the normal and compact gui controls and serves as
# the interface between gut.gd and the gui.  The GutRunner creates an instance
# of this and then this takes care of managing the different GUI controls.
# ##############################################################################
@onready var _normal_gui = $Normal
@onready var _compact_gui = $Compact

var gut = null :
	set(val):
		gut = val
		_set_gut(val)


func _ready():
	_normal_gui.switch_modes.connect(use_compact_mode.bind(true))
	_compact_gui.switch_modes.connect(use_compact_mode.bind(false))

	_normal_gui.set_title("GUT")
	_compact_gui.set_title("GUT")

	_normal_gui.align_right()
	_compact_gui.to_bottom_right()

	use_compact_mode(false)

	if(get_parent() == get_tree().root):
		_test_running_setup()

func _test_running_setup():
	set_font_size(100)
	_normal_gui.get_textbox().text = "hello world, how are you doing?"

# ------------------------
# Private
# ------------------------
func _set_gut(val):
	if(_normal_gui.get_gut() == val):
		return
	_normal_gui.set_gut(val)
	_compact_gui.set_gut(val)

	val.start_run.connect(_on_gut_start_run)
	val.end_run.connect(_on_gut_end_run)
	val.start_pause_before_teardown.connect(_on_gut_pause)
	val.end_pause_before_teardown.connect(_on_pause_end)

func _set_both_titles(text):
	_normal_gui.set_title(text)
	_compact_gui.set_title(text)


# ------------------------
# Events
# ------------------------
func _on_gut_start_run():
	_set_both_titles('Running')

func _on_gut_end_run():
	_set_both_titles('Finished')

func _on_gut_pause():
	_set_both_titles('-- Paused --')

func _on_pause_end():
	_set_both_titles('Running')


# ------------------------
# Public
# ------------------------
func get_textbox():
	return _normal_gui.get_textbox()


func set_font_size(new_size):
	var rtl = _normal_gui.get_textbox()

	rtl.set('theme_override_font_sizes/bold_italics_font_size', new_size)
	rtl.set('theme_override_font_sizes/bold_font_size', new_size)
	rtl.set('theme_override_font_sizes/italics_font_size', new_size)
	rtl.set('theme_override_font_sizes/normal_font_size', new_size)


func set_font(font_name):
	_set_all_fonts_in_rtl(_normal_gui.get_textbox(), font_name)


func _set_font(rtl, font_name, custom_name):
	if(font_name == null):
		rtl.remove_theme_font_override(custom_name)
	else:
		var dyn_font = FontFile.new()
		dyn_font.load_dynamic_font('res://addons/gut/fonts/' + font_name + '.ttf')
		rtl.add_theme_font_override(custom_name, dyn_font)


func _set_all_fonts_in_rtl(rtl, base_name):
	if(base_name == 'Default'):
		_set_font(rtl, null, 'normal_font')
		_set_font(rtl, null, 'bold_font')
		_set_font(rtl, null, 'italics_font')
		_set_font(rtl, null, 'bold_italics_font')
	else:
		_set_font(rtl, base_name + '-Regular', 'normal_font')
		_set_font(rtl, base_name + '-Bold', 'bold_font')
		_set_font(rtl, base_name + '-Italic', 'italics_font')
		_set_font(rtl, base_name + '-BoldItalic', 'bold_italics_font')


func set_default_font_color(color):
	_normal_gui.get_textbox().set('custom_colors/default_color', color)


func set_background_color(color):
	_normal_gui.set_bg_color(color)


func use_compact_mode(should=true):
	_compact_gui.visible = should
	_normal_gui.visible = !should


func set_opacity(val):
	_normal_gui.modulate.a = val
	_compact_gui.modulate.a = val

func set_title(text):
	_set_both_titles(text)
