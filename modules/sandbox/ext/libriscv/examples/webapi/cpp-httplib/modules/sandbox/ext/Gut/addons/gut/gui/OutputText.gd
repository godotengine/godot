@tool
extends VBoxContainer

var GutEditorGlobals = load('res://addons/gut/gui/editor_globals.gd')
var PanelControls = load('res://addons/gut/gui/panel_controls.gd')

# ##############################################################################
# Keeps search results from the TextEdit
# ##############################################################################
class TextEditSearcher:
	var te : TextEdit
	var _last_term = ''
	var _last_pos = Vector2(-1, -1)
	var _ignore_caret_change = false

	func set_text_edit(which):
		te = which
		te.caret_changed.connect(_on_caret_changed)


	func _on_caret_changed():
		if(_ignore_caret_change):
			_ignore_caret_change = false
		else:
			_last_pos = _get_caret();


	func _get_caret():
		return Vector2(te.get_caret_column(), te.get_caret_line())


	func _set_caret_and_sel(pos, len):
		te.set_caret_line(pos.y)
		te.set_caret_column(pos.x)
		if(len > 0):
			te.select(pos.y, pos.x, pos.y, pos.x + len)


	func _find(term, search_flags):
		var pos = _get_caret()
		if(term == _last_term):
			if(search_flags == 0):
				pos = _last_pos
				pos.x += 1
			else:
				pos = _last_pos
				pos.x -= 1

		var result = te.search(term, search_flags, pos.y, pos.x)
#		print('searching from ', pos, ' for "', term, '" = ', result)
		if(result.y != -1):
			_ignore_caret_change = true
			_set_caret_and_sel(result, term.length())
			_last_pos = result

		_last_term = term

	func find_next(term):
		_find(term, 0)

	func find_prev(term):
		_find(term, te.SEARCH_BACKWARDS)


# ##############################################################################
# Start OutputText control code
# ##############################################################################
@onready var _ctrls = {
	output = $Output,
	settings_bar = $Settings,
	use_colors = $Settings/UseColors,
	word_wrap = $Settings/WordWrap,

	copy_button = $Toolbar/CopyButton,
	clear_button = $Toolbar/ClearButton,
	show_search = $Toolbar/ShowSearch,
	caret_position = $Toolbar/LblPosition,

	search_bar = {
		bar = $Search,
		search_term = $Search/SearchTerm,
	}
}

var _sr = TextEditSearcher.new()
var _highlighter : CodeHighlighter
var _font_name = null
var _user_prefs = GutEditorGlobals.user_prefs
var _font_name_pctrl = null
var _font_size_pctrl = null

# Automatically used when running the OutputText scene from the editor.  Changes
# to this method only affect test-running the control through the editor.
func _test_running_setup():
	_ctrls.use_colors.text = 'use colors'
	_ctrls.show_search.text = 'search'
	_ctrls.word_wrap.text = 'ww'

	set_all_fonts("CourierPrime")
	set_font_size(30)

	_ctrls.output.queue_redraw()
	load_file('user://.gut_editor.bbcode')
	await get_tree().process_frame

	show_search(true)
	_ctrls.output.set_caret_line(0)
	_ctrls.output.scroll_vertical = 0
	_ctrls.output.caret_changed.connect(_on_caret_changed)


func _ready():
	_sr.set_text_edit(_ctrls.output)
	_ctrls.use_colors.icon = get_theme_icon('RichTextEffect', 'EditorIcons')
	_ctrls.show_search.icon = get_theme_icon('Search', 'EditorIcons')
	_ctrls.word_wrap.icon = get_theme_icon('Loop', 'EditorIcons')

	_setup_colors()
	_ctrls.use_colors.button_pressed = true
	_use_highlighting(true)

	if(get_parent() == get_tree().root):
		_test_running_setup()

	_ctrls.settings_bar.visible = false
	_add_other_ctrls()


func _add_other_ctrls():
	var fname = 'CourierNew'
	if(_user_prefs != null):
		fname = _user_prefs.output_font_name.value
	_font_name_pctrl = PanelControls.SelectControl.new('Font', fname, GutUtils.avail_fonts,
		"The font, you know, for the text below.  Change it, see what it does.")
	_font_name_pctrl.changed.connect(_on_font_name_changed)
	_font_name_pctrl.label.size_flags_horizontal = SIZE_SHRINK_BEGIN
	_ctrls.settings_bar.add_child(_font_name_pctrl)
	set_all_fonts(fname)

	var fsize = 30
	if(_user_prefs != null):
		fsize = _user_prefs.output_font_size.value
	_font_size_pctrl = PanelControls.NumberControl.new('Font Size', fsize , 5, 100,
		"The size of 'The Font'.")
	_font_size_pctrl.changed.connect(_on_font_size_changed)
	_font_size_pctrl.label.size_flags_horizontal = SIZE_SHRINK_BEGIN
	_ctrls.settings_bar.add_child(_font_size_pctrl)
	set_font_size(fsize)


# ------------------
# Private
# ------------------

# Call this after changes in colors and the like to get them to apply.  reloads
# the text of the output control.
func _refresh_output():
	var orig_pos = _ctrls.output.scroll_vertical
	var text = _ctrls.output.text

	_ctrls.output.text = text
	_ctrls.output.scroll_vertical = orig_pos


func _create_highlighter(default_color=Color(1, 1, 1, 1)):
	var to_return = CodeHighlighter.new()

	to_return.function_color = default_color
	to_return.number_color = default_color
	to_return.symbol_color = default_color
	to_return.member_variable_color = default_color

	var keywords = [
		['Failed', Color.RED],
		['Passed', Color.GREEN],
		['Pending', Color.YELLOW],
		['Orphans', Color.YELLOW],
		['WARNING', Color.YELLOW],
		['ERROR', Color.RED]
	]

	for keyword in keywords:
		to_return.add_keyword_color(keyword[0], keyword[1])

	return to_return


func _setup_colors():
	_ctrls.output.clear()

	var f_color = null
	if (_ctrls.output.theme == null) :
		f_color = get_theme_color("font_color")
	else :
		f_color = _ctrls.output.theme.font_color

	_highlighter = _create_highlighter()
	_ctrls.output.queue_redraw()



func _use_highlighting(should):
	if(should):
		_ctrls.output.syntax_highlighter = _highlighter
	else:
		_ctrls.output.syntax_highlighter = null
	_refresh_output()

# ------------------
# Events
# ------------------
func _on_caret_changed():
	var txt = str("line:",_ctrls.output.get_caret_line(), ' col:', _ctrls.output.get_caret_column())
	_ctrls.caret_position.text = str(txt)

func _on_font_size_changed():
	set_font_size(_font_size_pctrl.value)
	if(_user_prefs != null):
		_user_prefs.output_font_size.value = _font_size_pctrl.value
		_user_prefs.output_font_size.save_it()

func _on_font_name_changed():
	set_all_fonts(_font_name_pctrl.text)
	if(_user_prefs != null):
		_user_prefs.output_font_name.value = _font_name_pctrl.text
		_user_prefs.output_font_name.save_it()

func _on_CopyButton_pressed():
	copy_to_clipboard()

func _on_UseColors_pressed():
	_use_highlighting(_ctrls.use_colors.button_pressed)

func _on_ClearButton_pressed():
	clear()

func _on_ShowSearch_pressed():
	show_search(_ctrls.show_search.button_pressed)

func _on_SearchTerm_focus_entered():
	_ctrls.search_bar.search_term.call_deferred('select_all')

func _on_SearchNext_pressed():
	_sr.find_next(_ctrls.search_bar.search_term.text)

func _on_SearchPrev_pressed():
	_sr.find_prev(_ctrls.search_bar.search_term.text)

func _on_SearchTerm_text_changed(new_text):
	if(new_text == ''):
		_ctrls.output.deselect()
	else:
		_sr.find_next(new_text)

func _on_SearchTerm_text_entered(new_text):
	if(Input.is_physical_key_pressed(KEY_SHIFT)):
		_sr.find_prev(new_text)
	else:
		_sr.find_next(new_text)

func _on_SearchTerm_gui_input(event):
	if(event is InputEventKey and !event.pressed and event.keycode == KEY_ESCAPE):
		show_search(false)

func _on_WordWrap_pressed():
	if(_ctrls.word_wrap.button_pressed):
		_ctrls.output.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	else:
		_ctrls.output.wrap_mode = TextEdit.LINE_WRAPPING_NONE

	_ctrls.output.queue_redraw()

func _on_settings_pressed():
	_ctrls.settings_bar.visible = $Toolbar/ShowSettings.button_pressed

# ------------------
# Public
# ------------------
func show_search(should):
	_ctrls.search_bar.bar.visible = should
	if(should):
		_ctrls.search_bar.search_term.grab_focus()
		_ctrls.search_bar.search_term.select_all()
	_ctrls.show_search.button_pressed = should


func search(text, start_pos, highlight=true):
	return _sr.find_next(text)


func copy_to_clipboard():
	var selected = _ctrls.output.get_selected_text()
	if(selected != ''):
		DisplayServer.clipboard_set(selected)
	else:
		DisplayServer.clipboard_set(_ctrls.output.text)


func clear():
	_ctrls.output.text = ''


func _set_font(font_name, custom_name):
	var rtl = _ctrls.output
	if(font_name == null):
		rtl.remove_theme_font_override(custom_name)
	else:
		var dyn_font = FontFile.new()
		dyn_font.load_dynamic_font('res://addons/gut/fonts/' + font_name + '.ttf')
		rtl.add_theme_font_override(custom_name, dyn_font)


func set_all_fonts(base_name):
	_font_name = GutUtils.nvl(base_name, 'Default')

	if(base_name == 'Default'):
		_set_font(null, 'font')
		_set_font(null, 'normal_font')
		_set_font(null, 'bold_font')
		_set_font(null, 'italics_font')
		_set_font(null, 'bold_italics_font')
	else:
		_set_font(base_name + '-Regular', 'font')
		_set_font(base_name + '-Regular', 'normal_font')
		_set_font(base_name + '-Bold', 'bold_font')
		_set_font(base_name + '-Italic', 'italics_font')
		_set_font(base_name + '-BoldItalic', 'bold_italics_font')


func set_font_size(new_size):
	_ctrls.output.set("theme_override_font_sizes/font_size", new_size)


func set_use_colors(value):
	pass


func get_use_colors():
	return false;


func get_rich_text_edit():
	return _ctrls.output


func load_file(path):
	var f = FileAccess.open(path, FileAccess.READ)
	if(f == null):
		return

	var t = f.get_as_text()
	f = null # closes file
	_ctrls.output.text = t
	_ctrls.output.scroll_vertical = _ctrls.output.get_line_count()
	_ctrls.output.set_deferred('scroll_vertical', _ctrls.output.get_line_count())


func add_text(text):
	if(is_inside_tree()):
		_ctrls.output.text += text


func scroll_to_line(line):
	_ctrls.output.scroll_vertical = line
	_ctrls.output.set_caret_line(line)
