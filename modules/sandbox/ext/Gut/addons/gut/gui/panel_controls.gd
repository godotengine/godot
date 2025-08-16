# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class BaseGutPanelControl:
	extends HBoxContainer
	var label = Label.new()
	var _lbl_unsaved = Label.new()
	var _lbl_invalid = Label.new()

	var value = null:
		get: return get_value()
		set(val): set_value(val)

	signal changed

	func _init(title, val, hint=""):
		size_flags_horizontal = SIZE_EXPAND_FILL
		mouse_filter = MOUSE_FILTER_PASS

		label.size_flags_horizontal = label.SIZE_EXPAND_FILL
		label.mouse_filter = label.MOUSE_FILTER_STOP
		add_child(label)

		_lbl_unsaved.text = '*'
		_lbl_unsaved.visible = false
		add_child(_lbl_unsaved)

		_lbl_invalid.text = '!'
		_lbl_invalid.visible = false
		add_child(_lbl_invalid)

		label.text = title
		label.tooltip_text = hint


	func mark_unsaved(is_it=true):
		_lbl_unsaved.visible = is_it


	func mark_invalid(is_it):
		_lbl_invalid.visible = is_it

	# -- Virtual --
	#
	# value_ctrl (all should declare the value_ctrl)
	#
	func set_value(value):
		pass

	func get_value():
		pass


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class NumberControl:
	extends BaseGutPanelControl

	var value_ctrl = SpinBox.new()

	func _init(title, val, v_min, v_max, hint=""):
		super._init(title, val, hint)

		value_ctrl.value = val
		value_ctrl.size_flags_horizontal = value_ctrl.SIZE_EXPAND_FILL
		value_ctrl.min_value = v_min
		value_ctrl.max_value = v_max
		value_ctrl.value_changed.connect(_on_value_changed)
		value_ctrl.select_all_on_focus = true
		add_child(value_ctrl)

	func _on_value_changed(new_value):
		changed.emit()

	func get_value():
		return value_ctrl.value

	func set_value(val):
		value_ctrl.value = val


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class StringControl:
	extends BaseGutPanelControl

	var value_ctrl = LineEdit.new()

	func _init(title, val, hint=""):
		super._init(title, val, hint)

		value_ctrl.size_flags_horizontal = value_ctrl.SIZE_EXPAND_FILL
		value_ctrl.text = val
		value_ctrl.text_changed.connect(_on_text_changed)
		value_ctrl.select_all_on_focus = true
		add_child(value_ctrl)

	func _on_text_changed(new_value):
		changed.emit()

	func get_value():
		return value_ctrl.text

	func set_value(val):
		value_ctrl.text = val



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class BooleanControl:
	extends BaseGutPanelControl

	var value_ctrl = CheckBox.new()

	func _init(title, val, hint=""):
		super._init(title, val, hint)

		value_ctrl.button_pressed = val
		value_ctrl.toggled.connect(_on_button_toggled)
		add_child(value_ctrl)

	func _on_button_toggled(new_value):
		changed.emit()

	func get_value():
		return value_ctrl.button_pressed

	func set_value(val):
		value_ctrl.button_pressed = val


# ------------------------------------------------------------------------------
# value is "selected" and is gettable and settable
# text is the text value of the selected item, it is gettable only
# ------------------------------------------------------------------------------
class SelectControl:
	extends BaseGutPanelControl

	var value_ctrl = OptionButton.new()

	var text = '' :
		get: return value_ctrl.get_item_text(value_ctrl.selected)
		set(val): pass

	func _init(title, val, choices, hint=""):
		super._init(title, val, hint)

		var select_idx = 0
		for i in range(choices.size()):
			value_ctrl.add_item(choices[i])
			if(val == choices[i]):
				select_idx = i
		value_ctrl.selected = select_idx
		value_ctrl.size_flags_horizontal = value_ctrl.SIZE_EXPAND_FILL
		value_ctrl.item_selected.connect(_on_item_selected)
		add_child(value_ctrl)

	func _on_item_selected(idx):
		changed.emit()

	func get_value():
		return value_ctrl.selected

	func set_value(val):
		value_ctrl.selected = val


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class ColorControl:
	extends BaseGutPanelControl

	var value_ctrl = ColorPickerButton.new()

	func _init(title, val, hint=""):
		super._init(title, val, hint)
		value_ctrl.size_flags_horizontal = value_ctrl.SIZE_EXPAND_FILL
		value_ctrl.color = val
		add_child(value_ctrl)

	func get_value():
		return value_ctrl.color

	func set_value(val):
		value_ctrl.color = val


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class DirectoryControl:
	extends BaseGutPanelControl

	var value_ctrl := LineEdit.new()
	var dialog := FileDialog.new()
	var enabled_button = CheckButton.new()

	var _btn_dir := Button.new()

	func _init(title, val, hint=""):
		super._init(title, val, hint)

		label.size_flags_horizontal = Control.SIZE_SHRINK_BEGIN

		_btn_dir.text = '...'
		_btn_dir.pressed.connect(_on_dir_button_pressed)

		value_ctrl.text = val
		value_ctrl.size_flags_horizontal = value_ctrl.SIZE_EXPAND_FILL
		value_ctrl.select_all_on_focus = true
		value_ctrl.text_changed.connect(_on_value_changed)

		dialog.file_mode = dialog.FILE_MODE_OPEN_DIR
		dialog.unresizable = false
		dialog.dir_selected.connect(_on_selected)
		dialog.file_selected.connect(_on_selected)

		enabled_button.button_pressed = true
		enabled_button.visible = false

		add_child(enabled_button)
		add_child(value_ctrl)
		add_child(_btn_dir)
		add_child(dialog)

	func _update_display():
		var is_empty = value_ctrl.text == ''
		enabled_button.button_pressed = !is_empty
		enabled_button.disabled = is_empty


	func _ready():
		if(Engine.is_editor_hint()):
			dialog.size = Vector2(1000, 700)
		else:
			dialog.size = Vector2(500, 350)
		_update_display()

	func _on_value_changed(new_text):
		_update_display()

	func _on_selected(path):
		value_ctrl.text = path
		_update_display()

	func _on_dir_button_pressed():
		dialog.current_dir = value_ctrl.text
		dialog.popup_centered()

	func get_value():
		return value_ctrl.text

	func set_value(val):
		value_ctrl.text = val


# ------------------------------------------------------------------------------
# Features:
# 	Buttons to pick res://, user://, or anywhere on the OS.
# ------------------------------------------------------------------------------
class FileDialogSuperPlus:
	extends FileDialog

	var show_diretory_types = true :
		set(val) :
			show_diretory_types = val
			_update_display()

	var show_res = true :
		set(val) :
			show_res = val
			_update_display()

	var show_user = true :
		set(val) :
			show_user = val
			_update_display()

	var show_os = true :
		set(val) :
			show_os = val
			_update_display()

	var _dir_type_hbox = null
	var _btn_res = null
	var _btn_user = null
	var _btn_os = null

	func _ready():
		_init_controls()
		_update_display()


	func _init_controls():
		_dir_type_hbox = HBoxContainer.new()

		_btn_res = Button.new()
		_btn_user = Button.new()
		_btn_os = Button.new()
		var spacer1 = CenterContainer.new()
		spacer1.size_flags_horizontal = spacer1.SIZE_EXPAND_FILL
		var spacer2 = spacer1.duplicate()

		_dir_type_hbox.add_child(spacer1)
		_dir_type_hbox.add_child(_btn_res)
		_dir_type_hbox.add_child(_btn_user)
		_dir_type_hbox.add_child(_btn_os)
		_dir_type_hbox.add_child(spacer2)

		_btn_res.text = 'res://'
		_btn_user.text = 'user://'
		_btn_os.text = '  OS  '

		get_vbox().add_child(_dir_type_hbox)
		get_vbox().move_child(_dir_type_hbox, 0)

		_btn_res.pressed.connect(func(): access = ACCESS_RESOURCES)
		_btn_user.pressed.connect(func(): access = ACCESS_USERDATA)
		_btn_os.pressed.connect(func(): access = ACCESS_FILESYSTEM)


	func _update_display():
		if(is_inside_tree()):
			_dir_type_hbox.visible = show_diretory_types
			_btn_res.visible = show_res
			_btn_user.visible = show_user
			_btn_os.visible = show_os


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class SaveLoadControl:
	extends BaseGutPanelControl

	var btn_load = Button.new()
	var btn_save = Button.new()

	var dlg_load := FileDialogSuperPlus.new()
	var dlg_save := FileDialogSuperPlus.new()

	signal save_path_chosen(path)
	signal load_path_chosen(path)

	func _init(title, val, hint):
		super._init(title, val, hint)

		btn_load.text = "Load"
		btn_load.custom_minimum_size.x = 100
		btn_load.pressed.connect(_on_load_pressed)
		add_child(btn_load)

		btn_save.text = "Save As"
		btn_save.custom_minimum_size.x = 100
		btn_save.pressed.connect(_on_save_pressed)
		add_child(btn_save)

		dlg_load.file_mode = dlg_load.FILE_MODE_OPEN_FILE
		dlg_load.unresizable = false
		dlg_load.dir_selected.connect(_on_load_selected)
		dlg_load.file_selected.connect(_on_load_selected)
		add_child(dlg_load)

		dlg_save.file_mode = dlg_save.FILE_MODE_SAVE_FILE
		dlg_save.unresizable = false
		dlg_save.dir_selected.connect(_on_save_selected)
		dlg_save.file_selected.connect(_on_save_selected)
		add_child(dlg_save)


	func _ready():
		if(Engine.is_editor_hint()):
			dlg_load.size = Vector2(1000, 700)
			dlg_save.size = Vector2(1000, 700)
		else:
			dlg_load.size = Vector2(500, 350)
			dlg_save.size = Vector2(500, 350)

	func _on_load_selected(path):
		load_path_chosen.emit(path)

	func _on_save_selected(path):
		save_path_chosen.emit(path)

	func _on_load_pressed():
		dlg_load.popup_centered()

	func _on_save_pressed():
		dlg_save.popup_centered()

# ------------------------------------------------------------------------------
# This one was never used in gut_config_gui...but I put some work into it and
# I'm a sucker for that kinda thing.  Delete this when you get tired of looking
# at it.
# ------------------------------------------------------------------------------
# class Vector2Ctrl:
# 	extends VBoxContainer

# 	var value = Vector2(-1, -1) :
# 		get:
# 			return get_value()
# 		set(val):
# 			set_value(val)
# 	var disabled = false :
# 		get:
# 			return get_disabled()
# 		set(val):
# 			set_disabled(val)
# 	var x_spin = SpinBox.new()
# 	var y_spin = SpinBox.new()

# 	func _init():
# 		add_child(_make_one('x:  ', x_spin))
# 		add_child(_make_one('y:  ', y_spin))

# 	func _make_one(txt, spinner):
# 		var hbox = HBoxContainer.new()
# 		var lbl = Label.new()
# 		lbl.text = txt
# 		hbox.add_child(lbl)
# 		hbox.add_child(spinner)
# 		spinner.min_value = -1
# 		spinner.max_value = 10000
# 		spinner.size_flags_horizontal = spinner.SIZE_EXPAND_FILL
# 		return hbox

# 	func set_value(v):
# 		if(v != null):
# 			x_spin.value = v[0]
# 			y_spin.value = v[1]

# 	# Returns array instead of vector2 b/c that is what is stored in
# 	# in the dictionary and what is expected everywhere else.
# 	func get_value():
# 		return [x_spin.value, y_spin.value]

# 	func set_disabled(should):
# 		get_parent().visible = !should
# 		x_spin.visible = !should
# 		y_spin.visible = !should

# 	func get_disabled():
# 		pass
