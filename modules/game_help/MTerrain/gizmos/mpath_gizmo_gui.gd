@tool
extends Control


@onready var mirror_checkbox = find_child("mirror_checkbox")
@onready var mirror_lenght_checkbox = find_child("mirror_l_checkbox")
@onready var snap_checkbox = find_child("snap")
@onready var mode_option = find_child("mode")
@onready var collapse_btn = find_child("collapse")
@onready var toggle_connection_btn = find_child("toggle_connection")
@onready var connect_btn = find_child("Connect")
@onready var swap_points_btn = find_child("swap_points")
@onready var disconnect_btn = find_child("Disconnect")
@onready var remove_btn = find_child("remove")
@onready var tilt_num = find_child("tilt")
@onready var scale_num = find_child("scale")
@onready var depth_test_checkbox = find_child("depth_test")
#@onready var xz_handle_lock = find_child("xz_handle_lock")
@onready var select_lock = find_child("select_lock")
@onready var debug_col = find_child("debug_col")
@onready var sort_increasing_btn = find_child("sort_increasing")
@onready var sort_decreasing_btn = find_child("sort_decreasing")

@onready var show_rest_btn = find_child("show_rest")
@onready var settings_panel = find_child("settings_panel")

@onready var x_lock = find_child("x_lock")
@onready var y_lock = find_child("y_lock")
@onready var z_lock = find_child("z_lock")

@onready var active_point_label = find_child("active_point_label")

var mterrain_for_snap = null

var gizmo:
	set(value):
		gizmo = value
		gizmo.lock_mode_changed.connect(update_axis_lock)		
		gizmo.active_point_position_updated.connect(update_active_point_label)
		
		if is_instance_valid(x_lock):
			connect_lock_mode_signals()		
		
func connect_lock_mode_signals():
		x_lock.pressed.connect(func(): gizmo.update_lock_mode(x_lock.button_pressed, y_lock.button_pressed,z_lock.button_pressed))
		y_lock.pressed.connect(func(): gizmo.update_lock_mode(x_lock.button_pressed, y_lock.button_pressed,z_lock.button_pressed))
		z_lock.pressed.connect(func(): gizmo.update_lock_mode(x_lock.button_pressed, y_lock.button_pressed,z_lock.button_pressed))


enum MODE {
	EDIT = 0,
	CREATE = 1,
}
func update_active_point_label(new_text):
	active_point_label.text = new_text
	
func is_mirror()->bool:
	return mirror_checkbox.button_pressed

func is_mirror_lenght()->bool:
	return mirror_lenght_checkbox.button_pressed

func is_xz_handle_lock()->bool:
	return y_lock.button_pressed

func get_terrain_for_snap():
	if mterrain_for_snap and snap_checkbox.button_pressed:
		return mterrain_for_snap
	else:
		return null	

func get_mode():
	return mode_option.selected


func _input(event):
	if not visible:
		return		
	if Input.is_action_just_pressed( "mpath_toggle_mode" ):
		toggle_mode()
	if Input.is_action_just_pressed( "mpath_toggle_mirror" ):
		mirror_checkbox.button_pressed = not mirror_checkbox.button_pressed
	if Input.is_action_just_pressed( "mpath_toggle_mirror_length" ):
		mirror_lenght_checkbox.button_pressed = not mirror_lenght_checkbox.button_pressed

func _ready():
	tilt_num.__set_name("tilt")
	scale_num.__set_name("scale")
	tilt_num.set_value(0.0)
	scale_num.set_value(1.0)
	tilt_num.set_tooltip_text("Change Tilt\nHotkey: R")
	scale_num.set_tooltip_text("Change Tilt\nHotkey: E")
	settings_panel.visible = false
	
func update_axis_lock(lock_mode):		
	x_lock.button_pressed = true if lock_mode in [1,4,5,7] else false #x, xz, xy, xyz
	y_lock.button_pressed = true if lock_mode in [2,5,6,7] else false #y, xy, zy, xyz
	z_lock.button_pressed = true if lock_mode in [3,4,6,7] else false #z, xz, zy, xyz

func toggle_mode():
	if mode_option.selected == 0:
		mode_option.selected = 1
	else:
		mode_option.selected = 0

func is_select_lock()->bool:
	return select_lock.button_pressed

func is_debug_col()->bool:
	return debug_col.button_pressed

func _on_show_rest_toggled(toggle_on):
	settings_panel.visible = toggle_on
	settings_panel.position.x = -settings_panel.size.x + show_rest_btn.size.x 
	settings_panel.position.y = -settings_panel.size.y # - show_rest_btn.size.y
	
func set_terrain_snap(mterrain):
	if mterrain == null:
		snap_checkbox.button_pressed = false
		snap_checkbox.visible = false
		mterrain_for_snap = null
	else:	
		mterrain_for_snap = mterrain		
		snap_checkbox.button_pressed = true
		snap_checkbox.visible = true

func show_mpath_help_window():
	var help_window = preload("res://addons/m_terrain/gui/mpath_help_popup.tscn").instantiate()
	add_child(help_window)
	help_window.size.x = 25 * theme.default_font_size
	help_window.size.y = 24 * theme.default_font_size

func _on_show_rest_minimum_size_changed():
	_on_show_rest_toggled(settings_panel.visible)
