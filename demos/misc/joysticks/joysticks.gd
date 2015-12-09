
extends Node2D

# Joysticks demo, written by Dana Olson <dana@shineuponthee.com>
#
# This is a demo of joystick support, and doubles as a testing application
# inspired by and similar to jstest-gtk.
#
# Licensed under the MIT license

# member variables
var joy_num
var cur_joy
var axis_value
var btn_state


func _input(event):
	# get the joystick device number from the spinbox
	joy_num = get_node("joy_num").get_value()

	# display the name of the joystick if we haven't already
	if joy_num != cur_joy:
		cur_joy = joy_num
		get_node("joy_name").set_text(Input.get_joy_name(joy_num))

	# loop through the axes and show their current values
	for axis in range(0, 8):
		axis_value = Input.get_joy_axis(joy_num, axis)
		get_node("axis_prog" + str(axis)).set_value(100*axis_value)
		get_node("axis_val" + str(axis)).set_text(str(axis_value))

	# loop through the buttons and highlight the ones that are pressed
	for btn in range(0, 17):
		btn_state = 1
		if (Input.is_joy_button_pressed(joy_num, btn)):
			get_node("btn" + str(btn)).add_color_override("font_color", Color(1, 1, 1, 1))
		else:
			get_node("btn" + str(btn)).add_color_override("font_color", Color(0.2, 0.1, 0.3, 1))


func _ready():
	set_process_input(true)
