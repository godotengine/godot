
extends Node2D

# Joysticks demo, written by Dana Olson <dana@shineuponthee.com>
#
# This is a demo of joystick support, and doubles as a testing application
# inspired by and similar to jstest-gtk.
#
# Licensed under the MIT license

# Member variables
var joy_num
var cur_joy
var axis_value
var btn_state

const DEADZONE = 0.2

func _fixed_process(delta):
	# Get the joystick device number from the spinbox
	joy_num = get_node("joy_num").get_value()

	# Display the name of the joystick if we haven't already
	if joy_num != cur_joy:
		cur_joy = joy_num
		get_node("joy_name").set_text(Input.get_joy_name(joy_num))

	# Loop through the axes and show their current values
	for axis in range(0, 8):
		axis_value = Input.get_joy_axis(joy_num, axis)
		get_node("axis_prog" + str(axis)).set_value(100*axis_value)
		get_node("axis_val" + str(axis)).set_text(str(axis_value))
		if (axis < 4):
			if (abs(axis_value) < DEADZONE):
				get_node("diagram/axes/" + str(axis) + "+").hide()
				get_node("diagram/axes/" + str(axis) + "-").hide()
			elif (axis_value > 0):
				get_node("diagram/axes/" + str(axis) + "+").show()
			else:
				get_node("diagram/axes/" + str(axis) + "-").show()

	# Loop through the buttons and highlight the ones that are pressed
	for btn in range(0, 16):
		btn_state = 1
		if (Input.is_joy_button_pressed(joy_num, btn)):
			get_node("btn" + str(btn)).add_color_override("font_color", Color(1, 1, 1, 1))
			get_node("diagram/buttons/" + str(btn)).show()
		else:
			get_node("btn" + str(btn)).add_color_override("font_color", Color(0.2, 0.1, 0.3, 1))
			get_node("diagram/buttons/" + str(btn)).hide()

func _ready():
	set_fixed_process(true)
	Input.connect("joy_connection_changed", self, "_on_joy_connection_changed")

#Called whenever a joystick has been connected or disconnected.
func _on_joy_connection_changed(device_id, connected):
	if device_id == cur_joy:
		if connected:
			get_node("joy_name").set_text(Input.get_joy_name(device_id))
		else:
			get_node("joy_name").set_text("")
