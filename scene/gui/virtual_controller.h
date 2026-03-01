/**************************************************************************/
/*  virtual_controller.h                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "scene/gui/button.h"
#include "scene/gui/virtual_joystick.h"

class VirtualController : public Control {
	GDCLASS(VirtualController, Control);

private:
	int device_id = -1;

	VirtualJoystick *left_joystick = nullptr;
	VirtualJoystick *right_joystick = nullptr;
	Button *left_joystick_button = nullptr;
	Button *right_joystick_button = nullptr;
	Button *dpad_up = nullptr;
	Button *dpad_down = nullptr;
	Button *dpad_left = nullptr;
	Button *dpad_right = nullptr;
	Button *button_a = nullptr;
	Button *button_b = nullptr;
	Button *button_x = nullptr;
	Button *button_y = nullptr;
	Button *left_shoulder = nullptr;
	Button *right_shoulder = nullptr;
	Button *left_trigger = nullptr;
	Button *right_trigger = nullptr;
	Button *start_button = nullptr;
	Button *back_button = nullptr;
	Button *guide_button = nullptr;

	void _setup_controls();

	void _create_button_event(JoyButton p_button, bool p_pressed);
	void _create_motion_event(JoyAxis p_axis, float p_value);

	void _on_left_joystick_motion(Vector2 p_value);
	void _on_right_joystick_motion(Vector2 p_value);
	void _on_left_joystick_pressed();
	void _on_left_joystick_released();
	void _on_right_joystick_pressed();
	void _on_right_joystick_released();
	void _on_dpad_up_pressed();
	void _on_dpad_up_released();
	void _on_dpad_down_pressed();
	void _on_dpad_down_released();
	void _on_dpad_left_pressed();
	void _on_dpad_left_released();
	void _on_dpad_right_pressed();
	void _on_dpad_right_released();
	void _on_button_a_pressed();
	void _on_button_a_released();
	void _on_button_b_pressed();
	void _on_button_b_released();
	void _on_button_x_pressed();
	void _on_button_x_released();
	void _on_button_y_pressed();
	void _on_button_y_released();
	void _on_left_shoulder_pressed();
	void _on_left_shoulder_released();
	void _on_right_shoulder_pressed();
	void _on_right_shoulder_released();
	void _on_left_trigger_pressed();
	void _on_left_trigger_released();
	void _on_right_trigger_pressed();
	void _on_right_trigger_released();
	void _on_start_button_pressed();
	void _on_start_button_released();
	void _on_back_button_pressed();
	void _on_back_button_released();
	void _on_guide_button_pressed();
	void _on_guide_button_released();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	VirtualController();
};
