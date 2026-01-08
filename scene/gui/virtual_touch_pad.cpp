/**************************************************************************/
/*  virtual_touch_pad.cpp                                                 */
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

#include "virtual_touch_pad.h"

#include "core/config/engine.h"
#include "core/input/input.h"
#include "scene/theme/theme_db.h"

void VirtualTouchPad::_notification(int p_what) {
	VirtualDevice::_notification(p_what);
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (Engine::get_singleton()->is_editor_hint()) {
				draw_rect(Rect2(Point2(), get_size()), Color(0.5, 0.5, 0.5, 0.2), true);
				draw_rect(Rect2(Point2(), get_size()), Color(1, 1, 1, 0.5), false);
			} else if (is_pressed() && last_pos != current_pos) {
				// Draw movement trail in runtime
				draw_line(last_pos, current_pos, Color(1, 1, 1, 0.8), 2.0);
				draw_circle(current_pos, 5.0, Color(1, 1, 1, 0.6));
			}
		} break;
	}
}

void VirtualTouchPad::_on_drag(int p_index, const Vector2 &p_pos, const Vector2 &p_relative) {
	// Send Relative Motion
	// This acts like a mouse trackpad or relative joystick

	if (p_relative.x != 0) {
		int axis = (hand == HAND_LEFT) ? 0 : 2;
		Ref<InputEventVirtualMotion> ie_x;
		ie_x.instantiate();
		ie_x->set_device(get_device());
		ie_x->set_axis(axis);
		ie_x->set_axis_value(p_relative.x * sensitivity);
		// Wait, InputEventVirtualMotion uses "axis_value" which is typically absolute (-1 to 1).
		// Relative motion (mouse) is usually an InputEventMouseMotion.
		// If VirtualTouchPad is for Camera Pan, it should probably emit 'relative' values?
		// But VirtualMotion is designed for Joystick-like axes.
		// If we want it to act like a Mouse, maybe we should emit standard MouseMotion?
		// OR: We treat it as a "Rate" control?
		// "touchpad é focado em camera e movimentação pan".
		// Typically touchpads map relative movement to... movement.
		// If I swipe left, camera rotates left.

		// Let's assume we map the RELATIVE movement directly to the axis value for this frame.
		// NOTE: Receiver must handle this as "delta".
		Input::get_singleton()->parse_input_event(ie_x);
	}

	if (p_relative.y != 0) {
		int axis = (hand == HAND_LEFT) ? 1 : 3;
		Ref<InputEventVirtualMotion> ie_y;
		ie_y.instantiate();
		ie_y->set_device(get_device());
		ie_y->set_axis(axis);
		ie_y->set_axis_value(p_relative.y * sensitivity);
		Input::get_singleton()->parse_input_event(ie_y);
	}

	// Update trail positions
	last_pos = current_pos;
	current_pos = p_pos;
	queue_redraw();
}

void VirtualTouchPad::_reset_touchpad() {
	Ref<InputEventVirtualMotion> ie_x;
	ie_x.instantiate();
	ie_x->set_device(get_device());
	ie_x->set_axis(x_axis);
	ie_x->set_axis_value(0.0);
	Input::get_singleton()->parse_input_event(ie_x);

	Ref<InputEventVirtualMotion> ie_y;
	ie_y.instantiate();
	ie_y->set_device(get_device());
	ie_y->set_axis(y_axis);
	ie_y->set_axis_value(0.0);
	Input::get_singleton()->parse_input_event(ie_y);
}

void VirtualTouchPad::pressed_state_changed() {
	if (!is_pressed()) {
		_reset_touchpad();
	}
}

void VirtualTouchPad::set_hand(TouchPadHand p_hand) {
	hand = p_hand;
}

void VirtualTouchPad::_on_touch_down(int p_index, const Vector2 &p_pos) {
	last_pos = p_pos;
	current_pos = p_pos;
	queue_redraw();
}

void VirtualTouchPad::_on_touch_up(int p_index, const Vector2 &p_pos) {
	_reset_touchpad();
	queue_redraw();
}

void VirtualTouchPad::set_sensitivity(float p_sensitivity) {
	sensitivity = p_sensitivity;
}

float VirtualTouchPad::get_sensitivity() const {
	return sensitivity;
}

void VirtualTouchPad::set_x_axis(int p_axis) {
	x_axis = p_axis;
}

int VirtualTouchPad::get_x_axis() const {
	return x_axis;
}

void VirtualTouchPad::set_y_axis(int p_axis) {
	y_axis = p_axis;
}

int VirtualTouchPad::get_y_axis() const {
	return y_axis;
}

VirtualTouchPad::VirtualTouchPad() {
}

Size2 VirtualTouchPad::get_minimum_size() const {
	return Size2(20, 20);
}

void VirtualTouchPad::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sensitivity", "sensitivity"), &VirtualTouchPad::set_sensitivity);
	ClassDB::bind_method(D_METHOD("get_sensitivity"), &VirtualTouchPad::get_sensitivity);
	ClassDB::bind_method(D_METHOD("set_x_axis", "axis"), &VirtualTouchPad::set_x_axis);
	ClassDB::bind_method(D_METHOD("get_x_axis"), &VirtualTouchPad::get_x_axis);
	ClassDB::bind_method(D_METHOD("set_y_axis", "axis"), &VirtualTouchPad::set_y_axis);
	ClassDB::bind_method(D_METHOD("get_y_axis"), &VirtualTouchPad::get_y_axis);
	ClassDB::bind_method(D_METHOD("set_hand", "hand"), &VirtualTouchPad::set_hand);
	ClassDB::bind_method(D_METHOD("get_hand"), &VirtualTouchPad::get_hand);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sensitivity"), "set_sensitivity", "get_sensitivity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hand", PROPERTY_HINT_ENUM, "Left,Right"), "set_hand", "get_hand");

	ADD_GROUP("Axis Mapping", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "x_axis"), "set_x_axis", "get_x_axis");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "y_axis"), "set_y_axis", "get_y_axis");

	BIND_ENUM_CONSTANT(HAND_LEFT);
	BIND_ENUM_CONSTANT(HAND_RIGHT);
}
