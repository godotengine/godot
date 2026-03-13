/**************************************************************************/
/*  virtual_controller.cpp                                                */
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

#include "virtual_controller.h"

#include "core/config/engine.h"
#include "core/input/input.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/gui/button.h"
#include "scene/gui/virtual_joystick.h"
#include "scene/theme/theme_db.h"

void VirtualController::_setup_controls() {
	float min_size = MIN(get_size().x, get_size().y);
	int margin = min_size * 0.05f;
	Size2 joystick_size(min_size * 0.25f, min_size * 0.25f);

	left_joystick = memnew(VirtualJoystick);
	left_joystick->set_joystick_size(joystick_size.x);
	left_joystick->set_h_grow_direction(Control::GROW_DIRECTION_END);
	left_joystick->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	left_joystick->set_custom_minimum_size(joystick_size);
	left_joystick->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
	left_joystick->set_offset(Side::SIDE_BOTTOM, -margin);
	left_joystick->set_offset(Side::SIDE_LEFT, margin + min_size * 0.3f);
	add_child(left_joystick, true, Node::INTERNAL_MODE_FRONT);

	right_joystick = memnew(VirtualJoystick);
	right_joystick->set_joystick_size(joystick_size.x);
	right_joystick->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	right_joystick->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	right_joystick->set_custom_minimum_size(joystick_size);
	right_joystick->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_RIGHT);
	right_joystick->set_offset(Side::SIDE_BOTTOM, -margin);
	right_joystick->set_offset(Side::SIDE_RIGHT, -margin + min_size * -0.3f);
	add_child(right_joystick, true, Node::INTERNAL_MODE_FRONT);

	Size2 button_size(min_size * 0.125f, min_size * 0.125f);

	left_joystick_button = memnew(Button);
	left_joystick_button->set_text("L3");
	left_joystick_button->set_focus_mode(FOCUS_NONE);
	left_joystick_button->set_h_grow_direction(Control::GROW_DIRECTION_END);
	left_joystick_button->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	left_joystick_button->set_custom_minimum_size(button_size);
	left_joystick_button->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
	left_joystick_button->set_offset(SIDE_BOTTOM, -margin);
	left_joystick_button->set_offset(SIDE_LEFT, margin);
	add_child(left_joystick_button, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(left_joystick_button);

	right_joystick_button = memnew(Button);
	right_joystick_button->set_text("R3");
	right_joystick_button->set_focus_mode(FOCUS_NONE);
	right_joystick_button->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	right_joystick_button->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	right_joystick_button->set_custom_minimum_size(button_size);
	right_joystick_button->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_RIGHT);
	right_joystick_button->set_offset(SIDE_BOTTOM, -margin);
	right_joystick_button->set_offset(SIDE_RIGHT, -margin);
	add_child(right_joystick_button, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(right_joystick_button);

	float start_bottom_offset = min_size * -0.25f;

	dpad_down = memnew(Button);
	dpad_down->set_text("D");
	dpad_down->set_focus_mode(FOCUS_NONE);
	dpad_down->set_h_grow_direction(Control::GROW_DIRECTION_END);
	dpad_down->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	dpad_down->set_custom_minimum_size(button_size);
	dpad_down->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
	dpad_down->set_offset(SIDE_BOTTOM, start_bottom_offset);
	dpad_down->set_offset(SIDE_LEFT, margin + button_size.x);
	add_child(dpad_down, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(dpad_down);

	dpad_up = memnew(Button);
	dpad_up->set_text("U");
	dpad_up->set_focus_mode(FOCUS_NONE);
	dpad_up->set_h_grow_direction(Control::GROW_DIRECTION_END);
	dpad_up->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	dpad_up->set_custom_minimum_size(button_size);
	dpad_up->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
	dpad_up->set_offset(SIDE_BOTTOM, start_bottom_offset - 2 * button_size.y);
	dpad_up->set_offset(SIDE_LEFT, margin + button_size.x);
	add_child(dpad_up, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(dpad_up);

	dpad_left = memnew(Button);
	dpad_left->set_text("L");
	dpad_left->set_focus_mode(FOCUS_NONE);
	dpad_left->set_h_grow_direction(Control::GROW_DIRECTION_END);
	dpad_left->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	dpad_left->set_custom_minimum_size(button_size);
	dpad_left->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
	dpad_left->set_offset(SIDE_BOTTOM, start_bottom_offset - button_size.y);
	dpad_left->set_offset(SIDE_LEFT, margin);
	add_child(dpad_left, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(dpad_left);

	dpad_right = memnew(Button);
	dpad_right->set_text("R");
	dpad_right->set_focus_mode(FOCUS_NONE);
	dpad_right->set_h_grow_direction(Control::GROW_DIRECTION_END);
	dpad_right->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	dpad_right->set_custom_minimum_size(button_size);
	dpad_right->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
	dpad_right->set_offset(SIDE_BOTTOM, start_bottom_offset - button_size.y);
	dpad_right->set_offset(SIDE_LEFT, margin + 2 * button_size.x);
	add_child(dpad_right, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(dpad_right);

	button_a = memnew(Button);
	button_a->set_text("A");
	button_a->set_focus_mode(FOCUS_NONE);
	button_a->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	button_a->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	button_a->set_custom_minimum_size(button_size);
	button_a->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_RIGHT);
	button_a->set_offset(SIDE_BOTTOM, start_bottom_offset);
	button_a->set_offset(SIDE_RIGHT, -margin - button_size.x);
	add_child(button_a, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(button_a);

	button_y = memnew(Button);
	button_y->set_text("Y");
	button_y->set_focus_mode(FOCUS_NONE);
	button_y->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	button_y->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	button_y->set_custom_minimum_size(button_size);
	button_y->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_RIGHT);
	button_y->set_offset(SIDE_BOTTOM, start_bottom_offset - 2 * button_size.y);
	button_y->set_offset(SIDE_RIGHT, -margin - button_size.x);
	add_child(button_y, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(button_y);

	button_b = memnew(Button);
	button_b->set_text("B");
	button_b->set_focus_mode(FOCUS_NONE);
	button_b->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	button_b->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	button_b->set_custom_minimum_size(button_size);
	button_b->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_RIGHT);
	button_b->set_offset(SIDE_BOTTOM, start_bottom_offset - button_size.y);
	button_b->set_offset(SIDE_RIGHT, -margin);
	add_child(button_b, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(button_b);

	button_x = memnew(Button);
	button_x->set_text("X");
	button_x->set_focus_mode(FOCUS_NONE);
	button_x->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	button_x->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	button_x->set_custom_minimum_size(button_size);
	button_x->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_RIGHT);
	button_x->set_offset(SIDE_BOTTOM, start_bottom_offset - button_size.y);
	button_x->set_offset(SIDE_RIGHT, -margin - 2 * button_size.x);
	add_child(button_x, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(button_x);

	left_trigger = memnew(Button);
	left_trigger->set_text("LT");
	left_trigger->set_focus_mode(FOCUS_NONE);
	left_trigger->set_h_grow_direction(Control::GROW_DIRECTION_END);
	left_trigger->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	left_trigger->set_custom_minimum_size(button_size);
	left_trigger->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
	left_trigger->set_offset(SIDE_BOTTOM, start_bottom_offset - 3 * button_size.y - margin);
	left_trigger->set_offset(SIDE_LEFT, margin);
	add_child(left_trigger, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(left_trigger);

	left_shoulder = memnew(Button);
	left_shoulder->set_text("LB");
	left_shoulder->set_focus_mode(FOCUS_NONE);
	left_shoulder->set_h_grow_direction(Control::GROW_DIRECTION_END);
	left_shoulder->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	left_shoulder->set_custom_minimum_size(button_size);
	left_shoulder->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
	left_shoulder->set_offset(SIDE_BOTTOM, start_bottom_offset - 3 * button_size.y - margin);
	left_shoulder->set_offset(SIDE_LEFT, margin + 2 * button_size.x);
	add_child(left_shoulder, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(left_shoulder);

	right_trigger = memnew(Button);
	right_trigger->set_text("RT");
	right_trigger->set_focus_mode(FOCUS_NONE);
	right_trigger->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	right_trigger->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	right_trigger->set_custom_minimum_size(button_size);
	right_trigger->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_RIGHT);
	right_trigger->set_offset(SIDE_BOTTOM, start_bottom_offset - 3 * button_size.y - margin);
	right_trigger->set_offset(SIDE_RIGHT, -margin);
	add_child(right_trigger, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(right_trigger);

	right_shoulder = memnew(Button);
	right_shoulder->set_text("RB");
	right_shoulder->set_focus_mode(FOCUS_NONE);
	right_shoulder->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	right_shoulder->set_v_grow_direction(Control::GROW_DIRECTION_BEGIN);
	right_shoulder->set_custom_minimum_size(button_size);
	right_shoulder->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_RIGHT);
	right_shoulder->set_offset(SIDE_BOTTOM, start_bottom_offset - 3 * button_size.y - margin);
	right_shoulder->set_offset(SIDE_RIGHT, -margin - 2 * button_size.x);
	add_child(right_shoulder, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(right_shoulder);

	guide_button = memnew(Button);
	guide_button->set_text("G");
	guide_button->set_focus_mode(FOCUS_NONE);
	guide_button->set_h_grow_direction(Control::GROW_DIRECTION_BOTH);
	guide_button->set_v_grow_direction(Control::GROW_DIRECTION_END);
	guide_button->set_custom_minimum_size(button_size);
	guide_button->set_anchors_and_offsets_preset(Control::PRESET_CENTER_TOP);
	guide_button->set_offset(SIDE_TOP, margin);
	add_child(guide_button, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(guide_button);

	back_button = memnew(Button);
	back_button->set_text("<-");
	back_button->set_focus_mode(FOCUS_NONE);
	back_button->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
	back_button->set_v_grow_direction(Control::GROW_DIRECTION_END);
	back_button->set_custom_minimum_size(button_size);
	back_button->set_anchors_and_offsets_preset(Control::PRESET_CENTER_TOP);
	back_button->set_offset(SIDE_TOP, margin);
	back_button->set_offset(SIDE_RIGHT, -button_size.x - margin);
	add_child(back_button, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(back_button);

	start_button = memnew(Button);
	start_button->set_text("->");
	start_button->set_focus_mode(FOCUS_NONE);
	start_button->set_h_grow_direction(Control::GROW_DIRECTION_END);
	start_button->set_v_grow_direction(Control::GROW_DIRECTION_END);
	start_button->set_custom_minimum_size(button_size);
	start_button->set_anchors_and_offsets_preset(Control::PRESET_CENTER_TOP);
	start_button->set_offset(SIDE_TOP, margin);
	start_button->set_offset(SIDE_LEFT, button_size.x + margin);
	add_child(start_button, true, Node::INTERNAL_MODE_FRONT);
	_update_button_style(start_button);

	left_joystick->connect("motion", callable_mp(this, &VirtualController::_on_left_joystick_motion));
	left_joystick_button->connect("button_down", callable_mp(this, &VirtualController::_on_left_joystick_pressed));
	left_joystick_button->connect("button_up", callable_mp(this, &VirtualController::_on_left_joystick_released));
	right_joystick->connect("motion", callable_mp(this, &VirtualController::_on_right_joystick_motion));
	right_joystick_button->connect("button_down", callable_mp(this, &VirtualController::_on_right_joystick_pressed));
	right_joystick_button->connect("button_up", callable_mp(this, &VirtualController::_on_right_joystick_released));
	dpad_up->connect("button_down", callable_mp(this, &VirtualController::_on_dpad_up_pressed));
	dpad_up->connect("button_up", callable_mp(this, &VirtualController::_on_dpad_up_released));
	dpad_down->connect("button_down", callable_mp(this, &VirtualController::_on_dpad_down_pressed));
	dpad_down->connect("button_up", callable_mp(this, &VirtualController::_on_dpad_down_released));
	dpad_left->connect("button_down", callable_mp(this, &VirtualController::_on_dpad_left_pressed));
	dpad_left->connect("button_up", callable_mp(this, &VirtualController::_on_dpad_left_released));
	dpad_right->connect("button_down", callable_mp(this, &VirtualController::_on_dpad_right_pressed));
	dpad_right->connect("button_up", callable_mp(this, &VirtualController::_on_dpad_right_released));
	button_a->connect("button_down", callable_mp(this, &VirtualController::_on_button_a_pressed));
	button_a->connect("button_up", callable_mp(this, &VirtualController::_on_button_a_released));
	button_b->connect("button_down", callable_mp(this, &VirtualController::_on_button_b_pressed));
	button_b->connect("button_up", callable_mp(this, &VirtualController::_on_button_b_released));
	button_x->connect("button_down", callable_mp(this, &VirtualController::_on_button_x_pressed));
	button_x->connect("button_up", callable_mp(this, &VirtualController::_on_button_x_released));
	button_y->connect("button_down", callable_mp(this, &VirtualController::_on_button_y_pressed));
	button_y->connect("button_up", callable_mp(this, &VirtualController::_on_button_y_released));
	left_shoulder->connect("button_down", callable_mp(this, &VirtualController::_on_left_shoulder_pressed));
	left_shoulder->connect("button_up", callable_mp(this, &VirtualController::_on_left_shoulder_released));
	right_shoulder->connect("button_down", callable_mp(this, &VirtualController::_on_right_shoulder_pressed));
	right_shoulder->connect("button_up", callable_mp(this, &VirtualController::_on_right_shoulder_released));
	left_trigger->connect("button_down", callable_mp(this, &VirtualController::_on_left_trigger_pressed));
	left_trigger->connect("button_up", callable_mp(this, &VirtualController::_on_left_trigger_released));
	right_trigger->connect("button_down", callable_mp(this, &VirtualController::_on_right_trigger_pressed));
	right_trigger->connect("button_up", callable_mp(this, &VirtualController::_on_right_trigger_released));
	start_button->connect("button_down", callable_mp(this, &VirtualController::_on_start_button_pressed));
	start_button->connect("button_up", callable_mp(this, &VirtualController::_on_start_button_released));
	back_button->connect("button_down", callable_mp(this, &VirtualController::_on_back_button_pressed));
	back_button->connect("button_up", callable_mp(this, &VirtualController::_on_back_button_released));
	guide_button->connect("button_down", callable_mp(this, &VirtualController::_on_guide_button_pressed));
	guide_button->connect("button_up", callable_mp(this, &VirtualController::_on_guide_button_released));
}

void VirtualController::_bind_methods() {
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, VirtualController, button);
}

void VirtualController::_notification(int p_notification) {
	ERR_MAIN_THREAD_GUARD;
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint()) {
				return;
			}

			device_id = Input::get_singleton()->get_unused_joy_id();
			Dictionary info;
			info["mapping_handled"] = true;
			Input::get_singleton()->joy_connection_changed(device_id, true, "Virtual Controller", "virtual_controller", info);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (Engine::get_singleton()->is_editor_hint()) {
				return;
			}

			Input::get_singleton()->joy_connection_changed(device_id, false, "", "", Dictionary());
			device_id = -1;
		} break;

		case NOTIFICATION_PROCESS: {
			if (Engine::get_singleton()->is_editor_hint()) {
				return;
			}
			set_visible(Input::get_singleton()->is_virtual_controller_enabled());
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_update_button_style(left_joystick_button);
			_update_button_style(right_joystick_button);
			_update_button_style(dpad_up);
			_update_button_style(dpad_down);
			_update_button_style(dpad_left);
			_update_button_style(dpad_right);
			_update_button_style(button_a);
			_update_button_style(button_b);
			_update_button_style(button_x);
			_update_button_style(button_y);
			_update_button_style(left_shoulder);
			_update_button_style(right_shoulder);
			_update_button_style(left_trigger);
			_update_button_style(right_trigger);
			_update_button_style(start_button);
			_update_button_style(back_button);
			_update_button_style(guide_button);
		} break;
	}
}

void VirtualController::_create_button_event(JoyButton p_button, bool p_pressed) {
	Ref<InputEventJoypadButton> button_event;
	button_event.instantiate();
	button_event->set_button_index(p_button);
	button_event->set_pressed(p_pressed);
	button_event->set_device(device_id);

	Input::get_singleton()->parse_input_event(button_event);
}

void VirtualController::_create_motion_event(JoyAxis p_axis, float p_value) {
	Ref<InputEventJoypadMotion> motion_event;
	motion_event.instantiate();
	motion_event->set_axis(p_axis);
	motion_event->set_axis_value(p_value);
	motion_event->set_device(device_id);

	Input::get_singleton()->parse_input_event(motion_event);
}

void VirtualController::_update_button_style(Button *p_button) {
	if (p_button == nullptr) {
		return;
	}

	p_button->add_theme_style_override("normal", theme_cache.button);
	p_button->add_theme_style_override("hover", theme_cache.button);
	p_button->add_theme_style_override("pressed", theme_cache.button);
}

void VirtualController::_on_left_joystick_motion(Vector2 p_value) {
	_create_motion_event(JoyAxis::LEFT_X, p_value.x);
	_create_motion_event(JoyAxis::LEFT_Y, p_value.y);
}

void VirtualController::_on_left_joystick_pressed() {
	_create_button_event(JoyButton::LEFT_STICK, true);
}

void VirtualController::_on_left_joystick_released() {
	_create_button_event(JoyButton::LEFT_STICK, false);
}

void VirtualController::_on_right_joystick_motion(Vector2 p_value) {
	_create_motion_event(JoyAxis::RIGHT_X, p_value.x);
	_create_motion_event(JoyAxis::RIGHT_Y, p_value.y);
}

void VirtualController::_on_right_joystick_pressed() {
	_create_button_event(JoyButton::RIGHT_STICK, true);
}

void VirtualController::_on_right_joystick_released() {
	_create_button_event(JoyButton::RIGHT_STICK, false);
}

void VirtualController::_on_dpad_up_pressed() {
	_create_button_event(JoyButton::DPAD_UP, true);
}

void VirtualController::_on_dpad_up_released() {
	_create_button_event(JoyButton::DPAD_UP, false);
}

void VirtualController::_on_dpad_down_pressed() {
	_create_button_event(JoyButton::DPAD_DOWN, true);
}

void VirtualController::_on_dpad_down_released() {
	_create_button_event(JoyButton::DPAD_DOWN, false);
}

void VirtualController::_on_dpad_left_pressed() {
	_create_button_event(JoyButton::DPAD_LEFT, true);
}

void VirtualController::_on_dpad_left_released() {
	_create_button_event(JoyButton::DPAD_LEFT, false);
}

void VirtualController::_on_dpad_right_pressed() {
	_create_button_event(JoyButton::DPAD_RIGHT, true);
}

void VirtualController::_on_dpad_right_released() {
	_create_button_event(JoyButton::DPAD_RIGHT, false);
}

void VirtualController::_on_button_a_pressed() {
	_create_button_event(JoyButton::A, true);
}

void VirtualController::_on_button_a_released() {
	_create_button_event(JoyButton::A, false);
}

void VirtualController::_on_button_b_pressed() {
	_create_button_event(JoyButton::B, true);
}

void VirtualController::_on_button_b_released() {
	_create_button_event(JoyButton::B, false);
}

void VirtualController::_on_button_x_pressed() {
	_create_button_event(JoyButton::X, true);
}

void VirtualController::_on_button_x_released() {
	_create_button_event(JoyButton::X, false);
}

void VirtualController::_on_button_y_pressed() {
	_create_button_event(JoyButton::Y, true);
}

void VirtualController::_on_button_y_released() {
	_create_button_event(JoyButton::Y, false);
}

void VirtualController::_on_left_shoulder_pressed() {
	_create_button_event(JoyButton::LEFT_SHOULDER, true);
}

void VirtualController::_on_left_shoulder_released() {
	_create_button_event(JoyButton::LEFT_SHOULDER, false);
}

void VirtualController::_on_right_shoulder_pressed() {
	_create_button_event(JoyButton::RIGHT_SHOULDER, true);
}

void VirtualController::_on_right_shoulder_released() {
	_create_button_event(JoyButton::RIGHT_SHOULDER, false);
}

void VirtualController::_on_left_trigger_pressed() {
	_create_motion_event(JoyAxis::TRIGGER_LEFT, 1.0f);
}

void VirtualController::_on_left_trigger_released() {
	_create_motion_event(JoyAxis::TRIGGER_LEFT, 0.0f);
}

void VirtualController::_on_right_trigger_pressed() {
	_create_motion_event(JoyAxis::TRIGGER_RIGHT, 1.0f);
}

void VirtualController::_on_right_trigger_released() {
	_create_motion_event(JoyAxis::TRIGGER_RIGHT, 0.0f);
}

void VirtualController::_on_start_button_pressed() {
	_create_button_event(JoyButton::START, true);
}

void VirtualController::_on_start_button_released() {
	_create_button_event(JoyButton::START, false);
}

void VirtualController::_on_back_button_pressed() {
	_create_button_event(JoyButton::BACK, true);
}

void VirtualController::_on_back_button_released() {
	_create_button_event(JoyButton::BACK, false);
}

void VirtualController::_on_guide_button_pressed() {
	_create_button_event(JoyButton::GUIDE, true);
}

void VirtualController::_on_guide_button_released() {
	_create_button_event(JoyButton::GUIDE, false);
}

VirtualController::VirtualController() {
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	set_process(true);
	set_focus_mode(FOCUS_NONE);
	set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	callable_mp(this, &VirtualController::_setup_controls).call_deferred();
}
