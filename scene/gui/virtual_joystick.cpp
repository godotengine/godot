/**************************************************************************/
/*  virtual_joystick.cpp                                                  */
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

#include "virtual_joystick.h"

#include "core/input/input_map.h"
#include "scene/theme/theme_db.h"

void VirtualJoystick::gui_input(const Ref<InputEvent> &p_event) {
	if (visibility == VISIBILITY_TOUCHSCREEN_ONLY && !DisplayServer::get_singleton()->is_touchscreen_available()) {
		return;
	}

	Ref<InputEventScreenTouch> touch = p_event;
	if (touch.is_valid()) {
		if (touch->is_pressed()) {
			if (touch_index == -1 && has_point(touch->get_position())) {
				Rect2 base_rect = Rect2(joystick_pos - Vector2(0.5, 0.5) * joystick_size, Vector2(joystick_size, joystick_size));
				if (joystick_mode == JOYSTICK_DYNAMIC || joystick_mode == JOYSTICK_FOLLOWING || (base_rect.has_point(touch->get_position()) && joystick_mode == JOYSTICK_FIXED)) {
					if (joystick_mode == JOYSTICK_DYNAMIC || joystick_mode == JOYSTICK_FOLLOWING) {
						joystick_pos = touch->get_position();
					}

					emit_signal(SceneStringName(pressed));

					is_pressed = true;
					touch_index = touch->get_index();
					_update_joystick(touch->get_position());
				}
			}
		} else if (touch->get_index() == touch_index) {
			is_pressed = false;
			emit_signal(SNAME("released"), input_vector);

			if (!is_flick_canceled && !has_moved) {
				emit_signal(SNAME("tapped"));
			} else if (has_input && has_moved) {
				emit_signal(SNAME("flicked"), input_vector);
			}
			_reset();
		}
	}

	Ref<InputEventScreenDrag> drag = p_event;
	if (drag.is_valid() && drag->get_index() == touch_index) {
		has_moved = true;
		_update_joystick(drag->get_position());
	}
}

void VirtualJoystick::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (!Engine::get_singleton()->is_editor_hint() && visibility == VISIBILITY_TOUCHSCREEN_ONLY && !DisplayServer::get_singleton()->is_touchscreen_available()) {
				return;
			}

			if (!Engine::get_singleton()->is_editor_hint() && visibility == VISIBILITY_WHEN_TOUCHED && !is_pressed) {
				return;
			}

			if (joystick_texture.is_valid()) {
				Rect2 rect = Rect2(joystick_pos - Vector2(0.5, 0.5) * joystick_size, Vector2(joystick_size, joystick_size));
				draw_texture_rect(joystick_texture, rect);
			} else {
				draw_circle(joystick_pos, joystick_size * 0.5, is_pressed ? theme_cache.ring_pressed_color : theme_cache.ring_normal_color, false, joystick_size * 0.05, true);
			}

			if (tip_texture.is_valid()) {
				Rect2 rect = Rect2(tip_pos - Vector2(0.5, 0.5) * tip_size, Vector2(tip_size, tip_size));
				draw_texture_rect(tip_texture, rect);
			} else {
				draw_circle(tip_pos, tip_size * 0.5, is_pressed ? theme_cache.tip_pressed_color : theme_cache.tip_normal_color, true, -1, true);
			}
		} break;

		case NOTIFICATION_ENTER_TREE: {
			joystick_pos = get_size() * initial_offset_ratio;
			tip_pos = joystick_pos;
		} break;

		case NOTIFICATION_RESIZED: {
			_reset();
		} break;
	}
}

void VirtualJoystick::_update_joystick(const Vector2 &p_pos) {
	Vector2 offset = p_pos - joystick_pos;
	float length = offset.length();
	Vector2 direction = offset.normalized();

	float clampzone_radius = joystick_size * 0.5f * clampzone_ratio;

	if (joystick_mode == JOYSTICK_FOLLOWING && length > clampzone_radius) {
		if (has_point(p_pos)) {
			joystick_pos = p_pos - direction * clampzone_radius;
		}
	}

	if (length > clampzone_radius) {
		length = clampzone_radius;
		offset = direction * length;
	}

	tip_pos = joystick_pos + offset;

	bool was_pressed = has_input;
	raw_input_vector = offset / clampzone_radius;
	if (length > deadzone_ratio * clampzone_radius) {
		has_input = true;
		float scaled = Math::inverse_lerp(deadzone_ratio * clampzone_radius, clampzone_radius, length);
		input_vector = direction * scaled;
	} else {
		has_input = false;
		input_vector = Vector2();
	}

	if (!is_flick_canceled && was_pressed && !has_input) {
		is_flick_canceled = true;
		emit_signal(SNAME("flick_canceled"));
	} else if (is_flick_canceled && !was_pressed && has_input) {
		is_flick_canceled = false;
	}

	_handle_input_actions();

	queue_redraw();
}

void VirtualJoystick::_handle_input_actions() {
	Input *input = Input::get_singleton();

	if (raw_input_vector.x >= 0.0f && input->is_action_pressed(action_left)) {
		input->action_release(action_left);
	}
	if (raw_input_vector.x <= 0.0f && input->is_action_pressed(action_right)) {
		input->action_release(action_right);
	}
	if (raw_input_vector.y >= 0.0f && input->is_action_pressed(action_up)) {
		input->action_release(action_up);
	}
	if (raw_input_vector.y <= 0.0f && input->is_action_pressed(action_down)) {
		input->action_release(action_down);
	}

	if (raw_input_vector.x < 0.0f) {
		input->action_press(action_left, -raw_input_vector.x);
	} else if (raw_input_vector.x > 0.0f) {
		input->action_press(action_right, raw_input_vector.x);
	}
	if (raw_input_vector.y < 0.0f) {
		input->action_press(action_up, -raw_input_vector.y);
	} else if (raw_input_vector.y > 0.0f) {
		input->action_press(action_down, raw_input_vector.y);
	}
}

void VirtualJoystick::_reset() {
	is_pressed = false;
	has_input = false;
	has_moved = false;
	raw_input_vector = Vector2();
	input_vector = Vector2();
	is_flick_canceled = false;
	touch_index = -1;
	joystick_pos = get_size() * initial_offset_ratio;
	tip_pos = joystick_pos;

	Input *input = Input::get_singleton();
	for (const StringName &action : { action_left, action_right, action_down, action_up }) {
		if (input->is_action_pressed(action)) {
			input->action_release(action);
		}
	}

	queue_redraw();
}

void VirtualJoystick::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_joystick_mode", "mode"), &VirtualJoystick::set_joystick_mode);
	ClassDB::bind_method(D_METHOD("get_joystick_mode"), &VirtualJoystick::get_joystick_mode);

	ClassDB::bind_method(D_METHOD("set_joystick_size", "size"), &VirtualJoystick::set_joystick_size);
	ClassDB::bind_method(D_METHOD("get_joystick_size"), &VirtualJoystick::get_joystick_size);

	ClassDB::bind_method(D_METHOD("set_tip_size", "size"), &VirtualJoystick::set_tip_size);
	ClassDB::bind_method(D_METHOD("get_tip_size"), &VirtualJoystick::get_tip_size);

	ClassDB::bind_method(D_METHOD("set_deadzone_ratio", "ratio"), &VirtualJoystick::set_deadzone_ratio);
	ClassDB::bind_method(D_METHOD("get_deadzone_ratio"), &VirtualJoystick::get_deadzone_ratio);

	ClassDB::bind_method(D_METHOD("set_clampzone_ratio", "ratio"), &VirtualJoystick::set_clampzone_ratio);
	ClassDB::bind_method(D_METHOD("get_clampzone_ratio"), &VirtualJoystick::get_clampzone_ratio);

	ClassDB::bind_method(D_METHOD("set_initial_offset_ratio", "ratio"), &VirtualJoystick::set_initial_offset_ratio);
	ClassDB::bind_method(D_METHOD("get_initial_offset_ratio"), &VirtualJoystick::get_initial_offset_ratio);

	ClassDB::bind_method(D_METHOD("set_action_left", "action"), &VirtualJoystick::set_action_left);
	ClassDB::bind_method(D_METHOD("get_action_left"), &VirtualJoystick::get_action_left);

	ClassDB::bind_method(D_METHOD("set_action_right", "action"), &VirtualJoystick::set_action_right);
	ClassDB::bind_method(D_METHOD("get_action_right"), &VirtualJoystick::get_action_right);

	ClassDB::bind_method(D_METHOD("set_action_up", "action"), &VirtualJoystick::set_action_up);
	ClassDB::bind_method(D_METHOD("get_action_up"), &VirtualJoystick::get_action_up);

	ClassDB::bind_method(D_METHOD("set_action_down", "action"), &VirtualJoystick::set_action_down);
	ClassDB::bind_method(D_METHOD("get_action_down"), &VirtualJoystick::get_action_down);

	ClassDB::bind_method(D_METHOD("set_visibility_mode", "mode"), &VirtualJoystick::set_visibility_mode);
	ClassDB::bind_method(D_METHOD("get_visibility_mode"), &VirtualJoystick::get_visibility_mode);

	ClassDB::bind_method(D_METHOD("set_joystick_texture", "texture"), &VirtualJoystick::set_joystick_texture);
	ClassDB::bind_method(D_METHOD("get_joystick_texture"), &VirtualJoystick::get_joystick_texture);
	ClassDB::bind_method(D_METHOD("set_tip_texture", "texture"), &VirtualJoystick::set_tip_texture);
	ClassDB::bind_method(D_METHOD("get_tip_texture"), &VirtualJoystick::get_tip_texture);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "joystick_mode", PROPERTY_HINT_ENUM, "Fixed,Dynamic,Following"), "set_joystick_mode", "get_joystick_mode");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "joystick_size", PROPERTY_HINT_RANGE, "10,500,1"), "set_joystick_size", "get_joystick_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tip_size", PROPERTY_HINT_RANGE, "5,250,1"), "set_tip_size", "get_tip_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "deadzone_ratio", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_deadzone_ratio", "get_deadzone_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "clampzone_ratio", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_clampzone_ratio", "get_clampzone_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "initial_offset_ratio", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_initial_offset_ratio", "get_initial_offset_ratio");

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "action_left", PROPERTY_HINT_INPUT_NAME, "show_builtin,loose_mode"), "set_action_left", "get_action_left");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "action_right", PROPERTY_HINT_INPUT_NAME, "show_builtin,loose_mode"), "set_action_right", "get_action_right");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "action_up", PROPERTY_HINT_INPUT_NAME, "show_builtin,loose_mode"), "set_action_up", "get_action_up");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "action_down", PROPERTY_HINT_INPUT_NAME, "show_builtin,loose_mode"), "set_action_down", "get_action_down");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "visibility_mode", PROPERTY_HINT_ENUM, "Always,Touchscreen Only,When Touched"), "set_visibility_mode", "get_visibility_mode");

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "joystick_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_joystick_texture", "get_joystick_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tip_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_tip_texture", "get_tip_texture");

	ADD_SIGNAL(MethodInfo("pressed"));
	ADD_SIGNAL(MethodInfo("tapped"));
	ADD_SIGNAL(MethodInfo("released", PropertyInfo(Variant::VECTOR2, "input_vector")));
	ADD_SIGNAL(MethodInfo("flicked", PropertyInfo(Variant::VECTOR2, "input_vector")));
	ADD_SIGNAL(MethodInfo("flick_canceled"));

	BIND_ENUM_CONSTANT(JOYSTICK_FIXED);
	BIND_ENUM_CONSTANT(JOYSTICK_DYNAMIC);
	BIND_ENUM_CONSTANT(JOYSTICK_FOLLOWING);
	BIND_ENUM_CONSTANT(VISIBILITY_ALWAYS);
	BIND_ENUM_CONSTANT(VISIBILITY_TOUCHSCREEN_ONLY);
	BIND_ENUM_CONSTANT(VISIBILITY_WHEN_TOUCHED);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, VirtualJoystick, ring_normal_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, VirtualJoystick, tip_normal_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, VirtualJoystick, ring_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, VirtualJoystick, tip_pressed_color);
}

void VirtualJoystick::set_joystick_size(float p_size) {
	if (joystick_size == p_size) {
		return;
	}
	joystick_size = p_size;
	_reset();
}

float VirtualJoystick::get_joystick_size() const {
	return joystick_size;
}

void VirtualJoystick::set_tip_size(float p_size) {
	if (tip_size == p_size) {
		return;
	}
	tip_size = p_size;
	_reset();
}

float VirtualJoystick::get_tip_size() const {
	return tip_size;
}

void VirtualJoystick::set_deadzone_ratio(float p_ratio) {
	deadzone_ratio = p_ratio;
}

float VirtualJoystick::get_deadzone_ratio() const {
	return deadzone_ratio;
}

void VirtualJoystick::set_clampzone_ratio(float p_ratio) {
	clampzone_ratio = p_ratio;
}

float VirtualJoystick::get_clampzone_ratio() const {
	return clampzone_ratio;
}

void VirtualJoystick::set_initial_offset_ratio(const Vector2 &p_ratio) {
	if (initial_offset_ratio == p_ratio) {
		return;
	}
	initial_offset_ratio = p_ratio;
	_reset();
}

Vector2 VirtualJoystick::get_initial_offset_ratio() const {
	return initial_offset_ratio;
}

void VirtualJoystick::set_joystick_mode(JoystickMode p_mode) {
	joystick_mode = p_mode;
}

VirtualJoystick::JoystickMode VirtualJoystick::get_joystick_mode() const {
	return joystick_mode;
}

void VirtualJoystick::set_action_left(const StringName &p_action) {
	action_left = p_action;
}

StringName VirtualJoystick::get_action_left() const {
	return action_left;
}

void VirtualJoystick::set_action_right(const StringName &p_action) {
	action_right = p_action;
}

StringName VirtualJoystick::get_action_right() const {
	return action_right;
}

void VirtualJoystick::set_action_up(const StringName &p_action) {
	action_up = p_action;
}

StringName VirtualJoystick::get_action_up() const {
	return action_up;
}

void VirtualJoystick::set_action_down(const StringName &p_action) {
	action_down = p_action;
}

StringName VirtualJoystick::get_action_down() const {
	return action_down;
}

void VirtualJoystick::set_visibility_mode(VisibilityMode p_mode) {
	visibility = p_mode;
}

VirtualJoystick::VisibilityMode VirtualJoystick::get_visibility_mode() const {
	return visibility;
}

void VirtualJoystick::set_joystick_texture(const Ref<Texture2D> &p_texture) {
	if (joystick_texture == p_texture) {
		return;
	}
	joystick_texture = p_texture;
	queue_redraw();
}

Ref<Texture2D> VirtualJoystick::get_joystick_texture() const {
	return joystick_texture;
}

void VirtualJoystick::set_tip_texture(const Ref<Texture2D> &p_texture) {
	if (tip_texture == p_texture) {
		return;
	}
	tip_texture = p_texture;
	queue_redraw();
}

Ref<Texture2D> VirtualJoystick::get_tip_texture() const {
	return tip_texture;
}

VirtualJoystick::VirtualJoystick() {
}
