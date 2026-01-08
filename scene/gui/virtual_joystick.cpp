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
#include "core/config/engine.h"
#include "core/input/input.h"
#include "scene/resources/style_box.h"
#include "scene/resources/texture.h"
#include "scene/theme/theme_db.h"

void VirtualJoystick::_notification(int p_what) {
	VirtualDevice::_notification(p_what);
	switch (p_what) {
		case NOTIFICATION_READY: {
			original_base_pos = get_size() / 2.0;
			_reset_joystick();
		} break;
		case NOTIFICATION_DRAW: {
			_draw_joystick();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme_item_cache();
			queue_redraw();
		} break;
		case NOTIFICATION_RESIZED: {
			if (!is_pressed()) {
				original_base_pos = get_size() / 2.0;
				_reset_joystick();
			}
			queue_redraw();
		} break;
	}
}

void VirtualJoystick::_draw_joystick() {
	if (joystick_mode == JOYSTICK_MODE_DYNAMIC && !is_pressed() && !Engine::get_singleton()->is_editor_hint()) {
		// Don't draw dynamic joystick if not touched (runtime)
		return;
	}

	Size2 s = get_size();
	float radius = MIN(s.x, s.y) / 2.0;

	if (joystick_mode == JOYSTICK_MODE_DYNAMIC) {
		radius = clamp_zone_size;
	}

	// In FIXED mode, we draw at the center of the control
	Vector2 drawing_base = (joystick_mode == JOYSTICK_MODE_FIXED) ? (s / 2.0) : base_pos;
	Vector2 drawing_tip = (joystick_mode == JOYSTICK_MODE_FIXED && !is_pressed()) ? drawing_base : tip_pos;

	// Resolve Textures
	Ref<Texture2D> b_tex = base_texture.is_valid() ? base_texture : theme_cache.base_texture_theme;
	Ref<Texture2D> t_tex = tip_texture.is_valid() ? tip_texture : theme_cache.tip_texture_theme;

	// Resolve Colors (property multiplies theme color if both set, or just use property)
	Color b_col = base_color * theme_cache.base_color_theme;
	Color t_col = tip_color * theme_cache.tip_color_theme;

	// Draw Base
	if (b_tex.is_valid()) {
		Size2 base_size = Size2(radius * 2, radius * 2);
		draw_texture_rect(b_tex, Rect2(drawing_base - base_size / 2, base_size), false, b_col);
	} else {
		draw_circle(drawing_base, radius, b_col);
	}

	// Draw Tip
	if (t_tex.is_valid()) {
		float tip_scale = 0.4;
		Size2 tip_size = Size2(radius * 2 * tip_scale, radius * 2 * tip_scale);
		draw_texture_rect(t_tex, Rect2(drawing_tip - tip_size / 2, tip_size), false, t_col);
	} else {
		draw_circle(drawing_tip, radius * 0.4, t_col);
	}
}

void VirtualJoystick::_on_touch_down(int p_index, const Vector2 &p_pos) {
	if (joystick_mode == JOYSTICK_MODE_DYNAMIC) {
		base_pos = p_pos;
		tip_pos = p_pos;
	} else {
		tip_pos = p_pos;
	}
	_update_input_vector();
	queue_redraw();
}

void VirtualJoystick::_on_touch_up(int p_index, const Vector2 &p_pos) {
	_reset_joystick();
	queue_redraw();
}

void VirtualJoystick::_on_drag(int p_index, const Vector2 &p_pos, const Vector2 &p_relative) {
	tip_pos = p_pos;

	// Dynamics are handled by VirtualJoystickDynamic area

	_update_input_vector();
	queue_redraw();
}

void VirtualJoystick::_update_input_vector() {
	Size2 s = get_size();
	float radius = MIN(s.x, s.y) / 2.0;
	if (joystick_mode == JOYSTICK_MODE_DYNAMIC) {
		radius = clamp_zone_size;
	}

	Vector2 current_base = (joystick_mode == JOYSTICK_MODE_FIXED) ? (s / 2.0) : base_pos;
	Vector2 diff = tip_pos - current_base;
	float dist = diff.length();

	if (dist <= deadzone_size) {
		input_vector = Vector2();
	} else {
		// Clamp visual tip
		if (dist > radius) {
			diff = diff.normalized() * radius;
			tip_pos = current_base + diff; // Update visual position too
		}
		input_vector = diff / radius;
	}

	// Emit Input Event
	int axis_x = (joystick_hand == JOYSTICK_HAND_LEFT) ? 0 : 2;
	int axis_y = (joystick_hand == JOYSTICK_HAND_LEFT) ? 1 : 3;

	// X Axis
	Ref<InputEventVirtualMotion> ie_x;
	ie_x.instantiate();
	ie_x->set_device(get_device());
	ie_x->set_axis(axis_x);
	ie_x->set_axis_value(input_vector.x);
	Input::get_singleton()->parse_input_event(ie_x);

	// Y Axis
	Ref<InputEventVirtualMotion> ie_y;
	ie_y.instantiate();
	ie_y->set_device(get_device());
	ie_y->set_axis(axis_y);
	ie_y->set_axis_value(input_vector.y);
	Input::get_singleton()->parse_input_event(ie_y);
}

void VirtualJoystick::_reset_joystick() {
	input_vector = Vector2();
	base_pos = original_base_pos;
	tip_pos = original_base_pos;

	int axis_x = (joystick_hand == JOYSTICK_HAND_LEFT) ? 0 : 2;
	int axis_y = (joystick_hand == JOYSTICK_HAND_LEFT) ? 1 : 3;

	// Reset inputs to 0
	Ref<InputEventVirtualMotion> ie_x;
	ie_x.instantiate();
	ie_x->set_device(get_device());
	ie_x->set_axis(axis_x);
	ie_x->set_axis_value(0.0);
	Input::get_singleton()->parse_input_event(ie_x);

	Ref<InputEventVirtualMotion> ie_y;
	ie_y.instantiate();
	ie_y->set_device(get_device());
	ie_y->set_axis(axis_y);
	ie_y->set_axis_value(0.0);
	Input::get_singleton()->parse_input_event(ie_y);
}

void VirtualJoystick::pressed_state_changed() {
	if (!is_pressed()) {
		_reset_joystick();
	}
}

void VirtualJoystick::set_deadzone_size(float p_size) {
	deadzone_size = p_size;
}

float VirtualJoystick::get_deadzone_size() const {
	return deadzone_size;
}

void VirtualJoystick::set_clamp_zone_size(float p_size) {
	clamp_zone_size = p_size;
	queue_redraw();
}

float VirtualJoystick::get_clamp_zone_size() const {
	return clamp_zone_size;
}

Vector2 VirtualJoystick::get_output() const {
	return input_vector;
}

Size2 VirtualJoystick::get_minimum_size() const {
	return Size2(20, 20);
}

void VirtualJoystick::set_joystick_mode(JoystickMode p_mode) {
	joystick_mode = p_mode;
	queue_redraw();
}

void VirtualJoystick::set_joystick_hand(JoystickHand p_hand) {
	joystick_hand = p_hand;
}

void VirtualJoystick::set_base_texture(const Ref<Texture2D> &p_texture) {
	base_texture = p_texture;
	queue_redraw();
}

Ref<Texture2D> VirtualJoystick::get_base_texture() const {
	return base_texture;
}

void VirtualJoystick::set_tip_texture(const Ref<Texture2D> &p_texture) {
	tip_texture = p_texture;
	queue_redraw();
}

Ref<Texture2D> VirtualJoystick::get_tip_texture() const {
	return tip_texture;
}

void VirtualJoystick::set_base_color(const Color &p_color) {
	base_color = p_color;
	queue_redraw();
}

Color VirtualJoystick::get_base_color() const {
	return base_color;
}

void VirtualJoystick::set_tip_color(const Color &p_color) {
	tip_color = p_color;
	queue_redraw();
}

Color VirtualJoystick::get_tip_color() const {
	return tip_color;
}

void VirtualJoystick::_update_theme_item_cache() {
	VirtualDevice::_update_theme_item_cache();
}

VirtualJoystick::VirtualJoystick() {
	set_mouse_filter(MOUSE_FILTER_PASS); // Let users click
}

void VirtualJoystick::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_deadzone_size", "deadzone_size"), &VirtualJoystick::set_deadzone_size);
	ClassDB::bind_method(D_METHOD("get_deadzone_size"), &VirtualJoystick::get_deadzone_size);

	ClassDB::bind_method(D_METHOD("set_clamp_zone_size", "size"), &VirtualJoystick::set_clamp_zone_size);
	ClassDB::bind_method(D_METHOD("get_clamp_zone_size"), &VirtualJoystick::get_clamp_zone_size);

	ClassDB::bind_method(D_METHOD("get_output"), &VirtualJoystick::get_output);

	ClassDB::bind_method(D_METHOD("set_joystick_mode", "mode"), &VirtualJoystick::set_joystick_mode);
	ClassDB::bind_method(D_METHOD("get_joystick_mode"), &VirtualJoystick::get_joystick_mode);

	ClassDB::bind_method(D_METHOD("set_joystick_hand", "hand"), &VirtualJoystick::set_joystick_hand);
	ClassDB::bind_method(D_METHOD("get_joystick_hand"), &VirtualJoystick::get_joystick_hand);

	ClassDB::bind_method(D_METHOD("set_base_texture", "texture"), &VirtualJoystick::set_base_texture);
	ClassDB::bind_method(D_METHOD("get_base_texture"), &VirtualJoystick::get_base_texture);

	ClassDB::bind_method(D_METHOD("set_tip_texture", "texture"), &VirtualJoystick::set_tip_texture);
	ClassDB::bind_method(D_METHOD("get_tip_texture"), &VirtualJoystick::get_tip_texture);

	ClassDB::bind_method(D_METHOD("set_base_color", "color"), &VirtualJoystick::set_base_color);
	ClassDB::bind_method(D_METHOD("get_base_color"), &VirtualJoystick::get_base_color);

	ClassDB::bind_method(D_METHOD("set_tip_color", "color"), &VirtualJoystick::set_tip_color);
	ClassDB::bind_method(D_METHOD("get_tip_color"), &VirtualJoystick::get_tip_color);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "deadzone_size"), "set_deadzone_size", "get_deadzone_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "clamp_zone_size"), "set_clamp_zone_size", "get_clamp_zone_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "joystick_hand", PROPERTY_HINT_ENUM, "Left,Right"), "set_joystick_hand", "get_joystick_hand");

	ADD_GROUP("Visuals", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "base_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_base_texture", "get_base_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tip_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_tip_texture", "get_tip_texture");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "base_color"), "set_base_color", "get_base_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "tip_color"), "set_tip_color", "get_tip_color");

	BIND_ENUM_CONSTANT(JOYSTICK_MODE_FIXED);
	BIND_ENUM_CONSTANT(JOYSTICK_MODE_DYNAMIC);

	BIND_ENUM_CONSTANT(JOYSTICK_HAND_LEFT);
	BIND_ENUM_CONSTANT(JOYSTICK_HAND_RIGHT);

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, VirtualJoystick, base_style);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, VirtualJoystick, tip_style);
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, VirtualJoystick, base_texture_theme, "base");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, VirtualJoystick, tip_texture_theme, "tip");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, VirtualJoystick, base_color_theme, "base_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, VirtualJoystick, tip_color_theme, "tip_color");
}
