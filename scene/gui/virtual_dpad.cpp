/**************************************************************************/
/*  virtual_dpad.cpp                                                      */
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

#include "virtual_dpad.h"
#include "core/input/input.h"
#include "core/math/math_funcs.h"
#include "scene/theme/theme_db.h"

void VirtualDPad::_update_theme_item_cache() {
	VirtualDevice::_update_theme_item_cache();
}

void VirtualDPad::_notification(int p_what) {
	VirtualDevice::_notification(p_what);
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (texture.is_valid()) {
				draw_texture_rect(texture, Rect2(Point2(), get_size()), false);
				return;
			}

			// Procedural Cross with Rectangular Arms
			Size2 s = get_size();
			Vector2 c = s / 2.0;
			float radius = MIN(s.width, s.height) / 2.0;
			float thickness = radius * 0.7; // Width of the arms

			Rect2 rect_center(c.x - thickness / 2.0, c.y - thickness / 2.0, thickness, thickness);
			Rect2 rect_up(c.x - thickness / 2.0, c.y - radius, thickness, radius - thickness / 2.0);
			Rect2 rect_down(c.x - thickness / 2.0, c.y + thickness / 2.0, thickness, radius - thickness / 2.0);
			Rect2 rect_left(c.x - radius, c.y - thickness / 2.0, radius - thickness / 2.0, thickness);
			Rect2 rect_right(c.x + thickness / 2.0, c.y - thickness / 2.0, radius - thickness / 2.0, thickness);

			Color base_color = theme_cache.normal_color;
			if (is_pressed()) {
				base_color = theme_cache.active_color;
			}

			// Draw Center
			draw_rect(rect_center, base_color);

			// Draw Arms
			draw_rect(rect_up, (current_direction == DIR_UP) ? theme_cache.highlight_color : base_color);
			draw_rect(rect_down, (current_direction == DIR_DOWN) ? theme_cache.highlight_color : base_color);
			draw_rect(rect_left, (current_direction == DIR_LEFT) ? theme_cache.highlight_color : base_color);
			draw_rect(rect_right, (current_direction == DIR_RIGHT) ? theme_cache.highlight_color : base_color);
		} break;
		case NOTIFICATION_RESIZED: {
			queue_redraw();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme_item_cache();
			queue_redraw();
		} break;
	}
}

// ... (methods) ...
void VirtualDPad::_on_touch_down(int p_index, const Vector2 &p_pos) {
	_update_dpad(p_pos);
}

void VirtualDPad::_on_touch_up(int p_index, const Vector2 &p_pos) {
	if (current_direction != DIR_NONE) {
		_release_direction(current_direction);
		current_direction = DIR_NONE;
		queue_redraw();
	}
}

void VirtualDPad::pressed_state_changed() {
	if (!is_pressed()) {
		if (current_direction != DIR_NONE) {
			_release_direction(current_direction);
			current_direction = DIR_NONE;
			queue_redraw();
		}
	}
}

void VirtualDPad::_on_drag(int p_index, const Vector2 &p_pos, const Vector2 &p_relative) {
	_update_dpad(p_pos);
}

void VirtualDPad::_update_dpad(const Vector2 &p_pos) {
	Vector2 center = get_size() / 2.0;
	Vector2 diff = p_pos - center;

	float angle = diff.angle();

	DPadDirection new_dir = DIR_NONE;

	// Simple sector logic (45 deg sectors)
	if (diff.length() < deadzone_size) {
		new_dir = DIR_NONE; // Dead center
	} else {
		if (angle < -Math::PI * 0.75 || angle > Math::PI * 0.75) {
			new_dir = DIR_LEFT;
		} else if (angle > -Math::PI * 0.75 && angle < -Math::PI * 0.25) {
			new_dir = DIR_UP;
		} else if (angle > -Math::PI * 0.25 && angle < Math::PI * 0.25) {
			new_dir = DIR_RIGHT;
		} else {
			new_dir = DIR_DOWN;
		}
	}

	if (new_dir != current_direction) {
		if (current_direction != DIR_NONE) {
			_release_direction(current_direction);
		}
		if (new_dir != DIR_NONE) {
			_press_direction(new_dir);
		}
		current_direction = new_dir;
		queue_redraw();
	}
}

void VirtualDPad::_press_direction(DPadDirection p_dir) {
	int btn = -1;
	switch (p_dir) {
		case DIR_UP: btn = up_button_index; break;
		case DIR_DOWN: btn = down_button_index; break;
		case DIR_LEFT: btn = left_button_index; break;
		case DIR_RIGHT: btn = right_button_index; break;
		default: break;
	}

	if (btn != -1) {
		Ref<InputEventVirtualButton> ie;
		ie.instantiate();
		ie->set_device(get_device());
		ie->set_button_index(btn);
		ie->set_pressed(true);
		ie->set_pressure(1.0);
		Input::get_singleton()->parse_input_event(ie);
	}
}

void VirtualDPad::_release_direction(DPadDirection p_dir) {
	int btn = -1;
	switch (p_dir) {
		case DIR_UP: btn = up_button_index; break;
		case DIR_DOWN: btn = down_button_index; break;
		case DIR_LEFT: btn = left_button_index; break;
		case DIR_RIGHT: btn = right_button_index; break;
		default: break;
	}

	if (btn != -1) {
		Ref<InputEventVirtualButton> ie;
		ie.instantiate();
		ie->set_device(get_device());
		ie->set_button_index(btn);
		ie->set_pressed(false);
		ie->set_pressure(0.0);
		Input::get_singleton()->parse_input_event(ie);
	}
}

void VirtualDPad::set_texture(const Ref<Texture2D> &p_texture) {
	texture = p_texture;
	update_minimum_size();
	queue_redraw();
}

Ref<Texture2D> VirtualDPad::get_texture() const {
	return texture;
}

void VirtualDPad::set_deadzone_size(float p_size) {
	deadzone_size = p_size;
	queue_redraw();
}

float VirtualDPad::get_deadzone_size() const {
	return deadzone_size;
}

void VirtualDPad::set_up_button_index(int p_index) {
	up_button_index = p_index;
}

int VirtualDPad::get_up_button_index() const {
	return up_button_index;
}

void VirtualDPad::set_down_button_index(int p_index) {
	down_button_index = p_index;
}

int VirtualDPad::get_down_button_index() const {
	return down_button_index;
}

void VirtualDPad::set_left_button_index(int p_index) {
	left_button_index = p_index;
}

int VirtualDPad::get_left_button_index() const {
	return left_button_index;
}

void VirtualDPad::set_right_button_index(int p_index) {
	right_button_index = p_index;
}

int VirtualDPad::get_right_button_index() const {
	return right_button_index;
}

Size2 VirtualDPad::get_minimum_size() const {
	return Size2(20, 20);
}

VirtualDPad::VirtualDPad() {
	up_button_index = 12;
	down_button_index = 13;
	left_button_index = 14;
	right_button_index = 15;
}

void VirtualDPad::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &VirtualDPad::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &VirtualDPad::get_texture);
	ClassDB::bind_method(D_METHOD("set_deadzone_size", "size"), &VirtualDPad::set_deadzone_size);
	ClassDB::bind_method(D_METHOD("get_deadzone_size"), &VirtualDPad::get_deadzone_size);

	ClassDB::bind_method(D_METHOD("set_up_button_index", "index"), &VirtualDPad::set_up_button_index);
	ClassDB::bind_method(D_METHOD("get_up_button_index"), &VirtualDPad::get_up_button_index);
	ClassDB::bind_method(D_METHOD("set_down_button_index", "index"), &VirtualDPad::set_down_button_index);
	ClassDB::bind_method(D_METHOD("get_down_button_index"), &VirtualDPad::get_down_button_index);
	ClassDB::bind_method(D_METHOD("set_left_button_index", "index"), &VirtualDPad::set_left_button_index);
	ClassDB::bind_method(D_METHOD("get_left_button_index"), &VirtualDPad::get_left_button_index);
	ClassDB::bind_method(D_METHOD("set_right_button_index", "index"), &VirtualDPad::set_right_button_index);
	ClassDB::bind_method(D_METHOD("get_right_button_index"), &VirtualDPad::get_right_button_index);


	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "deadzone_size"), "set_deadzone_size", "get_deadzone_size");

	ADD_GROUP("Button Indices", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "up_index"), "set_up_button_index", "get_up_button_index");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "down_index"), "set_down_button_index", "get_down_button_index");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "left_index"), "set_left_button_index", "get_left_button_index");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "right_index"), "set_right_button_index", "get_right_button_index");

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, VirtualDPad, normal_color, "normal_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, VirtualDPad, active_color, "active_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, VirtualDPad, highlight_color, "highlight_color");
}
