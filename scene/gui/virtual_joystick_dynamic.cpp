/**************************************************************************/
/*  virtual_joystick_dynamic.cpp                                          */
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

#include "virtual_joystick_dynamic.h"

#include "core/config/engine.h"

void VirtualJoystickDynamic::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
		case NOTIFICATION_RESIZED: {
			if (visible_by_default) {
				if (default_position == Vector2()) {
					default_position = get_size() / 2.0f;
				}
				base_pos = default_position;
				tip_pos = default_position;
			}
		} break;
	}
	// Delegate to parent for runtime behavior and other notifications
	VirtualJoystick::_notification(p_what);
}

void VirtualJoystickDynamic::_draw_joystick() {
	if (Engine::get_singleton()->is_editor_hint()) {
		// Draw the region debug rectangle in the editor
		draw_rect(Rect2(Point2(), get_size()), Color(1, 1, 1, 0.2), true);
		draw_rect(Rect2(Point2(), get_size()), Color(1, 1, 1, 0.5), false);

		if (!visible_by_default) {
			// In editor, if not visible by default, only draw the debug region
			return;
		}
	}
	// At runtime, we need to override the drawing to use our joystick_size
	if (!Engine::get_singleton()->is_editor_hint() && !is_pressed() && get_joystick_mode() == JOYSTICK_MODE_DYNAMIC && !visible_by_default) {
		// Don't draw dynamic joystick if not touched (runtime) and not visible by default
		return;
	}

	Size2 s = get_size();
	// joystick_size is the diameter, so radius is half
	float radius = joystick_size / 2.0f;

	Vector2 drawing_base = get_base_pos();
	Vector2 drawing_tip = is_pressed() ? get_tip_pos() : drawing_base;

	// Access parent's theme cache through a local variable reference
	auto &cache = VirtualJoystick::theme_cache;

	// Resolve Textures (respecting base class properties)
	Ref<Texture2D> b_tex = get_base_texture().is_valid() ? get_base_texture() : cache.base_texture_theme;
	Ref<Texture2D> t_tex = get_tip_texture().is_valid() ? get_tip_texture() : cache.tip_texture_theme;

	// Resolve Colors (property multiplies theme color if both set, or just use property)
	Color b_col = get_base_color() * cache.base_color_theme;
	Color t_col = get_tip_color() * cache.tip_color_theme;

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

void VirtualJoystickDynamic::set_joystick_size(float p_size) {
	joystick_size = p_size;
	// Sync clamp zone to half the diameter (the radius)
	// This allows tip center to reach the edge, so half the tip extends beyond
	clamp_zone_size = p_size / 2.0f;
	queue_redraw();
}

float VirtualJoystickDynamic::get_joystick_size() const {
	return joystick_size;
}

void VirtualJoystickDynamic::set_visible_by_default(bool p_visible) {
	visible_by_default = p_visible;
	if (visible_by_default && !is_pressed()) {
		if (default_position == Vector2()) {
			default_position = get_size() / 2.0f;
		}
		base_pos = default_position;
		tip_pos = default_position;
	}
	queue_redraw();
}

bool VirtualJoystickDynamic::is_visible_by_default() const {
	return visible_by_default;
}

void VirtualJoystickDynamic::set_default_position(const Vector2 &p_pos) {
	default_position = p_pos;
	if (visible_by_default && !is_pressed()) {
		base_pos = default_position;
		tip_pos = default_position;
	}
	queue_redraw();
}

Vector2 VirtualJoystickDynamic::get_default_position() const {
	return default_position;
}

void VirtualJoystickDynamic::_on_touch_up(int p_index, const Vector2 &p_pos) {
	VirtualJoystick::_on_touch_up(p_index, p_pos);
	if (visible_by_default) {
		Vector2 actual_pos = (default_position == Vector2()) ? (get_size() / 2.0f) : default_position;
		base_pos = actual_pos;
		tip_pos = actual_pos;
		queue_redraw();
	}
}

void VirtualJoystickDynamic::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_joystick_size", "size"), &VirtualJoystickDynamic::set_joystick_size);
	ClassDB::bind_method(D_METHOD("get_joystick_size"), &VirtualJoystickDynamic::get_joystick_size);

	ClassDB::bind_method(D_METHOD("set_visible_by_default", "visible"), &VirtualJoystickDynamic::set_visible_by_default);
	ClassDB::bind_method(D_METHOD("is_visible_by_default"), &VirtualJoystickDynamic::is_visible_by_default);

	ClassDB::bind_method(D_METHOD("set_default_position", "position"), &VirtualJoystickDynamic::set_default_position);
	ClassDB::bind_method(D_METHOD("get_default_position"), &VirtualJoystickDynamic::get_default_position);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "joystick_size"), "set_joystick_size", "get_joystick_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible_by_default"), "set_visible_by_default", "is_visible_by_default");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "default_position"), "set_default_position", "get_default_position");
}

VirtualJoystickDynamic::VirtualJoystickDynamic() {
	// Defaults to Dynamic mode to act as a region that spawns the joystick
	joystick_mode = JOYSTICK_MODE_DYNAMIC;

	// Sync clamp zone with joystick radius (half the diameter)
	clamp_zone_size = joystick_size / 2.0f;

	// Default to a transparent control that covers an area
	// Users should likely set the rect to be large (e.g. half screen)
}
