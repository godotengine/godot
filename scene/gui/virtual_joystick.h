/**************************************************************************/
/*  virtual_joystick.h                                                    */
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

#include "scene/gui/control.h"

class VirtualJoystick : public Control {
	GDCLASS(VirtualJoystick, Control);

public:
	enum JoystickMode {
		JOYSTICK_FIXED,
		JOYSTICK_DYNAMIC,
		JOYSTICK_FOLLOWING,
	};

	enum VisibilityMode {
		VISIBILITY_ALWAYS,
		VISIBILITY_TOUCHSCREEN_ONLY,
		VISIBILITY_WHEN_TOUCHED,
	};

private:
	JoystickMode joystick_mode = JOYSTICK_FIXED;
	float joystick_size = 100.0f;
	float tip_size = 50.0f;
	float deadzone_ratio = 0.25f;
	float clampzone_ratio = 1.0f;
	Vector2 initial_offset_ratio = Vector2(0.5, 0.5);
	StringName action_left = "ui_left";
	StringName action_right = "ui_right";
	StringName action_up = "ui_up";
	StringName action_down = "ui_down";
	VisibilityMode visibility = VISIBILITY_ALWAYS;

	struct ThemeCache {
		Color ring_normal_color;
		Color tip_normal_color;
		Color ring_pressed_color;
		Color tip_pressed_color;
	} theme_cache;

	bool is_pressed = false;
	bool has_input = false;
	bool has_moved = false;
	Vector2 raw_input_vector;
	Vector2 input_vector;
	bool is_flick_canceled = false;
	int touch_index = -1;

	Vector2 joystick_pos;
	Vector2 tip_pos;

	Ref<Texture2D> joystick_texture;
	Ref<Texture2D> tip_texture;

	void _update_joystick(const Vector2 &p_pos);
	void _handle_input_actions();
	void _reset();

protected:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_joystick_mode(JoystickMode p_mode);
	JoystickMode get_joystick_mode() const;

	void set_joystick_size(float p_size);
	float get_joystick_size() const;

	void set_tip_size(float p_size);
	float get_tip_size() const;

	void set_deadzone_ratio(float p_ratio);
	float get_deadzone_ratio() const;

	void set_clampzone_ratio(float p_ratio);
	float get_clampzone_ratio() const;

	void set_initial_offset_ratio(const Vector2 &p_ratio);
	Vector2 get_initial_offset_ratio() const;

	void set_action_left(const StringName &p_action);
	StringName get_action_left() const;
	void set_action_right(const StringName &p_action);
	StringName get_action_right() const;
	void set_action_up(const StringName &p_action);
	StringName get_action_up() const;
	void set_action_down(const StringName &p_action);
	StringName get_action_down() const;

	void set_visibility_mode(VisibilityMode p_mode);
	VisibilityMode get_visibility_mode() const;

	void set_joystick_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_joystick_texture() const;
	void set_tip_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_tip_texture() const;

	VirtualJoystick();
};

VARIANT_ENUM_CAST(VirtualJoystick::JoystickMode);
VARIANT_ENUM_CAST(VirtualJoystick::VisibilityMode);
