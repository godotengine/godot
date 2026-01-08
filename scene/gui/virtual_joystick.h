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

#ifndef VIRTUAL_JOYSTICK_H
#define VIRTUAL_JOYSTICK_H

#include "scene/gui/virtual_device.h"

class VirtualJoystick : public VirtualDevice {
	GDCLASS(VirtualJoystick, VirtualDevice);

public:
	enum JoystickMode {
		JOYSTICK_MODE_FIXED,
		JOYSTICK_MODE_DYNAMIC,
	};

	enum JoystickHand {
		JOYSTICK_HAND_LEFT,
		JOYSTICK_HAND_RIGHT,
	};

private:
	// Settings
	float deadzone_size = 10.0f;
	Vector2 input_vector; // Normalized output (-1 to 1)

protected:
	// Visualization
	Vector2 base_pos; // Visual position of the base (bg)
	Vector2 tip_pos; // Visual position of the tip (handle)

	Vector2 original_base_pos; // Where it is initially placed in editor

	Ref<Texture2D> base_texture;
	Ref<Texture2D> tip_texture;
	Color base_color = Color(1, 1, 1, 1);
	Color tip_color = Color(1, 1, 1, 1);
	bool use_theme_colors = true;

protected:
	float clamp_zone_size = 50.0f;
	JoystickMode joystick_mode = JOYSTICK_MODE_FIXED;
	JoystickHand joystick_hand = JOYSTICK_HAND_LEFT;
	struct ThemeCache {
		Ref<StyleBox> base_style;
		Ref<StyleBox> tip_style;
		Ref<Texture2D> base_texture_theme;
		Ref<Texture2D> tip_texture_theme;
		Color base_color_theme = Color(0.4, 0.4, 0.4, 0.6);
		Color tip_color_theme = Color(0.8, 0.8, 0.8, 1.0);
	} theme_cache;

	virtual void _update_theme_item_cache() override;
	void _notification(int p_what);
	static void _bind_methods();

	virtual void _on_touch_down(int p_index, const Vector2 &p_pos) override;
	virtual void _on_touch_up(int p_index, const Vector2 &p_pos) override;
	virtual void _on_drag(int p_index, const Vector2 &p_pos, const Vector2 &p_relative) override;

	virtual bool _should_force_square() const { return true; }
	virtual void _draw_joystick();
	void _update_input_vector();
	void _reset_joystick();

	virtual void pressed_state_changed() override;

public:
	void set_deadzone_size(float p_size);
	float get_deadzone_size() const;

	void set_clamp_zone_size(float p_size);
	float get_clamp_zone_size() const;

	Vector2 get_output() const;

	Vector2 get_base_pos() const { return base_pos; }
	Vector2 get_tip_pos() const { return tip_pos; }

	virtual Size2 get_minimum_size() const override;

	void set_joystick_mode(JoystickMode p_mode);
	JoystickMode get_joystick_mode() const { return joystick_mode; }

	void set_joystick_hand(JoystickHand p_hand);
	JoystickHand get_joystick_hand() const { return joystick_hand; }

	void set_base_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_base_texture() const;

	void set_tip_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_tip_texture() const;

	void set_base_color(const Color &p_color);
	Color get_base_color() const;

	void set_tip_color(const Color &p_color);
	Color get_tip_color() const;

	VirtualJoystick();
};

VARIANT_ENUM_CAST(VirtualJoystick::JoystickMode);
VARIANT_ENUM_CAST(VirtualJoystick::JoystickHand);

#endif // VIRTUAL_JOYSTICK_H
