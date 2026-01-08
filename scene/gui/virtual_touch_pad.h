/**************************************************************************/
/*  virtual_touch_pad.h                                                   */
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

#include "scene/gui/virtual_device.h"

class VirtualTouchPad : public VirtualDevice {
	GDCLASS(VirtualTouchPad, VirtualDevice);

public:
	enum TouchPadHand {
		HAND_LEFT,
		HAND_RIGHT,
	};

private:
	float sensitivity = 1.0f;

	// Axis Mapping
	int x_axis = 0;
	int y_axis = 1;

	// Trail visualization
	Vector2 last_pos;
	Vector2 current_pos;

protected:
	void _notification(int p_what);
	static void _bind_methods();

	virtual void _on_touch_down(int p_index, const Vector2 &p_pos) override;
	virtual void _on_touch_up(int p_index, const Vector2 &p_pos) override;
	virtual void _on_drag(int p_index, const Vector2 &p_pos, const Vector2 &p_relative) override;

	void _reset_touchpad();
	virtual void pressed_state_changed() override;
	virtual Size2 get_minimum_size() const override;

	TouchPadHand hand = HAND_LEFT;

public:
	void set_sensitivity(float p_sensitivity);
	float get_sensitivity() const;

	void set_x_axis(int p_axis);
	int get_x_axis() const;

	void set_y_axis(int p_axis);
	int get_y_axis() const;

	void set_hand(TouchPadHand p_hand);
	TouchPadHand get_hand() const { return hand; }

	VirtualTouchPad();
};

VARIANT_ENUM_CAST(VirtualTouchPad::TouchPadHand);
