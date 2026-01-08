/**************************************************************************/
/*  virtual_device.h                                                      */
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

#ifndef VIRTUAL_DEVICE_H
#define VIRTUAL_DEVICE_H

#include "scene/gui/control.h"

class VirtualDevice : public Control {
	GDCLASS(VirtualDevice, Control);

public:
	enum VisibilityMode {
		VISIBILITY_ALWAYS,
		VISIBILITY_TOUCHSCREEN_ONLY,
	};

private:
	int device = 0;
	VisibilityMode visibility_mode = VISIBILITY_ALWAYS;

	// Touch state
	int current_touch_index = -1; // -1 means no touch
	bool pressed = false;
	bool hovering = false;

	// Common properties matching BaseButton for familiarity
	bool disabled = false;
	BitField<MouseButtonMask> action_mask = MouseButtonMask::LEFT;

protected:
	void _notification(int p_what);
	static void _bind_methods();

	// Input handling core
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

	// virtual hooks for subclasses
	virtual void _on_touch_down(int p_index, const Vector2 &p_pos);
	virtual void _on_touch_up(int p_index, const Vector2 &p_pos);
	virtual void _on_drag(int p_index, const Vector2 &p_pos, const Vector2 &p_relative);

	virtual void pressed_state_changed(); // Called when pressed changes

public:
	enum DrawMode {
		DRAW_NORMAL,
		DRAW_PRESSED,
		DRAW_HOVER,
		DRAW_DISABLED,
		DRAW_HOVER_PRESSED,
	};

	DrawMode get_draw_mode() const;

	void set_device(int p_device);
	int get_device() const;

	void set_visibility_mode(VisibilityMode p_mode);
	VisibilityMode get_visibility_mode() const;

	bool is_pressed() const;
	bool is_hovered() const;

	void set_disabled(bool p_disabled);
	bool is_disabled() const;

	VirtualDevice();
};

VARIANT_ENUM_CAST(VirtualDevice::VisibilityMode);
VARIANT_ENUM_CAST(VirtualDevice::DrawMode);

#endif // VIRTUAL_DEVICE_H
