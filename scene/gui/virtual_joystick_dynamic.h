/**************************************************************************/
/*  virtual_joystick_dynamic.h                                            */
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

#ifndef VIRTUAL_JOYSTICK_DYNAMIC_H
#define VIRTUAL_JOYSTICK_DYNAMIC_H

#include "scene/gui/virtual_joystick.h"

class VirtualJoystickDynamic : public VirtualJoystick {
	GDCLASS(VirtualJoystickDynamic, VirtualJoystick);

	float joystick_size = 50.0f;
	bool visible_by_default = false;
	Vector2 default_position;

protected:
	void _notification(int p_what);
	virtual bool _should_force_square() const override { return false; }
	virtual void _draw_joystick() override;
	static void _bind_methods();

public:
	void set_joystick_size(float p_size);
	float get_joystick_size() const;

	void set_visible_by_default(bool p_visible);
	bool is_visible_by_default() const;

	void set_default_position(const Vector2 &p_pos);
	Vector2 get_default_position() const;

	virtual void _on_touch_up(int p_index, const Vector2 &p_pos) override;

	VirtualJoystickDynamic();
};

#endif // VIRTUAL_JOYSTICK_DYNAMIC_H
