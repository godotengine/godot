/**************************************************************************/
/*  input_event_screen_drag.hpp                                           */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/input_event_from_window.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class InputEventScreenDrag : public InputEventFromWindow {
	GDEXTENSION_CLASS(InputEventScreenDrag, InputEventFromWindow)

public:
	void set_index(int32_t p_index);
	int32_t get_index() const;
	void set_tilt(const Vector2 &p_tilt);
	Vector2 get_tilt() const;
	void set_pressure(float p_pressure);
	float get_pressure() const;
	void set_pen_inverted(bool p_pen_inverted);
	bool get_pen_inverted() const;
	void set_position(const Vector2 &p_position);
	Vector2 get_position() const;
	void set_relative(const Vector2 &p_relative);
	Vector2 get_relative() const;
	void set_screen_relative(const Vector2 &p_relative);
	Vector2 get_screen_relative() const;
	void set_velocity(const Vector2 &p_velocity);
	Vector2 get_velocity() const;
	void set_screen_velocity(const Vector2 &p_velocity);
	Vector2 get_screen_velocity() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		InputEventFromWindow::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

