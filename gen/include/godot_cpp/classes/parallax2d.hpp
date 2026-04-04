/**************************************************************************/
/*  parallax2d.hpp                                                        */
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

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Parallax2D : public Node2D {
	GDEXTENSION_CLASS(Parallax2D, Node2D)

public:
	void set_scroll_scale(const Vector2 &p_scale);
	Vector2 get_scroll_scale() const;
	void set_repeat_size(const Vector2 &p_repeat_size);
	Vector2 get_repeat_size() const;
	void set_repeat_times(int32_t p_repeat_times);
	int32_t get_repeat_times() const;
	void set_autoscroll(const Vector2 &p_autoscroll);
	Vector2 get_autoscroll() const;
	void set_scroll_offset(const Vector2 &p_offset);
	Vector2 get_scroll_offset() const;
	void set_screen_offset(const Vector2 &p_offset);
	Vector2 get_screen_offset() const;
	void set_limit_begin(const Vector2 &p_offset);
	Vector2 get_limit_begin() const;
	void set_limit_end(const Vector2 &p_offset);
	Vector2 get_limit_end() const;
	void set_follow_viewport(bool p_follow);
	bool get_follow_viewport();
	void set_ignore_camera_scroll(bool p_ignore);
	bool is_ignore_camera_scroll();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

