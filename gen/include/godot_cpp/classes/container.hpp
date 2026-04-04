/**************************************************************************/
/*  container.hpp                                                         */
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

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct Rect2;

class Container : public Control {
	GDEXTENSION_CLASS(Container, Control)

public:
	static const int NOTIFICATION_PRE_SORT_CHILDREN = 50;
	static const int NOTIFICATION_SORT_CHILDREN = 51;

	void queue_sort();
	void fit_child_in_rect(Control *p_child, const Rect2 &p_rect);
	virtual PackedInt32Array _get_allowed_size_flags_horizontal() const;
	virtual PackedInt32Array _get_allowed_size_flags_vertical() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_allowed_size_flags_horizontal), decltype(&T::_get_allowed_size_flags_horizontal)>) {
			BIND_VIRTUAL_METHOD(T, _get_allowed_size_flags_horizontal, 1930428628);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_allowed_size_flags_vertical), decltype(&T::_get_allowed_size_flags_vertical)>) {
			BIND_VIRTUAL_METHOD(T, _get_allowed_size_flags_vertical, 1930428628);
		}
	}

public:
};

} // namespace godot

