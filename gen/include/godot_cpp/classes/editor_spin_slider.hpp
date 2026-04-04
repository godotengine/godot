/**************************************************************************/
/*  editor_spin_slider.hpp                                                */
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

#include <godot_cpp/classes/range.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class EditorSpinSlider : public Range {
	GDEXTENSION_CLASS(EditorSpinSlider, Range)

public:
	enum ControlState {
		CONTROL_STATE_DEFAULT = 0,
		CONTROL_STATE_PREFER_SLIDER = 1,
		CONTROL_STATE_HIDE = 2,
	};

	void set_label(const String &p_label);
	String get_label() const;
	void set_suffix(const String &p_suffix);
	String get_suffix() const;
	void set_read_only(bool p_read_only);
	bool is_read_only() const;
	void set_flat(bool p_flat);
	bool is_flat() const;
	void set_control_state(EditorSpinSlider::ControlState p_state);
	EditorSpinSlider::ControlState get_control_state() const;
	void set_hide_slider(bool p_hide_slider);
	bool is_hiding_slider() const;
	void set_editing_integer(bool p_editing_integer);
	bool is_editing_integer() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Range::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorSpinSlider::ControlState);

