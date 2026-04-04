/**************************************************************************/
/*  graph_frame.hpp                                                       */
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

#include <godot_cpp/classes/graph_element.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class HBoxContainer;

class GraphFrame : public GraphElement {
	GDEXTENSION_CLASS(GraphFrame, GraphElement)

public:
	void set_title(const String &p_title);
	String get_title() const;
	HBoxContainer *get_titlebar_hbox();
	void set_autoshrink_enabled(bool p_shrink);
	bool is_autoshrink_enabled() const;
	void set_autoshrink_margin(int32_t p_autoshrink_margin);
	int32_t get_autoshrink_margin() const;
	void set_drag_margin(int32_t p_drag_margin);
	int32_t get_drag_margin() const;
	void set_tint_color_enabled(bool p_enable);
	bool is_tint_color_enabled() const;
	void set_tint_color(const Color &p_color);
	Color get_tint_color() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		GraphElement::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

