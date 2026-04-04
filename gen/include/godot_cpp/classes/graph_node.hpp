/**************************************************************************/
/*  graph_node.hpp                                                        */
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
#include <godot_cpp/classes/graph_element.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class HBoxContainer;
struct Vector2i;

class GraphNode : public GraphElement {
	GDEXTENSION_CLASS(GraphNode, GraphElement)

public:
	void set_title(const String &p_title);
	String get_title() const;
	HBoxContainer *get_titlebar_hbox();
	void set_slot(int32_t p_slot_index, bool p_enable_left_port, int32_t p_type_left, const Color &p_color_left, bool p_enable_right_port, int32_t p_type_right, const Color &p_color_right, const Ref<Texture2D> &p_custom_icon_left = nullptr, const Ref<Texture2D> &p_custom_icon_right = nullptr, bool p_draw_stylebox = true);
	void clear_slot(int32_t p_slot_index);
	void clear_all_slots();
	bool is_slot_enabled_left(int32_t p_slot_index) const;
	void set_slot_enabled_left(int32_t p_slot_index, bool p_enable);
	void set_slot_type_left(int32_t p_slot_index, int32_t p_type);
	int32_t get_slot_type_left(int32_t p_slot_index) const;
	void set_slot_color_left(int32_t p_slot_index, const Color &p_color);
	Color get_slot_color_left(int32_t p_slot_index) const;
	void set_slot_custom_icon_left(int32_t p_slot_index, const Ref<Texture2D> &p_custom_icon);
	Ref<Texture2D> get_slot_custom_icon_left(int32_t p_slot_index) const;
	void set_slot_metadata_left(int32_t p_slot_index, const Variant &p_value);
	Variant get_slot_metadata_left(int32_t p_slot_index) const;
	bool is_slot_enabled_right(int32_t p_slot_index) const;
	void set_slot_enabled_right(int32_t p_slot_index, bool p_enable);
	void set_slot_type_right(int32_t p_slot_index, int32_t p_type);
	int32_t get_slot_type_right(int32_t p_slot_index) const;
	void set_slot_color_right(int32_t p_slot_index, const Color &p_color);
	Color get_slot_color_right(int32_t p_slot_index) const;
	void set_slot_custom_icon_right(int32_t p_slot_index, const Ref<Texture2D> &p_custom_icon);
	Ref<Texture2D> get_slot_custom_icon_right(int32_t p_slot_index) const;
	void set_slot_metadata_right(int32_t p_slot_index, const Variant &p_value);
	Variant get_slot_metadata_right(int32_t p_slot_index) const;
	bool is_slot_draw_stylebox(int32_t p_slot_index) const;
	void set_slot_draw_stylebox(int32_t p_slot_index, bool p_enable);
	void set_ignore_invalid_connection_type(bool p_ignore);
	bool is_ignoring_valid_connection_type() const;
	void set_slots_focus_mode(Control::FocusMode p_focus_mode);
	Control::FocusMode get_slots_focus_mode() const;
	int32_t get_input_port_count();
	Vector2 get_input_port_position(int32_t p_port_idx);
	int32_t get_input_port_type(int32_t p_port_idx);
	Color get_input_port_color(int32_t p_port_idx);
	int32_t get_input_port_slot(int32_t p_port_idx);
	int32_t get_output_port_count();
	Vector2 get_output_port_position(int32_t p_port_idx);
	int32_t get_output_port_type(int32_t p_port_idx);
	Color get_output_port_color(int32_t p_port_idx);
	int32_t get_output_port_slot(int32_t p_port_idx);
	virtual void _draw_port(int32_t p_slot_index, const Vector2i &p_position, bool p_left, const Color &p_color);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		GraphElement::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_draw_port), decltype(&T::_draw_port)>) {
			BIND_VIRTUAL_METHOD(T, _draw_port, 93366828);
		}
	}

public:
};

} // namespace godot

