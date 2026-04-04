/**************************************************************************/
/*  label_settings.hpp                                                    */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Font;

class LabelSettings : public Resource {
	GDEXTENSION_CLASS(LabelSettings, Resource)

public:
	void set_line_spacing(float p_spacing);
	float get_line_spacing() const;
	void set_paragraph_spacing(float p_spacing);
	float get_paragraph_spacing() const;
	void set_font(const Ref<Font> &p_font);
	Ref<Font> get_font() const;
	void set_font_size(int32_t p_size);
	int32_t get_font_size() const;
	void set_font_color(const Color &p_color);
	Color get_font_color() const;
	void set_outline_size(int32_t p_size);
	int32_t get_outline_size() const;
	void set_outline_color(const Color &p_color);
	Color get_outline_color() const;
	void set_shadow_size(int32_t p_size);
	int32_t get_shadow_size() const;
	void set_shadow_color(const Color &p_color);
	Color get_shadow_color() const;
	void set_shadow_offset(const Vector2 &p_offset);
	Vector2 get_shadow_offset() const;
	int32_t get_stacked_outline_count() const;
	void set_stacked_outline_count(int32_t p_count);
	void add_stacked_outline(int32_t p_index = -1);
	void move_stacked_outline(int32_t p_from_index, int32_t p_to_position);
	void remove_stacked_outline(int32_t p_index);
	void set_stacked_outline_size(int32_t p_index, int32_t p_size);
	int32_t get_stacked_outline_size(int32_t p_index) const;
	void set_stacked_outline_color(int32_t p_index, const Color &p_color);
	Color get_stacked_outline_color(int32_t p_index) const;
	int32_t get_stacked_shadow_count() const;
	void set_stacked_shadow_count(int32_t p_count);
	void add_stacked_shadow(int32_t p_index = -1);
	void move_stacked_shadow(int32_t p_from_index, int32_t p_to_position);
	void remove_stacked_shadow(int32_t p_index);
	void set_stacked_shadow_offset(int32_t p_index, const Vector2 &p_offset);
	Vector2 get_stacked_shadow_offset(int32_t p_index) const;
	void set_stacked_shadow_color(int32_t p_index, const Color &p_color);
	Color get_stacked_shadow_color(int32_t p_index) const;
	void set_stacked_shadow_outline_size(int32_t p_index, int32_t p_size);
	int32_t get_stacked_shadow_outline_size(int32_t p_index) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

