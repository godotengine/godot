/**************************************************************************/
/*  style_box_flat.hpp                                                    */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/style_box.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class StyleBoxFlat : public StyleBox {
	GDEXTENSION_CLASS(StyleBoxFlat, StyleBox)

public:
	void set_bg_color(const Color &p_color);
	Color get_bg_color() const;
	void set_border_color(const Color &p_color);
	Color get_border_color() const;
	void set_border_width_all(int32_t p_width);
	int32_t get_border_width_min() const;
	void set_border_width(Side p_margin, int32_t p_width);
	int32_t get_border_width(Side p_margin) const;
	void set_border_blend(bool p_blend);
	bool get_border_blend() const;
	void set_corner_radius_all(int32_t p_radius);
	void set_corner_radius(Corner p_corner, int32_t p_radius);
	int32_t get_corner_radius(Corner p_corner) const;
	void set_expand_margin(Side p_margin, float p_size);
	void set_expand_margin_all(float p_size);
	float get_expand_margin(Side p_margin) const;
	void set_draw_center(bool p_draw_center);
	bool is_draw_center_enabled() const;
	void set_skew(const Vector2 &p_skew);
	Vector2 get_skew() const;
	void set_shadow_color(const Color &p_color);
	Color get_shadow_color() const;
	void set_shadow_size(int32_t p_size);
	int32_t get_shadow_size() const;
	void set_shadow_offset(const Vector2 &p_offset);
	Vector2 get_shadow_offset() const;
	void set_anti_aliased(bool p_anti_aliased);
	bool is_anti_aliased() const;
	void set_aa_size(float p_size);
	float get_aa_size() const;
	void set_corner_detail(int32_t p_detail);
	int32_t get_corner_detail() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		StyleBox::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

