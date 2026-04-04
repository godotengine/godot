/**************************************************************************/
/*  char_fx_transform.hpp                                                 */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CharFXTransform : public RefCounted {
	GDEXTENSION_CLASS(CharFXTransform, RefCounted)

public:
	Transform2D get_transform();
	void set_transform(const Transform2D &p_transform);
	Vector2i get_range();
	void set_range(const Vector2i &p_range);
	double get_elapsed_time();
	void set_elapsed_time(double p_time);
	bool is_visible();
	void set_visibility(bool p_visibility);
	bool is_outline();
	void set_outline(bool p_outline);
	Vector2 get_offset();
	void set_offset(const Vector2 &p_offset);
	Color get_color();
	void set_color(const Color &p_color);
	Dictionary get_environment();
	void set_environment(const Dictionary &p_environment);
	uint32_t get_glyph_index() const;
	void set_glyph_index(uint32_t p_glyph_index);
	int32_t get_relative_index() const;
	void set_relative_index(int32_t p_relative_index);
	uint8_t get_glyph_count() const;
	void set_glyph_count(uint8_t p_glyph_count);
	uint16_t get_glyph_flags() const;
	void set_glyph_flags(uint16_t p_glyph_flags);
	RID get_font() const;
	void set_font(const RID &p_font);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

