/**************************************************************************/
/*  text_line.hpp                                                         */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Font;
class PackedFloat32Array;

class TextLine : public RefCounted {
	GDEXTENSION_CLASS(TextLine, RefCounted)

public:
	void clear();
	Ref<TextLine> duplicate() const;
	void set_direction(TextServer::Direction p_direction);
	TextServer::Direction get_direction() const;
	TextServer::Direction get_inferred_direction() const;
	void set_orientation(TextServer::Orientation p_orientation);
	TextServer::Orientation get_orientation() const;
	void set_preserve_invalid(bool p_enabled);
	bool get_preserve_invalid() const;
	void set_preserve_control(bool p_enabled);
	bool get_preserve_control() const;
	void set_bidi_override(const Array &p_override);
	bool add_string(const String &p_text, const Ref<Font> &p_font, int32_t p_font_size, const String &p_language = String(), const Variant &p_meta = nullptr);
	bool add_object(const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align = (InlineAlignment)5, int32_t p_length = 1, float p_baseline = 0.0);
	bool resize_object(const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align = (InlineAlignment)5, float p_baseline = 0.0);
	bool has_object(const Variant &p_key) const;
	void set_width(float p_width);
	float get_width() const;
	void set_horizontal_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_horizontal_alignment() const;
	void tab_align(const PackedFloat32Array &p_tab_stops);
	void set_flags(BitField<TextServer::JustificationFlag> p_flags);
	BitField<TextServer::JustificationFlag> get_flags() const;
	void set_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior);
	TextServer::OverrunBehavior get_text_overrun_behavior() const;
	void set_ellipsis_char(const String &p_char);
	String get_ellipsis_char() const;
	Array get_objects() const;
	Rect2 get_object_rect(const Variant &p_key) const;
	Vector2 get_size() const;
	RID get_rid() const;
	float get_line_ascent() const;
	float get_line_descent() const;
	float get_line_width() const;
	float get_line_underline_position() const;
	float get_line_underline_thickness() const;
	void draw(const RID &p_canvas, const Vector2 &p_pos, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	void draw_outline(const RID &p_canvas, const Vector2 &p_pos, int32_t p_outline_size = 1, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	int32_t hit_test(float p_coords) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

