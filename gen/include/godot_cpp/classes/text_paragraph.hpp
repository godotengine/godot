/**************************************************************************/
/*  text_paragraph.hpp                                                    */
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
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Font;
class PackedFloat32Array;

class TextParagraph : public RefCounted {
	GDEXTENSION_CLASS(TextParagraph, RefCounted)

public:
	void clear();
	Ref<TextParagraph> duplicate() const;
	void set_direction(TextServer::Direction p_direction);
	TextServer::Direction get_direction() const;
	TextServer::Direction get_inferred_direction() const;
	void set_custom_punctuation(const String &p_custom_punctuation);
	String get_custom_punctuation() const;
	void set_orientation(TextServer::Orientation p_orientation);
	TextServer::Orientation get_orientation() const;
	void set_preserve_invalid(bool p_enabled);
	bool get_preserve_invalid() const;
	void set_preserve_control(bool p_enabled);
	bool get_preserve_control() const;
	void set_bidi_override(const Array &p_override);
	bool set_dropcap(const String &p_text, const Ref<Font> &p_font, int32_t p_font_size, const Rect2 &p_dropcap_margins = Rect2(0, 0, 0, 0), const String &p_language = String());
	void clear_dropcap();
	bool add_string(const String &p_text, const Ref<Font> &p_font, int32_t p_font_size, const String &p_language = String(), const Variant &p_meta = nullptr);
	bool add_object(const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align = (InlineAlignment)5, int32_t p_length = 1, float p_baseline = 0.0);
	bool resize_object(const Variant &p_key, const Vector2 &p_size, InlineAlignment p_inline_align = (InlineAlignment)5, float p_baseline = 0.0);
	bool has_object(const Variant &p_key) const;
	void set_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_alignment() const;
	void tab_align(const PackedFloat32Array &p_tab_stops);
	void set_break_flags(BitField<TextServer::LineBreakFlag> p_flags);
	BitField<TextServer::LineBreakFlag> get_break_flags() const;
	void set_justification_flags(BitField<TextServer::JustificationFlag> p_flags);
	BitField<TextServer::JustificationFlag> get_justification_flags() const;
	void set_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior);
	TextServer::OverrunBehavior get_text_overrun_behavior() const;
	void set_ellipsis_char(const String &p_char);
	String get_ellipsis_char() const;
	void set_width(float p_width);
	float get_width() const;
	Vector2 get_non_wrapped_size() const;
	Vector2 get_size() const;
	RID get_rid() const;
	RID get_line_rid(int32_t p_line) const;
	RID get_dropcap_rid() const;
	Vector2i get_range() const;
	int32_t get_line_count() const;
	void set_max_lines_visible(int32_t p_max_lines_visible);
	int32_t get_max_lines_visible() const;
	void set_line_spacing(float p_line_spacing);
	float get_line_spacing() const;
	Array get_line_objects(int32_t p_line) const;
	Rect2 get_line_object_rect(int32_t p_line, const Variant &p_key) const;
	Vector2 get_line_size(int32_t p_line) const;
	Vector2i get_line_range(int32_t p_line) const;
	float get_line_ascent(int32_t p_line) const;
	float get_line_descent(int32_t p_line) const;
	float get_line_width(int32_t p_line) const;
	float get_line_underline_position(int32_t p_line) const;
	float get_line_underline_thickness(int32_t p_line) const;
	Vector2 get_dropcap_size() const;
	int32_t get_dropcap_lines() const;
	void draw(const RID &p_canvas, const Vector2 &p_pos, const Color &p_color = Color(1, 1, 1, 1), const Color &p_dc_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	void draw_outline(const RID &p_canvas, const Vector2 &p_pos, int32_t p_outline_size = 1, const Color &p_color = Color(1, 1, 1, 1), const Color &p_dc_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	void draw_line(const RID &p_canvas, const Vector2 &p_pos, int32_t p_line, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	void draw_line_outline(const RID &p_canvas, const Vector2 &p_pos, int32_t p_line, int32_t p_outline_size = 1, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	void draw_dropcap(const RID &p_canvas, const Vector2 &p_pos, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	void draw_dropcap_outline(const RID &p_canvas, const Vector2 &p_pos, int32_t p_outline_size = 1, const Color &p_color = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	int32_t hit_test(const Vector2 &p_coords) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

