/**************************************************************************/
/*  font.hpp                                                              */
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
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Font : public Resource {
	GDEXTENSION_CLASS(Font, Resource)

public:
	void set_fallbacks(const TypedArray<Ref<Font>> &p_fallbacks);
	TypedArray<Ref<Font>> get_fallbacks() const;
	RID find_variation(const Dictionary &p_variation_coordinates, int32_t p_face_index = 0, float p_strength = 0.0, const Transform2D &p_transform = Transform2D(), int32_t p_spacing_top = 0, int32_t p_spacing_bottom = 0, int32_t p_spacing_space = 0, int32_t p_spacing_glyph = 0, float p_baseline_offset = 0.0) const;
	TypedArray<RID> get_rids() const;
	float get_height(int32_t p_font_size = 16) const;
	float get_ascent(int32_t p_font_size = 16) const;
	float get_descent(int32_t p_font_size = 16) const;
	float get_underline_position(int32_t p_font_size = 16) const;
	float get_underline_thickness(int32_t p_font_size = 16) const;
	String get_font_name() const;
	String get_font_style_name() const;
	Dictionary get_ot_name_strings() const;
	BitField<TextServer::FontStyle> get_font_style() const;
	int32_t get_font_weight() const;
	int32_t get_font_stretch() const;
	int32_t get_spacing(TextServer::SpacingType p_spacing) const;
	Dictionary get_opentype_features() const;
	void set_cache_capacity(int32_t p_single_line, int32_t p_multi_line);
	Vector2 get_string_size(const String &p_text, HorizontalAlignment p_alignment = (HorizontalAlignment)0, float p_width = -1, int32_t p_font_size = 16, BitField<TextServer::JustificationFlag> p_justification_flags = (BitField<TextServer::JustificationFlag>)3, TextServer::Direction p_direction = (TextServer::Direction)0, TextServer::Orientation p_orientation = (TextServer::Orientation)0) const;
	Vector2 get_multiline_string_size(const String &p_text, HorizontalAlignment p_alignment = (HorizontalAlignment)0, float p_width = -1, int32_t p_font_size = 16, int32_t p_max_lines = -1, BitField<TextServer::LineBreakFlag> p_brk_flags = (BitField<TextServer::LineBreakFlag>)3, BitField<TextServer::JustificationFlag> p_justification_flags = (BitField<TextServer::JustificationFlag>)3, TextServer::Direction p_direction = (TextServer::Direction)0, TextServer::Orientation p_orientation = (TextServer::Orientation)0) const;
	void draw_string(const RID &p_canvas_item, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment = (HorizontalAlignment)0, float p_width = -1, int32_t p_font_size = 16, const Color &p_modulate = Color(1, 1, 1, 1), BitField<TextServer::JustificationFlag> p_justification_flags = (BitField<TextServer::JustificationFlag>)3, TextServer::Direction p_direction = (TextServer::Direction)0, TextServer::Orientation p_orientation = (TextServer::Orientation)0, float p_oversampling = 0.0) const;
	void draw_multiline_string(const RID &p_canvas_item, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment = (HorizontalAlignment)0, float p_width = -1, int32_t p_font_size = 16, int32_t p_max_lines = -1, const Color &p_modulate = Color(1, 1, 1, 1), BitField<TextServer::LineBreakFlag> p_brk_flags = (BitField<TextServer::LineBreakFlag>)3, BitField<TextServer::JustificationFlag> p_justification_flags = (BitField<TextServer::JustificationFlag>)3, TextServer::Direction p_direction = (TextServer::Direction)0, TextServer::Orientation p_orientation = (TextServer::Orientation)0, float p_oversampling = 0.0) const;
	void draw_string_outline(const RID &p_canvas_item, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment = (HorizontalAlignment)0, float p_width = -1, int32_t p_font_size = 16, int32_t p_size = 1, const Color &p_modulate = Color(1, 1, 1, 1), BitField<TextServer::JustificationFlag> p_justification_flags = (BitField<TextServer::JustificationFlag>)3, TextServer::Direction p_direction = (TextServer::Direction)0, TextServer::Orientation p_orientation = (TextServer::Orientation)0, float p_oversampling = 0.0) const;
	void draw_multiline_string_outline(const RID &p_canvas_item, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment = (HorizontalAlignment)0, float p_width = -1, int32_t p_font_size = 16, int32_t p_max_lines = -1, int32_t p_size = 1, const Color &p_modulate = Color(1, 1, 1, 1), BitField<TextServer::LineBreakFlag> p_brk_flags = (BitField<TextServer::LineBreakFlag>)3, BitField<TextServer::JustificationFlag> p_justification_flags = (BitField<TextServer::JustificationFlag>)3, TextServer::Direction p_direction = (TextServer::Direction)0, TextServer::Orientation p_orientation = (TextServer::Orientation)0, float p_oversampling = 0.0) const;
	Vector2 get_char_size(char32_t p_char, int32_t p_font_size) const;
	float draw_char(const RID &p_canvas_item, const Vector2 &p_pos, char32_t p_char, int32_t p_font_size, const Color &p_modulate = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	float draw_char_outline(const RID &p_canvas_item, const Vector2 &p_pos, char32_t p_char, int32_t p_font_size, int32_t p_size = -1, const Color &p_modulate = Color(1, 1, 1, 1), float p_oversampling = 0.0) const;
	bool has_char(char32_t p_char) const;
	String get_supported_chars() const;
	bool is_language_supported(const String &p_language) const;
	bool is_script_supported(const String &p_script) const;
	Dictionary get_supported_feature_list() const;
	Dictionary get_supported_variation_list() const;
	int64_t get_face_count() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

