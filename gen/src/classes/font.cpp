/**************************************************************************/
/*  font.cpp                                                              */
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

#include <godot_cpp/classes/font.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void Font::set_fallbacks(const TypedArray<Ref<Font>> &p_fallbacks) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("set_fallbacks")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fallbacks);
}

TypedArray<Ref<Font>> Font::get_fallbacks() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_fallbacks")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Font>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Font>>>(_gde_method_bind, _owner);
}

RID Font::find_variation(const Dictionary &p_variation_coordinates, int32_t p_face_index, float p_strength, const Transform2D &p_transform, int32_t p_spacing_top, int32_t p_spacing_bottom, int32_t p_spacing_space, int32_t p_spacing_glyph, float p_baseline_offset) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("find_variation")._native_ptr(), 2553855095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_face_index_encoded;
	PtrToArg<int64_t>::encode(p_face_index, &p_face_index_encoded);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	int64_t p_spacing_top_encoded;
	PtrToArg<int64_t>::encode(p_spacing_top, &p_spacing_top_encoded);
	int64_t p_spacing_bottom_encoded;
	PtrToArg<int64_t>::encode(p_spacing_bottom, &p_spacing_bottom_encoded);
	int64_t p_spacing_space_encoded;
	PtrToArg<int64_t>::encode(p_spacing_space, &p_spacing_space_encoded);
	int64_t p_spacing_glyph_encoded;
	PtrToArg<int64_t>::encode(p_spacing_glyph, &p_spacing_glyph_encoded);
	double p_baseline_offset_encoded;
	PtrToArg<double>::encode(p_baseline_offset, &p_baseline_offset_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_variation_coordinates, &p_face_index_encoded, &p_strength_encoded, &p_transform, &p_spacing_top_encoded, &p_spacing_bottom_encoded, &p_spacing_space_encoded, &p_spacing_glyph_encoded, &p_baseline_offset_encoded);
}

TypedArray<RID> Font::get_rids() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_rids")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner);
}

float Font::get_height(int32_t p_font_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_height")._native_ptr(), 378113874);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_size_encoded);
}

float Font::get_ascent(int32_t p_font_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_ascent")._native_ptr(), 378113874);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_size_encoded);
}

float Font::get_descent(int32_t p_font_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_descent")._native_ptr(), 378113874);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_size_encoded);
}

float Font::get_underline_position(int32_t p_font_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_underline_position")._native_ptr(), 378113874);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_size_encoded);
}

float Font::get_underline_thickness(int32_t p_font_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_underline_thickness")._native_ptr(), 378113874);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_font_size_encoded);
}

String Font::get_font_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_font_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String Font::get_font_style_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_font_style_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

Dictionary Font::get_ot_name_strings() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_ot_name_strings")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

BitField<TextServer::FontStyle> Font::get_font_style() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_font_style")._native_ptr(), 2520224254);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<TextServer::FontStyle>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t Font::get_font_weight() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_font_weight")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t Font::get_font_stretch() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_font_stretch")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t Font::get_spacing(TextServer::SpacingType p_spacing) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_spacing")._native_ptr(), 1310880908);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_spacing_encoded;
	PtrToArg<int64_t>::encode(p_spacing, &p_spacing_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_spacing_encoded);
}

Dictionary Font::get_opentype_features() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_opentype_features")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

void Font::set_cache_capacity(int32_t p_single_line, int32_t p_multi_line) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("set_cache_capacity")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_single_line_encoded;
	PtrToArg<int64_t>::encode(p_single_line, &p_single_line_encoded);
	int64_t p_multi_line_encoded;
	PtrToArg<int64_t>::encode(p_multi_line, &p_multi_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_single_line_encoded, &p_multi_line_encoded);
}

Vector2 Font::get_string_size(const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_string_size")._native_ptr(), 1868866121);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded);
}

Vector2 Font::get_multiline_string_size(const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, int32_t p_max_lines, BitField<TextServer::LineBreakFlag> p_brk_flags, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_multiline_string_size")._native_ptr(), 519636710);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_max_lines_encoded;
	PtrToArg<int64_t>::encode(p_max_lines, &p_max_lines_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_max_lines_encoded, &p_brk_flags, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded);
}

void Font::draw_string(const RID &p_canvas_item, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, const Color &p_modulate, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("draw_string")._native_ptr(), 1976686372);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_item, &p_pos, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_modulate, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded, &p_oversampling_encoded);
}

void Font::draw_multiline_string(const RID &p_canvas_item, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, int32_t p_max_lines, const Color &p_modulate, BitField<TextServer::LineBreakFlag> p_brk_flags, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("draw_multiline_string")._native_ptr(), 2686601589);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_max_lines_encoded;
	PtrToArg<int64_t>::encode(p_max_lines, &p_max_lines_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_item, &p_pos, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_max_lines_encoded, &p_modulate, &p_brk_flags, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded, &p_oversampling_encoded);
}

void Font::draw_string_outline(const RID &p_canvas_item, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, int32_t p_size, const Color &p_modulate, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("draw_string_outline")._native_ptr(), 701417663);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_item, &p_pos, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_size_encoded, &p_modulate, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded, &p_oversampling_encoded);
}

void Font::draw_multiline_string_outline(const RID &p_canvas_item, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, int32_t p_max_lines, int32_t p_size, const Color &p_modulate, BitField<TextServer::LineBreakFlag> p_brk_flags, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("draw_multiline_string_outline")._native_ptr(), 4147839237);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_max_lines_encoded;
	PtrToArg<int64_t>::encode(p_max_lines, &p_max_lines_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_item, &p_pos, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_max_lines_encoded, &p_size_encoded, &p_modulate, &p_brk_flags, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded, &p_oversampling_encoded);
}

Vector2 Font::get_char_size(char32_t p_char, int32_t p_font_size) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_char_size")._native_ptr(), 3016396712);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_char_encoded;
	PtrToArg<int64_t>::encode(p_char, &p_char_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_char_encoded, &p_font_size_encoded);
}

float Font::draw_char(const RID &p_canvas_item, const Vector2 &p_pos, char32_t p_char, int32_t p_font_size, const Color &p_modulate, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("draw_char")._native_ptr(), 3500170256);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_char_encoded;
	PtrToArg<int64_t>::encode(p_char, &p_char_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_canvas_item, &p_pos, &p_char_encoded, &p_font_size_encoded, &p_modulate, &p_oversampling_encoded);
}

float Font::draw_char_outline(const RID &p_canvas_item, const Vector2 &p_pos, char32_t p_char, int32_t p_font_size, int32_t p_size, const Color &p_modulate, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("draw_char_outline")._native_ptr(), 1684114874);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_char_encoded;
	PtrToArg<int64_t>::encode(p_char, &p_char_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_canvas_item, &p_pos, &p_char_encoded, &p_font_size_encoded, &p_size_encoded, &p_modulate, &p_oversampling_encoded);
}

bool Font::has_char(char32_t p_char) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("has_char")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_char_encoded;
	PtrToArg<int64_t>::encode(p_char, &p_char_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_char_encoded);
}

String Font::get_supported_chars() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_supported_chars")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool Font::is_language_supported(const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("is_language_supported")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_language);
}

bool Font::is_script_supported(const String &p_script) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("is_script_supported")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_script);
}

Dictionary Font::get_supported_feature_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_supported_feature_list")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

Dictionary Font::get_supported_variation_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_supported_variation_list")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

int64_t Font::get_face_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Font::get_class_static()._native_ptr(), StringName("get_face_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
