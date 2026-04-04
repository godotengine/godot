/**************************************************************************/
/*  rich_text_label.cpp                                                   */
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

#include <godot_cpp/classes/rich_text_label.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/popup_menu.hpp>
#include <godot_cpp/classes/rich_text_effect.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/v_scroll_bar.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

String RichTextLabel::get_parsed_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_parsed_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void RichTextLabel::add_text(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("add_text")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

void RichTextLabel::set_text(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_text")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

void RichTextLabel::add_hr(int32_t p_width, int32_t p_height, const Color &p_color, HorizontalAlignment p_alignment, bool p_width_in_percent, bool p_height_in_percent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("add_hr")._native_ptr(), 16816895);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	int8_t p_width_in_percent_encoded;
	PtrToArg<bool>::encode(p_width_in_percent, &p_width_in_percent_encoded);
	int8_t p_height_in_percent_encoded;
	PtrToArg<bool>::encode(p_height_in_percent, &p_height_in_percent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded, &p_height_encoded, &p_color, &p_alignment_encoded, &p_width_in_percent_encoded, &p_height_in_percent_encoded);
}

void RichTextLabel::add_image(const Ref<Texture2D> &p_image, int32_t p_width, int32_t p_height, const Color &p_color, InlineAlignment p_inline_align, const Rect2 &p_region, const Variant &p_key, bool p_pad, const String &p_tooltip, bool p_width_in_percent, bool p_height_in_percent, const String &p_alt_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("add_image")._native_ptr(), 1390915033);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_inline_align_encoded;
	PtrToArg<int64_t>::encode(p_inline_align, &p_inline_align_encoded);
	int8_t p_pad_encoded;
	PtrToArg<bool>::encode(p_pad, &p_pad_encoded);
	int8_t p_width_in_percent_encoded;
	PtrToArg<bool>::encode(p_width_in_percent, &p_width_in_percent_encoded);
	int8_t p_height_in_percent_encoded;
	PtrToArg<bool>::encode(p_height_in_percent, &p_height_in_percent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr), &p_width_encoded, &p_height_encoded, &p_color, &p_inline_align_encoded, &p_region, &p_key, &p_pad_encoded, &p_tooltip, &p_width_in_percent_encoded, &p_height_in_percent_encoded, &p_alt_text);
}

void RichTextLabel::update_image(const Variant &p_key, BitField<RichTextLabel::ImageUpdateMask> p_mask, const Ref<Texture2D> &p_image, int32_t p_width, int32_t p_height, const Color &p_color, InlineAlignment p_inline_align, const Rect2 &p_region, bool p_pad, const String &p_tooltip, bool p_width_in_percent, bool p_height_in_percent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("update_image")._native_ptr(), 6389170);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_inline_align_encoded;
	PtrToArg<int64_t>::encode(p_inline_align, &p_inline_align_encoded);
	int8_t p_pad_encoded;
	PtrToArg<bool>::encode(p_pad, &p_pad_encoded);
	int8_t p_width_in_percent_encoded;
	PtrToArg<bool>::encode(p_width_in_percent, &p_width_in_percent_encoded);
	int8_t p_height_in_percent_encoded;
	PtrToArg<bool>::encode(p_height_in_percent, &p_height_in_percent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_key, &p_mask, (p_image != nullptr ? &p_image->_owner : nullptr), &p_width_encoded, &p_height_encoded, &p_color, &p_inline_align_encoded, &p_region, &p_pad_encoded, &p_tooltip, &p_width_in_percent_encoded, &p_height_in_percent_encoded);
}

void RichTextLabel::newline() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("newline")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool RichTextLabel::remove_paragraph(int32_t p_paragraph, bool p_no_invalidate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("remove_paragraph")._native_ptr(), 3262369265);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_paragraph_encoded;
	PtrToArg<int64_t>::encode(p_paragraph, &p_paragraph_encoded);
	int8_t p_no_invalidate_encoded;
	PtrToArg<bool>::encode(p_no_invalidate, &p_no_invalidate_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_paragraph_encoded, &p_no_invalidate_encoded);
}

bool RichTextLabel::invalidate_paragraph(int32_t p_paragraph) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("invalidate_paragraph")._native_ptr(), 3067735520);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_paragraph_encoded;
	PtrToArg<int64_t>::encode(p_paragraph, &p_paragraph_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_paragraph_encoded);
}

void RichTextLabel::push_font(const Ref<Font> &p_font, int32_t p_font_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_font")._native_ptr(), 2347424842);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr), &p_font_size_encoded);
}

void RichTextLabel::push_font_size(int32_t p_font_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_font_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_font_size_encoded);
}

void RichTextLabel::push_normal() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_normal")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::push_bold() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_bold")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::push_bold_italics() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_bold_italics")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::push_italics() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_italics")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::push_mono() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_mono")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::push_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void RichTextLabel::push_outline_size(int32_t p_outline_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_outline_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_outline_size_encoded;
	PtrToArg<int64_t>::encode(p_outline_size, &p_outline_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_outline_size_encoded);
}

void RichTextLabel::push_outline_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_outline_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void RichTextLabel::push_paragraph(HorizontalAlignment p_alignment, Control::TextDirection p_base_direction, const String &p_language, TextServer::StructuredTextParser p_st_parser, BitField<TextServer::JustificationFlag> p_justification_flags, const PackedFloat32Array &p_tab_stops) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_paragraph")._native_ptr(), 3089306873);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	int64_t p_base_direction_encoded;
	PtrToArg<int64_t>::encode(p_base_direction, &p_base_direction_encoded);
	int64_t p_st_parser_encoded;
	PtrToArg<int64_t>::encode(p_st_parser, &p_st_parser_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded, &p_base_direction_encoded, &p_language, &p_st_parser_encoded, &p_justification_flags, &p_tab_stops);
}

void RichTextLabel::push_indent(int32_t p_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_indent")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_level_encoded;
	PtrToArg<int64_t>::encode(p_level, &p_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_level_encoded);
}

void RichTextLabel::push_list(int32_t p_level, RichTextLabel::ListType p_type, bool p_capitalize, const String &p_bullet) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_list")._native_ptr(), 3017143144);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_level_encoded;
	PtrToArg<int64_t>::encode(p_level, &p_level_encoded);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int8_t p_capitalize_encoded;
	PtrToArg<bool>::encode(p_capitalize, &p_capitalize_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_level_encoded, &p_type_encoded, &p_capitalize_encoded, &p_bullet);
}

void RichTextLabel::push_meta(const Variant &p_data, RichTextLabel::MetaUnderline p_underline_mode, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_meta")._native_ptr(), 3765356747);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_underline_mode_encoded;
	PtrToArg<int64_t>::encode(p_underline_mode, &p_underline_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data, &p_underline_mode_encoded, &p_tooltip);
}

void RichTextLabel::push_hint(const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_hint")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_description);
}

void RichTextLabel::push_language(const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_language")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_language);
}

void RichTextLabel::push_underline(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_underline")._native_ptr(), 1458098034);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void RichTextLabel::push_strikethrough(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_strikethrough")._native_ptr(), 1458098034);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void RichTextLabel::push_table(int32_t p_columns, InlineAlignment p_inline_align, int32_t p_align_to_row, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_table")._native_ptr(), 3426862026);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_columns_encoded;
	PtrToArg<int64_t>::encode(p_columns, &p_columns_encoded);
	int64_t p_inline_align_encoded;
	PtrToArg<int64_t>::encode(p_inline_align, &p_inline_align_encoded);
	int64_t p_align_to_row_encoded;
	PtrToArg<int64_t>::encode(p_align_to_row, &p_align_to_row_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_columns_encoded, &p_inline_align_encoded, &p_align_to_row_encoded, &p_name);
}

void RichTextLabel::push_dropcap(const String &p_string, const Ref<Font> &p_font, int32_t p_size, const Rect2 &p_dropcap_margins, const Color &p_color, int32_t p_outline_size, const Color &p_outline_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_dropcap")._native_ptr(), 4061635501);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_outline_size_encoded;
	PtrToArg<int64_t>::encode(p_outline_size, &p_outline_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_string, (p_font != nullptr ? &p_font->_owner : nullptr), &p_size_encoded, &p_dropcap_margins, &p_color, &p_outline_size_encoded, &p_outline_color);
}

void RichTextLabel::set_table_column_expand(int32_t p_column, bool p_expand, int32_t p_ratio, bool p_shrink) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_table_column_expand")._native_ptr(), 117236061);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_expand_encoded;
	PtrToArg<bool>::encode(p_expand, &p_expand_encoded);
	int64_t p_ratio_encoded;
	PtrToArg<int64_t>::encode(p_ratio, &p_ratio_encoded);
	int8_t p_shrink_encoded;
	PtrToArg<bool>::encode(p_shrink, &p_shrink_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_expand_encoded, &p_ratio_encoded, &p_shrink_encoded);
}

void RichTextLabel::set_table_column_name(int32_t p_column, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_table_column_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_name);
}

void RichTextLabel::set_cell_row_background_color(const Color &p_odd_row_bg, const Color &p_even_row_bg) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_cell_row_background_color")._native_ptr(), 3465483165);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_odd_row_bg, &p_even_row_bg);
}

void RichTextLabel::set_cell_border_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_cell_border_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void RichTextLabel::set_cell_size_override(const Vector2 &p_min_size, const Vector2 &p_max_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_cell_size_override")._native_ptr(), 3108078480);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_min_size, &p_max_size);
}

void RichTextLabel::set_cell_padding(const Rect2 &p_padding) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_cell_padding")._native_ptr(), 2046264180);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_padding);
}

void RichTextLabel::push_cell() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_cell")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::push_fgcolor(const Color &p_fgcolor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_fgcolor")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fgcolor);
}

void RichTextLabel::push_bgcolor(const Color &p_bgcolor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_bgcolor")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bgcolor);
}

void RichTextLabel::push_customfx(const Ref<RichTextEffect> &p_effect, const Dictionary &p_env) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_customfx")._native_ptr(), 2337942958);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_effect != nullptr ? &p_effect->_owner : nullptr), &p_env);
}

void RichTextLabel::push_context() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("push_context")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::pop_context() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("pop_context")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::pop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("pop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::pop_all() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("pop_all")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_structured_text_bidi_override")._native_ptr(), 55961453);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_parser_encoded;
	PtrToArg<int64_t>::encode(p_parser, &p_parser_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_parser_encoded);
}

TextServer::StructuredTextParser RichTextLabel::get_structured_text_bidi_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_structured_text_bidi_override")._native_ptr(), 3385126229);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::StructuredTextParser(0)));
	return (TextServer::StructuredTextParser)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_structured_text_bidi_override_options(const Array &p_args) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_structured_text_bidi_override_options")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_args);
}

Array RichTextLabel::get_structured_text_bidi_override_options() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_structured_text_bidi_override_options")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void RichTextLabel::set_text_direction(Control::TextDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_text_direction")._native_ptr(), 119160795);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

Control::TextDirection RichTextLabel::get_text_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_text_direction")._native_ptr(), 797257663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::TextDirection(0)));
	return (Control::TextDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_language(const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_language")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_language);
}

String RichTextLabel::get_language() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_language")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void RichTextLabel::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_horizontal_alignment")._native_ptr(), 2312603777);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded);
}

HorizontalAlignment RichTextLabel::get_horizontal_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_horizontal_alignment")._native_ptr(), 341400642);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (HorizontalAlignment(0)));
	return (HorizontalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_vertical_alignment(VerticalAlignment p_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_vertical_alignment")._native_ptr(), 1796458609);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded);
}

VerticalAlignment RichTextLabel::get_vertical_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_vertical_alignment")._native_ptr(), 3274884059);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VerticalAlignment(0)));
	return (VerticalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_justification_flags(BitField<TextServer::JustificationFlag> p_justification_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_justification_flags")._native_ptr(), 2877345813);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_justification_flags);
}

BitField<TextServer::JustificationFlag> RichTextLabel::get_justification_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_justification_flags")._native_ptr(), 1583363614);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<TextServer::JustificationFlag>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_tab_stops(const PackedFloat32Array &p_tab_stops) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_tab_stops")._native_ptr(), 2899603908);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_stops);
}

PackedFloat32Array RichTextLabel::get_tab_stops() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_tab_stops")._native_ptr(), 675695659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner);
}

void RichTextLabel::set_autowrap_mode(TextServer::AutowrapMode p_autowrap_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_autowrap_mode")._native_ptr(), 3289138044);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_autowrap_mode_encoded;
	PtrToArg<int64_t>::encode(p_autowrap_mode, &p_autowrap_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_autowrap_mode_encoded);
}

TextServer::AutowrapMode RichTextLabel::get_autowrap_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_autowrap_mode")._native_ptr(), 1549071663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::AutowrapMode(0)));
	return (TextServer::AutowrapMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_autowrap_trim_flags(BitField<TextServer::LineBreakFlag> p_autowrap_trim_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_autowrap_trim_flags")._native_ptr(), 2809697122);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_autowrap_trim_flags);
}

BitField<TextServer::LineBreakFlag> RichTextLabel::get_autowrap_trim_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_autowrap_trim_flags")._native_ptr(), 2340632602);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<TextServer::LineBreakFlag>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_meta_underline(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_meta_underline")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool RichTextLabel::is_meta_underlined() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_meta_underlined")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_hint_underline(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_hint_underline")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool RichTextLabel::is_hint_underlined() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_hint_underlined")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_scroll_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_scroll_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

bool RichTextLabel::is_scroll_active() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_scroll_active")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_scroll_follow_visible_characters(bool p_follow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_scroll_follow_visible_characters")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_follow_encoded;
	PtrToArg<bool>::encode(p_follow, &p_follow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_follow_encoded);
}

bool RichTextLabel::is_scroll_following_visible_characters() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_scroll_following_visible_characters")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_scroll_follow(bool p_follow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_scroll_follow")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_follow_encoded;
	PtrToArg<bool>::encode(p_follow, &p_follow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_follow_encoded);
}

bool RichTextLabel::is_scroll_following() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_scroll_following")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

VScrollBar *RichTextLabel::get_v_scroll_bar() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_v_scroll_bar")._native_ptr(), 2630340773);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<VScrollBar>(_gde_method_bind, _owner);
}

void RichTextLabel::scroll_to_line(int32_t p_line) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("scroll_to_line")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded);
}

void RichTextLabel::scroll_to_paragraph(int32_t p_paragraph) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("scroll_to_paragraph")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_paragraph_encoded;
	PtrToArg<int64_t>::encode(p_paragraph, &p_paragraph_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_paragraph_encoded);
}

void RichTextLabel::scroll_to_selection() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("scroll_to_selection")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::set_tab_size(int32_t p_spaces) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_tab_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_spaces_encoded;
	PtrToArg<int64_t>::encode(p_spaces, &p_spaces_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_spaces_encoded);
}

int32_t RichTextLabel::get_tab_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_tab_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_fit_content(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_fit_content")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool RichTextLabel::is_fit_content_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_fit_content_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_selection_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_selection_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool RichTextLabel::is_selection_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_selection_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_context_menu_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_context_menu_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool RichTextLabel::is_context_menu_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_context_menu_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_shortcut_keys_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_shortcut_keys_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool RichTextLabel::is_shortcut_keys_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_shortcut_keys_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_deselect_on_focus_loss_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_deselect_on_focus_loss_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool RichTextLabel::is_deselect_on_focus_loss_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_deselect_on_focus_loss_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_drag_and_drop_selection_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_drag_and_drop_selection_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool RichTextLabel::is_drag_and_drop_selection_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_drag_and_drop_selection_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_selection_from() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_selection_from")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_selection_to() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_selection_to")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

float RichTextLabel::get_selection_line_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_selection_line_offset")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RichTextLabel::select_all() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("select_all")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

String RichTextLabel::get_selected_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_selected_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void RichTextLabel::deselect() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("deselect")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RichTextLabel::parse_bbcode(const String &p_bbcode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("parse_bbcode")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bbcode);
}

void RichTextLabel::append_text(const String &p_bbcode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("append_text")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bbcode);
}

String RichTextLabel::get_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool RichTextLabel::is_ready() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_ready")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool RichTextLabel::is_finished() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_finished")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_threaded(bool p_threaded) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_threaded")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_threaded_encoded;
	PtrToArg<bool>::encode(p_threaded, &p_threaded_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threaded_encoded);
}

bool RichTextLabel::is_threaded() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_threaded")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_progress_bar_delay(int32_t p_delay_ms) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_progress_bar_delay")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_delay_ms_encoded;
	PtrToArg<int64_t>::encode(p_delay_ms, &p_delay_ms_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_delay_ms_encoded);
}

int32_t RichTextLabel::get_progress_bar_delay() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_progress_bar_delay")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_visible_characters(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_visible_characters")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

int32_t RichTextLabel::get_visible_characters() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_visible_characters")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

TextServer::VisibleCharactersBehavior RichTextLabel::get_visible_characters_behavior() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_visible_characters_behavior")._native_ptr(), 258789322);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::VisibleCharactersBehavior(0)));
	return (TextServer::VisibleCharactersBehavior)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_visible_characters_behavior(TextServer::VisibleCharactersBehavior p_behavior) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_visible_characters_behavior")._native_ptr(), 3383839701);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_behavior_encoded;
	PtrToArg<int64_t>::encode(p_behavior, &p_behavior_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_behavior_encoded);
}

void RichTextLabel::set_visible_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_visible_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

float RichTextLabel::get_visible_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_visible_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_character_line(int32_t p_character) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_character_line")._native_ptr(), 3744713108);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_character_encoded;
	PtrToArg<int64_t>::encode(p_character, &p_character_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_character_encoded);
}

int32_t RichTextLabel::get_character_paragraph(int32_t p_character) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_character_paragraph")._native_ptr(), 3744713108);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_character_encoded;
	PtrToArg<int64_t>::encode(p_character, &p_character_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_character_encoded);
}

int32_t RichTextLabel::get_total_character_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_total_character_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RichTextLabel::set_use_bbcode(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_use_bbcode")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool RichTextLabel::is_using_bbcode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_using_bbcode")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_line_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_line_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Vector2i RichTextLabel::get_line_range(int32_t p_line) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_line_range")._native_ptr(), 3665014314);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_line_encoded);
}

int32_t RichTextLabel::get_visible_line_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_visible_line_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_paragraph_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_paragraph_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_visible_paragraph_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_visible_paragraph_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_content_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_content_height")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_content_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_content_width")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t RichTextLabel::get_line_height(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_line_height")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded);
}

int32_t RichTextLabel::get_line_width(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_line_width")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded);
}

Rect2i RichTextLabel::get_visible_content_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_visible_content_rect")._native_ptr(), 410525958);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner);
}

float RichTextLabel::get_line_offset(int32_t p_line) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_line_offset")._native_ptr(), 4025615559);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_line_encoded);
}

float RichTextLabel::get_paragraph_offset(int32_t p_paragraph) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_paragraph_offset")._native_ptr(), 4025615559);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_paragraph_encoded;
	PtrToArg<int64_t>::encode(p_paragraph, &p_paragraph_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_paragraph_encoded);
}

Dictionary RichTextLabel::parse_expressions_for_values(const PackedStringArray &p_expressions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("parse_expressions_for_values")._native_ptr(), 1522900837);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_expressions);
}

void RichTextLabel::set_effects(const Array &p_effects) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("set_effects")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_effects);
}

Array RichTextLabel::get_effects() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_effects")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void RichTextLabel::install_effect(const Variant &p_effect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("install_effect")._native_ptr(), 1114965689);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_effect);
}

void RichTextLabel::reload_effects() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("reload_effects")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

PopupMenu *RichTextLabel::get_menu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("get_menu")._native_ptr(), 229722558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<PopupMenu>(_gde_method_bind, _owner);
}

bool RichTextLabel::is_menu_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("is_menu_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RichTextLabel::menu_option(int32_t p_option) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RichTextLabel::get_class_static()._native_ptr(), StringName("menu_option")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_option_encoded;
	PtrToArg<int64_t>::encode(p_option, &p_option_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_option_encoded);
}

} // namespace godot
