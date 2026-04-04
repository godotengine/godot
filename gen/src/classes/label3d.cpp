/**************************************************************************/
/*  label3d.cpp                                                           */
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

#include <godot_cpp/classes/label3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/triangle_mesh.hpp>

namespace godot {

void Label3D::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_horizontal_alignment")._native_ptr(), 2312603777);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded);
}

HorizontalAlignment Label3D::get_horizontal_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_horizontal_alignment")._native_ptr(), 341400642);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (HorizontalAlignment(0)));
	return (HorizontalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_vertical_alignment(VerticalAlignment p_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_vertical_alignment")._native_ptr(), 1796458609);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded);
}

VerticalAlignment Label3D::get_vertical_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_vertical_alignment")._native_ptr(), 3274884059);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VerticalAlignment(0)));
	return (VerticalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_modulate(const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_modulate")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_modulate);
}

Color Label3D::get_modulate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_modulate")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void Label3D::set_outline_modulate(const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_outline_modulate")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_modulate);
}

Color Label3D::get_outline_modulate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_outline_modulate")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void Label3D::set_text(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_text")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

String Label3D::get_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Label3D::set_text_direction(TextServer::Direction p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_text_direction")._native_ptr(), 1418190634);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

TextServer::Direction Label3D::get_text_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_text_direction")._native_ptr(), 2516697328);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::Direction(0)));
	return (TextServer::Direction)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_language(const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_language")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_language);
}

String Label3D::get_language() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_language")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Label3D::set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_structured_text_bidi_override")._native_ptr(), 55961453);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_parser_encoded;
	PtrToArg<int64_t>::encode(p_parser, &p_parser_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_parser_encoded);
}

TextServer::StructuredTextParser Label3D::get_structured_text_bidi_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_structured_text_bidi_override")._native_ptr(), 3385126229);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::StructuredTextParser(0)));
	return (TextServer::StructuredTextParser)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_structured_text_bidi_override_options(const Array &p_args) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_structured_text_bidi_override_options")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_args);
}

Array Label3D::get_structured_text_bidi_override_options() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_structured_text_bidi_override_options")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void Label3D::set_uppercase(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_uppercase")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Label3D::is_uppercase() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("is_uppercase")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Label3D::set_render_priority(int32_t p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_render_priority")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_encoded;
	PtrToArg<int64_t>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_priority_encoded);
}

int32_t Label3D::get_render_priority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_render_priority")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_outline_render_priority(int32_t p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_outline_render_priority")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_encoded;
	PtrToArg<int64_t>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_priority_encoded);
}

int32_t Label3D::get_outline_render_priority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_outline_render_priority")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_font(const Ref<Font> &p_font) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_font")._native_ptr(), 1262170328);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr));
}

Ref<Font> Label3D::get_font() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_font")._native_ptr(), 3229501585);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Font>()));
	return Ref<Font>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Font>(_gde_method_bind, _owner));
}

void Label3D::set_font_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_font_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t Label3D::get_font_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_font_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_outline_size(int32_t p_outline_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_outline_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_outline_size_encoded;
	PtrToArg<int64_t>::encode(p_outline_size, &p_outline_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_outline_size_encoded);
}

int32_t Label3D::get_outline_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_outline_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_line_spacing(float p_line_spacing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_line_spacing")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_line_spacing_encoded;
	PtrToArg<double>::encode(p_line_spacing, &p_line_spacing_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_spacing_encoded);
}

float Label3D::get_line_spacing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_line_spacing")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Label3D::set_autowrap_mode(TextServer::AutowrapMode p_autowrap_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_autowrap_mode")._native_ptr(), 3289138044);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_autowrap_mode_encoded;
	PtrToArg<int64_t>::encode(p_autowrap_mode, &p_autowrap_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_autowrap_mode_encoded);
}

TextServer::AutowrapMode Label3D::get_autowrap_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_autowrap_mode")._native_ptr(), 1549071663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::AutowrapMode(0)));
	return (TextServer::AutowrapMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_autowrap_trim_flags(BitField<TextServer::LineBreakFlag> p_autowrap_trim_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_autowrap_trim_flags")._native_ptr(), 2809697122);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_autowrap_trim_flags);
}

BitField<TextServer::LineBreakFlag> Label3D::get_autowrap_trim_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_autowrap_trim_flags")._native_ptr(), 2340632602);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<TextServer::LineBreakFlag>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_justification_flags(BitField<TextServer::JustificationFlag> p_justification_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_justification_flags")._native_ptr(), 2877345813);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_justification_flags);
}

BitField<TextServer::JustificationFlag> Label3D::get_justification_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_justification_flags")._native_ptr(), 1583363614);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<TextServer::JustificationFlag>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_width(float p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_width")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded);
}

float Label3D::get_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_width")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Label3D::set_pixel_size(float p_pixel_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_pixel_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pixel_size_encoded;
	PtrToArg<double>::encode(p_pixel_size, &p_pixel_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pixel_size_encoded);
}

float Label3D::get_pixel_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_pixel_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Label3D::set_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 Label3D::get_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Label3D::set_draw_flag(Label3D::DrawFlags p_flag, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_draw_flag")._native_ptr(), 1285833066);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flag_encoded, &p_enabled_encoded);
}

bool Label3D::get_draw_flag(Label3D::DrawFlags p_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_draw_flag")._native_ptr(), 259226453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_flag_encoded);
}

void Label3D::set_billboard_mode(BaseMaterial3D::BillboardMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_billboard_mode")._native_ptr(), 4202036497);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

BaseMaterial3D::BillboardMode Label3D::get_billboard_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_billboard_mode")._native_ptr(), 1283840139);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::BillboardMode(0)));
	return (BaseMaterial3D::BillboardMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_alpha_cut_mode(Label3D::AlphaCutMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_alpha_cut_mode")._native_ptr(), 2549142916);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Label3D::AlphaCutMode Label3D::get_alpha_cut_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_alpha_cut_mode")._native_ptr(), 219468601);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Label3D::AlphaCutMode(0)));
	return (Label3D::AlphaCutMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_alpha_scissor_threshold(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_alpha_scissor_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float Label3D::get_alpha_scissor_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_alpha_scissor_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Label3D::set_alpha_hash_scale(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_alpha_hash_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float Label3D::get_alpha_hash_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_alpha_hash_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Label3D::set_alpha_antialiasing(BaseMaterial3D::AlphaAntiAliasing p_alpha_aa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_alpha_antialiasing")._native_ptr(), 3212649852);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alpha_aa_encoded;
	PtrToArg<int64_t>::encode(p_alpha_aa, &p_alpha_aa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alpha_aa_encoded);
}

BaseMaterial3D::AlphaAntiAliasing Label3D::get_alpha_antialiasing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_alpha_antialiasing")._native_ptr(), 2889939400);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::AlphaAntiAliasing(0)));
	return (BaseMaterial3D::AlphaAntiAliasing)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Label3D::set_alpha_antialiasing_edge(float p_edge) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_alpha_antialiasing_edge")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_edge_encoded;
	PtrToArg<double>::encode(p_edge, &p_edge_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_edge_encoded);
}

float Label3D::get_alpha_antialiasing_edge() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_alpha_antialiasing_edge")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Label3D::set_texture_filter(BaseMaterial3D::TextureFilter p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("set_texture_filter")._native_ptr(), 22904437);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

BaseMaterial3D::TextureFilter Label3D::get_texture_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("get_texture_filter")._native_ptr(), 3289213076);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::TextureFilter(0)));
	return (BaseMaterial3D::TextureFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<TriangleMesh> Label3D::generate_triangle_mesh() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Label3D::get_class_static()._native_ptr(), StringName("generate_triangle_mesh")._native_ptr(), 3476533166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TriangleMesh>()));
	return Ref<TriangleMesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TriangleMesh>(_gde_method_bind, _owner));
}

} // namespace godot
