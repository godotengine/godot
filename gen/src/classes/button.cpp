/**************************************************************************/
/*  button.cpp                                                            */
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

#include <godot_cpp/classes/button.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void Button::set_text(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_text")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

String Button::get_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Button::set_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_text_overrun_behavior")._native_ptr(), 1008890932);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_overrun_behavior_encoded;
	PtrToArg<int64_t>::encode(p_overrun_behavior, &p_overrun_behavior_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_overrun_behavior_encoded);
}

TextServer::OverrunBehavior Button::get_text_overrun_behavior() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_text_overrun_behavior")._native_ptr(), 3779142101);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::OverrunBehavior(0)));
	return (TextServer::OverrunBehavior)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Button::set_autowrap_mode(TextServer::AutowrapMode p_autowrap_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_autowrap_mode")._native_ptr(), 3289138044);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_autowrap_mode_encoded;
	PtrToArg<int64_t>::encode(p_autowrap_mode, &p_autowrap_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_autowrap_mode_encoded);
}

TextServer::AutowrapMode Button::get_autowrap_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_autowrap_mode")._native_ptr(), 1549071663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::AutowrapMode(0)));
	return (TextServer::AutowrapMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Button::set_autowrap_trim_flags(BitField<TextServer::LineBreakFlag> p_autowrap_trim_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_autowrap_trim_flags")._native_ptr(), 2809697122);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_autowrap_trim_flags);
}

BitField<TextServer::LineBreakFlag> Button::get_autowrap_trim_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_autowrap_trim_flags")._native_ptr(), 2340632602);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<TextServer::LineBreakFlag>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Button::set_text_direction(Control::TextDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_text_direction")._native_ptr(), 119160795);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

Control::TextDirection Button::get_text_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_text_direction")._native_ptr(), 797257663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::TextDirection(0)));
	return (Control::TextDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Button::set_language(const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_language")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_language);
}

String Button::get_language() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_language")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Button::set_button_icon(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_button_icon")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> Button::get_button_icon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_button_icon")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void Button::set_flat(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_flat")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Button::is_flat() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("is_flat")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Button::set_clip_text(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_clip_text")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Button::get_clip_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_clip_text")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Button::set_text_alignment(HorizontalAlignment p_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_text_alignment")._native_ptr(), 2312603777);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded);
}

HorizontalAlignment Button::get_text_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_text_alignment")._native_ptr(), 341400642);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (HorizontalAlignment(0)));
	return (HorizontalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Button::set_icon_alignment(HorizontalAlignment p_icon_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_icon_alignment")._native_ptr(), 2312603777);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_icon_alignment_encoded;
	PtrToArg<int64_t>::encode(p_icon_alignment, &p_icon_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_icon_alignment_encoded);
}

HorizontalAlignment Button::get_icon_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_icon_alignment")._native_ptr(), 341400642);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (HorizontalAlignment(0)));
	return (HorizontalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Button::set_vertical_icon_alignment(VerticalAlignment p_vertical_icon_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_vertical_icon_alignment")._native_ptr(), 1796458609);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_vertical_icon_alignment_encoded;
	PtrToArg<int64_t>::encode(p_vertical_icon_alignment, &p_vertical_icon_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertical_icon_alignment_encoded);
}

VerticalAlignment Button::get_vertical_icon_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("get_vertical_icon_alignment")._native_ptr(), 3274884059);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VerticalAlignment(0)));
	return (VerticalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Button::set_expand_icon(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("set_expand_icon")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Button::is_expand_icon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Button::get_class_static()._native_ptr(), StringName("is_expand_icon")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
