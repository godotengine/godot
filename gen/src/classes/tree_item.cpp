/**************************************************************************/
/*  tree_item.cpp                                                         */
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

#include <godot_cpp/classes/tree_item.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/style_box.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/tree.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void TreeItem::set_cell_mode(int32_t p_column, TreeItem::TreeCellMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_cell_mode")._native_ptr(), 289920701);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_mode_encoded);
}

TreeItem::TreeCellMode TreeItem::get_cell_mode(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_cell_mode")._native_ptr(), 3406114978);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TreeItem::TreeCellMode(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (TreeItem::TreeCellMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_auto_translate_mode(int32_t p_column, Node::AutoTranslateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_auto_translate_mode")._native_ptr(), 287402019);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_mode_encoded);
}

Node::AutoTranslateMode TreeItem::get_auto_translate_mode(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_auto_translate_mode")._native_ptr(), 906302372);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node::AutoTranslateMode(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (Node::AutoTranslateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_edit_multiline(int32_t p_column, bool p_multiline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_edit_multiline")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_multiline_encoded;
	PtrToArg<bool>::encode(p_multiline, &p_multiline_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_multiline_encoded);
}

bool TreeItem::is_edit_multiline(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_edit_multiline")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_checked(int32_t p_column, bool p_checked) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_checked")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_checked_encoded;
	PtrToArg<bool>::encode(p_checked, &p_checked_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_checked_encoded);
}

void TreeItem::set_indeterminate(int32_t p_column, bool p_indeterminate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_indeterminate")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_indeterminate_encoded;
	PtrToArg<bool>::encode(p_indeterminate, &p_indeterminate_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_indeterminate_encoded);
}

bool TreeItem::is_checked(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_checked")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

bool TreeItem::is_indeterminate(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_indeterminate")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::propagate_check(int32_t p_column, bool p_emit_signal) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("propagate_check")._native_ptr(), 972357352);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_emit_signal_encoded;
	PtrToArg<bool>::encode(p_emit_signal, &p_emit_signal_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_emit_signal_encoded);
}

void TreeItem::set_text(int32_t p_column, const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_text")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_text);
}

String TreeItem::get_text(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_text")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_description(int32_t p_column, const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_description")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_description);
}

String TreeItem::get_description(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_description")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_text_direction(int32_t p_column, Control::TextDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_text_direction")._native_ptr(), 1707680378);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_direction_encoded);
}

Control::TextDirection TreeItem::get_text_direction(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_text_direction")._native_ptr(), 4235602388);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::TextDirection(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (Control::TextDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_autowrap_mode(int32_t p_column, TextServer::AutowrapMode p_autowrap_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_autowrap_mode")._native_ptr(), 3633006561);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_autowrap_mode_encoded;
	PtrToArg<int64_t>::encode(p_autowrap_mode, &p_autowrap_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_autowrap_mode_encoded);
}

TextServer::AutowrapMode TreeItem::get_autowrap_mode(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_autowrap_mode")._native_ptr(), 2902757236);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::AutowrapMode(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (TextServer::AutowrapMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_text_overrun_behavior(int32_t p_column, TextServer::OverrunBehavior p_overrun_behavior) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_text_overrun_behavior")._native_ptr(), 1940772195);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_overrun_behavior_encoded;
	PtrToArg<int64_t>::encode(p_overrun_behavior, &p_overrun_behavior_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_overrun_behavior_encoded);
}

TextServer::OverrunBehavior TreeItem::get_text_overrun_behavior(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_text_overrun_behavior")._native_ptr(), 3782727860);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::OverrunBehavior(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (TextServer::OverrunBehavior)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_structured_text_bidi_override(int32_t p_column, TextServer::StructuredTextParser p_parser) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_structured_text_bidi_override")._native_ptr(), 868756907);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_parser_encoded;
	PtrToArg<int64_t>::encode(p_parser, &p_parser_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_parser_encoded);
}

TextServer::StructuredTextParser TreeItem::get_structured_text_bidi_override(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_structured_text_bidi_override")._native_ptr(), 3377823772);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::StructuredTextParser(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (TextServer::StructuredTextParser)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_structured_text_bidi_override_options(int32_t p_column, const Array &p_args) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_structured_text_bidi_override_options")._native_ptr(), 537221740);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_args);
}

Array TreeItem::get_structured_text_bidi_override_options(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_structured_text_bidi_override_options")._native_ptr(), 663333327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_language(int32_t p_column, const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_language")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_language);
}

String TreeItem::get_language(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_language")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_suffix(int32_t p_column, const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_suffix")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_text);
}

String TreeItem::get_suffix(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_suffix")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_icon(int32_t p_column, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_icon")._native_ptr(), 666127730);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> TreeItem::get_icon(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_icon")._native_ptr(), 3536238170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_column_encoded));
}

void TreeItem::set_icon_overlay(int32_t p_column, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_icon_overlay")._native_ptr(), 666127730);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> TreeItem::get_icon_overlay(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_icon_overlay")._native_ptr(), 3536238170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_column_encoded));
}

void TreeItem::set_icon_region(int32_t p_column, const Rect2 &p_region) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_icon_region")._native_ptr(), 1356297692);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_region);
}

Rect2 TreeItem::get_icon_region(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_icon_region")._native_ptr(), 3327874267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_icon_max_width(int32_t p_column, int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_icon_max_width")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_width_encoded);
}

int32_t TreeItem::get_icon_max_width(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_icon_max_width")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_icon_modulate(int32_t p_column, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_icon_modulate")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_modulate);
}

Color TreeItem::get_icon_modulate(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_icon_modulate")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_range(int32_t p_column, double p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_range")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_value_encoded);
}

double TreeItem::get_range(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_range")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_range_config(int32_t p_column, double p_min, double p_max, double p_step, bool p_expr) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_range_config")._native_ptr(), 1547181014);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	double p_step_encoded;
	PtrToArg<double>::encode(p_step, &p_step_encoded);
	int8_t p_expr_encoded;
	PtrToArg<bool>::encode(p_expr, &p_expr_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_min_encoded, &p_max_encoded, &p_step_encoded, &p_expr_encoded);
}

Dictionary TreeItem::get_range_config(int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_range_config")._native_ptr(), 3554694381);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_metadata(int32_t p_column, const Variant &p_meta) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_metadata")._native_ptr(), 2152698145);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_meta);
}

Variant TreeItem::get_metadata(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_metadata")._native_ptr(), 4227898402);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_custom_draw(int32_t p_column, Object *p_object, const StringName &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_draw")._native_ptr(), 272420368);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, (p_object != nullptr ? &p_object->_owner : nullptr), &p_callback);
}

void TreeItem::set_custom_draw_callback(int32_t p_column, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_draw_callback")._native_ptr(), 957362965);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_callback);
}

Callable TreeItem::get_custom_draw_callback(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_custom_draw_callback")._native_ptr(), 1317077508);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Callable()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Callable>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_custom_stylebox(int32_t p_column, const Ref<StyleBox> &p_stylebox) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_stylebox")._native_ptr(), 1433009359);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, (p_stylebox != nullptr ? &p_stylebox->_owner : nullptr));
}

Ref<StyleBox> TreeItem::get_custom_stylebox(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_custom_stylebox")._native_ptr(), 3362509644);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<StyleBox>()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return Ref<StyleBox>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<StyleBox>(_gde_method_bind, _owner, &p_column_encoded));
}

void TreeItem::set_collapsed(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_collapsed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TreeItem::is_collapsed() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_collapsed")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TreeItem::set_collapsed_recursive(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_collapsed_recursive")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TreeItem::is_any_collapsed(bool p_only_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_any_collapsed")._native_ptr(), 2595650253);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_only_visible_encoded;
	PtrToArg<bool>::encode(p_only_visible, &p_only_visible_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_only_visible_encoded);
}

void TreeItem::set_visible(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TreeItem::is_visible() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_visible")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool TreeItem::is_visible_in_tree() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_visible_in_tree")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TreeItem::uncollapse_tree() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("uncollapse_tree")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TreeItem::set_custom_minimum_height(int32_t p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_minimum_height")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

int32_t TreeItem::get_custom_minimum_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_custom_minimum_height")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TreeItem::set_selectable(int32_t p_column, bool p_selectable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_selectable")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_selectable_encoded;
	PtrToArg<bool>::encode(p_selectable, &p_selectable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_selectable_encoded);
}

bool TreeItem::is_selectable(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_selectable")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

bool TreeItem::is_selected(int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_selected")._native_ptr(), 3067735520);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::select(int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("select")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::deselect(int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("deselect")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_editable(int32_t p_column, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_editable")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_enabled_encoded);
}

bool TreeItem::is_editable(int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_editable")._native_ptr(), 3067735520);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_custom_color(int32_t p_column, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_color")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_color);
}

Color TreeItem::get_custom_color(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_custom_color")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::clear_custom_color(int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("clear_custom_color")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_custom_font(int32_t p_column, const Ref<Font> &p_font) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_font")._native_ptr(), 2637609184);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, (p_font != nullptr ? &p_font->_owner : nullptr));
}

Ref<Font> TreeItem::get_custom_font(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_custom_font")._native_ptr(), 4244553094);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Font>()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return Ref<Font>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Font>(_gde_method_bind, _owner, &p_column_encoded));
}

void TreeItem::set_custom_font_size(int32_t p_column, int32_t p_font_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_font_size")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_font_size_encoded);
}

int32_t TreeItem::get_custom_font_size(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_custom_font_size")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_custom_bg_color(int32_t p_column, const Color &p_color, bool p_just_outline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_bg_color")._native_ptr(), 894174518);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_just_outline_encoded;
	PtrToArg<bool>::encode(p_just_outline, &p_just_outline_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_color, &p_just_outline_encoded);
}

void TreeItem::clear_custom_bg_color(int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("clear_custom_bg_color")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded);
}

Color TreeItem::get_custom_bg_color(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_custom_bg_color")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_custom_as_button(int32_t p_column, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_custom_as_button")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_enable_encoded);
}

bool TreeItem::is_custom_set_as_button(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_custom_set_as_button")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::clear_buttons() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("clear_buttons")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TreeItem::add_button(int32_t p_column, const Ref<Texture2D> &p_button, int32_t p_id, bool p_disabled, const String &p_tooltip_text, const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("add_button")._native_ptr(), 973481897);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, (p_button != nullptr ? &p_button->_owner : nullptr), &p_id_encoded, &p_disabled_encoded, &p_tooltip_text, &p_description);
}

int32_t TreeItem::get_button_count(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_button_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

String TreeItem::get_button_tooltip_text(int32_t p_column, int32_t p_button_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_button_tooltip_text")._native_ptr(), 1391810591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded);
}

int32_t TreeItem::get_button_id(int32_t p_column, int32_t p_button_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_button_id")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded);
}

int32_t TreeItem::get_button_by_id(int32_t p_column, int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_button_by_id")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded, &p_id_encoded);
}

Color TreeItem::get_button_color(int32_t p_column, int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_button_color")._native_ptr(), 2165839948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_column_encoded, &p_id_encoded);
}

Ref<Texture2D> TreeItem::get_button(int32_t p_column, int32_t p_button_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_button")._native_ptr(), 2584904275);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded));
}

void TreeItem::set_button_tooltip_text(int32_t p_column, int32_t p_button_index, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_button_tooltip_text")._native_ptr(), 2285447957);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded, &p_tooltip);
}

void TreeItem::set_button(int32_t p_column, int32_t p_button_index, const Ref<Texture2D> &p_button) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_button")._native_ptr(), 176101966);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded, (p_button != nullptr ? &p_button->_owner : nullptr));
}

void TreeItem::erase_button(int32_t p_column, int32_t p_button_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("erase_button")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded);
}

void TreeItem::set_button_description(int32_t p_column, int32_t p_button_index, const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_button_description")._native_ptr(), 2285447957);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded, &p_description);
}

void TreeItem::set_button_disabled(int32_t p_column, int32_t p_button_index, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_button_disabled")._native_ptr(), 1383440665);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded, &p_disabled_encoded);
}

void TreeItem::set_button_color(int32_t p_column, int32_t p_button_index, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_button_color")._native_ptr(), 3733378741);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded, &p_color);
}

bool TreeItem::is_button_disabled(int32_t p_column, int32_t p_button_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_button_disabled")._native_ptr(), 2522259332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded, &p_button_index_encoded);
}

void TreeItem::set_tooltip_text(int32_t p_column, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_tooltip_text")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_tooltip);
}

String TreeItem::get_tooltip_text(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_tooltip_text")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_text_alignment(int32_t p_column, HorizontalAlignment p_text_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_text_alignment")._native_ptr(), 3276431499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_text_alignment_encoded;
	PtrToArg<int64_t>::encode(p_text_alignment, &p_text_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_text_alignment_encoded);
}

HorizontalAlignment TreeItem::get_text_alignment(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_text_alignment")._native_ptr(), 4171562184);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (HorizontalAlignment(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (HorizontalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_expand_right(int32_t p_column, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_expand_right")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_enable_encoded);
}

bool TreeItem::get_expand_right(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_expand_right")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void TreeItem::set_disable_folding(bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("set_disable_folding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_disable_encoded);
}

bool TreeItem::is_folding_disabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("is_folding_disabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

TreeItem *TreeItem::create_child(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("create_child")._native_ptr(), 954243986);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, &p_index_encoded);
}

void TreeItem::add_child(TreeItem *p_child) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("add_child")._native_ptr(), 1819951137);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_child != nullptr ? &p_child->_owner : nullptr));
}

void TreeItem::remove_child(TreeItem *p_child) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("remove_child")._native_ptr(), 1819951137);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_child != nullptr ? &p_child->_owner : nullptr));
}

Tree *TreeItem::get_tree() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_tree")._native_ptr(), 2243340556);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Tree>(_gde_method_bind, _owner);
}

TreeItem *TreeItem::get_next() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_next")._native_ptr(), 1514277247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner);
}

TreeItem *TreeItem::get_prev() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_prev")._native_ptr(), 2768121250);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner);
}

TreeItem *TreeItem::get_parent() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_parent")._native_ptr(), 1514277247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner);
}

TreeItem *TreeItem::get_first_child() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_first_child")._native_ptr(), 1514277247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner);
}

TreeItem *TreeItem::get_next_in_tree(bool p_wrap) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_next_in_tree")._native_ptr(), 1666920593);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int8_t p_wrap_encoded;
	PtrToArg<bool>::encode(p_wrap, &p_wrap_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, &p_wrap_encoded);
}

TreeItem *TreeItem::get_prev_in_tree(bool p_wrap) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_prev_in_tree")._native_ptr(), 1666920593);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int8_t p_wrap_encoded;
	PtrToArg<bool>::encode(p_wrap, &p_wrap_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, &p_wrap_encoded);
}

TreeItem *TreeItem::get_next_visible(bool p_wrap) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_next_visible")._native_ptr(), 1666920593);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int8_t p_wrap_encoded;
	PtrToArg<bool>::encode(p_wrap, &p_wrap_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, &p_wrap_encoded);
}

TreeItem *TreeItem::get_prev_visible(bool p_wrap) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_prev_visible")._native_ptr(), 1666920593);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int8_t p_wrap_encoded;
	PtrToArg<bool>::encode(p_wrap, &p_wrap_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, &p_wrap_encoded);
}

TreeItem *TreeItem::get_child(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_child")._native_ptr(), 306700752);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t TreeItem::get_child_count() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_child_count")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

TypedArray<TreeItem> TreeItem::get_children() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_children")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<TreeItem>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<TreeItem>>(_gde_method_bind, _owner);
}

int32_t TreeItem::get_index() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("get_index")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TreeItem::move_before(TreeItem *p_item) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("move_before")._native_ptr(), 1819951137);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_item != nullptr ? &p_item->_owner : nullptr));
}

void TreeItem::move_after(TreeItem *p_item) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("move_after")._native_ptr(), 1819951137);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_item != nullptr ? &p_item->_owner : nullptr));
}

void TreeItem::call_recursive_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TreeItem::get_class_static()._native_ptr(), StringName("call_recursive")._native_ptr(), 2866548813);
	CHECK_METHOD_BIND(_gde_method_bind);
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
}

} // namespace godot
