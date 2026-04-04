/**************************************************************************/
/*  editor_property.cpp                                                   */
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

#include <godot_cpp/classes/editor_property.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

void EditorProperty::set_label(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_label")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

String EditorProperty::get_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("get_label")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void EditorProperty::set_read_only(bool p_read_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_read_only")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_read_only_encoded;
	PtrToArg<bool>::encode(p_read_only, &p_read_only_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_read_only_encoded);
}

bool EditorProperty::is_read_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_read_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_draw_label(bool p_draw_label) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_draw_label")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_draw_label_encoded;
	PtrToArg<bool>::encode(p_draw_label, &p_draw_label_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_label_encoded);
}

bool EditorProperty::is_draw_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_draw_label")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_draw_background(bool p_draw_background) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_draw_background")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_draw_background_encoded;
	PtrToArg<bool>::encode(p_draw_background, &p_draw_background_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_background_encoded);
}

bool EditorProperty::is_draw_background() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_draw_background")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_checkable(bool p_checkable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_checkable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_checkable_encoded;
	PtrToArg<bool>::encode(p_checkable, &p_checkable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_checkable_encoded);
}

bool EditorProperty::is_checkable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_checkable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_checked(bool p_checked) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_checked")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_checked_encoded;
	PtrToArg<bool>::encode(p_checked, &p_checked_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_checked_encoded);
}

bool EditorProperty::is_checked() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_checked")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_draw_warning(bool p_draw_warning) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_draw_warning")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_draw_warning_encoded;
	PtrToArg<bool>::encode(p_draw_warning, &p_draw_warning_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_warning_encoded);
}

bool EditorProperty::is_draw_warning() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_draw_warning")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_keying(bool p_keying) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_keying")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keying_encoded;
	PtrToArg<bool>::encode(p_keying, &p_keying_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_keying_encoded);
}

bool EditorProperty::is_keying() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_keying")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_deletable(bool p_deletable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_deletable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_deletable_encoded;
	PtrToArg<bool>::encode(p_deletable, &p_deletable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_deletable_encoded);
}

bool EditorProperty::is_deletable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_deletable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

StringName EditorProperty::get_edited_property() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("get_edited_property")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

Object *EditorProperty::get_edited_object() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("get_edited_object")._native_ptr(), 2050059866);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Object>(_gde_method_bind, _owner);
}

void EditorProperty::update_property() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("update_property")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorProperty::add_focusable(Control *p_control) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("add_focusable")._native_ptr(), 1496901182);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_control != nullptr ? &p_control->_owner : nullptr));
}

void EditorProperty::set_bottom_editor(Control *p_editor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_bottom_editor")._native_ptr(), 1496901182);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_editor != nullptr ? &p_editor->_owner : nullptr));
}

void EditorProperty::set_selectable(bool p_selectable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_selectable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_selectable_encoded;
	PtrToArg<bool>::encode(p_selectable, &p_selectable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_selectable_encoded);
}

bool EditorProperty::is_selectable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_selectable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_use_folding(bool p_use_folding) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_use_folding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_folding_encoded;
	PtrToArg<bool>::encode(p_use_folding, &p_use_folding_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_folding_encoded);
}

bool EditorProperty::is_using_folding() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_using_folding")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::set_name_split_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_name_split_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

float EditorProperty::get_name_split_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("get_name_split_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void EditorProperty::deselect() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("deselect")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool EditorProperty::is_selected() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("is_selected")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorProperty::select(int32_t p_focusable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("select")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_focusable_encoded;
	PtrToArg<int64_t>::encode(p_focusable, &p_focusable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_focusable_encoded);
}

void EditorProperty::set_object_and_property(Object *p_object, const StringName &p_property) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_object_and_property")._native_ptr(), 4157606280);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_property);
}

void EditorProperty::set_label_reference(Control *p_control) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("set_label_reference")._native_ptr(), 1496901182);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_control != nullptr ? &p_control->_owner : nullptr));
}

void EditorProperty::emit_changed(const StringName &p_property, const Variant &p_value, const StringName &p_field, bool p_changing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorProperty::get_class_static()._native_ptr(), StringName("emit_changed")._native_ptr(), 1822500399);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_changing_encoded;
	PtrToArg<bool>::encode(p_changing, &p_changing_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_property, &p_value, &p_field, &p_changing_encoded);
}

void EditorProperty::_update_property() {}

void EditorProperty::_set_read_only(bool p_read_only) {}

} // namespace godot
