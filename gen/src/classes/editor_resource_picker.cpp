/**************************************************************************/
/*  editor_resource_picker.cpp                                            */
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

#include <godot_cpp/classes/editor_resource_picker.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/object.hpp>

namespace godot {

void EditorResourcePicker::set_base_type(const String &p_base_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("set_base_type")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_base_type);
}

String EditorResourcePicker::get_base_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("get_base_type")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

PackedStringArray EditorResourcePicker::get_allowed_types() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("get_allowed_types")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void EditorResourcePicker::set_edited_resource(const Ref<Resource> &p_resource) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("set_edited_resource")._native_ptr(), 968641751);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_resource != nullptr ? &p_resource->_owner : nullptr));
}

Ref<Resource> EditorResourcePicker::get_edited_resource() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("get_edited_resource")._native_ptr(), 2674603643);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Resource>()));
	return Ref<Resource>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Resource>(_gde_method_bind, _owner));
}

void EditorResourcePicker::set_toggle_mode(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("set_toggle_mode")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool EditorResourcePicker::is_toggle_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("is_toggle_mode")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorResourcePicker::set_toggle_pressed(bool p_pressed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("set_toggle_pressed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pressed_encoded;
	PtrToArg<bool>::encode(p_pressed, &p_pressed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pressed_encoded);
}

void EditorResourcePicker::set_editable(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("set_editable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool EditorResourcePicker::is_editable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorResourcePicker::get_class_static()._native_ptr(), StringName("is_editable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorResourcePicker::_set_create_options(Object *p_menu_node) {}

bool EditorResourcePicker::_handle_menu_selected(int32_t p_id) {
	return false;
}

} // namespace godot
