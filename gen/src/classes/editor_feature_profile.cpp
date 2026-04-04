/**************************************************************************/
/*  editor_feature_profile.cpp                                            */
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

#include <godot_cpp/classes/editor_feature_profile.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void EditorFeatureProfile::set_disable_class(const StringName &p_class_name, bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("set_disable_class")._native_ptr(), 2524380260);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_class_name, &p_disable_encoded);
}

bool EditorFeatureProfile::is_class_disabled(const StringName &p_class_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("is_class_disabled")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_class_name);
}

void EditorFeatureProfile::set_disable_class_editor(const StringName &p_class_name, bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("set_disable_class_editor")._native_ptr(), 2524380260);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_class_name, &p_disable_encoded);
}

bool EditorFeatureProfile::is_class_editor_disabled(const StringName &p_class_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("is_class_editor_disabled")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_class_name);
}

void EditorFeatureProfile::set_disable_class_property(const StringName &p_class_name, const StringName &p_property, bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("set_disable_class_property")._native_ptr(), 865197084);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_class_name, &p_property, &p_disable_encoded);
}

bool EditorFeatureProfile::is_class_property_disabled(const StringName &p_class_name, const StringName &p_property) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("is_class_property_disabled")._native_ptr(), 471820014);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_class_name, &p_property);
}

void EditorFeatureProfile::set_disable_feature(EditorFeatureProfile::Feature p_feature, bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("set_disable_feature")._native_ptr(), 1884871044);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_feature_encoded, &p_disable_encoded);
}

bool EditorFeatureProfile::is_feature_disabled(EditorFeatureProfile::Feature p_feature) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("is_feature_disabled")._native_ptr(), 2974403161);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_feature_encoded);
}

String EditorFeatureProfile::get_feature_name(EditorFeatureProfile::Feature p_feature) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("get_feature_name")._native_ptr(), 3401335809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_feature_encoded);
}

Error EditorFeatureProfile::save_to_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("save_to_file")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

Error EditorFeatureProfile::load_from_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorFeatureProfile::get_class_static()._native_ptr(), StringName("load_from_file")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

} // namespace godot
