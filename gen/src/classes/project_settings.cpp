/**************************************************************************/
/*  project_settings.cpp                                                  */
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

#include <godot_cpp/classes/project_settings.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string_name.hpp>

namespace godot {

ProjectSettings *ProjectSettings::singleton = nullptr;

ProjectSettings *ProjectSettings::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(ProjectSettings::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<ProjectSettings *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &ProjectSettings::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(ProjectSettings::get_class_static(), singleton);
		}
	}
	return singleton;
}

ProjectSettings::~ProjectSettings() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(ProjectSettings::get_class_static());
		singleton = nullptr;
	}
}

bool ProjectSettings::has_setting(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("has_setting")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

void ProjectSettings::set_setting(const String &p_name, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("set_setting")._native_ptr(), 402577236);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_value);
}

Variant ProjectSettings::get_setting(const String &p_name, const Variant &p_default_value) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("get_setting")._native_ptr(), 223050753);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name, &p_default_value);
}

Variant ProjectSettings::get_setting_with_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("get_setting_with_override")._native_ptr(), 2760726917);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name);
}

TypedArray<Dictionary> ProjectSettings::get_global_class_list() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("get_global_class_list")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner);
}

Variant ProjectSettings::get_setting_with_override_and_custom_features(const StringName &p_name, const PackedStringArray &p_features) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("get_setting_with_override_and_custom_features")._native_ptr(), 2434817427);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name, &p_features);
}

void ProjectSettings::set_order(const String &p_name, int32_t p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("set_order")._native_ptr(), 2956805083);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_position_encoded;
	PtrToArg<int64_t>::encode(p_position, &p_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_position_encoded);
}

int32_t ProjectSettings::get_order(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("get_order")._native_ptr(), 1321353865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name);
}

void ProjectSettings::set_initial_value(const String &p_name, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("set_initial_value")._native_ptr(), 402577236);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_value);
}

void ProjectSettings::set_as_basic(const String &p_name, bool p_basic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("set_as_basic")._native_ptr(), 2678287736);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_basic_encoded;
	PtrToArg<bool>::encode(p_basic, &p_basic_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_basic_encoded);
}

void ProjectSettings::set_as_internal(const String &p_name, bool p_internal) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("set_as_internal")._native_ptr(), 2678287736);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_internal_encoded;
	PtrToArg<bool>::encode(p_internal, &p_internal_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_internal_encoded);
}

void ProjectSettings::add_property_info(const Dictionary &p_hint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("add_property_info")._native_ptr(), 4155329257);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hint);
}

void ProjectSettings::set_restart_if_changed(const String &p_name, bool p_restart) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("set_restart_if_changed")._native_ptr(), 2678287736);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_restart_encoded;
	PtrToArg<bool>::encode(p_restart, &p_restart_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_restart_encoded);
}

void ProjectSettings::clear(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

String ProjectSettings::localize_path(const String &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("localize_path")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_path);
}

String ProjectSettings::globalize_path(const String &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("globalize_path")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_path);
}

Error ProjectSettings::save() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("save")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool ProjectSettings::load_resource_pack(const String &p_pack, bool p_replace_files, int32_t p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("load_resource_pack")._native_ptr(), 708980503);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_replace_files_encoded;
	PtrToArg<bool>::encode(p_replace_files, &p_replace_files_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_pack, &p_replace_files_encoded, &p_offset_encoded);
}

Error ProjectSettings::save_custom(const String &p_file) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("save_custom")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_file);
}

PackedStringArray ProjectSettings::get_changed_settings() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("get_changed_settings")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

bool ProjectSettings::check_changed_settings_in_group(const String &p_setting_prefix) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ProjectSettings::get_class_static()._native_ptr(), StringName("check_changed_settings_in_group")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_setting_prefix);
}

} // namespace godot
