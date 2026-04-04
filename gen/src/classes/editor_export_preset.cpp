/**************************************************************************/
/*  editor_export_preset.cpp                                              */
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

#include <godot_cpp/classes/editor_export_preset.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string_name.hpp>

namespace godot {

bool EditorExportPreset::has(const StringName &p_property) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("has")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_property);
}

PackedStringArray EditorExportPreset::get_files_to_export() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_files_to_export")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

Dictionary EditorExportPreset::get_customized_files() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_customized_files")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

int32_t EditorExportPreset::get_customized_files_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_customized_files_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool EditorExportPreset::has_export_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("has_export_file")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

EditorExportPreset::FileExportMode EditorExportPreset::get_file_export_mode(const String &p_path, EditorExportPreset::FileExportMode p_default) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_file_export_mode")._native_ptr(), 407825436);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (EditorExportPreset::FileExportMode(0)));
	int64_t p_default_encoded;
	PtrToArg<int64_t>::encode(p_default, &p_default_encoded);
	return (EditorExportPreset::FileExportMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_default_encoded);
}

Variant EditorExportPreset::get_project_setting(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_project_setting")._native_ptr(), 2138907829);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name);
}

String EditorExportPreset::get_preset_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_preset_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool EditorExportPreset::is_runnable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("is_runnable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool EditorExportPreset::are_advanced_options_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("are_advanced_options_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool EditorExportPreset::is_dedicated_server() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("is_dedicated_server")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

EditorExportPreset::ExportFilter EditorExportPreset::get_export_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_export_filter")._native_ptr(), 4227045696);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (EditorExportPreset::ExportFilter(0)));
	return (EditorExportPreset::ExportFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String EditorExportPreset::get_include_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_include_filter")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String EditorExportPreset::get_exclude_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_exclude_filter")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String EditorExportPreset::get_custom_features() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_custom_features")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

PackedStringArray EditorExportPreset::get_patches() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_patches")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

String EditorExportPreset::get_export_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_export_path")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String EditorExportPreset::get_encryption_in_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_encryption_in_filter")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String EditorExportPreset::get_encryption_ex_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_encryption_ex_filter")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool EditorExportPreset::get_encrypt_pck() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_encrypt_pck")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool EditorExportPreset::get_encrypt_directory() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_encrypt_directory")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

String EditorExportPreset::get_encryption_key() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_encryption_key")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

EditorExportPreset::ScriptExportMode EditorExportPreset::get_script_export_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_script_export_mode")._native_ptr(), 2835358398);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (EditorExportPreset::ScriptExportMode(0)));
	return (EditorExportPreset::ScriptExportMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Variant EditorExportPreset::get_or_env(const StringName &p_name, const String &p_env_var) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_or_env")._native_ptr(), 389838787);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name, &p_env_var);
}

String EditorExportPreset::get_version(const StringName &p_name, bool p_windows_version) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPreset::get_class_static()._native_ptr(), StringName("get_version")._native_ptr(), 1132184663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_windows_version_encoded;
	PtrToArg<bool>::encode(p_windows_version, &p_windows_version_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_name, &p_windows_version_encoded);
}

} // namespace godot
