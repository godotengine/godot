/**************************************************************************/
/*  dir_access.cpp                                                        */
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

#include <godot_cpp/classes/dir_access.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Ref<DirAccess> DirAccess::open(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("open")._native_ptr(), 1923528528);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<DirAccess>()));
	return Ref<DirAccess>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<DirAccess>(_gde_method_bind, nullptr, &p_path));
}

Error DirAccess::get_open_error() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_open_error")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, nullptr);
}

Ref<DirAccess> DirAccess::create_temp(const String &p_prefix, bool p_keep) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("create_temp")._native_ptr(), 812913566);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<DirAccess>()));
	int8_t p_keep_encoded;
	PtrToArg<bool>::encode(p_keep, &p_keep_encoded);
	return Ref<DirAccess>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<DirAccess>(_gde_method_bind, nullptr, &p_prefix, &p_keep_encoded));
}

Error DirAccess::list_dir_begin() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("list_dir_begin")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String DirAccess::get_next() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_next")._native_ptr(), 2841200299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool DirAccess::current_is_dir() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("current_is_dir")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void DirAccess::list_dir_end() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("list_dir_end")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

PackedStringArray DirAccess::get_files() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_files")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

PackedStringArray DirAccess::get_files_at(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_files_at")._native_ptr(), 3538744774);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, nullptr, &p_path);
}

PackedStringArray DirAccess::get_directories() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_directories")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

PackedStringArray DirAccess::get_directories_at(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_directories_at")._native_ptr(), 3538744774);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, nullptr, &p_path);
}

int32_t DirAccess::get_drive_count() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_drive_count")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, nullptr);
}

String DirAccess::get_drive_name(int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_drive_name")._native_ptr(), 990163283);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, nullptr, &p_idx_encoded);
}

int32_t DirAccess::get_current_drive() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_current_drive")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Error DirAccess::change_dir(const String &p_to_dir) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("change_dir")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_to_dir);
}

String DirAccess::get_current_dir(bool p_include_drive) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_current_dir")._native_ptr(), 1287308131);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_include_drive_encoded;
	PtrToArg<bool>::encode(p_include_drive, &p_include_drive_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_include_drive_encoded);
}

Error DirAccess::make_dir(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("make_dir")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

Error DirAccess::make_dir_absolute(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("make_dir_absolute")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, nullptr, &p_path);
}

Error DirAccess::make_dir_recursive(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("make_dir_recursive")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

Error DirAccess::make_dir_recursive_absolute(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("make_dir_recursive_absolute")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, nullptr, &p_path);
}

bool DirAccess::file_exists(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("file_exists")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

bool DirAccess::dir_exists(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("dir_exists")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

bool DirAccess::dir_exists_absolute(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("dir_exists_absolute")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, nullptr, &p_path);
}

uint64_t DirAccess::get_space_left() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_space_left")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

Error DirAccess::copy(const String &p_from, const String &p_to, int32_t p_chmod_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("copy")._native_ptr(), 1063198817);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_chmod_flags_encoded;
	PtrToArg<int64_t>::encode(p_chmod_flags, &p_chmod_flags_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_from, &p_to, &p_chmod_flags_encoded);
}

Error DirAccess::copy_absolute(const String &p_from, const String &p_to, int32_t p_chmod_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("copy_absolute")._native_ptr(), 1063198817);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_chmod_flags_encoded;
	PtrToArg<int64_t>::encode(p_chmod_flags, &p_chmod_flags_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, nullptr, &p_from, &p_to, &p_chmod_flags_encoded);
}

Error DirAccess::rename(const String &p_from, const String &p_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("rename")._native_ptr(), 852856452);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_from, &p_to);
}

Error DirAccess::rename_absolute(const String &p_from, const String &p_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("rename_absolute")._native_ptr(), 852856452);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, nullptr, &p_from, &p_to);
}

Error DirAccess::remove(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("remove")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

Error DirAccess::remove_absolute(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("remove_absolute")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, nullptr, &p_path);
}

bool DirAccess::is_link(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("is_link")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

String DirAccess::read_link(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("read_link")._native_ptr(), 1703090593);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_path);
}

Error DirAccess::create_link(const String &p_source, const String &p_target) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("create_link")._native_ptr(), 852856452);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_source, &p_target);
}

bool DirAccess::is_bundle(const String &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("is_bundle")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

void DirAccess::set_include_navigational(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("set_include_navigational")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool DirAccess::get_include_navigational() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_include_navigational")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void DirAccess::set_include_hidden(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("set_include_hidden")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool DirAccess::get_include_hidden() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_include_hidden")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

String DirAccess::get_filesystem_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("get_filesystem_type")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool DirAccess::is_case_sensitive(const String &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("is_case_sensitive")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

bool DirAccess::is_equivalent(const String &p_path_a, const String &p_path_b) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DirAccess::get_class_static()._native_ptr(), StringName("is_equivalent")._native_ptr(), 820780508);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path_a, &p_path_b);
}

} // namespace godot
