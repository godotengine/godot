/**************************************************************************/
/*  file_dialog.cpp                                                       */
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

#include <godot_cpp/classes/file_dialog.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/line_edit.hpp>
#include <godot_cpp/classes/v_box_container.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

void FileDialog::clear_filters() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("clear_filters")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void FileDialog::add_filter(const String &p_filter, const String &p_description, const String &p_mime_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("add_filter")._native_ptr(), 914921954);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter, &p_description, &p_mime_type);
}

void FileDialog::set_filters(const PackedStringArray &p_filters) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_filters")._native_ptr(), 4015028928);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filters);
}

PackedStringArray FileDialog::get_filters() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_filters")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void FileDialog::clear_filename_filter() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("clear_filename_filter")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void FileDialog::set_filename_filter(const String &p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_filename_filter")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter);
}

String FileDialog::get_filename_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_filename_filter")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String FileDialog::get_option_name(int32_t p_option) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_option_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_option_encoded;
	PtrToArg<int64_t>::encode(p_option, &p_option_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_option_encoded);
}

PackedStringArray FileDialog::get_option_values(int32_t p_option) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_option_values")._native_ptr(), 647634434);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	int64_t p_option_encoded;
	PtrToArg<int64_t>::encode(p_option, &p_option_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_option_encoded);
}

int32_t FileDialog::get_option_default(int32_t p_option) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_option_default")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_option_encoded;
	PtrToArg<int64_t>::encode(p_option, &p_option_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_option_encoded);
}

void FileDialog::set_option_name(int32_t p_option, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_option_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_option_encoded;
	PtrToArg<int64_t>::encode(p_option, &p_option_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_option_encoded, &p_name);
}

void FileDialog::set_option_values(int32_t p_option, const PackedStringArray &p_values) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_option_values")._native_ptr(), 3353661094);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_option_encoded;
	PtrToArg<int64_t>::encode(p_option, &p_option_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_option_encoded, &p_values);
}

void FileDialog::set_option_default(int32_t p_option, int32_t p_default_value_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_option_default")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_option_encoded;
	PtrToArg<int64_t>::encode(p_option, &p_option_encoded);
	int64_t p_default_value_index_encoded;
	PtrToArg<int64_t>::encode(p_default_value_index, &p_default_value_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_option_encoded, &p_default_value_index_encoded);
}

void FileDialog::set_option_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_option_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t FileDialog::get_option_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_option_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FileDialog::add_option(const String &p_name, const PackedStringArray &p_values, int32_t p_default_value_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("add_option")._native_ptr(), 149592325);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_default_value_index_encoded;
	PtrToArg<int64_t>::encode(p_default_value_index, &p_default_value_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_values, &p_default_value_index_encoded);
}

Dictionary FileDialog::get_selected_options() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_selected_options")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

String FileDialog::get_current_dir() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_current_dir")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String FileDialog::get_current_file() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_current_file")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String FileDialog::get_current_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_current_path")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void FileDialog::set_current_dir(const String &p_dir) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_current_dir")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_dir);
}

void FileDialog::set_current_file(const String &p_file) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_current_file")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_file);
}

void FileDialog::set_current_path(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_current_path")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void FileDialog::set_mode_overrides_title(bool p_override) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_mode_overrides_title")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_override_encoded;
	PtrToArg<bool>::encode(p_override, &p_override_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_override_encoded);
}

bool FileDialog::is_mode_overriding_title() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("is_mode_overriding_title")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FileDialog::set_file_mode(FileDialog::FileMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_file_mode")._native_ptr(), 3654936397);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

FileDialog::FileMode FileDialog::get_file_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_file_mode")._native_ptr(), 4074825319);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (FileDialog::FileMode(0)));
	return (FileDialog::FileMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FileDialog::set_display_mode(FileDialog::DisplayMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_display_mode")._native_ptr(), 2692197101);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

FileDialog::DisplayMode FileDialog::get_display_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_display_mode")._native_ptr(), 1092104624);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (FileDialog::DisplayMode(0)));
	return (FileDialog::DisplayMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

VBoxContainer *FileDialog::get_vbox() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_vbox")._native_ptr(), 915758477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<VBoxContainer>(_gde_method_bind, _owner);
}

LineEdit *FileDialog::get_line_edit() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_line_edit")._native_ptr(), 4071694264);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<LineEdit>(_gde_method_bind, _owner);
}

void FileDialog::set_access(FileDialog::Access p_access) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_access")._native_ptr(), 4104413466);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_access_encoded;
	PtrToArg<int64_t>::encode(p_access, &p_access_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_access_encoded);
}

FileDialog::Access FileDialog::get_access() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_access")._native_ptr(), 3344081076);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (FileDialog::Access(0)));
	return (FileDialog::Access)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FileDialog::set_root_subfolder(const String &p_dir) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_root_subfolder")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_dir);
}

String FileDialog::get_root_subfolder() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_root_subfolder")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void FileDialog::set_show_hidden_files(bool p_show) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_show_hidden_files")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_show_encoded;
	PtrToArg<bool>::encode(p_show, &p_show_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_show_encoded);
}

bool FileDialog::is_showing_hidden_files() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("is_showing_hidden_files")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FileDialog::set_use_native_dialog(bool p_native) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_use_native_dialog")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_native_encoded;
	PtrToArg<bool>::encode(p_native, &p_native_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_native_encoded);
}

bool FileDialog::get_use_native_dialog() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_use_native_dialog")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FileDialog::set_customization_flag_enabled(FileDialog::Customization p_flag, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_customization_flag_enabled")._native_ptr(), 3849177100);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flag_encoded, &p_enabled_encoded);
}

bool FileDialog::is_customization_flag_enabled(FileDialog::Customization p_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("is_customization_flag_enabled")._native_ptr(), 3722277863);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_flag_encoded);
}

void FileDialog::deselect_all() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("deselect_all")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void FileDialog::set_favorite_list(const PackedStringArray &p_favorites) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_favorite_list")._native_ptr(), 4015028928);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, &p_favorites);
}

PackedStringArray FileDialog::get_favorite_list() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_favorite_list")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, nullptr);
}

void FileDialog::set_recent_list(const PackedStringArray &p_recents) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_recent_list")._native_ptr(), 4015028928);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, &p_recents);
}

PackedStringArray FileDialog::get_recent_list() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("get_recent_list")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, nullptr);
}

void FileDialog::set_get_icon_callback(const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_get_icon_callback")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, &p_callback);
}

void FileDialog::set_get_thumbnail_callback(const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("set_get_thumbnail_callback")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, &p_callback);
}

void FileDialog::popup_file_dialog() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("popup_file_dialog")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void FileDialog::invalidate() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FileDialog::get_class_static()._native_ptr(), StringName("invalidate")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
