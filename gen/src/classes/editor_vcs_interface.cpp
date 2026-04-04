/**************************************************************************/
/*  editor_vcs_interface.cpp                                              */
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

#include <godot_cpp/classes/editor_vcs_interface.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Dictionary EditorVCSInterface::create_diff_line(int32_t p_new_line_no, int32_t p_old_line_no, const String &p_content, const String &p_status) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorVCSInterface::get_class_static()._native_ptr(), StringName("create_diff_line")._native_ptr(), 2901184053);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_new_line_no_encoded;
	PtrToArg<int64_t>::encode(p_new_line_no, &p_new_line_no_encoded);
	int64_t p_old_line_no_encoded;
	PtrToArg<int64_t>::encode(p_old_line_no, &p_old_line_no_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_new_line_no_encoded, &p_old_line_no_encoded, &p_content, &p_status);
}

Dictionary EditorVCSInterface::create_diff_hunk(int32_t p_old_start, int32_t p_new_start, int32_t p_old_lines, int32_t p_new_lines) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorVCSInterface::get_class_static()._native_ptr(), StringName("create_diff_hunk")._native_ptr(), 3784842090);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_old_start_encoded;
	PtrToArg<int64_t>::encode(p_old_start, &p_old_start_encoded);
	int64_t p_new_start_encoded;
	PtrToArg<int64_t>::encode(p_new_start, &p_new_start_encoded);
	int64_t p_old_lines_encoded;
	PtrToArg<int64_t>::encode(p_old_lines, &p_old_lines_encoded);
	int64_t p_new_lines_encoded;
	PtrToArg<int64_t>::encode(p_new_lines, &p_new_lines_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_old_start_encoded, &p_new_start_encoded, &p_old_lines_encoded, &p_new_lines_encoded);
}

Dictionary EditorVCSInterface::create_diff_file(const String &p_new_file, const String &p_old_file) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorVCSInterface::get_class_static()._native_ptr(), StringName("create_diff_file")._native_ptr(), 2723227684);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_new_file, &p_old_file);
}

Dictionary EditorVCSInterface::create_commit(const String &p_msg, const String &p_author, const String &p_id, int64_t p_unix_timestamp, int64_t p_offset_minutes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorVCSInterface::get_class_static()._native_ptr(), StringName("create_commit")._native_ptr(), 1075983584);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_unix_timestamp_encoded;
	PtrToArg<int64_t>::encode(p_unix_timestamp, &p_unix_timestamp_encoded);
	int64_t p_offset_minutes_encoded;
	PtrToArg<int64_t>::encode(p_offset_minutes, &p_offset_minutes_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_msg, &p_author, &p_id, &p_unix_timestamp_encoded, &p_offset_minutes_encoded);
}

Dictionary EditorVCSInterface::create_status_file(const String &p_file_path, EditorVCSInterface::ChangeType p_change_type, EditorVCSInterface::TreeArea p_area) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorVCSInterface::get_class_static()._native_ptr(), StringName("create_status_file")._native_ptr(), 1083471673);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_change_type_encoded;
	PtrToArg<int64_t>::encode(p_change_type, &p_change_type_encoded);
	int64_t p_area_encoded;
	PtrToArg<int64_t>::encode(p_area, &p_area_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_file_path, &p_change_type_encoded, &p_area_encoded);
}

Dictionary EditorVCSInterface::add_diff_hunks_into_diff_file(const Dictionary &p_diff_file, const TypedArray<Dictionary> &p_diff_hunks) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorVCSInterface::get_class_static()._native_ptr(), StringName("add_diff_hunks_into_diff_file")._native_ptr(), 4015243225);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_diff_file, &p_diff_hunks);
}

Dictionary EditorVCSInterface::add_line_diffs_into_diff_hunk(const Dictionary &p_diff_hunk, const TypedArray<Dictionary> &p_line_diffs) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorVCSInterface::get_class_static()._native_ptr(), StringName("add_line_diffs_into_diff_hunk")._native_ptr(), 4015243225);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_diff_hunk, &p_line_diffs);
}

void EditorVCSInterface::popup_error(const String &p_msg) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorVCSInterface::get_class_static()._native_ptr(), StringName("popup_error")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msg);
}

bool EditorVCSInterface::_initialize(const String &p_project_path) {
	return false;
}

void EditorVCSInterface::_set_credentials(const String &p_username, const String &p_password, const String &p_ssh_public_key_path, const String &p_ssh_private_key_path, const String &p_ssh_passphrase) {}

TypedArray<Dictionary> EditorVCSInterface::_get_modified_files_data() {
	return TypedArray<Dictionary>();
}

void EditorVCSInterface::_stage_file(const String &p_file_path) {}

void EditorVCSInterface::_unstage_file(const String &p_file_path) {}

void EditorVCSInterface::_discard_file(const String &p_file_path) {}

void EditorVCSInterface::_commit(const String &p_msg) {}

TypedArray<Dictionary> EditorVCSInterface::_get_diff(const String &p_identifier, int32_t p_area) {
	return TypedArray<Dictionary>();
}

bool EditorVCSInterface::_shut_down() {
	return false;
}

String EditorVCSInterface::_get_vcs_name() {
	return String();
}

TypedArray<Dictionary> EditorVCSInterface::_get_previous_commits(int32_t p_max_commits) {
	return TypedArray<Dictionary>();
}

TypedArray<String> EditorVCSInterface::_get_branch_list() {
	return TypedArray<String>();
}

TypedArray<String> EditorVCSInterface::_get_remotes() {
	return TypedArray<String>();
}

void EditorVCSInterface::_create_branch(const String &p_branch_name) {}

void EditorVCSInterface::_remove_branch(const String &p_branch_name) {}

void EditorVCSInterface::_create_remote(const String &p_remote_name, const String &p_remote_url) {}

void EditorVCSInterface::_remove_remote(const String &p_remote_name) {}

String EditorVCSInterface::_get_current_branch_name() {
	return String();
}

bool EditorVCSInterface::_checkout_branch(const String &p_branch_name) {
	return false;
}

void EditorVCSInterface::_pull(const String &p_remote) {}

void EditorVCSInterface::_push(const String &p_remote, bool p_force) {}

void EditorVCSInterface::_fetch(const String &p_remote) {}

TypedArray<Dictionary> EditorVCSInterface::_get_line_diff(const String &p_file_path, const String &p_text) {
	return TypedArray<Dictionary>();
}

} // namespace godot
