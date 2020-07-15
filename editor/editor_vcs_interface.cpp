/*************************************************************************/
/*  editor_vcs_interface.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "editor_vcs_interface.h"

EditorVCSInterface *EditorVCSInterface::singleton = nullptr;

void EditorVCSInterface::_bind_methods() {
	// Proxy end points that act as fallbacks to unavailability of a function in the VCS addon
	ClassDB::bind_method(D_METHOD("_initialize", "project_root_path"), &EditorVCSInterface::_initialize);
	ClassDB::bind_method(D_METHOD("_is_vcs_initialized"), &EditorVCSInterface::_is_vcs_initialized);
	ClassDB::bind_method(D_METHOD("_get_vcs_name"), &EditorVCSInterface::_get_vcs_name);
	ClassDB::bind_method(D_METHOD("_shut_down"), &EditorVCSInterface::_shut_down);
	ClassDB::bind_method(D_METHOD("_get_project_name"), &EditorVCSInterface::_get_project_name);
	ClassDB::bind_method(D_METHOD("_get_modified_files_data"), &EditorVCSInterface::_get_modified_files_data);
	ClassDB::bind_method(D_METHOD("_commit", "msg"), &EditorVCSInterface::_commit);
	ClassDB::bind_method(D_METHOD("_get_file_diff", "file_path", "area"), &EditorVCSInterface::_get_file_diff);
	ClassDB::bind_method(D_METHOD("_stage_file", "file_path"), &EditorVCSInterface::_stage_file);
	ClassDB::bind_method(D_METHOD("_unstage_file", "file_path"), &EditorVCSInterface::_unstage_file);
	ClassDB::bind_method(D_METHOD("_discard_file", "file_path"), &EditorVCSInterface::_discard_file);
	ClassDB::bind_method(D_METHOD("_get_previous_commits"), &EditorVCSInterface::_get_previous_commits);
	ClassDB::bind_method(D_METHOD("_get_branch_list"), &EditorVCSInterface::_get_branch_list);
	ClassDB::bind_method(D_METHOD("_checkout_branch", "branch"), &EditorVCSInterface::_checkout_branch);
	ClassDB::bind_method(D_METHOD("_push"), &EditorVCSInterface::_push);
	ClassDB::bind_method(D_METHOD("_pull"), &EditorVCSInterface::_pull);
	ClassDB::bind_method(D_METHOD("_fetch"), &EditorVCSInterface::_fetch);
	ClassDB::bind_method(D_METHOD("_set_up_credentials"), &EditorVCSInterface::_set_up_credentials);

	ClassDB::bind_method(D_METHOD("is_addon_ready"), &EditorVCSInterface::is_addon_ready);

	// API methods that redirect calls to the proxy end points
	ClassDB::bind_method(D_METHOD("initialize", "project_root_path"), &EditorVCSInterface::initialize);
	ClassDB::bind_method(D_METHOD("is_vcs_initialized"), &EditorVCSInterface::is_vcs_initialized);
	ClassDB::bind_method(D_METHOD("get_modified_files_data"), &EditorVCSInterface::get_modified_files_data);
	ClassDB::bind_method(D_METHOD("stage_file", "file_path"), &EditorVCSInterface::stage_file);
	ClassDB::bind_method(D_METHOD("unstage_file", "file_path"), &EditorVCSInterface::unstage_file);
	ClassDB::bind_method(D_METHOD("discard_file", "file_path"), &EditorVCSInterface::discard_file);
	ClassDB::bind_method(D_METHOD("commit", "msg"), &EditorVCSInterface::commit);
	ClassDB::bind_method(D_METHOD("get_file_diff", "file_path", "area"), &EditorVCSInterface::get_file_diff);
	ClassDB::bind_method(D_METHOD("shut_down"), &EditorVCSInterface::shut_down);
	ClassDB::bind_method(D_METHOD("get_project_name"), &EditorVCSInterface::get_project_name);
	ClassDB::bind_method(D_METHOD("get_vcs_name"), &EditorVCSInterface::get_vcs_name);
	ClassDB::bind_method(D_METHOD("get_previous_commits"), &EditorVCSInterface::get_previous_commits);
	ClassDB::bind_method(D_METHOD("get_branch_list"), &EditorVCSInterface::get_branch_list);
	ClassDB::bind_method(D_METHOD("checkout_branch", "branch"), &EditorVCSInterface::checkout_branch);
	ClassDB::bind_method(D_METHOD("push"), &EditorVCSInterface::push);
	ClassDB::bind_method(D_METHOD("pull"), &EditorVCSInterface::pull);
	ClassDB::bind_method(D_METHOD("fetch"), &EditorVCSInterface::fetch);
	ClassDB::bind_method(D_METHOD("set_up_credentials"), &EditorVCSInterface::set_up_credentials);

	BIND_ENUM_CONSTANT(CHANGE_TYPE_NEW);
	BIND_ENUM_CONSTANT(CHANGE_TYPE_MODIFIED);
	BIND_ENUM_CONSTANT(CHANGE_TYPE_RENAMED);
	BIND_ENUM_CONSTANT(CHANGE_TYPE_DELETED);
	BIND_ENUM_CONSTANT(CHANGE_TYPE_TYPECHANGE);
	BIND_ENUM_CONSTANT(CHANGE_TYPE_UNMERGED);

	BIND_ENUM_CONSTANT(TREE_AREA_COMMIT);
	BIND_ENUM_CONSTANT(TREE_AREA_STAGED);
	BIND_ENUM_CONSTANT(TREE_AREA_UNSTAGED);
}

bool EditorVCSInterface::_initialize(String p_project_root_path) {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return true;
}

bool EditorVCSInterface::_is_vcs_initialized() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return false;
}

Dictionary EditorVCSInterface::_get_modified_files_data() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return Dictionary();
}

void EditorVCSInterface::_stage_file(String p_file_path) {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
}

void EditorVCSInterface::_unstage_file(String p_file_path) {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
}

void EditorVCSInterface::_commit(String p_msg) {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
}

void EditorVCSInterface::_discard_file(String p_file_path) {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
}

Array EditorVCSInterface::_get_file_diff(String p_identifier, TreeArea area) {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return Array();
}

Array EditorVCSInterface::_get_branch_list() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return Array();
}

bool EditorVCSInterface::_checkout_branch(String p_branch) {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return false;
}

Array EditorVCSInterface::_get_previous_commits() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return Array();
}

bool EditorVCSInterface::_shut_down() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return false;
}

String EditorVCSInterface::_get_project_name() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return String();
}

String EditorVCSInterface::_get_vcs_name() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return "";
}

void EditorVCSInterface::_pull() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return;
}

void EditorVCSInterface::_push() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return;
}

void EditorVCSInterface::_fetch() {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
	return;
}

void EditorVCSInterface::_set_up_credentials(String p_username, String p_password) {
	ERR_PRINT("Selected VCS addon does not implement \"" + String(__FUNCTION__) + "\"function. This warning will be suppressed.");
}

bool EditorVCSInterface::initialize(String p_project_root_path) {
	is_initialized = call("_initialize", p_project_root_path);
	return is_initialized;
}

bool EditorVCSInterface::is_vcs_initialized() {
	return call("_is_vcs_initialized");
}

Dictionary EditorVCSInterface::get_modified_files_data() {
	return call("_get_modified_files_data");
}

void EditorVCSInterface::stage_file(String p_file_path) {
	if (is_addon_ready()) {
		call("_stage_file", p_file_path);
	}
}

void EditorVCSInterface::unstage_file(String p_file_path) {
	if (is_addon_ready()) {
		call("_unstage_file", p_file_path);
	}
}

void EditorVCSInterface::discard_file(String p_file_path) {
	if (is_addon_ready()) {

		call("_discard_file", p_file_path);
	}
}

bool EditorVCSInterface::is_addon_ready() {
	return is_initialized;
}

void EditorVCSInterface::commit(String p_msg) {
	if (is_addon_ready()) {
		call("_commit", p_msg);
	}
}

Array EditorVCSInterface::get_file_diff(String p_identifier, TreeArea area) {

	if (is_addon_ready()) {

		return call("_get_file_diff", p_identifier, area);
	}
	return Array();
}

Array EditorVCSInterface::get_previous_commits() {

	if (is_addon_ready()) {

		return call("_get_previous_commits");
	}
	return Array();
}

Array EditorVCSInterface::get_branch_list() {

	if (is_addon_ready()) {

		return call("_get_branch_list");
	}
	return Array();
}

bool EditorVCSInterface::checkout_branch(String p_branch) {

	if (is_addon_ready()) {

		return call("_checkout_branch", p_branch);
	}
	return false;
}

void EditorVCSInterface::pull() {
	if (is_addon_ready()) {

		call("_pull");
	}
}

void EditorVCSInterface::push() {
	if (is_addon_ready()) {

		call("_push");
	}
}

void EditorVCSInterface::fetch() {
	if (is_addon_ready()) {

		call("_fetch");
	}
}

bool EditorVCSInterface::shut_down() {
	return call("_shut_down");
}

String EditorVCSInterface::get_project_name() {
	return call("_get_project_name");
}

String EditorVCSInterface::get_vcs_name() {
	return call("_get_vcs_name");
}

void EditorVCSInterface::set_up_credentials(String p_username, String p_password) {

	call("_set_up_credentials", p_username, p_password);
}

EditorVCSInterface::EditorVCSInterface() {
	is_initialized = false;
}

EditorVCSInterface::~EditorVCSInterface() {
}

EditorVCSInterface *EditorVCSInterface::get_singleton() {
	return singleton;
}

void EditorVCSInterface::set_singleton(EditorVCSInterface *p_singleton) {
	singleton = p_singleton;
}
