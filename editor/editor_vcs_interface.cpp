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
#include "editor_node.h"

EditorVCSInterface *EditorVCSInterface::singleton = nullptr;

bool EditorVCSInterface::_initialize(String p_project_root_path) {
	_not_implemented_function(__FUNCTION__);
	return true;
}

bool EditorVCSInterface::_is_vcs_initialized() {
	_not_implemented_function(__FUNCTION__);
	return false;
}

Array EditorVCSInterface::_get_modified_files_data() {
	_not_implemented_function(__FUNCTION__);
	return Array();
}

void EditorVCSInterface::_stage_file(String p_file_path) {
	_not_implemented_function(__FUNCTION__);
}

void EditorVCSInterface::_unstage_file(String p_file_path) {
	_not_implemented_function(__FUNCTION__);
}

void EditorVCSInterface::_commit(String p_msg) {
	_not_implemented_function(__FUNCTION__);
}

void EditorVCSInterface::_discard_file(String p_file_path) {
	_not_implemented_function(__FUNCTION__);
}

Array EditorVCSInterface::_get_file_diff(String p_identifier, TreeArea area) {
	_not_implemented_function(__FUNCTION__);
	return Array();
}

Array EditorVCSInterface::_get_branch_list() {
	_not_implemented_function(__FUNCTION__);
	return Array();
}

bool EditorVCSInterface::_checkout_branch(String p_branch) {
	_not_implemented_function(__FUNCTION__);
	return false;
}

Array EditorVCSInterface::_get_previous_commits() {
	_not_implemented_function(__FUNCTION__);
	return Array();
}

bool EditorVCSInterface::_shut_down() {
	_not_implemented_function(__FUNCTION__);
	return false;
}

String EditorVCSInterface::_get_project_name() {
	_not_implemented_function(__FUNCTION__);
	return String();
}

String EditorVCSInterface::_get_vcs_name() {
	_not_implemented_function(__FUNCTION__);
	return "";
}

void EditorVCSInterface::_pull() {
	_not_implemented_function(__FUNCTION__);
	return;
}

void EditorVCSInterface::_push() {
	_not_implemented_function(__FUNCTION__);
	return;
}

void EditorVCSInterface::_fetch() {
	_not_implemented_function(__FUNCTION__);
	return;
}

void EditorVCSInterface::_set_up_credentials(String p_username, String p_password) {
	_not_implemented_function(__FUNCTION__);
}

void EditorVCSInterface::_popup_error(String p_msg) {
	EditorNode::get_singleton()->show_warning(p_msg, get_vcs_name() + TTR(": Error"));
}

bool EditorVCSInterface::initialize(String p_project_root_path) {
	is_initialized = call("_initialize", p_project_root_path);
	return is_initialized;
}

bool EditorVCSInterface::is_vcs_initialized() {
	return call("_is_vcs_initialized");
}

List<EditorVCSInterface::StatusFile> EditorVCSInterface::get_modified_files_data() {
	List<EditorVCSInterface::StatusFile> status_files;

	Array result = call("_get_modified_files_data");
	for (int i = 0; i < result.size(); i++) {
		status_files.push_back(_convert_status_file(result[i]));
	}

	return status_files;
}

void EditorVCSInterface::stage_file(String p_file_path) {
	if (is_plugin_ready()) {
		call("_stage_file", p_file_path);
	}
}

void EditorVCSInterface::unstage_file(String p_file_path) {
	if (is_plugin_ready()) {
		call("_unstage_file", p_file_path);
	}
}

void EditorVCSInterface::discard_file(String p_file_path) {
	if (is_plugin_ready()) {
		call("_discard_file", p_file_path);
	}
}

bool EditorVCSInterface::is_plugin_ready() {
	return is_initialized;
}

void EditorVCSInterface::commit(String p_msg) {
	if (is_plugin_ready()) {
		call("_commit", p_msg);
	}
}

List<EditorVCSInterface::DiffFile> EditorVCSInterface::get_file_diff(String p_identifier, TreeArea p_area) {
	List<DiffFile> diff_files = List<DiffFile>();

	if (is_plugin_ready()) {
		Array result = call("_get_file_diff", p_identifier, p_area);
		for (int i = 0; i < result.size(); i++) {
			diff_files.push_back(_convert_diff_file(result[i]));
		}
	}

	return diff_files;
}

List<EditorVCSInterface::Commit> EditorVCSInterface::get_previous_commits() {
	List<EditorVCSInterface::Commit> commits = List<EditorVCSInterface::Commit>();
	if (is_plugin_ready()) {
		Array result = call("_get_previous_commits");
		for (int i = 0; i < result.size(); i++) {
			commits.push_back(_convert_commit(result[i]));
		}
	}
	return commits;
}

List<String> EditorVCSInterface::get_branch_list() {
	List<String> branch_list = List<String>();

	if (is_plugin_ready()) {
		Array result = call("_get_branch_list");
		for (int i = 0; i < result.size(); i++) {
			branch_list.push_back(result[i]);
		}
	}
	return branch_list;
}

bool EditorVCSInterface::checkout_branch(String p_branch) {
	if (is_plugin_ready()) {
		return call("_checkout_branch", p_branch);
	}
	return false;
}

void EditorVCSInterface::pull() {
	if (is_plugin_ready()) {
		call("_pull");
	}
}

void EditorVCSInterface::push() {
	if (is_plugin_ready()) {
		call("_push");
	}
}

void EditorVCSInterface::fetch() {
	if (is_plugin_ready()) {
		call("_fetch");
	}
}

List<EditorVCSInterface::DiffHunk> EditorVCSInterface::get_line_diff(String p_file_path, String p_text) {
	List<DiffHunk> diff_hunks = List<DiffHunk>();
	if (is_plugin_ready()) {
		Array result = call("_get_line_diff", p_file_path, p_text);
		for (int i = 0; i < result.size(); i++) {
			diff_hunks.push_back(_convert_diff_hunk(result[i]));
		}
	}

	return diff_hunks;
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

Dictionary EditorVCSInterface::_create_diff_line(int new_line_no, int old_line_no, String content, String status) {
	Dictionary diff_line;
	diff_line["new_line_no"] = new_line_no;
	diff_line["old_line_no"] = old_line_no;
	diff_line["content"] = content;
	diff_line["status"] = status;

	return diff_line;
}

Dictionary EditorVCSInterface::_create_diff_hunk(int old_start, int new_start, int old_lines, int new_lines) {
	Dictionary diff_hunk;
	diff_hunk["new_lines"] = new_lines;
	diff_hunk["old_lines"] = old_lines;
	diff_hunk["new_start"] = new_start;
	diff_hunk["old_start"] = old_start;
	diff_hunk["diff_lines"] = Array();
	return diff_hunk;
}

Dictionary EditorVCSInterface::_add_line_diffs_into_diff_hunk(Dictionary diff_hunk, Array line_diffs) {
	diff_hunk["diff_lines"] = line_diffs;
	return diff_hunk;
}

Dictionary EditorVCSInterface::_create_diff_file(String new_file, String old_file) {
	Dictionary file_diff;
	file_diff["new_file"] = new_file;
	file_diff["old_file"] = old_file;
	file_diff["diff_hunks"] = Array();
	return file_diff;
}

Dictionary EditorVCSInterface::_create_commit(String p_msg, String p_author, String p_hex_id, int16_t p_when) {
	Dictionary commit_info;
	commit_info["message"] = p_msg;
	commit_info["author"] = p_author;
	commit_info["when"] = p_when; // Epoch time in seconds
	commit_info["id"] = p_hex_id;
	return commit_info;
}

Dictionary EditorVCSInterface::_add_diff_hunks_into_diff_file(Dictionary diff_file, Array diff_hunks) {
	diff_file["diff_hunks"] = diff_hunks;
	return diff_file;
}

Dictionary EditorVCSInterface::_create_status_file(String p_file_path, ChangeType p_change, TreeArea p_area) {
	Dictionary sf;
	sf["file_path"] = p_file_path;
	sf["chanage_type"] = p_change;
	sf["area"] = p_area;
	return sf;
}

EditorVCSInterface::DiffLine EditorVCSInterface::_convert_diff_line(Dictionary p_diff_line) {
	DiffLine d;
	d.new_line_no = p_diff_line["new_line_no"];
	d.old_line_no = p_diff_line["old_line_no"];
	d.content = p_diff_line["content"];
	d.status = p_diff_line["status"];
	return d;
}

EditorVCSInterface::DiffHunk EditorVCSInterface::_convert_diff_hunk(Dictionary p_diff_hunk) {
	DiffHunk dh;
	dh.new_lines = p_diff_hunk["new_lines"];
	dh.old_lines = p_diff_hunk["old_lines"];
	dh.new_start = p_diff_hunk["new_start"];
	dh.old_start = p_diff_hunk["old_start"];
	dh.diff_lines = List<DiffLine>();
	Array diff_lines = p_diff_hunk["diff_lines"];
	for (int i = 0; i < diff_lines.size(); i++) {
		DiffLine dl = _convert_diff_line(diff_lines[i]);
		dh.diff_lines.push_back(dl);
	}
	return dh;
}

EditorVCSInterface::DiffFile EditorVCSInterface::_convert_diff_file(Dictionary p_diff_file) {
	DiffFile df;
	df.new_file = p_diff_file["new_file"];
	df.old_file = p_diff_file["old_file"];
	df.diff_hunks = List<DiffHunk>();
	Array diff_hunks = p_diff_file["diff_hunks"];
	for (int i = 0; i < diff_hunks.size(); i++) {
		DiffHunk dh = _convert_diff_hunk(diff_hunks[i]);
		df.diff_hunks.push_back(dh);
	}
	return df;
}

EditorVCSInterface::Commit EditorVCSInterface::_convert_commit(Dictionary p_commit) {
	EditorVCSInterface::Commit c;
	c.msg = p_commit["message"];
	c.author = p_commit["author"];
	c.time = p_commit["when"]; // Epoch time in seconds
	c.hex_id = p_commit["id"];
	return c;
}

EditorVCSInterface::StatusFile EditorVCSInterface::_convert_status_file(Dictionary p_status_file) {
	StatusFile sf;
	sf.file_path = p_status_file["file_path"];
	sf.change_type = (ChangeType)(int)p_status_file["chanage_type"];
	sf.area = (TreeArea)(int)p_status_file["area"];
	return sf;
}

void EditorVCSInterface::_not_implemented_function(String p_function) {
	ERR_PRINT("Selected VCS plugin does not implement " + p_function + " function");
}

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
	// ClassDB::bind_method(D_METHOD("_get_line_diff"), &EditorVCSInterface::_get_line_diff);

	ClassDB::bind_method(D_METHOD("is_plugin_ready"), &EditorVCSInterface::is_plugin_ready);

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
	ClassDB::bind_method(D_METHOD("get_line_diff"), &EditorVCSInterface::get_line_diff);

	ClassDB::bind_method(D_METHOD("_create_diff_line", "new_line_no", "old_line_no", "p_content", "p_status"), &EditorVCSInterface::_create_diff_line);
	ClassDB::bind_method(D_METHOD("_create_diff_hunk", "old_start", "new_start", "old_lines", "new_lines"), &EditorVCSInterface::_create_diff_hunk);
	ClassDB::bind_method(D_METHOD("_create_diff_file", "p_new_file", "p_old_file"), &EditorVCSInterface::_create_diff_file);
	ClassDB::bind_method(D_METHOD("_create_commit", "p_msg", "p_author", "p_hex_id", "p_time"), &EditorVCSInterface::_create_commit);
	ClassDB::bind_method(D_METHOD("_create_status_file", "p_file_path", "p_change_type", "p_area"), &EditorVCSInterface::_create_status_file);

	ClassDB::bind_method(D_METHOD("_add_diff_hunks_into_diff_file", "p_diff_hunk", "p_line_diffs"), &EditorVCSInterface::_add_diff_hunks_into_diff_file);
	ClassDB::bind_method(D_METHOD("_add_line_diffs_into_diff_hunk", "p_diff_files", "p_diff_hunks"), &EditorVCSInterface::_add_line_diffs_into_diff_hunk);
	ClassDB::bind_method(D_METHOD("_popup_error", "p_msg"), &EditorVCSInterface::_popup_error);
	ADD_SIGNAL(MethodInfo("stage_area_refreshed"));

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
