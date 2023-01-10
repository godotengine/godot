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

#include "editor_vcs_interface.h"

#include "editor_node.h"

EditorVCSInterface *EditorVCSInterface::singleton = nullptr;

void EditorVCSInterface::popup_error(String p_msg) {
	// TRANSLATORS: %s refers to the name of a version control system (e.g. "Git").
	EditorNode::get_singleton()->show_warning(p_msg.strip_edges(), vformat(TTR("%s Error"), get_vcs_name()));
}

bool EditorVCSInterface::initialize(String p_project_path) {
	return call("_initialize", p_project_path);
}

void EditorVCSInterface::set_credentials(String p_username, String p_password, String p_ssh_public_key, String p_ssh_private_key, String p_ssh_passphrase) {
	call("_set_credentials", p_username, p_password, p_ssh_public_key, p_ssh_private_key, p_ssh_passphrase);
}

List<String> EditorVCSInterface::get_remotes() {
	List<String> remotes;

	Array result = call("_get_remotes");
	for (int i = 0; i < result.size(); i++) {
		remotes.push_back(result[i]);
	}

	return remotes;
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
	call("_stage_file", p_file_path);
}

void EditorVCSInterface::unstage_file(String p_file_path) {
	call("_unstage_file", p_file_path);
}

void EditorVCSInterface::discard_file(String p_file_path) {
	call("_discard_file", p_file_path);
}

void EditorVCSInterface::commit(String p_msg) {
	call("_commit", p_msg);
}

List<EditorVCSInterface::DiffFile> EditorVCSInterface::get_diff(String p_identifier, TreeArea p_area) {
	List<DiffFile> diff_files;

	Array result = call("_get_diff", p_identifier, p_area);
	for (int i = 0; i < result.size(); i++) {
		diff_files.push_back(_convert_diff_file(result[i]));
	}

	return diff_files;
}

List<EditorVCSInterface::Commit> EditorVCSInterface::get_previous_commits(int p_max_commits) {
	List<EditorVCSInterface::Commit> commits;

	Array result = call("_get_previous_commits", p_max_commits);
	for (int i = 0; i < result.size(); i++) {
		commits.push_back(_convert_commit(result[i]));
	}

	return commits;
}

List<String> EditorVCSInterface::get_branch_list() {
	List<String> branch_list;

	Array result = call("_get_branch_list");
	for (int i = 0; i < result.size(); i++) {
		branch_list.push_back(result[i]);
	}

	return branch_list;
}

void EditorVCSInterface::create_branch(String p_branch_name) {
	call("_create_branch", p_branch_name);
}

void EditorVCSInterface::create_remote(String p_remote_name, String p_remote_url) {
	call("_create_remote", p_remote_name, p_remote_url);
}

void EditorVCSInterface::remove_branch(String p_branch_name) {
	call("_remove_branch", p_branch_name);
}

void EditorVCSInterface::remove_remote(String p_remote_name) {
	call("_remove_remote", p_remote_name);
}

String EditorVCSInterface::get_current_branch_name() {
	return call("_get_current_branch_name");
}

bool EditorVCSInterface::checkout_branch(String p_branch_name) {
	return call("_checkout_branch", p_branch_name);
}

void EditorVCSInterface::pull(String p_remote) {
	call("_pull", p_remote);
}

void EditorVCSInterface::push(String p_remote, bool p_force) {
	call("_push", p_remote, p_force);
}

void EditorVCSInterface::fetch(String p_remote) {
	call("_fetch", p_remote);
}

List<EditorVCSInterface::DiffHunk> EditorVCSInterface::get_line_diff(String p_file_path, String p_text) {
	List<DiffHunk> diff_hunks;

	Array result = call("_get_line_diff", p_file_path, p_text);
	for (int i = 0; i < result.size(); i++) {
		diff_hunks.push_back(_convert_diff_hunk(result[i]));
	}

	return diff_hunks;
}

bool EditorVCSInterface::shut_down() {
	return call("_shut_down");
}

String EditorVCSInterface::get_vcs_name() {
	return call("_get_vcs_name");
}

Dictionary EditorVCSInterface::create_diff_line(int p_new_line_no, int p_old_line_no, String p_content, String p_status) {
	Dictionary diff_line;
	diff_line["new_line_no"] = p_new_line_no;
	diff_line["old_line_no"] = p_old_line_no;
	diff_line["content"] = p_content;
	diff_line["status"] = p_status;

	return diff_line;
}

Dictionary EditorVCSInterface::create_diff_hunk(int p_old_start, int p_new_start, int p_old_lines, int p_new_lines) {
	Dictionary diff_hunk;
	diff_hunk["new_lines"] = p_new_lines;
	diff_hunk["old_lines"] = p_old_lines;
	diff_hunk["new_start"] = p_new_start;
	diff_hunk["old_start"] = p_old_start;
	diff_hunk["diff_lines"] = Array();
	return diff_hunk;
}

Dictionary EditorVCSInterface::add_line_diffs_into_diff_hunk(Dictionary p_diff_hunk, Array p_line_diffs) {
	p_diff_hunk["diff_lines"] = p_line_diffs;
	return p_diff_hunk;
}

Dictionary EditorVCSInterface::create_diff_file(String p_new_file, String p_old_file) {
	Dictionary file_diff;
	file_diff["new_file"] = p_new_file;
	file_diff["old_file"] = p_old_file;
	file_diff["diff_hunks"] = Array();
	return file_diff;
}

Dictionary EditorVCSInterface::create_commit(String p_msg, String p_author, String p_id, int64_t p_unix_timestamp, int64_t p_offset_minutes) {
	Dictionary commit_info;
	commit_info["message"] = p_msg;
	commit_info["author"] = p_author;
	commit_info["unix_timestamp"] = p_unix_timestamp;
	commit_info["offset_minutes"] = p_offset_minutes;
	commit_info["id"] = p_id;
	return commit_info;
}

Dictionary EditorVCSInterface::add_diff_hunks_into_diff_file(Dictionary p_diff_file, Array p_diff_hunks) {
	p_diff_file["diff_hunks"] = p_diff_hunks;
	return p_diff_file;
}

Dictionary EditorVCSInterface::create_status_file(String p_file_path, ChangeType p_change, TreeArea p_area) {
	Dictionary sf;
	sf["file_path"] = p_file_path;
	sf["change_type"] = p_change;
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
	c.unix_timestamp = p_commit["unix_timestamp"];
	c.offset_minutes = p_commit["offset_minutes"];
	c.id = p_commit["id"];
	return c;
}

EditorVCSInterface::StatusFile EditorVCSInterface::_convert_status_file(Dictionary p_status_file) {
	StatusFile sf;
	sf.file_path = p_status_file["file_path"];
	sf.change_type = (ChangeType)(int)p_status_file["change_type"];
	sf.area = (TreeArea)(int)p_status_file["area"];
	return sf;
}

void EditorVCSInterface::_bind_methods() {
	// Proxy end points that implement the VCS specific operations that the editor demands.
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_initialize", PropertyInfo(Variant::STRING, "project_path")));
	BIND_VMETHOD(MethodInfo("_set_credentials", PropertyInfo(Variant::STRING, "username"), PropertyInfo(Variant::STRING, "password"), PropertyInfo(Variant::STRING, "ssh_public_key_path"), PropertyInfo(Variant::STRING, "ssh_private_key_path"), PropertyInfo(Variant::STRING, "ssh_passphrase")));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_remotes"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_vcs_name"));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_shut_down"));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_modified_files_data"));
	BIND_VMETHOD(MethodInfo("_commit", PropertyInfo(Variant::STRING, "msg")));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_diff", PropertyInfo(Variant::STRING, "identifier"), PropertyInfo(Variant::INT, "area")));
	BIND_VMETHOD(MethodInfo("_stage_file", PropertyInfo(Variant::STRING, "file_path")));
	BIND_VMETHOD(MethodInfo("_unstage_file", PropertyInfo(Variant::STRING, "file_path")));
	BIND_VMETHOD(MethodInfo("_discard_file", PropertyInfo(Variant::STRING, "file_path")));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_previous_commits", PropertyInfo(Variant::INT, "max_commits")));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_branch_list"));
	BIND_VMETHOD(MethodInfo("_create_branch", PropertyInfo(Variant::STRING, "branch_name")));
	BIND_VMETHOD(MethodInfo("_remove_branch", PropertyInfo(Variant::STRING, "branch_name")));
	BIND_VMETHOD(MethodInfo("_create_remote", PropertyInfo(Variant::STRING, "remote_name"), PropertyInfo(Variant::STRING, "remote_url")));
	BIND_VMETHOD(MethodInfo("_remove_remote", PropertyInfo(Variant::STRING, "remote_name")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_current_branch_name"));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_checkout_branch", PropertyInfo(Variant::STRING, "branch_name")));
	BIND_VMETHOD(MethodInfo("_push", PropertyInfo(Variant::STRING, "remote"), PropertyInfo(Variant::BOOL, "force")));
	BIND_VMETHOD(MethodInfo("_pull", PropertyInfo(Variant::STRING, "remote")));
	BIND_VMETHOD(MethodInfo("_fetch", PropertyInfo(Variant::STRING, "remote")));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_line_diff", PropertyInfo(Variant::STRING, "file_path"), PropertyInfo(Variant::STRING, "text")));

	ClassDB::bind_method(D_METHOD("create_diff_line", "new_line_no", "old_line_no", "content", "status"), &EditorVCSInterface::create_diff_line);
	ClassDB::bind_method(D_METHOD("create_diff_hunk", "old_start", "new_start", "old_lines", "new_lines"), &EditorVCSInterface::create_diff_hunk);
	ClassDB::bind_method(D_METHOD("create_diff_file", "new_file", "old_file"), &EditorVCSInterface::create_diff_file);
	ClassDB::bind_method(D_METHOD("create_commit", "msg", "author", "id", "unix_timestamp", "offset_minutes"), &EditorVCSInterface::create_commit);
	ClassDB::bind_method(D_METHOD("create_status_file", "file_path", "change_type", "area"), &EditorVCSInterface::create_status_file);
	ClassDB::bind_method(D_METHOD("add_diff_hunks_into_diff_file", "diff_file", "diff_hunks"), &EditorVCSInterface::add_diff_hunks_into_diff_file);
	ClassDB::bind_method(D_METHOD("add_line_diffs_into_diff_hunk", "diff_hunk", "line_diffs"), &EditorVCSInterface::add_line_diffs_into_diff_hunk);
	ClassDB::bind_method(D_METHOD("popup_error", "msg"), &EditorVCSInterface::popup_error);

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

EditorVCSInterface *EditorVCSInterface::get_singleton() {
	return singleton;
}

void EditorVCSInterface::set_singleton(EditorVCSInterface *p_singleton) {
	singleton = p_singleton;
}
