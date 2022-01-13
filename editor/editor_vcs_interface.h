/*************************************************************************/
/*  editor_vcs_interface.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EDITOR_VCS_INTERFACE_H
#define EDITOR_VCS_INTERFACE_H

#include "core/object.h"
#include "core/ustring.h"
#include "scene/gui/panel_container.h"

class EditorVCSInterface : public Object {
	GDCLASS(EditorVCSInterface, Object)

public:
	enum ChangeType {
		CHANGE_TYPE_NEW = 0,
		CHANGE_TYPE_MODIFIED = 1,
		CHANGE_TYPE_RENAMED = 2,
		CHANGE_TYPE_DELETED = 3,
		CHANGE_TYPE_TYPECHANGE = 4,
		CHANGE_TYPE_UNMERGED = 5
	};

	enum TreeArea {
		TREE_AREA_COMMIT = 0,
		TREE_AREA_STAGED = 1,
		TREE_AREA_UNSTAGED = 2
	};

	struct DiffLine {
		int new_line_no;
		int old_line_no;
		String content;
		String status;

		String old_text;
		String new_text;
	};

	struct DiffHunk {
		int new_start;
		int old_start;
		int new_lines;
		int old_lines;
		List<DiffLine> diff_lines;
	};

	struct DiffFile {
		String new_file;
		String old_file;
		List<DiffHunk> diff_hunks;
	};

	struct Commit {
		String author;
		String msg;
		String id;
		String date;
	};

	struct StatusFile {
		TreeArea area;
		ChangeType change_type;
		String file_path;
	};

protected:
	static EditorVCSInterface *singleton;

	static void _bind_methods();

	DiffLine _convert_diff_line(Dictionary p_diff_line);
	DiffHunk _convert_diff_hunk(Dictionary p_diff_hunk);
	DiffFile _convert_diff_file(Dictionary p_diff_file);
	Commit _convert_commit(Dictionary p_commit);
	StatusFile _convert_status_file(Dictionary p_status_file);

public:
	static EditorVCSInterface *get_singleton();
	static void set_singleton(EditorVCSInterface *p_singleton);

	// Proxy functions to the editor for use
	bool initialize(String p_project_path);
	void set_credentials(String p_username, String p_password, String p_ssh_public_key_path, String p_ssh_private_key_path, String p_ssh_passphrase);
	List<StatusFile> get_modified_files_data();
	void stage_file(String p_file_path);
	void unstage_file(String p_file_path);
	void discard_file(String p_file_path);
	void commit(String p_msg);
	List<DiffFile> get_diff(String p_identifier, TreeArea p_area);
	bool shut_down();
	String get_vcs_name();
	List<Commit> get_previous_commits(int p_max_commits);
	List<String> get_branch_list();
	List<String> get_remotes();
	void create_branch(String p_branch_name);
	void remove_branch(String p_branch_name);
	void create_remote(String p_remote_name, String p_remote_url);
	void remove_remote(String p_remote_name);
	String get_current_branch_name();
	bool checkout_branch(String p_branch_name);
	void pull(String p_remote);
	void push(String p_remote, bool p_force);
	void fetch(String p_remote);
	List<DiffHunk> get_line_diff(String p_file_path, String p_text);

	// Helper functions to create and convert Dictionary into data structures
	Dictionary create_diff_line(int p_new_line_no, int p_old_line_no, String p_content, String p_status);
	Dictionary create_diff_hunk(int p_old_start, int p_new_start, int p_old_lines, int p_new_lines);
	Dictionary create_diff_file(String p_new_file, String p_old_file);
	Dictionary create_commit(String p_msg, String p_author, String p_id, String p_date);
	Dictionary create_status_file(String p_file_path, ChangeType p_change, TreeArea p_area);
	Dictionary add_line_diffs_into_diff_hunk(Dictionary p_diff_hunk, Array p_line_diffs);
	Dictionary add_diff_hunks_into_diff_file(Dictionary p_diff_file, Array p_diff_hunks);

	void popup_error(String p_msg);
};

VARIANT_ENUM_CAST(EditorVCSInterface::ChangeType);
VARIANT_ENUM_CAST(EditorVCSInterface::TreeArea);

#endif // !EDITOR_VCS_INTERFACE_H
