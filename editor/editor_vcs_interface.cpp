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
	ClassDB::bind_method(D_METHOD("_get_file_diff", "file_path"), &EditorVCSInterface::_get_file_diff);
	ClassDB::bind_method(D_METHOD("_stage_file", "file_path"), &EditorVCSInterface::_stage_file);
	ClassDB::bind_method(D_METHOD("_unstage_file", "file_path"), &EditorVCSInterface::_unstage_file);

	ClassDB::bind_method(D_METHOD("is_addon_ready"), &EditorVCSInterface::is_addon_ready);

	// API methods that redirect calls to the proxy end points
	ClassDB::bind_method(D_METHOD("initialize", "project_root_path"), &EditorVCSInterface::initialize);
	ClassDB::bind_method(D_METHOD("is_vcs_initialized"), &EditorVCSInterface::is_vcs_initialized);
	ClassDB::bind_method(D_METHOD("get_modified_files_data"), &EditorVCSInterface::get_modified_files_data);
	ClassDB::bind_method(D_METHOD("stage_file", "file_path"), &EditorVCSInterface::stage_file);
	ClassDB::bind_method(D_METHOD("unstage_file", "file_path"), &EditorVCSInterface::unstage_file);
	ClassDB::bind_method(D_METHOD("commit", "msg"), &EditorVCSInterface::commit);
	ClassDB::bind_method(D_METHOD("get_file_diff", "file_path"), &EditorVCSInterface::get_file_diff);
	ClassDB::bind_method(D_METHOD("shut_down"), &EditorVCSInterface::shut_down);
	ClassDB::bind_method(D_METHOD("get_project_name"), &EditorVCSInterface::get_project_name);
	ClassDB::bind_method(D_METHOD("get_vcs_name"), &EditorVCSInterface::get_vcs_name);
}

bool EditorVCSInterface::_initialize(String p_project_root_path) {
	WARN_PRINT("Selected VCS addon does not implement an initialization function. This warning will be suppressed.");
	return true;
}

bool EditorVCSInterface::_is_vcs_initialized() {
	return false;
}

Dictionary EditorVCSInterface::_get_modified_files_data() {
	return Dictionary();
}

void EditorVCSInterface::_stage_file(String p_file_path) {
}

void EditorVCSInterface::_unstage_file(String p_file_path) {
}

void EditorVCSInterface::_commit(String p_msg) {
}

Array EditorVCSInterface::_get_file_diff(String p_file_path) {
	return Array();
}

bool EditorVCSInterface::_shut_down() {
	return false;
}

String EditorVCSInterface::_get_project_name() {
	return String();
}

String EditorVCSInterface::_get_vcs_name() {
	return "";
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

bool EditorVCSInterface::is_addon_ready() {
	return is_initialized;
}

void EditorVCSInterface::commit(String p_msg) {
	if (is_addon_ready()) {
		call("_commit", p_msg);
	}
}

Array EditorVCSInterface::get_file_diff(String p_file_path) {
	if (is_addon_ready()) {
		return call("_get_file_diff", p_file_path);
	}
	return Array();
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

void EditorVCSInterface::create_vcs_metadata_files(VCSMetadata p_vcs_metadata_type, String &p_dir) {
	if (p_vcs_metadata_type == VCSMetadata::GIT) {
		FileAccess *f = FileAccess::open(p_dir.plus_file(".gitignore"), FileAccess::WRITE);
		if (!f) {
			ERR_FAIL_MSG(TTR("Couldn't create .gitignore in project path."));
		} else {
			f->store_line("# Godot 4+ specific ignores");
			f->store_line(".godot/");
			memdelete(f);
		}
		f = FileAccess::open(p_dir.plus_file(".gitattributes"), FileAccess::WRITE);
		if (!f) {
			ERR_FAIL_MSG(TTR("Couldn't create .gitattributes in project path."));
		} else {
			f->store_line("# Normalize EOL for all files that Git considers text files.");
			f->store_line("* text=auto eol=lf");
			memdelete(f);
		}
	}
}
