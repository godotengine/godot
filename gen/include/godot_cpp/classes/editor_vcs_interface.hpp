/**************************************************************************/
/*  editor_vcs_interface.hpp                                              */
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

#pragma once

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class EditorVCSInterface : public Object {
	GDEXTENSION_CLASS(EditorVCSInterface, Object)

public:
	enum ChangeType {
		CHANGE_TYPE_NEW = 0,
		CHANGE_TYPE_MODIFIED = 1,
		CHANGE_TYPE_RENAMED = 2,
		CHANGE_TYPE_DELETED = 3,
		CHANGE_TYPE_TYPECHANGE = 4,
		CHANGE_TYPE_UNMERGED = 5,
	};

	enum TreeArea {
		TREE_AREA_COMMIT = 0,
		TREE_AREA_STAGED = 1,
		TREE_AREA_UNSTAGED = 2,
	};

	Dictionary create_diff_line(int32_t p_new_line_no, int32_t p_old_line_no, const String &p_content, const String &p_status);
	Dictionary create_diff_hunk(int32_t p_old_start, int32_t p_new_start, int32_t p_old_lines, int32_t p_new_lines);
	Dictionary create_diff_file(const String &p_new_file, const String &p_old_file);
	Dictionary create_commit(const String &p_msg, const String &p_author, const String &p_id, int64_t p_unix_timestamp, int64_t p_offset_minutes);
	Dictionary create_status_file(const String &p_file_path, EditorVCSInterface::ChangeType p_change_type, EditorVCSInterface::TreeArea p_area);
	Dictionary add_diff_hunks_into_diff_file(const Dictionary &p_diff_file, const TypedArray<Dictionary> &p_diff_hunks);
	Dictionary add_line_diffs_into_diff_hunk(const Dictionary &p_diff_hunk, const TypedArray<Dictionary> &p_line_diffs);
	void popup_error(const String &p_msg);
	virtual bool _initialize(const String &p_project_path);
	virtual void _set_credentials(const String &p_username, const String &p_password, const String &p_ssh_public_key_path, const String &p_ssh_private_key_path, const String &p_ssh_passphrase);
	virtual TypedArray<Dictionary> _get_modified_files_data();
	virtual void _stage_file(const String &p_file_path);
	virtual void _unstage_file(const String &p_file_path);
	virtual void _discard_file(const String &p_file_path);
	virtual void _commit(const String &p_msg);
	virtual TypedArray<Dictionary> _get_diff(const String &p_identifier, int32_t p_area);
	virtual bool _shut_down();
	virtual String _get_vcs_name();
	virtual TypedArray<Dictionary> _get_previous_commits(int32_t p_max_commits);
	virtual TypedArray<String> _get_branch_list();
	virtual TypedArray<String> _get_remotes();
	virtual void _create_branch(const String &p_branch_name);
	virtual void _remove_branch(const String &p_branch_name);
	virtual void _create_remote(const String &p_remote_name, const String &p_remote_url);
	virtual void _remove_remote(const String &p_remote_name);
	virtual String _get_current_branch_name();
	virtual bool _checkout_branch(const String &p_branch_name);
	virtual void _pull(const String &p_remote);
	virtual void _push(const String &p_remote, bool p_force);
	virtual void _fetch(const String &p_remote);
	virtual TypedArray<Dictionary> _get_line_diff(const String &p_file_path, const String &p_text);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_initialize), decltype(&T::_initialize)>) {
			BIND_VIRTUAL_METHOD(T, _initialize, 2323990056);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_credentials), decltype(&T::_set_credentials)>) {
			BIND_VIRTUAL_METHOD(T, _set_credentials, 1336744649);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_modified_files_data), decltype(&T::_get_modified_files_data)>) {
			BIND_VIRTUAL_METHOD(T, _get_modified_files_data, 2915620761);
		}
		if constexpr (!std::is_same_v<decltype(&B::_stage_file), decltype(&T::_stage_file)>) {
			BIND_VIRTUAL_METHOD(T, _stage_file, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_unstage_file), decltype(&T::_unstage_file)>) {
			BIND_VIRTUAL_METHOD(T, _unstage_file, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_discard_file), decltype(&T::_discard_file)>) {
			BIND_VIRTUAL_METHOD(T, _discard_file, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_commit), decltype(&T::_commit)>) {
			BIND_VIRTUAL_METHOD(T, _commit, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_diff), decltype(&T::_get_diff)>) {
			BIND_VIRTUAL_METHOD(T, _get_diff, 1366379175);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shut_down), decltype(&T::_shut_down)>) {
			BIND_VIRTUAL_METHOD(T, _shut_down, 2240911060);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_vcs_name), decltype(&T::_get_vcs_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_vcs_name, 2841200299);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_previous_commits), decltype(&T::_get_previous_commits)>) {
			BIND_VIRTUAL_METHOD(T, _get_previous_commits, 1171824711);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_branch_list), decltype(&T::_get_branch_list)>) {
			BIND_VIRTUAL_METHOD(T, _get_branch_list, 2915620761);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_remotes), decltype(&T::_get_remotes)>) {
			BIND_VIRTUAL_METHOD(T, _get_remotes, 2915620761);
		}
		if constexpr (!std::is_same_v<decltype(&B::_create_branch), decltype(&T::_create_branch)>) {
			BIND_VIRTUAL_METHOD(T, _create_branch, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_remove_branch), decltype(&T::_remove_branch)>) {
			BIND_VIRTUAL_METHOD(T, _remove_branch, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_create_remote), decltype(&T::_create_remote)>) {
			BIND_VIRTUAL_METHOD(T, _create_remote, 3186203200);
		}
		if constexpr (!std::is_same_v<decltype(&B::_remove_remote), decltype(&T::_remove_remote)>) {
			BIND_VIRTUAL_METHOD(T, _remove_remote, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_current_branch_name), decltype(&T::_get_current_branch_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_current_branch_name, 2841200299);
		}
		if constexpr (!std::is_same_v<decltype(&B::_checkout_branch), decltype(&T::_checkout_branch)>) {
			BIND_VIRTUAL_METHOD(T, _checkout_branch, 2323990056);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pull), decltype(&T::_pull)>) {
			BIND_VIRTUAL_METHOD(T, _pull, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_push), decltype(&T::_push)>) {
			BIND_VIRTUAL_METHOD(T, _push, 2678287736);
		}
		if constexpr (!std::is_same_v<decltype(&B::_fetch), decltype(&T::_fetch)>) {
			BIND_VIRTUAL_METHOD(T, _fetch, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_line_diff), decltype(&T::_get_line_diff)>) {
			BIND_VIRTUAL_METHOD(T, _get_line_diff, 2796572089);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorVCSInterface::ChangeType);
VARIANT_ENUM_CAST(EditorVCSInterface::TreeArea);

