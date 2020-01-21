/*************************************************************************/
/*  file_access_resources.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef FILE_ACCESS_RESOURCES_H
#define FILE_ACCESS_RESOURCES_H

#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"

#ifdef TOOLS_ENABLED

#include "core/engine.h"
#include "core/io/file_access_pack.h"
#include "editor/addons_fs_manager.h"
#include "editor/editor_settings.h"
#include "editor/plugins_db.h"

namespace {

template <typename T>
class PathAccessResources : public T {
protected:
	Error _check_path_availability(String p_path, int p_addons_min_level = 2, bool p_for_creation = false) {

		if (p_path.is_rel_path()) {
			p_path = _get_default_directory().plus_file(p_path);
		}

		if (p_path.begins_with("res://")) {
			Vector<String> parts = p_path.strip_filesystem_prefix().split("/", false, 2);
			// Consider res://addons/<subdir> inexistent if it's a new model plugin
			if (parts.size() >= 2 && parts[0] == "addons") {
				if (PluginsDB::get_singleton()->has_universal_plugin(parts[1])) {
					return ERR_DOES_NOT_EXIST;
				}
			}
		} else if (p_path.begins_with("addons://")) {
			Vector<String> parts = p_path.strip_filesystem_prefix().split("/", false, 1);
			if (parts.size() < p_addons_min_level) {
				// No operations allowed on addons://<subdir>/ or more shallow; only deeper
				return ERR_UNAVAILABLE;
			}
			if (!AddonsFileSystemManager::get_singleton()->has_subdirectory(parts[0])) {
				// No operation allowed on addons://<nonexistent>/
				return p_for_creation ? ERR_UNAVAILABLE : ERR_DOES_NOT_EXIST;
			}
		}

		return OK;
	}

	virtual String _get_default_directory() = 0;

	virtual void _get_path_bases_for_unfix(const String &p_path, String *r_logical_base, String *r_physical_base) const {

		String subdir;
		if (AddonsFileSystemManager::get_singleton()->find_subdirectory_for_path(p_path, &subdir, r_physical_base)) {
			*r_logical_base = String("addons://").plus_file(subdir);
			return;
		}

		*r_logical_base = "res://";
		*r_physical_base = this->_get_resource_path();
	}

public:
	virtual String fix_path(const String &p_path) const {

		if (p_path.begins_with("res://")) {

			return p_path.replace_first("res:/", this->_get_resource_path());
		} else if (p_path.begins_with("addons://")) {

			Vector<String> parts = p_path.strip_filesystem_prefix().split("/", false, 1);
			if (parts.size() >= 1) {
				if (AddonsFileSystemManager::get_singleton()->has_subdirectory(parts[0])) {
					String location = AddonsFileSystemManager::get_singleton()->get_subdirectory_location(parts[0]);
					if (parts.size() >= 2) {
						return location.plus_file(parts[1]);
					} else {
						return location;
					}
				}
			}
		}

		return p_path;
	}
};

} // namespace

//////////////////////////////////////////////////////////////////////////////////

template <class T>
class FileAccessResources : public PathAccessResources<T> {
protected:
	virtual String _get_default_directory() {

		return this->get_path_absolute();
	}

public:
	virtual Error _open(const String &p_path, int p_mode_flags) {

		Error result = this->_check_path_availability(p_path, 2, p_mode_flags == FileAccess::WRITE || p_mode_flags == FileAccess::WRITE_READ);
		if (result != OK) {
			return result;
		}

		return T::_open(p_path, p_mode_flags);
	}

	virtual uint64_t _get_modified_time(const String &p_file) {

		if (this->_check_path_availability(p_file) != OK) {
			return 0;
		} else {
			// Since we are controlling the visibility of res://addons/ contents, we return 0
			// to force refresh on every scan.
			// TODO: Collaborate with PluginsDB so we have an actual timestamp of the latest virtual
			// FS change and return the higher between that and the real one from the FS.
			if (p_file == "res://") {
				return 0;
			}
			String file = p_file.rstrip("/");
			if (file == "" || file == "res://addons") {
				return 0;
			}
		}

		return T::_get_modified_time(p_file);
	}

	virtual bool file_exists(const String &p_name) {

		if (this->_check_path_availability(p_name) != OK) {
			return false;
		}

		return T::file_exists(p_name);
	}

	virtual Error reopen(const String &p_path, int p_mode_flags) {

		Error result = this->_check_path_availability(p_path, 2, p_mode_flags == FileAccess::WRITE || p_mode_flags == FileAccess::WRITE_READ);
		if (result != OK) {
			return result;
		}

		return T::reopen(p_path, p_mode_flags);
	}

	virtual uint32_t _get_unix_permissions(const String &p_file) {

		Error result = this->_check_path_availability(p_file);
		if (result != OK) {
			return 0;
		}

		return T::_get_unix_permissions(p_file);
	}

	virtual Error _set_unix_permissions(const String &p_file, uint32_t p_permissions) {

		Error result = this->_check_path_availability(p_file);
		if (result != OK) {
			return result;
		}

		return T::_set_unix_permissions(p_file, p_permissions);
	}
};

//////////////////////////////////////////////////////////////////////////////////
// DIR ACCESS
//////////////////////////////////////////////////////////////////////////////////

/*
 * NOTE: DirAccessResources adds further abstraction, to make it possible to jump
 * between packed and non-packed locations. That makes it much easier for consumers
 * to browse the file system, seamlessly jumping between directories.
 * For instance, that's useful for the editor file scan and the file dialog.
 */

template <class T>
class DirAccessResources : public PathAccessResources<DirAccess> {
	DirAccess *da_normal;
	DirAccess *da_packed;
	DirAccess *current_da; // NULL means we are at addons:// root

	// For directory listing
	enum ListingMode {
		LISTING_MODE_NONE,
		LISTING_MODE_PASS_THROUGH,
		LISTING_MODE_RES_ADDONS,
		LISTING_MODE_ADDONS_FS_ROOT,
		LISTING_MODE_ADDONS_FS_SUBDIR_OR_BELOW,
	} listing_mode;
	DirAccess *listing_da;
	Vector<String> addons_fs_children;
	Vector<String> editor_only_paths;
	bool current_item_is_dir;
	bool current_item_is_hidden;

protected:
	virtual bool _is_valid_dir_change(const String &p_curr_dir, const String &p_new_dir) const {

		// This ensures that you  end up on a physical directory exposed by the sandbox.
		// TODO: Although this is good enough, it would be more correct checking that the path
		// traversal doesn't cross the sandboxed root. Even if you end up in again a sandboxed
		// path, that wouldn't be correct. In other words, the most correct behavior would involve
		// giving the impression that FS sandboxes are actually roots, and nothing at all exists
		// above them.

		return this->unfix_path(p_new_dir).begins_with("addons://") || p_new_dir.begins_with(this->_get_resource_path());
	}

	virtual String _get_default_directory() {

		return get_current_dir();
	}

	String _resolve_full_path(const String &p_path, bool p_da_depends_on_parent, DirAccess **r_relevant_da_impl) {

		String full_path;
		if (p_path.get_filesystem_prefix() != "") {
			full_path = p_path;
		} else if (p_path.begins_with("/")) {
			full_path = get_current_dir().get_filesystem_prefix().plus_file(p_path);
		} else {
			Vector<String> resolved_parts = get_current_dir_without_drive().split("/", false);
			Vector<String> rel_parts = p_path.split("/", false);

			for (int i = 0; i < rel_parts.size(); ++i) {
				if (rel_parts[i] == ".") {
					continue;
				}
				if (rel_parts[i] == "..") {
					if (resolved_parts.size() != 0) {
						resolved_parts.resize(resolved_parts.size() - 1);
					} else {
						return ""; // Invalid
					}
				} else {
					resolved_parts.push_back(rel_parts[i]);
				}
			}

			full_path = get_current_dir().get_filesystem_prefix().plus_file(String("/").join(resolved_parts));
		}

		String full_path_for_da;
		if (p_da_depends_on_parent) {
			int p = full_path.trim_suffix("/").find_last("/");
			full_path_for_da = full_path.left(p);
		} else {
			full_path_for_da = full_path;
		}

		if (full_path_for_da == "addons://") {
			*r_relevant_da_impl = NULL;
		} else if (full_path_for_da.begins_with("addons://")) {
			if (AddonsFileSystemManager::get_singleton()->is_path_packed(full_path_for_da)) {
				*r_relevant_da_impl = da_packed;
			} else {
				*r_relevant_da_impl = da_normal;
			}
		} else {
			*r_relevant_da_impl = da_normal;
		}

		return full_path;
	}

public:
	virtual Error list_dir_begin() {

		if (listing_mode != LISTING_MODE_NONE) {
			list_dir_end();
		}

		ListingMode intended_listing_mode = LISTING_MODE_NONE;
		Error result = OK;

		if (!current_da) {
			AddonsFileSystemManager::get_singleton()->get_all_subdirectories(&addons_fs_children);
			intended_listing_mode = LISTING_MODE_ADDONS_FS_ROOT;
		} else if (get_current_dir() == "res://addons") {
			result = da_normal->list_dir_begin();
			if (result == OK) {
				intended_listing_mode = LISTING_MODE_RES_ADDONS;
			}
		} else {
			intended_listing_mode = LISTING_MODE_PASS_THROUGH;
			if (get_current_dir().begins_with("addons://")) {
				Vector<String> parts = get_current_dir().strip_filesystem_prefix().split("/", false, 1);
				if (parts.size() >= 1) {
					PluginsDB::PluginInfo info;
					if (PluginsDB::get_singleton()->get_plugin_info(parts[0], &info)) {
						if (Engine::get_singleton()->is_editor_hint()) {
							if (info.is_pack) {
								editor_only_paths = info.editor_only_paths;
							} else {
								Vector<String> show_editor_only_plugins = EditorSettings::get_singleton()->get_project_metadata("editor_plugins", "show_editor_only", Vector<String>());
								if (show_editor_only_plugins.find(parts[0]) == -1) {
									editor_only_paths = info.editor_only_paths;
								}
							}
						}
						intended_listing_mode = LISTING_MODE_ADDONS_FS_SUBDIR_OR_BELOW;
						listing_da = current_da;
					}
				}
			}
			if (listing_da) {
				result = listing_da->list_dir_begin();
			} else {
				result = current_da->list_dir_begin();
			}
		}

		// This must be set after calling superclass' list_dir_begin() since it may call list_dir_end()
		listing_mode = intended_listing_mode;

		return result;
	}

	virtual String get_next() {

		String result;

		switch (listing_mode) {

			case LISTING_MODE_NONE: {
			} break;

			case LISTING_MODE_PASS_THROUGH: {
				result = da_normal->get_next();
				current_item_is_dir = da_normal->current_is_dir();
				current_item_is_hidden = da_normal->current_is_hidden();
			} break;

			case LISTING_MODE_RES_ADDONS: {
				bool skip;
				String f;
				do {
					skip = false;
					f = da_normal->get_next();
					if (f != "" && da_normal->current_is_dir()) {
						// Consider res://addons/<subdir> inexistent if is a new model plugin
						if (PluginsDB::get_singleton()->has_universal_plugin(f)) {
							skip = true;
						}
					}
				} while (skip);
				result = f;
				current_item_is_dir = da_normal->current_is_dir();
				current_item_is_hidden = da_normal->current_is_hidden();
			} break;

			case LISTING_MODE_ADDONS_FS_ROOT: {
				if (addons_fs_children.size()) {
					String subdir = addons_fs_children[0];
					addons_fs_children.remove(0);
					result = subdir;
					current_item_is_dir = true;
					current_item_is_hidden = AddonsFileSystemManager::get_singleton()->is_subdirectory_hidden(subdir);
				}
			} break;

			case LISTING_MODE_ADDONS_FS_SUBDIR_OR_BELOW: {
				String f = listing_da->get_next();
				result = f;
				current_item_is_dir = listing_da->current_is_dir();
				current_item_is_hidden = listing_da->current_is_hidden();
				if (!current_item_is_hidden) {
					current_item_is_hidden = PluginsDB::get_singleton()->is_editor_only_path(unfix_path(listing_da->get_current_dir()).plus_file(f), &editor_only_paths);
				}
			}
		}

		return result;
	}

	virtual void list_dir_end() {

		switch (listing_mode) {

			case LISTING_MODE_NONE: {
			} break;

			case LISTING_MODE_PASS_THROUGH: {
				da_normal->list_dir_end();
			} break;

			case LISTING_MODE_RES_ADDONS: {
				da_normal->list_dir_end();
			} break;

			case LISTING_MODE_ADDONS_FS_ROOT: {
				addons_fs_children.clear();
			} break;

			case LISTING_MODE_ADDONS_FS_SUBDIR_OR_BELOW: {
				listing_da->list_dir_end();
				listing_da = NULL;

				editor_only_paths = Vector<String>();
			}
		}

		listing_mode = LISTING_MODE_NONE;
	}

	virtual bool current_is_dir() const {

		return current_item_is_dir;
	}

	virtual bool current_is_hidden() const {

		return current_item_is_hidden;
	}

	virtual int get_drive_count() {

		return 2;
	}

	virtual String get_drive(int p_drive) {

		switch (p_drive) {
			case 0: return "res://";
			case 1: return "addons://";
			default: return "";
		}
	}

	virtual bool drives_are_shortcuts() {

		return false;
	}

	virtual Error change_dir(String p_dir) {

		DirAccess *new_da = NULL;
		String new_dir = _resolve_full_path(p_dir, false, &new_da);
		if (new_dir == "") {
			return ERR_INVALID_PARAMETER;
		}

		if (!new_da) {
			current_da = NULL;
			return OK;
		}

		Error err = new_da->change_dir(fix_path(new_dir));
		if (err == OK) {
			current_da = new_da;
		}

		return err;
	}

	virtual String get_current_dir() {

		if (!current_da) {
			return "addons://";
		} else {
			return unfix_path(current_da->get_current_dir());
		}
	}

	virtual String get_current_dir_without_drive() {

		return "/" + get_current_dir().strip_filesystem_prefix();
	}

	virtual Error make_dir(String p_dir) {

		Error result = this->_check_path_availability(p_dir, 2, true);
		if (result != OK) {
			return result;
		}

		DirAccess *relevant_da = NULL;
		String full_dir = _resolve_full_path(p_dir, true, &relevant_da);
		if (full_dir == "") {
			return ERR_INVALID_PARAMETER;
		}
		if (relevant_da == NULL) {
			return ERR_UNAVAILABLE;
		} else if (relevant_da == da_packed) {
			return ERR_CANT_CREATE;
		} else { // da_normal
			return relevant_da->make_dir(fix_path(full_dir));
		}
	}

	virtual bool file_exists(String p_file) {

		if (this->_check_path_availability(p_file) != OK) {
			return false;
		}

		DirAccess *relevant_da = NULL;
		String full_path = _resolve_full_path(p_file, true, &relevant_da);
		if (full_path == "") {
			return false;
		}
		if (!relevant_da) {
			return false;
		} else {
			return relevant_da->file_exists(fix_path(full_path));
		}
	}

	virtual bool dir_exists(String p_dir) {

		if (p_dir == "res://" || p_dir == "addons://") {
			return true;
		}

		if (this->_check_path_availability(p_dir, 1) != OK) {
			return false;
		}

		DirAccess *relevant_da = NULL;
		String full_dir = _resolve_full_path(p_dir, false, &relevant_da);
		if (full_dir == "") {
			return false;
		}
		ERR_FAIL_COND_V(!relevant_da, false);
		return relevant_da->dir_exists(fix_path(full_dir));
	}

	virtual Error rename(String p_from, String p_to) {

		Error result = MAX(
				this->_check_path_availability(p_from),
				this->_check_path_availability(p_to));
		if (result != OK) {
			return result;
		}

		DirAccess *relevant_da_from = NULL;
		String full_path_from = _resolve_full_path(p_from, false, &relevant_da_from);
		if (full_path_from == "") {
			return ERR_INVALID_PARAMETER;
		}
		ERR_FAIL_COND_V(!relevant_da_from, ERR_BUG);

		DirAccess *relevant_da_to = NULL;
		String full_path_to = _resolve_full_path(p_to, false, &relevant_da_to);
		if (full_path_to == "") {
			return ERR_INVALID_PARAMETER;
		}
		ERR_FAIL_COND_V(!relevant_da_to, ERR_BUG);

		if (relevant_da_from != relevant_da_to) {
			return ERR_UNAVAILABLE;
		}

		return relevant_da_from->rename(fix_path(p_from), fix_path(p_to));
	}

	virtual Error remove(String p_name) {

		Error result = this->_check_path_availability(p_name);
		if (result != OK) {
			return result;
		}

		DirAccess *relevant_da = NULL;
		String full_path = _resolve_full_path(p_name, false, &relevant_da);
		if (full_path == "") {
			return ERR_INVALID_PARAMETER;
		}
		ERR_FAIL_COND_V(!relevant_da, ERR_BUG);
		return relevant_da->remove(fix_path(p_name));
	}

	virtual String get_filesystem_type() const {

		if (!current_da) {
			return "ADDONS";
		} else {
			return current_da->get_filesystem_type();
		}
	}

	virtual size_t get_space_left() {

		if (!current_da) {
			return 0;
		} else {
			return current_da->get_space_left();
		}
	}

	DirAccessResources() :
			da_normal(memnew(T)),
			da_packed(memnew(DirAccessPack)),
			current_da(da_normal),
			listing_mode(LISTING_MODE_NONE),
			listing_da(NULL),
			current_item_is_dir(false),
			current_item_is_hidden(false) {
	}

	~DirAccessResources() {

		memdelete(da_normal);
		memdelete(da_packed);
	}
};

#else // TOOLS_ENABLED

namespace {

template <typename T>
class PathAccessResources : public T {
	virtual void _get_path_bases_for_unfix(const String &p_path, String *r_logical_base, String *r_physical_base) const {

		*r_logical_base = "res://";
		*r_physical_base = this->_get_resource_path();
	}

public:
	virtual String fix_path(const String &p_path) const {

		if (p_path.begins_with("res://")) {
			return p_path.replace_first("res:/", this->_get_resource_path());
		} else if (p_path.begins_with("addons://")) {
			Vector<String> parts = p_path.strip_filesystem_prefix().split("/", false, 1);
			if (parts.size() >= 1) {
				return "res://addons/" + String("/").join(parts);
			}
		}

		return p_path;
	}
};

} // namespace

template <class T>
class FileAccessResources : public PathAccessResources<T> {
};

template <class T>
class DirAccessResources : public PathAccessResources<T> {
};

#endif // TOOLS_ENABLED

#endif
