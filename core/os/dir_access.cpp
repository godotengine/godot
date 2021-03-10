/*************************************************************************/
/*  dir_access.cpp                                                       */
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

#include "dir_access.h"

#include "core/config/project_settings.h"
#include "core/os/file_access.h"
#include "core/os/memory.h"
#include "core/os/os.h"

String DirAccess::_get_root_path() const {
	switch (_access_type) {
		case ACCESS_RESOURCES:
			return ProjectSettings::get_singleton()->get_resource_path();
		case ACCESS_USERDATA:
			return OS::get_singleton()->get_user_data_dir();
		default:
			return "";
	}
}

String DirAccess::_get_root_string() const {
	switch (_access_type) {
		case ACCESS_RESOURCES:
			return "res://";
		case ACCESS_USERDATA:
			return "user://";
		default:
			return "";
	}
}

int DirAccess::get_current_drive() {
	String path = get_current_dir().to_lower();
	for (int i = 0; i < get_drive_count(); i++) {
		String d = get_drive(i).to_lower();
		if (path.begins_with(d)) {
			return i;
		}
	}

	return 0;
}

bool DirAccess::drives_are_shortcuts() {
	return false;
}

static Error _erase_recursive(DirAccess *da) {
	List<String> dirs;
	List<String> files;

	da->list_dir_begin();
	String n = da->get_next();
	while (n != String()) {
		if (n != "." && n != "..") {
			if (da->current_is_dir()) {
				dirs.push_back(n);
			} else {
				files.push_back(n);
			}
		}

		n = da->get_next();
	}

	da->list_dir_end();

	for (List<String>::Element *E = dirs.front(); E; E = E->next()) {
		Error err = da->change_dir(E->get());
		if (err == OK) {
			err = _erase_recursive(da);
			if (err) {
				da->change_dir("..");
				return err;
			}
			err = da->change_dir("..");
			if (err) {
				return err;
			}
			err = da->remove(da->get_current_dir().plus_file(E->get()));
			if (err) {
				return err;
			}
		} else {
			return err;
		}
	}

	for (List<String>::Element *E = files.front(); E; E = E->next()) {
		Error err = da->remove(da->get_current_dir().plus_file(E->get()));
		if (err) {
			return err;
		}
	}

	return OK;
}

Error DirAccess::erase_contents_recursive() {
	return _erase_recursive(this);
}

Error DirAccess::make_dir_recursive(String p_dir) {
	if (p_dir.length() < 1) {
		return OK;
	}

	String full_dir;

	if (p_dir.is_rel_path()) {
		//append current
		full_dir = get_current_dir().plus_file(p_dir);

	} else {
		full_dir = p_dir;
	}

	full_dir = full_dir.replace("\\", "/");

	//int slices = full_dir.get_slice_count("/");

	String base;

	if (full_dir.begins_with("res://")) {
		base = "res://";
	} else if (full_dir.begins_with("user://")) {
		base = "user://";
	} else if (full_dir.begins_with("/")) {
		base = "/";
	} else if (full_dir.find(":/") != -1) {
		base = full_dir.substr(0, full_dir.find(":/") + 2);
	} else {
		ERR_FAIL_V(ERR_INVALID_PARAMETER);
	}

	full_dir = full_dir.replace_first(base, "").simplify_path();

	Vector<String> subdirs = full_dir.split("/");

	String curpath = base;
	for (int i = 0; i < subdirs.size(); i++) {
		curpath = curpath.plus_file(subdirs[i]);
		Error err = make_dir(curpath);
		if (err != OK && err != ERR_ALREADY_EXISTS) {
			ERR_FAIL_V_MSG(err, "Could not create directory: " + curpath);
		}
	}

	return OK;
}

String DirAccess::fix_path(String p_path) const {
	switch (_access_type) {
		case ACCESS_RESOURCES: {
			if (ProjectSettings::get_singleton()) {
				if (p_path.begins_with("res://")) {
					String resource_path = ProjectSettings::get_singleton()->get_resource_path();
					if (resource_path != "") {
						return p_path.replace_first("res:/", resource_path);
					}
					return p_path.replace_first("res://", "");
				}
			}

		} break;
		case ACCESS_USERDATA: {
			if (p_path.begins_with("user://")) {
				String data_dir = OS::get_singleton()->get_user_data_dir();
				if (data_dir != "") {
					return p_path.replace_first("user:/", data_dir);
				}
				return p_path.replace_first("user://", "");
			}

		} break;
		case ACCESS_FILESYSTEM: {
			return p_path;
		} break;
		case ACCESS_MAX:
			break; // Can't happen, but silences warning
	}

	return p_path;
}

DirAccess::CreateFunc DirAccess::create_func[ACCESS_MAX] = { nullptr, nullptr, nullptr };

DirAccess *DirAccess::create_for_path(const String &p_path) {
	DirAccess *da = nullptr;
	if (p_path.begins_with("res://")) {
		da = create(ACCESS_RESOURCES);
	} else if (p_path.begins_with("user://")) {
		da = create(ACCESS_USERDATA);
	} else {
		da = create(ACCESS_FILESYSTEM);
	}

	return da;
}

DirAccess *DirAccess::open(const String &p_path, Error *r_error) {
	DirAccess *da = create_for_path(p_path);

	ERR_FAIL_COND_V_MSG(!da, nullptr, "Cannot create DirAccess for path '" + p_path + "'.");
	Error err = da->change_dir(p_path);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		memdelete(da);
		return nullptr;
	}

	return da;
}

DirAccess *DirAccess::create(AccessType p_access) {
	DirAccess *da = create_func[p_access] ? create_func[p_access]() : nullptr;
	if (da) {
		da->_access_type = p_access;
	}

	return da;
}

String DirAccess::get_full_path(const String &p_path, AccessType p_access) {
	DirAccess *d = DirAccess::create(p_access);
	if (!d) {
		return p_path;
	}

	d->change_dir(p_path);
	String full = d->get_current_dir();
	memdelete(d);
	return full;
}

Error DirAccess::copy(String p_from, String p_to, int p_chmod_flags) {
	//printf("copy %s -> %s\n",p_from.ascii().get_data(),p_to.ascii().get_data());
	Error err;
	FileAccess *fsrc = FileAccess::open(p_from, FileAccess::READ, &err);

	if (err) {
		ERR_PRINT("Failed to open " + p_from);
		return err;
	}

	FileAccess *fdst = FileAccess::open(p_to, FileAccess::WRITE, &err);
	if (err) {
		fsrc->close();
		memdelete(fsrc);
		ERR_PRINT("Failed to open " + p_to);
		return err;
	}

	fsrc->seek_end(0);
	int size = fsrc->get_position();
	fsrc->seek(0);
	err = OK;
	while (size--) {
		if (fsrc->get_error() != OK) {
			err = fsrc->get_error();
			break;
		}
		if (fdst->get_error() != OK) {
			err = fdst->get_error();
			break;
		}

		fdst->store_8(fsrc->get_8());
	}

	if (err == OK && p_chmod_flags != -1) {
		fdst->close();
		err = FileAccess::set_unix_permissions(p_to, p_chmod_flags);
		// If running on a platform with no chmod support (i.e., Windows), don't fail
		if (err == ERR_UNAVAILABLE) {
			err = OK;
		}
	}

	memdelete(fsrc);
	memdelete(fdst);

	return err;
}

// Changes dir for the current scope, returning back to the original dir
// when scope exits
class DirChanger {
	DirAccess *da;
	String original_dir;

public:
	DirChanger(DirAccess *p_da, String p_dir) :
			da(p_da),
			original_dir(p_da->get_current_dir()) {
		p_da->change_dir(p_dir);
	}

	~DirChanger() {
		da->change_dir(original_dir);
	}
};

Error DirAccess::_copy_dir(DirAccess *p_target_da, String p_to, int p_chmod_flags, bool p_copy_links) {
	List<String> dirs;

	String curdir = get_current_dir();
	list_dir_begin();
	String n = get_next();
	while (n != String()) {
		if (n != "." && n != "..") {
			if (p_copy_links && is_link(get_current_dir().plus_file(n))) {
				create_link(read_link(get_current_dir().plus_file(n)), p_to + n);
			} else if (current_is_dir()) {
				dirs.push_back(n);
			} else {
				const String &rel_path = n;
				if (!n.is_rel_path()) {
					list_dir_end();
					return ERR_BUG;
				}
				Error err = copy(get_current_dir().plus_file(n), p_to + rel_path, p_chmod_flags);
				if (err) {
					list_dir_end();
					return err;
				}
			}
		}

		n = get_next();
	}

	list_dir_end();

	for (List<String>::Element *E = dirs.front(); E; E = E->next()) {
		String rel_path = E->get();
		String target_dir = p_to + rel_path;
		if (!p_target_da->dir_exists(target_dir)) {
			Error err = p_target_da->make_dir(target_dir);
			ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot create directory '" + target_dir + "'.");
		}

		Error err = change_dir(E->get());
		ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot change current directory to '" + E->get() + "'.");

		err = _copy_dir(p_target_da, p_to + rel_path + "/", p_chmod_flags, p_copy_links);
		if (err) {
			change_dir("..");
			ERR_FAIL_V_MSG(err, "Failed to copy recursively.");
		}
		err = change_dir("..");
		ERR_FAIL_COND_V_MSG(err != OK, err, "Failed to go back.");
	}

	return OK;
}

Error DirAccess::copy_dir(String p_from, String p_to, int p_chmod_flags, bool p_copy_links) {
	ERR_FAIL_COND_V_MSG(!dir_exists(p_from), ERR_FILE_NOT_FOUND, "Source directory doesn't exist.");

	DirAccess *target_da = DirAccess::create_for_path(p_to);
	ERR_FAIL_COND_V_MSG(!target_da, ERR_CANT_CREATE, "Cannot create DirAccess for path '" + p_to + "'.");

	if (!target_da->dir_exists(p_to)) {
		Error err = target_da->make_dir_recursive(p_to);
		if (err) {
			memdelete(target_da);
		}
		ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot create directory '" + p_to + "'.");
	}

	if (!p_to.ends_with("/")) {
		p_to = p_to + "/";
	}

	DirChanger dir_changer(this, p_from);
	Error err = _copy_dir(target_da, p_to, p_chmod_flags, p_copy_links);
	memdelete(target_da);

	return err;
}

bool DirAccess::exists(String p_dir) {
	DirAccess *da = DirAccess::create_for_path(p_dir);
	bool valid = da->change_dir(p_dir) == OK;
	memdelete(da);
	return valid;
}
