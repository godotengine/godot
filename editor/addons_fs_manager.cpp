/*************************************************************************/
/*  addons_fs_manager.cpp                                                */
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

#ifdef TOOLS_ENABLED

#include "addons_fs_manager.h"

#include "core/os/mutex.h"

AddonsFileSystemManager *AddonsFileSystemManager::singleton;

AddonsFileSystemManager *AddonsFileSystemManager::get_singleton() {

	return singleton;
}

void AddonsFileSystemManager::start_building() {

	_THREAD_SAFE_LOCK_

	subdirs.clear();
	has_any_pack = false;
}

void AddonsFileSystemManager::add_subdirectory(const String &p_subdir, const String &p_location) {

	Subdirectory subdir;
	subdir.location = p_location;
	subdir.is_pack = false;
	subdir.hidden = false;
	subdirs.insert(p_subdir, subdir);
}

void AddonsFileSystemManager::add_pack_subdirectory(const String &p_subdir, const String &p_pack_location) {

	Subdirectory subdir;
	subdir.location = p_pack_location;
	subdir.is_pack = true;
	subdir.hidden = false;
	subdirs.insert(p_subdir, subdir);

	has_any_pack = true;
}

void AddonsFileSystemManager::end_building() {

	_THREAD_SAFE_UNLOCK_
}

bool AddonsFileSystemManager::has_subdirectory(const String &p_subdir) {

	_THREAD_SAFE_METHOD_

	return subdirs.has(p_subdir);
}

String AddonsFileSystemManager::get_subdirectory_location(const String &p_subdir) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!subdirs.has(p_subdir), "");
	if (subdirs[p_subdir].is_pack) {
		return "addons://" + p_subdir;
	} else {
		return subdirs[p_subdir].location;
	}
}

void AddonsFileSystemManager::get_all_subdirectories(Vector<String> *r_subdir_list) {

	_THREAD_SAFE_METHOD_

	r_subdir_list->clear();
	for (Map<String, Subdirectory>::Element *E = subdirs.front(); E; E = E->next()) {
		r_subdir_list->push_back(E->key());
	}
	r_subdir_list->sort();
}

bool AddonsFileSystemManager::find_subdirectory_for_path(const String &p_path, String *r_subdir, String *r_location) {

	_THREAD_SAFE_METHOD_

	for (Map<String, Subdirectory>::Element *E = subdirs.front(); E; E = E->next()) {

		if (p_path.begins_with(E->value().location)) {
			*r_subdir = E->key();
			*r_location = E->value().location;
			return true;
		}
	}

	return false;
}

void AddonsFileSystemManager::set_subdirectory_hidden(const String &p_subdir, bool p_hidden) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!subdirs.has(p_subdir));
	subdirs[p_subdir].hidden = p_hidden;
}

bool AddonsFileSystemManager::is_subdirectory_hidden(const String &p_subdir) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!subdirs.has(p_subdir), true);
	return subdirs[p_subdir].hidden;
}

bool AddonsFileSystemManager::is_path_packed(const String &p_path) {

	_THREAD_SAFE_METHOD_

	// Fast path for the case where not PCK plugins are mounted
	if (!has_any_pack) {
		return false;
	}

	if (!p_path.begins_with("addons://")) {
		return false;
	}

	Vector<String> parts = p_path.strip_filesystem_prefix().split("/", false, 1);
	if (parts.size() == 0) {
		return false;
	}

	Map<String, Subdirectory>::Element *subdir = subdirs.find(parts[0]);
	if (!subdir) {
		return false;
	}

	return subdir->value().is_pack;
}

AddonsFileSystemManager::AddonsFileSystemManager() :
		has_any_pack(false) {

	singleton = this;
}

#endif
