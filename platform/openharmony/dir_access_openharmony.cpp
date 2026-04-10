/**************************************************************************/
/*  dir_access_openharmony.cpp                                            */
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

#include "dir_access_openharmony.h"

#include "os_openharmony.h"

#include <rawfile/raw_dir.h>
#include <rawfile/raw_file_manager.h>

NativeResourceManager *DirAccessOpenHarmony::resource_manager = nullptr;

void DirAccessOpenHarmony::setup(NativeResourceManager *p_resource_manager) {
	DirAccessOpenHarmony::resource_manager = p_resource_manager;
}

String DirAccessOpenHarmony::get_absolute_path(String p_path) {
	if (p_path.is_relative_path()) {
		p_path = current_dir.path_join(p_path);
	}

	return fix_path(p_path);
}

bool DirAccessOpenHarmony::is_in_bundle(String p_path) {
	return p_path.begins_with(OS_OpenHarmony::get_singleton()->get_bundle_resource_dir());
}

Error DirAccessOpenHarmony::list_dir_begin() {
	if (_is_rawdir) {
		list_dir_end();

		String raw_dir = current_dir.trim_prefix(OS_OpenHarmony::get_singleton()->get_bundle_resource_dir());
		_rawdir = OH_ResourceManager_OpenRawDir(resource_manager, raw_dir.utf8().get_data());
		_rawfile_count = OH_ResourceManager_GetRawFileCount(_rawdir);
		_rawdir_counter = 0;
		return OK;
	}
	return DirAccessUnix::list_dir_begin();
}

String DirAccessOpenHarmony::get_next() {
	if (_is_rawdir) {
		if (!_rawdir) {
			return "";
		}
		if (_rawdir_counter >= _rawfile_count) {
			list_dir_end();
			return "";
		}

		_cpath = OH_ResourceManager_GetRawFileName(_rawdir, _rawdir_counter);
		_rawdir_counter++;
		return _cpath.get_file();
	}
	return DirAccessUnix::get_next();
}

bool DirAccessOpenHarmony::current_is_dir() const {
	if (_is_rawdir) {
		if (!_rawdir) {
			return false;
		}
		return !OH_ResourceManager_IsRawDir(resource_manager, _cpath.utf8().get_data());
	}

	return DirAccessUnix::current_is_dir();
}

bool DirAccessOpenHarmony::current_is_hidden() const {
	if (_is_rawdir) {
		return false;
	}
	return DirAccessUnix::current_is_hidden();
}

void DirAccessOpenHarmony::list_dir_end() {
	if (_is_rawdir) {
		if (_rawdir) {
			OH_ResourceManager_CloseRawDir(_rawdir);
			_rawdir = nullptr;
		}
		return;
	}
	DirAccessUnix::list_dir_end();
}

Error DirAccessOpenHarmony::change_dir(String p_dir) {
	p_dir = get_absolute_path(p_dir);
	if (is_in_bundle(p_dir)) {
		if (!dir_exists(p_dir)) {
			return ERR_INVALID_PARAMETER;
		}
		list_dir_end();
		current_dir = p_dir;
		_is_rawdir = true;
		return OK;
	}
	return DirAccessUnix::change_dir(p_dir);
}

String DirAccessOpenHarmony::get_current_dir(bool p_include_drive) const {
	if (_is_rawdir) {
		return current_dir;
	}
	return DirAccessUnix::get_current_dir(p_include_drive);
}

Error DirAccessOpenHarmony::make_dir(String p_dir) {
	p_dir = get_absolute_path(p_dir);
	if (is_in_bundle(p_dir)) {
		return ERR_UNAVAILABLE;
	}
	return DirAccessUnix::make_dir(p_dir);
}

bool DirAccessOpenHarmony::file_exists(String p_file) {
	p_file = get_absolute_path(p_file);
	if (is_in_bundle(p_file)) {
		p_file = p_file.trim_prefix(OS_OpenHarmony::get_singleton()->get_bundle_resource_dir());
		String p_dir = p_file.get_base_dir();
		String p_name = p_file.get_file();
		if (OH_ResourceManager_IsRawDir(resource_manager, p_dir.utf8().get_data())) {
			return false;
		}
		RawDir *rawdir = OH_ResourceManager_OpenRawDir(resource_manager, p_dir.utf8().get_data());
		int count = OH_ResourceManager_GetRawFileCount(rawdir);
		for (int i = 0; i < count; i++) {
			String file_name = OH_ResourceManager_GetRawFileName(rawdir, i);
			if (file_name == p_name) {
				return !OH_ResourceManager_IsRawDir(resource_manager, p_file.utf8().get_data());
			}
		}
		return false;
	}
	return DirAccessUnix::file_exists(p_file);
}

bool DirAccessOpenHarmony::dir_exists(String p_dir) {
	p_dir = get_absolute_path(p_dir);
	if (is_in_bundle(p_dir)) {
		p_dir = p_dir.trim_prefix(OS_OpenHarmony::get_singleton()->get_bundle_resource_dir());
		return OH_ResourceManager_IsRawDir(resource_manager, p_dir.utf8().get_data());
	}
	return DirAccessUnix::dir_exists(p_dir);
}

bool DirAccessOpenHarmony::is_readable(String p_dir) {
	p_dir = get_absolute_path(p_dir);
	if (is_in_bundle(p_dir)) {
		return true;
	}
	return DirAccessUnix::is_readable(p_dir);
}

bool DirAccessOpenHarmony::is_writable(String p_dir) {
	p_dir = get_absolute_path(p_dir);
	if (is_in_bundle(p_dir)) {
		return false;
	}
	return DirAccessUnix::is_writable(p_dir);
}

uint64_t DirAccessOpenHarmony::get_modified_time(String p_file) {
	p_file = get_absolute_path(p_file);
	if (is_in_bundle(p_file)) {
		return 0;
	}
	return DirAccessUnix::get_modified_time(p_file);
}

Error DirAccessOpenHarmony::rename(String p_path, String p_new_path) {
	p_path = get_absolute_path(p_path);
	p_new_path = get_absolute_path(p_new_path);
	if (is_in_bundle(p_path) || is_in_bundle(p_new_path)) {
		return ERR_UNAVAILABLE;
	}
	return DirAccessUnix::rename(p_path, p_new_path);
}

Error DirAccessOpenHarmony::remove(String p_path) {
	p_path = get_absolute_path(p_path);
	if (is_in_bundle(p_path)) {
		return ERR_UNAVAILABLE;
	}

	return DirAccessUnix::remove(p_path);
}

bool DirAccessOpenHarmony::is_link(String p_file) {
	p_file = get_absolute_path(p_file);

	if (is_in_bundle(p_file)) {
		return false;
	}
	return DirAccessUnix::is_link(p_file);
}

String DirAccessOpenHarmony::read_link(String p_file) {
	p_file = get_absolute_path(p_file);
	if (is_in_bundle(p_file)) {
		return "";
	}
	return DirAccessUnix::read_link(p_file);
}

Error DirAccessOpenHarmony::create_link(String p_source, String p_target) {
	p_source = get_absolute_path(p_source);
	p_target = get_absolute_path(p_target);
	if (is_in_bundle(p_source) || is_in_bundle(p_target)) {
		return ERR_UNAVAILABLE;
	}
	return DirAccessUnix::create_link(p_source, p_target);
}

bool DirAccessOpenHarmony::is_case_sensitive(const String &p_path) const {
	return true;
}

uint64_t DirAccessOpenHarmony::get_space_left() {
	if (_is_rawdir) {
		return 0;
	}
	return DirAccessUnix::get_space_left();
}

String DirAccessOpenHarmony::get_filesystem_type() const {
	return "";
}

DirAccessOpenHarmony::DirAccessOpenHarmony() {
}

DirAccessOpenHarmony::~DirAccessOpenHarmony() {
	list_dir_end();
}
