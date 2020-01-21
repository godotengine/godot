/*************************************************************************/
/*  addons_fs_manager.h                                                  */
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

#ifndef ADDONS_FS_MANAGER_H
#define ADDONS_FS_MANAGER_H

#include "core/map.h"
#include "core/os/thread_safe.h"
#include "core/ustring.h"

class AddonsFileSystemManager {
	_THREAD_SAFE_CLASS_

	static AddonsFileSystemManager *singleton;

	struct Subdirectory {
		String location; // Full real location
		bool is_pack;
		bool hidden;
	};
	Map<String, Subdirectory> subdirs;
	bool has_any_pack;

public:
	static AddonsFileSystemManager *get_singleton();

	void start_building();
	void add_subdirectory(const String &p_subdir, const String &p_location);
	void add_pack_subdirectory(const String &p_subdir, const String &p_pack_location);
	void end_building();

	bool has_subdirectory(const String &p_subdir);
	String get_subdirectory_location(const String &p_subdir);
	void get_all_subdirectories(Vector<String> *r_subdir_list);
	bool find_subdirectory_for_path(const String &p_path, String *r_subdir, String *r_location);

	bool is_path_packed(const String &p_path);
	bool is_path_read_only(const String &p_path);
	bool is_path_or_parent_read_only(const String &p_path);

	void set_subdirectory_hidden(const String &p_subdir, bool p_hidden);
	bool is_subdirectory_hidden(const String &p_subdir);

	AddonsFileSystemManager();
};

#endif
