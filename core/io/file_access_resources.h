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

template <class T>
class FileAccessResources : public T {
protected:
	virtual String fix_path(const String &p_path) const {

		String r_path = p_path.replace("\\", "/");

		if (r_path.begins_with("res://")) {
			r_path = r_path.replace("res:/", _get_resource_path());
		}

		return r_path;
	}

	virtual String unfix_path(const String &p_path) const {

		String resource_path = _get_resource_path();
		return "res://" + p_path.right(resource_path.length());
	}
};

//////////////////////////////////////////////////////////////////////////////////
// DIR ACCESS
//////////////////////////////////////////////////////////////////////////////////

template <class T>
class DirAccessResources : public T {
protected:
	virtual String _get_root_path() const {

		return _get_resource_path();
	}

	virtual String fix_path(const String &p_path) const {

		if (p_path.begins_with("res://")) {
			return p_path.replace_first("res:/", _get_resource_path());
		} else {
			return p_path;
		}
	}

	virtual String unfix_path(const String &p_path) const {

		String resource_path = _get_resource_path();
		return String("res://").plus_file(p_path.right(resource_path.length()));
	}

public:
	virtual int get_drive_count() {

		return 1;
	}

	virtual String get_drive(int p_drive) {

		if (p_drive == 0) {
			return "res://";
		} else {
			return "";
		}
	}

	virtual bool drives_are_shortcuts() {

		return false;
	}

	virtual String get_current_dir_without_drive() {

		String d = T::get_current_dir();
		int p = d.find("://");
		if (p == -1) {
			return d;
		}
		return d.right(p + 3);
	}
};

#endif
