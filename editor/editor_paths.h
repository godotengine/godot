/*************************************************************************/
/*  editor_paths.h                                                       */
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

#ifndef EDITOR_PATHS_H
#define EDITOR_PATHS_H

#include "core/object/class_db.h"
#include "core/string/ustring.h"

class EditorPaths : public Object {
	GDCLASS(EditorPaths, Object)

	bool paths_valid = false; // If any of the paths can't be created, this is false.
	String data_dir; // Editor data (templates, shader cache, etc.).
	String config_dir; // Editor config (settings, profiles, themes, etc.).
	String cache_dir; // Editor cache (thumbnails, tmp generated files).
	String project_data_dir; // Project-specific data (metadata, shader cache, etc.).
	bool self_contained = false; // Self-contained means everything goes to `editor_data` dir.
	String self_contained_file; // Self-contained file with configuration.

	static EditorPaths *singleton;

protected:
	static void _bind_methods();

public:
	bool are_paths_valid() const;

	String get_data_dir() const;
	String get_config_dir() const;
	String get_cache_dir() const;
	String get_project_data_dir() const;

	bool is_self_contained() const;
	String get_self_contained_file() const;

	static EditorPaths *get_singleton() {
		return singleton;
	}

	static void create();
	static void free();

	EditorPaths();
};

#endif // EDITOR_PATHS_H
