/*************************************************************************/
/*  editor_paths.h                                                       */
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

#ifndef EDITORPATHS_H
#define EDITORPATHS_H

#include "core/config/engine.h"

class EditorPaths : public Object {
	GDCLASS(EditorPaths, Object)

	bool paths_valid = false;
	String settings_dir;
	String data_dir; //editor data dir
	String config_dir; //editor config dir
	String cache_dir; //editor cache dir
	bool self_contained = false; //true if running self contained
	String self_contained_file; //self contained file with configuration

	static EditorPaths *singleton;

protected:
	static void _bind_methods();

public:
	bool are_paths_valid() const;

	String get_settings_dir() const;
	String get_data_dir() const;
	String get_config_dir() const;
	String get_cache_dir() const;
	bool is_self_contained() const;
	String get_self_contained_file() const;

	static EditorPaths *get_singleton() {
		return singleton;
	}

	static void create(bool p_for_project_manager);
	static void free();

	EditorPaths(bool p_for_project_mamanger = false);
};

#endif // EDITORPATHS_H
