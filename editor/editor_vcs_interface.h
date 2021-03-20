/*************************************************************************/
/*  editor_vcs_interface.h                                               */
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

#ifndef EDITOR_VCS_INTERFACE_H
#define EDITOR_VCS_INTERFACE_H

#include "core/object/class_db.h"
#include "core/string/ustring.h"
#include "scene/gui/panel_container.h"

class EditorVCSInterface : public Object {
	GDCLASS(EditorVCSInterface, Object)

	bool is_initialized;

protected:
	static EditorVCSInterface *singleton;

	static void _bind_methods();

	// Implemented by addons as end points for the proxy functions
	bool _initialize(String p_project_root_path);
	bool _is_vcs_initialized();
	Dictionary _get_modified_files_data();
	void _stage_file(String p_file_path);
	void _unstage_file(String p_file_path);
	void _commit(String p_msg);
	Array _get_file_diff(String p_file_path);
	bool _shut_down();
	String _get_project_name();
	String _get_vcs_name();

public:
	static EditorVCSInterface *get_singleton();
	static void set_singleton(EditorVCSInterface *p_singleton);

	bool is_addon_ready();

	// Proxy functions to the editor for use
	bool initialize(String p_project_root_path);
	bool is_vcs_initialized();
	Dictionary get_modified_files_data();
	void stage_file(String p_file_path);
	void unstage_file(String p_file_path);
	void commit(String p_msg);
	Array get_file_diff(String p_file_path);
	bool shut_down();
	String get_project_name();
	String get_vcs_name();

	EditorVCSInterface();
	virtual ~EditorVCSInterface();
};

#endif // !EDITOR_VCS_INTERFACE_H
