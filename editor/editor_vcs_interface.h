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

#include "core/object.h"
#include "core/ustring.h"
#include "scene/gui/panel_container.h"

class EditorVCSInterface : public Object {
	GDCLASS(EditorVCSInterface, Object)

	bool is_initialized;

protected:
	static EditorVCSInterface *singleton;

	static void _bind_methods();

	// Implemented by addons as end points for the proxy functions
	virtual bool _initialize(String p_project_root_path);
	virtual bool _is_vcs_initialized();
	virtual Dictionary _get_modified_files_data();
	virtual void _stage_file(String p_file_path);
	virtual void _discard_file(String p_file_path);
	virtual void _unstage_file(String p_file_path);
	virtual void _commit(String p_msg);
	virtual Array _get_file_diff(String p_file_path);
	virtual bool _shut_down();
	virtual String _get_project_name();
	virtual String _get_vcs_name();
	virtual Array _get_previous_commits();
	virtual Array _get_branch_list();
	virtual void _pull();
	virtual void _push();
	virtual void _fetch();
	virtual bool _checkout_branch(String p_branch);
	virtual void _set_up_credentials(String p_username, String p_password);

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
	void discard_file(String p_file_path);
	void commit(String p_msg);
	Array get_file_diff(String p_file_path);
	bool shut_down();
	String get_project_name();
	String get_vcs_name();
	Array get_previous_commits();
	Array get_branch_list();
	bool checkout_branch(String p_branch);
	void pull();
	void push();
	void fetch();
	void set_up_credentials(String p_username, String p_password);

	EditorVCSInterface();
	virtual ~EditorVCSInterface();
};

#endif // !EDITOR_VCS_INTERFACE_H
