/*************************************************************************/
/*  godotsharp_editor.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef GODOTSHARP_EDITOR_H
#define GODOTSHARP_EDITOR_H

#include "godotsharp_builds.h"

#include "monodevelop_instance.h"

class GodotSharpEditor : public Node {
	GDCLASS(GodotSharpEditor, Object)

	EditorNode *editor;

	MenuButton *menu_button;
	PopupMenu *menu_popup;

	AcceptDialog *error_dialog;

	ToolButton *bottom_panel_btn;

	GodotSharpBuilds *godotsharp_builds;

	MonoDevelopInstance *monodevel_instance;

	bool _create_project_solution();

	void _remove_create_sln_menu_option();

	void _menu_option_pressed(int p_id);

	static GodotSharpEditor *singleton;

protected:
	static void _bind_methods();

public:
	enum MenuOptions {
		MENU_CREATE_SLN
	};

	enum ExternalEditor {
		EDITOR_NONE,
		EDITOR_MONODEVELOP,
		EDITOR_CODE,
	};

	_FORCE_INLINE_ static GodotSharpEditor *get_singleton() { return singleton; }

	void show_error_dialog(const String &p_message, const String &p_title = "Error");

	Error open_in_external_editor(const Ref<Script> &p_script, int p_line, int p_col);
	bool overrides_external_editor();

	GodotSharpEditor(EditorNode *p_editor);
	~GodotSharpEditor();
};

class MonoReloadNode : public Node {
	GDCLASS(MonoReloadNode, Node)

	Timer *reload_timer;

	void _reload_timer_timeout();

	static MonoReloadNode *singleton;

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	_FORCE_INLINE_ static MonoReloadNode *get_singleton() { return singleton; }

	void restart_reload_timer();

	MonoReloadNode();
	~MonoReloadNode();
};

#endif // GODOTSHARP_EDITOR_H
