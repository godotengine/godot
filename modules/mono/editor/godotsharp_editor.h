/*************************************************************************/
/*  godotsharp_editor.h                                                  */
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
	AcceptDialog *about_dialog;
	CheckBox *about_dialog_checkbox;

	ToolButton *bottom_panel_btn;

	GodotSharpBuilds *godotsharp_builds;

	MonoDevelopInstance *monodevelop_instance;
#ifdef OSX_ENABLED
	MonoDevelopInstance *visualstudio_mac_instance;
#endif

	bool _create_project_solution();
	void _make_api_solutions_if_needed();
	void _make_api_solutions_if_needed_impl();

	void _remove_create_sln_menu_option();
	void _show_about_dialog();
	void _toggle_about_dialog_on_start(bool p_enabled);

	void _menu_option_pressed(int p_id);

	void _build_solution_pressed();

	static GodotSharpEditor *singleton;

protected:
	void _notification(int p_notification);
	static void _bind_methods();

public:
	enum MenuOptions {
		MENU_CREATE_SLN,
		MENU_ABOUT_CSHARP,
	};

	enum ExternalEditor {
		EDITOR_NONE,
#if defined(WINDOWS_ENABLED)
		//EDITOR_VISUALSTUDIO, // TODO
		EDITOR_MONODEVELOP,
		EDITOR_VSCODE
#elif defined(OSX_ENABLED)
		EDITOR_VISUALSTUDIO_MAC,
		EDITOR_MONODEVELOP,
		EDITOR_VSCODE
#elif defined(UNIX_ENABLED)
		EDITOR_MONODEVELOP,
		EDITOR_VSCODE
#endif
	};

	_FORCE_INLINE_ static GodotSharpEditor *get_singleton() { return singleton; }

	static void register_internal_calls();

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
