/*************************************************************************/
/*  mono_bottom_panel.h                                                  */
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
#ifndef MONO_BOTTOM_PANEL_H
#define MONO_BOTTOM_PANEL_H

#include "editor/editor_node.h"
#include "scene/gui/control.h"

#include "mono_build_info.h"

class MonoBuildTab;

class MonoBottomPanel : public VBoxContainer {

	GDCLASS(MonoBottomPanel, VBoxContainer)

	EditorNode *editor;

	TabContainer *panel_tabs;

	VBoxContainer *panel_builds_tab;

	ItemList *build_tabs_list;
	TabContainer *build_tabs;

	Button *warnings_btn;
	Button *errors_btn;

	void _update_build_tabs_list();

	void _build_tab_item_selected(int p_idx);
	void _build_tab_changed(int p_idx);

	void _warnings_toggled(bool p_pressed);
	void _errors_toggled(bool p_pressed);

	void _build_project_pressed();

	static MonoBottomPanel *singleton;

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	_FORCE_INLINE_ static MonoBottomPanel *get_singleton() { return singleton; }

	void add_build_tab(MonoBuildTab *p_build_tab);
	void raise_build_tab(MonoBuildTab *p_build_tab);

	void show_build_tab();

	MonoBottomPanel(EditorNode *p_editor = NULL);
	~MonoBottomPanel();
};

class MonoBuildTab : public VBoxContainer {

	GDCLASS(MonoBuildTab, VBoxContainer)

public:
	enum BuildResult {
		RESULT_ERROR,
		RESULT_SUCCESS
	};

	struct BuildIssue {
		bool warning;
		String file;
		int line;
		int column;
		String code;
		String message;
		String project_file;
	};

private:
	friend class MonoBottomPanel;

	bool build_exited;
	BuildResult build_result;

	Vector<BuildIssue> issues;
	ItemList *issues_list;

	int error_count;
	int warning_count;

	bool errors_visible;
	bool warnings_visible;

	String logs_dir;

	MonoBuildInfo build_info;

	void _load_issues_from_file(const String &p_csv_file);
	void _update_issues_list();

	void _issue_activated(int p_idx);

protected:
	static void _bind_methods();

public:
	Ref<Texture> get_icon_texture() const;

	MonoBuildInfo get_build_info();

	void on_build_start();
	void on_build_exit(BuildResult result);
	void on_build_exec_failed(const String &p_cause);

	void restart_build();
	void stop_build();

	MonoBuildTab(const MonoBuildInfo &p_build_info, const String &p_logs_dir);
};

#endif // MONO_BOTTOM_PANEL_H
