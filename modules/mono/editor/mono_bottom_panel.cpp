/*************************************************************************/
/*  mono_bottom_panel.cpp                                                */
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
#include "mono_bottom_panel.h"

#include "../csharp_script.h"
#include "godotsharp_editor.h"

MonoBottomPanel *MonoBottomPanel::singleton = NULL;

void MonoBottomPanel::_update_build_tabs_list() {

	build_tabs_list->clear();

	int current_tab = build_tabs->get_current_tab();

	bool no_current_tab = current_tab < 0 || current_tab >= build_tabs->get_tab_count();

	for (int i = 0; i < build_tabs->get_child_count(); i++) {

		MonoBuildTab *tab = Object::cast_to<MonoBuildTab>(build_tabs->get_child(i));

		if (tab) {
			String item_name = tab->build_info.solution.get_file().get_basename();
			item_name += " [" + tab->build_info.configuration + "]";

			build_tabs_list->add_item(item_name, tab->get_icon_texture());

			String item_tooltip = String("Solution: ") + tab->build_info.solution;
			item_tooltip += String("\nConfiguration: ") + tab->build_info.configuration;
			item_tooltip += String("\nStatus: ");

			if (tab->build_exited) {
				item_tooltip += tab->build_result == MonoBuildTab::RESULT_SUCCESS ? "Succeeded" : "Errored";
			} else {
				item_tooltip += "Running";
			}

			if (!tab->build_exited || !tab->build_result == MonoBuildTab::RESULT_SUCCESS) {
				item_tooltip += "\nErrors: " + itos(tab->error_count);
			}

			item_tooltip += "\nWarnings: " + itos(tab->warning_count);

			build_tabs_list->set_item_tooltip(i, item_tooltip);

			if (no_current_tab || current_tab == i) {
				build_tabs_list->select(i);
				_build_tab_item_selected(i);
			}
		}
	}
}

void MonoBottomPanel::add_build_tab(MonoBuildTab *p_build_tab) {

	build_tabs->add_child(p_build_tab);
	raise_build_tab(p_build_tab);
}

void MonoBottomPanel::raise_build_tab(MonoBuildTab *p_build_tab) {

	ERR_FAIL_COND(p_build_tab->get_parent() != build_tabs);
	build_tabs->move_child(p_build_tab, 0);
	_update_build_tabs_list();
}

void MonoBottomPanel::show_build_tab() {

	for (int i = 0; i < panel_tabs->get_tab_count(); i++) {
		if (panel_tabs->get_tab_control(i) == panel_builds_tab) {
			panel_tabs->set_current_tab(i);
			editor->make_bottom_panel_item_visible(this);
			return;
		}
	}

	ERR_PRINT("Builds tab not found");
}

void MonoBottomPanel::_build_tab_item_selected(int p_idx) {

	ERR_FAIL_INDEX(p_idx, build_tabs->get_tab_count());
	build_tabs->set_current_tab(p_idx);
}

void MonoBottomPanel::_build_tab_changed(int p_idx) {

	if (p_idx < 0 || p_idx >= build_tabs->get_tab_count()) {
		warnings_btn->set_visible(false);
		errors_btn->set_visible(false);
	} else {
		warnings_btn->set_visible(true);
		errors_btn->set_visible(true);
	}
}

void MonoBottomPanel::_warnings_toggled(bool p_pressed) {

	int current_tab = build_tabs->get_current_tab();
	ERR_FAIL_INDEX(current_tab, build_tabs->get_tab_count());
	MonoBuildTab *build_tab = Object::cast_to<MonoBuildTab>(build_tabs->get_child(current_tab));
	build_tab->warnings_visible = p_pressed;
	build_tab->_update_issues_list();
}

void MonoBottomPanel::_errors_toggled(bool p_pressed) {

	int current_tab = build_tabs->get_current_tab();
	ERR_FAIL_INDEX(current_tab, build_tabs->get_tab_count());
	MonoBuildTab *build_tab = Object::cast_to<MonoBuildTab>(build_tabs->get_child(current_tab));
	build_tab->errors_visible = p_pressed;
	build_tab->_update_issues_list();
}

void MonoBottomPanel::_build_project_pressed() {

	GodotSharpBuilds::get_singleton()->build_project_blocking();

	MonoReloadNode::get_singleton()->restart_reload_timer();
	CSharpLanguage::get_singleton()->reload_assemblies_if_needed(true);
}

void MonoBottomPanel::_notification(int p_what) {

	switch (p_what) {

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			panel_tabs->add_style_override("panel", editor->get_gui_base()->get_stylebox("DebuggerPanel", "EditorStyles"));
			panel_tabs->add_style_override("tab_fg", editor->get_gui_base()->get_stylebox("DebuggerTabFG", "EditorStyles"));
			panel_tabs->add_style_override("tab_bg", editor->get_gui_base()->get_stylebox("DebuggerTabBG", "EditorStyles"));
		} break;
	}
}

void MonoBottomPanel::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_build_project_pressed"), &MonoBottomPanel::_build_project_pressed);
	ClassDB::bind_method(D_METHOD("_warnings_toggled", "pressed"), &MonoBottomPanel::_warnings_toggled);
	ClassDB::bind_method(D_METHOD("_errors_toggled", "pressed"), &MonoBottomPanel::_errors_toggled);
	ClassDB::bind_method(D_METHOD("_build_tab_item_selected", "idx"), &MonoBottomPanel::_build_tab_item_selected);
	ClassDB::bind_method(D_METHOD("_build_tab_changed", "idx"), &MonoBottomPanel::_build_tab_changed);
}

MonoBottomPanel::MonoBottomPanel(EditorNode *p_editor) {

	singleton = this;

	editor = p_editor;

	set_v_size_flags(SIZE_EXPAND_FILL);
	set_anchors_and_margins_preset(Control::PRESET_WIDE);

	panel_tabs = memnew(TabContainer);
	panel_tabs->set_tab_align(TabContainer::ALIGN_LEFT);
	panel_tabs->add_style_override("panel", editor->get_gui_base()->get_stylebox("DebuggerPanel", "EditorStyles"));
	panel_tabs->add_style_override("tab_fg", editor->get_gui_base()->get_stylebox("DebuggerTabFG", "EditorStyles"));
	panel_tabs->add_style_override("tab_bg", editor->get_gui_base()->get_stylebox("DebuggerTabBG", "EditorStyles"));
	panel_tabs->set_custom_minimum_size(Size2(0, 228) * EDSCALE);
	panel_tabs->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(panel_tabs);

	{ // Builds
		panel_builds_tab = memnew(VBoxContainer);
		panel_builds_tab->set_name(TTR("Builds"));
		panel_builds_tab->set_h_size_flags(SIZE_EXPAND_FILL);
		panel_tabs->add_child(panel_builds_tab);

		HBoxContainer *toolbar_hbc = memnew(HBoxContainer);
		toolbar_hbc->set_h_size_flags(SIZE_EXPAND_FILL);
		panel_builds_tab->add_child(toolbar_hbc);

		ToolButton *build_project_btn = memnew(ToolButton);
		build_project_btn->set_text("Build Project");
		build_project_btn->set_focus_mode(FOCUS_NONE);
		build_project_btn->connect("pressed", this, "_build_project_pressed");
		toolbar_hbc->add_child(build_project_btn);

		toolbar_hbc->add_spacer();

		warnings_btn = memnew(ToolButton);
		warnings_btn->set_text("Warnings");
		warnings_btn->set_toggle_mode(true);
		warnings_btn->set_pressed(true);
		warnings_btn->set_visible(false);
		warnings_btn->set_focus_mode(FOCUS_NONE);
		warnings_btn->connect("toggled", this, "_warnings_toggled");
		toolbar_hbc->add_child(warnings_btn);

		errors_btn = memnew(ToolButton);
		errors_btn->set_text("Errors");
		errors_btn->set_toggle_mode(true);
		errors_btn->set_pressed(true);
		errors_btn->set_visible(false);
		errors_btn->set_focus_mode(FOCUS_NONE);
		errors_btn->connect("toggled", this, "_errors_toggled");
		toolbar_hbc->add_child(errors_btn);

		HSplitContainer *hsc = memnew(HSplitContainer);
		hsc->set_h_size_flags(SIZE_EXPAND_FILL);
		hsc->set_v_size_flags(SIZE_EXPAND_FILL);
		panel_builds_tab->add_child(hsc);

		build_tabs_list = memnew(ItemList);
		build_tabs_list->set_h_size_flags(SIZE_EXPAND_FILL);
		build_tabs_list->connect("item_selected", this, "_build_tab_item_selected");
		hsc->add_child(build_tabs_list);

		build_tabs = memnew(TabContainer);
		build_tabs->set_tab_align(TabContainer::ALIGN_LEFT);
		build_tabs->set_h_size_flags(SIZE_EXPAND_FILL);
		build_tabs->set_tabs_visible(false);
		build_tabs->connect("tab_changed", this, "_build_tab_changed");
		hsc->add_child(build_tabs);
	}
}

MonoBottomPanel::~MonoBottomPanel() {

	singleton = NULL;
}

void MonoBuildTab::_load_issues_from_file(const String &p_csv_file) {

	FileAccessRef f = FileAccess::open(p_csv_file, FileAccess::READ);

	if (!f)
		return;

	while (!f->eof_reached()) {
		Vector<String> csv_line = f->get_csv_line();

		if (csv_line.size() == 1 && csv_line[0].empty())
			return;

		ERR_CONTINUE(csv_line.size() != 7);

		BuildIssue issue;
		issue.warning = csv_line[0] == "warning";
		issue.file = csv_line[1];
		issue.line = csv_line[2].to_int();
		issue.column = csv_line[3].to_int();
		issue.code = csv_line[4];
		issue.message = csv_line[5];
		issue.project_file = csv_line[6];

		if (issue.warning)
			warning_count += 1;
		else
			error_count += 1;

		issues.push_back(issue);
	}
}

void MonoBuildTab::_update_issues_list() {

	issues_list->clear();

	Ref<Texture> warning_icon = get_icon("Warning", "EditorIcons");
	Ref<Texture> error_icon = get_icon("Error", "EditorIcons");

	for (int i = 0; i < issues.size(); i++) {

		const BuildIssue &issue = issues[i];

		if (!(issue.warning ? warnings_visible : errors_visible))
			continue;

		String tooltip;
		tooltip += String("Message: ") + issue.message;

		if (issue.code.length()) {
			tooltip += String("\nCode: ") + issue.code;
		}

		tooltip += String("\nType: ") + (issue.warning ? "warning" : "error");

		String text;

		if (issue.file.length()) {
			String sline = String::num_int64(issue.line);
			String scolumn = String::num_int64(issue.column);

			text += issue.file + "(";
			text += sline + ",";
			text += scolumn + "): ";

			tooltip += "\nFile: " + issue.file;
			tooltip += "\nLine: " + sline;
			tooltip += "\nColumn: " + scolumn;
		}

		if (issue.project_file.length()) {
			tooltip += "\nProject: " + issue.project_file;
		}

		text += issue.message;

		int line_break_idx = text.find("\n");
		issues_list->add_item(line_break_idx == -1 ? text : text.substr(0, line_break_idx),
				issue.warning ? warning_icon : error_icon);
		int index = issues_list->get_item_count() - 1;
		issues_list->set_item_tooltip(index, tooltip);
		issues_list->set_item_metadata(index, i);
	}
}

Ref<Texture> MonoBuildTab::get_icon_texture() const {

	// FIXME these icons were removed... find something better

	if (build_exited) {
		if (build_result == RESULT_ERROR) {
			return get_icon("DependencyChangedHl", "EditorIcons");
		} else {
			return get_icon("DependencyOkHl", "EditorIcons");
		}
	} else {
		return get_icon("GraphTime", "EditorIcons");
	}
}

MonoBuildInfo MonoBuildTab::get_build_info() {

	return build_info;
}

void MonoBuildTab::on_build_start() {

	build_exited = false;

	issues.clear();
	warning_count = 0;
	error_count = 0;
	_update_issues_list();

	MonoBottomPanel::get_singleton()->raise_build_tab(this);
}

void MonoBuildTab::on_build_exit(BuildResult result) {

	build_exited = true;
	build_result = result;

	_load_issues_from_file(logs_dir.plus_file("msbuild_issues.csv"));
	_update_issues_list();

	MonoBottomPanel::get_singleton()->raise_build_tab(this);
}

void MonoBuildTab::on_build_exec_failed(const String &p_cause) {

	build_exited = true;
	build_result = RESULT_ERROR;

	issues_list->clear();

	BuildIssue issue;
	issue.message = p_cause;
	issue.warning = false;

	error_count += 1;
	issues.push_back(issue);

	_update_issues_list();

	MonoBottomPanel::get_singleton()->raise_build_tab(this);
}

void MonoBuildTab::restart_build() {

	ERR_FAIL_COND(!build_exited);
	GodotSharpBuilds::get_singleton()->restart_build(this);
}

void MonoBuildTab::stop_build() {

	ERR_FAIL_COND(build_exited);
	GodotSharpBuilds::get_singleton()->stop_build(this);
}

void MonoBuildTab::_issue_activated(int p_idx) {

	ERR_FAIL_INDEX(p_idx, issues.size());

	const BuildIssue &issue = issues[p_idx];

	if (issue.project_file.empty() && issue.file.empty())
		return;

	String project_dir = issue.project_file.length() ? issue.project_file.get_base_dir() : build_info.solution.get_base_dir();

	String file = project_dir.simplify_path().plus_file(issue.file.simplify_path());

	if (!FileAccess::exists(file))
		return;

	file = ProjectSettings::get_singleton()->localize_path(file);

	if (file.begins_with("res://")) {
		Ref<Script> script = ResourceLoader::load(file, CSharpLanguage::get_singleton()->get_type());

		if (script.is_valid() && ScriptEditor::get_singleton()->edit(script, issue.line, issue.column)) {
			EditorNode::get_singleton()->call("_editor_select", EditorNode::EDITOR_SCRIPT);
		}
	}
}

void MonoBuildTab::_bind_methods() {

	ClassDB::bind_method("_issue_activated", &MonoBuildTab::_issue_activated);
}

MonoBuildTab::MonoBuildTab(const MonoBuildInfo &p_build_info, const String &p_logs_dir) {

	build_info = p_build_info;
	logs_dir = p_logs_dir;

	build_exited = false;

	issues_list = memnew(ItemList);
	issues_list->set_v_size_flags(SIZE_EXPAND_FILL);
	issues_list->connect("item_activated", this, "_issue_activated");
	add_child(issues_list);

	error_count = 0;
	warning_count = 0;

	errors_visible = true;
	warnings_visible = true;
}
