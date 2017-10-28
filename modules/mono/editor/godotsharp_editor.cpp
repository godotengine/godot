/*************************************************************************/
/*  godotsharp_editor.cpp                                                */
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
#include "godotsharp_editor.h"

#include "core/os/os.h"
#include "core/project_settings.h"
#include "scene/gui/control.h"
#include "scene/main/node.h"

#include "../csharp_script.h"
#include "../godotsharp_dirs.h"
#include "../mono_gd/gd_mono.h"
#include "../utils/path_utils.h"
#include "bindings_generator.h"
#include "csharp_project.h"
#include "net_solution.h"

#ifdef WINDOWS_ENABLED
#include "../utils/mono_reg_utils.h"
#endif

GodotSharpEditor *GodotSharpEditor::singleton = NULL;

bool GodotSharpEditor::_create_project_solution() {

	EditorProgress pr("create_csharp_solution", "Generating solution...", 2);

	pr.step("Generating C# project...");

	String path = OS::get_singleton()->get_resource_dir();
	String name = ProjectSettings::get_singleton()->get("application/config/name");
	if (name.empty()) {
		name = "UnnamedProject";
	}

	String guid = CSharpProject::generate_game_project(path, name);

	if (guid.length()) {

		NETSolution solution(name);

		if (!solution.set_path(path)) {
			show_error_dialog("Failed to create solution.");
			return false;
		}

		Vector<String> extra_configs;
		extra_configs.push_back("Tools");

		solution.add_new_project(name, guid, extra_configs);

		Error sln_error = solution.save();

		if (sln_error != OK) {
			show_error_dialog("Failed to save solution.");
			return false;
		}

		if (!GodotSharpBuilds::make_api_sln(GodotSharpBuilds::API_CORE))
			return false;

		if (!GodotSharpBuilds::make_api_sln(GodotSharpBuilds::API_EDITOR))
			return false;

		pr.step("Done");

		// Here, after all calls to progress_task_step
		call_deferred("_remove_create_sln_menu_option");

	} else {
		show_error_dialog("Failed to create C# project.");
	}

	return true;
}

void GodotSharpEditor::_remove_create_sln_menu_option() {

	menu_popup->remove_item(menu_popup->get_item_index(MENU_CREATE_SLN));

	if (menu_popup->get_item_count() == 0)
		menu_button->hide();

	bottom_panel_btn->show();
}

void GodotSharpEditor::_menu_option_pressed(int p_id) {

	switch (p_id) {
		case MENU_CREATE_SLN: {

			_create_project_solution();
		} break;
		default:
			ERR_FAIL();
	}
}

void GodotSharpEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_create_project_solution"), &GodotSharpEditor::_create_project_solution);
	ClassDB::bind_method(D_METHOD("_remove_create_sln_menu_option"), &GodotSharpEditor::_remove_create_sln_menu_option);
	ClassDB::bind_method(D_METHOD("_menu_option_pressed", "id"), &GodotSharpEditor::_menu_option_pressed);
}

void GodotSharpEditor::show_error_dialog(const String &p_message, const String &p_title) {

	error_dialog->set_title(p_title);
	error_dialog->set_text(p_message);
	error_dialog->popup_centered_minsize();
}

Error GodotSharpEditor::open_in_external_editor(const Ref<Script> &p_script, int p_line, int p_col) {

	ExternalEditor editor = ExternalEditor(int(EditorSettings::get_singleton()->get("mono/editor/external_editor")));

	switch (editor) {
		case EDITOR_CODE: {
			List<String> args;
			args.push_back(ProjectSettings::get_singleton()->get_resource_path());

			String script_path = ProjectSettings::get_singleton()->globalize_path(p_script->get_path());

			if (p_line >= 0) {
				args.push_back("-g");
				args.push_back(script_path + ":" + itos(p_line) + ":" + itos(p_col));
			} else {
				args.push_back(script_path);
			}

			static String program = path_which("code");

			Error err = OS::get_singleton()->execute(program.length() ? program : "code", args, false);

			if (err != OK) {
				ERR_PRINT("GodotSharp: Could not execute external editor");
				return err;
			}
		} break;
		case EDITOR_MONODEVELOP: {
			if (!monodevel_instance)
				monodevel_instance = memnew(MonoDevelopInstance(GodotSharpDirs::get_project_sln_path()));

			String script_path = ProjectSettings::get_singleton()->globalize_path(p_script->get_path());
			monodevel_instance->execute(script_path);
		} break;
		default:
			return ERR_UNAVAILABLE;
	}

	return OK;
}

bool GodotSharpEditor::overrides_external_editor() {

	return ExternalEditor(int(EditorSettings::get_singleton()->get("mono/editor/external_editor"))) != EDITOR_NONE;
}

GodotSharpEditor::GodotSharpEditor(EditorNode *p_editor) {

	singleton = this;

	monodevel_instance = NULL;

	editor = p_editor;

	error_dialog = memnew(AcceptDialog);
	editor->get_gui_base()->add_child(error_dialog);

	bottom_panel_btn = editor->add_bottom_panel_item("Mono", memnew(MonoBottomPanel(editor)));

	godotsharp_builds = memnew(GodotSharpBuilds);

	editor->add_child(memnew(MonoReloadNode));

	menu_button = memnew(MenuButton);
	menu_button->set_text("Mono");
	menu_popup = menu_button->get_popup();

	String sln_path = GodotSharpDirs::get_project_sln_path();
	String csproj_path = GodotSharpDirs::get_project_csproj_path();

	if (!FileAccess::exists(sln_path) || !FileAccess::exists(csproj_path)) {
		bottom_panel_btn->hide();
		menu_popup->add_item("Create C# solution", MENU_CREATE_SLN);
	}

	menu_popup->connect("id_pressed", this, "_menu_option_pressed");

	if (menu_popup->get_item_count() == 0)
		menu_button->hide();

	editor->get_menu_hb()->add_child(menu_button);

	// External editor settings
	EditorSettings *ed_settings = EditorSettings::get_singleton();
	EDITOR_DEF("mono/editor/external_editor", EDITOR_NONE);
	ed_settings->add_property_hint(PropertyInfo(Variant::INT, "mono/editor/external_editor", PROPERTY_HINT_ENUM, "None,MonoDevelop,Visual Studio Code"));
}

GodotSharpEditor::~GodotSharpEditor() {

	singleton = NULL;

	memdelete(godotsharp_builds);

	if (monodevel_instance) {
		memdelete(monodevel_instance);
		monodevel_instance = NULL;
	}
}

MonoReloadNode *MonoReloadNode::singleton = NULL;

void MonoReloadNode::_reload_timer_timeout() {

	CSharpLanguage::get_singleton()->reload_assemblies_if_needed(false);
}

void MonoReloadNode::restart_reload_timer() {

	reload_timer->stop();
	reload_timer->start();
}

void MonoReloadNode::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_reload_timer_timeout"), &MonoReloadNode::_reload_timer_timeout);
}

void MonoReloadNode::_notification(int p_what) {
	switch (p_what) {
		case MainLoop::NOTIFICATION_WM_FOCUS_IN: {
			restart_reload_timer();
			CSharpLanguage::get_singleton()->reload_assemblies_if_needed(true);
		} break;
		default: {
		} break;
	};
}

MonoReloadNode::MonoReloadNode() {

	singleton = this;

	reload_timer = memnew(Timer);
	add_child(reload_timer);
	reload_timer->set_one_shot(false);
	reload_timer->set_wait_time(EDITOR_DEF("mono/assembly_watch_interval_sec", 0.5));
	reload_timer->connect("timeout", this, "_reload_timer_timeout");
	reload_timer->start();
}

MonoReloadNode::~MonoReloadNode() {

	singleton = NULL;
}
