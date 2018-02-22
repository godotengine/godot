/*************************************************************************/
/*  godotsharp_editor.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "godotsharp_export.h"
#include "net_solution.h"

#ifdef WINDOWS_ENABLED
#include "../utils/mono_reg_utils.h"
#endif

GodotSharpEditor *GodotSharpEditor::singleton = NULL;

bool GodotSharpEditor::_create_project_solution() {

	EditorProgress pr("create_csharp_solution", TTR("Generating solution..."), 2);

	pr.step(TTR("Generating C# project..."));

	String path = OS::get_singleton()->get_resource_dir();
	String name = ProjectSettings::get_singleton()->get("application/config/name");
	if (name.empty()) {
		name = "UnnamedProject";
	}

	String guid = CSharpProject::generate_game_project(path, name);

	if (guid.length()) {

		NETSolution solution(name);

		if (!solution.set_path(path)) {
			show_error_dialog(TTR("Failed to create solution."));
			return false;
		}

		Vector<String> extra_configs;
		extra_configs.push_back("Tools");

		solution.add_new_project(name, guid, extra_configs);

		Error sln_error = solution.save();

		if (sln_error != OK) {
			show_error_dialog(TTR("Failed to save solution."));
			return false;
		}

		if (!GodotSharpBuilds::make_api_sln(APIAssembly::API_CORE))
			return false;

		if (!GodotSharpBuilds::make_api_sln(APIAssembly::API_EDITOR))
			return false;

		pr.step(TTR("Done"));

		// Here, after all calls to progress_task_step
		call_deferred("_remove_create_sln_menu_option");

	} else {
		show_error_dialog(TTR("Failed to create C# project."));
	}

	return true;
}

void GodotSharpEditor::_remove_create_sln_menu_option() {

	menu_popup->remove_item(menu_popup->get_item_index(MENU_CREATE_SLN));

	if (menu_popup->get_item_count() == 0)
		menu_button->hide();

	bottom_panel_btn->show();
}

void GodotSharpEditor::_show_about_dialog() {

	bool show_on_start = EDITOR_GET("mono/editor/show_info_on_start");
	about_dialog_checkbox->set_pressed(show_on_start);
	about_dialog->popup_centered_minsize();
}

void GodotSharpEditor::_toggle_about_dialog_on_start(bool p_enabled) {

	bool show_on_start = EDITOR_GET("mono/editor/show_info_on_start");
	if (show_on_start != p_enabled) {
		EditorSettings::get_singleton()->set_setting("mono/editor/show_info_on_start", p_enabled);
	}
}

void GodotSharpEditor::_menu_option_pressed(int p_id) {

	switch (p_id) {
		case MENU_CREATE_SLN: {

			_create_project_solution();
		} break;
		case MENU_ABOUT_CSHARP: {

			_show_about_dialog();
		} break;
		default:
			ERR_FAIL();
	}
}

void GodotSharpEditor::_notification(int p_notification) {

	switch (p_notification) {

		case NOTIFICATION_READY: {

			bool show_info_dialog = EDITOR_GET("mono/editor/show_info_on_start");
			if (show_info_dialog) {
				about_dialog->set_exclusive(true);
				_show_about_dialog();
				// Once shown a first time, it can be seen again via the Mono menu - it doesn't have to be exclusive then.
				about_dialog->set_exclusive(false);
			}
		}
	}
}

void GodotSharpEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_create_project_solution"), &GodotSharpEditor::_create_project_solution);
	ClassDB::bind_method(D_METHOD("_remove_create_sln_menu_option"), &GodotSharpEditor::_remove_create_sln_menu_option);
	ClassDB::bind_method(D_METHOD("_toggle_about_dialog_on_start"), &GodotSharpEditor::_toggle_about_dialog_on_start);
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
				args.push_back(script_path + ":" + itos(p_line + 1) + ":" + itos(p_col));
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

			if (p_line >= 0) {
				script_path += ";" + itos(p_line + 1) + ";" + itos(p_col);
			}

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

	bottom_panel_btn = editor->add_bottom_panel_item(TTR("Mono"), memnew(MonoBottomPanel(editor)));

	godotsharp_builds = memnew(GodotSharpBuilds);

	editor->add_child(memnew(MonoReloadNode));

	menu_button = memnew(MenuButton);
	menu_button->set_text(TTR("Mono"));
	menu_popup = menu_button->get_popup();

	// TODO: Remove or edit this info dialog once Mono support is no longer in alpha
	{
		menu_popup->add_item(TTR("About C# support"), MENU_ABOUT_CSHARP);
		about_dialog = memnew(AcceptDialog);
		editor->get_gui_base()->add_child(about_dialog);
		about_dialog->set_title("Important: C# support is not feature-complete");

		// We don't use set_text() as the default AcceptDialog Label doesn't play well with the TextureRect and CheckBox
		// we'll add. Instead we add containers and a new autowrapped Label inside.

		// Main VBoxContainer (icon + label on top, checkbox at bottom)
		VBoxContainer *about_vbc = memnew(VBoxContainer);
		about_dialog->add_child(about_vbc);

		// HBoxContainer for icon + label
		HBoxContainer *about_hbc = memnew(HBoxContainer);
		about_vbc->add_child(about_hbc);

		TextureRect *about_icon = memnew(TextureRect);
		about_hbc->add_child(about_icon);
		Ref<Texture> about_icon_tex = about_icon->get_icon("NodeWarning", "EditorIcons");
		about_icon->set_texture(about_icon_tex);

		Label *about_label = memnew(Label);
		about_hbc->add_child(about_label);
		about_label->set_custom_minimum_size(Size2(600, 150) * EDSCALE);
		about_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		about_label->set_autowrap(true);
		String about_text =
				String("C# support in Godot Engine is a brand new feature and a work in progress.\n") +
				"It is at the alpha stage and thus not suitable for use in production.\n\n" +
				"As of Godot 3.0, C# support is not feature-complete and may crash in some situations. " +
				"Bugs and usability issues will be addressed gradually over 3.0.x and 3.x releases, " +
				"including compatibility breaking changes as new features are implemented for a better overall C# experience.\n\n" +
				"The main missing feature is the ability to export games using C# assemblies - you will therefore be able to develop and run games in the editor, " +
				"but not to share them as standalone binaries yet. This feature is of course high on the priority list and should be available as soon as possible.\n\n" +
				"If you experience issues with this Mono build, please report them on Godot's issue tracker with details about your system, Mono version, IDE, etc.:\n\n" +
				"        https://github.com/godotengine/godot/issues\n\n" +
				"Your critical feedback at this stage will play a great role in shaping the C# support in future releases, so thank you!";
		about_label->set_text(about_text);

		EDITOR_DEF("mono/editor/show_info_on_start", true);

		// CheckBox in main container
		about_dialog_checkbox = memnew(CheckBox);
		about_vbc->add_child(about_dialog_checkbox);
		about_dialog_checkbox->set_text("Show this warning when starting the editor");
		about_dialog_checkbox->connect("toggled", this, "_toggle_about_dialog_on_start");
	}

	String sln_path = GodotSharpDirs::get_project_sln_path();
	String csproj_path = GodotSharpDirs::get_project_csproj_path();

	if (!FileAccess::exists(sln_path) || !FileAccess::exists(csproj_path)) {
		bottom_panel_btn->hide();
		menu_popup->add_item(TTR("Create C# solution"), MENU_CREATE_SLN);
	}

	menu_popup->connect("id_pressed", this, "_menu_option_pressed");

	if (menu_popup->get_item_count() == 0)
		menu_button->hide();

	editor->get_menu_hb()->add_child(menu_button);

	// External editor settings
	EditorSettings *ed_settings = EditorSettings::get_singleton();
	EDITOR_DEF("mono/editor/external_editor", EDITOR_NONE);
	ed_settings->add_property_hint(PropertyInfo(Variant::INT, "mono/editor/external_editor", PROPERTY_HINT_ENUM, "None,MonoDevelop,Visual Studio Code"));

	// Export plugin
	Ref<GodotSharpExport> godotsharp_export;
	godotsharp_export.instance();
	EditorExport::get_singleton()->add_export_plugin(godotsharp_export);
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
