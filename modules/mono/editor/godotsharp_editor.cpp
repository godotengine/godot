/*************************************************************************/
/*  godotsharp_editor.cpp                                                */
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

#include "godotsharp_editor.h"

#include "core/message_queue.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "scene/gui/control.h"
#include "scene/main/node.h"

#include "../csharp_script.h"
#include "../godotsharp_dirs.h"
#include "../mono_gd/gd_mono.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../utils/path_utils.h"
#include "bindings_generator.h"
#include "csharp_project.h"
#include "dotnet_solution.h"
#include "godotsharp_export.h"

#ifdef OSX_ENABLED
#include "../utils/osx_utils.h"
#endif

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

		DotNetSolution solution(name);

		if (!solution.set_path(path)) {
			show_error_dialog(TTR("Failed to create solution."));
			return false;
		}

		DotNetSolution::ProjectInfo proj_info;
		proj_info.guid = guid;
		proj_info.relpath = name + ".csproj";
		proj_info.configs.push_back("Debug");
		proj_info.configs.push_back("Release");
		proj_info.configs.push_back("Tools");

		solution.add_new_project(name, proj_info);

		Error sln_error = solution.save();

		if (sln_error != OK) {
			show_error_dialog(TTR("Failed to save solution."));
			return false;
		}

		if (!GodotSharpBuilds::make_api_assembly(APIAssembly::API_CORE))
			return false;

		if (!GodotSharpBuilds::make_api_assembly(APIAssembly::API_EDITOR))
			return false;

		pr.step(TTR("Done"));

		// Here, after all calls to progress_task_step
		call_deferred("_remove_create_sln_menu_option");

	} else {
		show_error_dialog(TTR("Failed to create C# project."));
	}

	return true;
}

void GodotSharpEditor::_make_api_solutions_if_needed() {
	// I'm sick entirely of ProgressDialog

	static int attempts_left = 100;

	if (MessageQueue::get_singleton()->is_flushing() || !SceneTree::get_singleton()) {
		ERR_FAIL_COND(attempts_left == 0); // You've got to be kidding

		if (SceneTree::get_singleton()) {
			SceneTree::get_singleton()->connect("idle_frame", this, "_make_api_solutions_if_needed", Vector<Variant>());
		} else {
			call_deferred("_make_api_solutions_if_needed");
		}

		attempts_left--;
		return;
	}

	// Recursion guard needed because signals don't play well with ProgressDialog either, but unlike
	// the message queue, with signals the collateral damage should be minimal in the worst case.
	static bool recursion_guard = false;
	if (!recursion_guard) {
		recursion_guard = true;

		// Oneshot signals don't play well with ProgressDialog either, so we do it this way instead
		SceneTree::get_singleton()->disconnect("idle_frame", this, "_make_api_solutions_if_needed");

		_make_api_solutions_if_needed_impl();

		recursion_guard = false;
	}
}

void GodotSharpEditor::_make_api_solutions_if_needed_impl() {
	// If the project has a solution and C# project make sure the API assemblies are present and up to date
	String res_assemblies_dir = GodotSharpDirs::get_res_assemblies_dir();

	if (!FileAccess::exists(res_assemblies_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll")) ||
			GDMono::get_singleton()->metadata_is_api_assembly_invalidated(APIAssembly::API_CORE)) {
		if (!GodotSharpBuilds::make_api_assembly(APIAssembly::API_CORE))
			return;
	}

	if (!FileAccess::exists(res_assemblies_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll")) ||
			GDMono::get_singleton()->metadata_is_api_assembly_invalidated(APIAssembly::API_EDITOR)) {
		if (!GodotSharpBuilds::make_api_assembly(APIAssembly::API_EDITOR))
			return; // Redundant? I don't think so
	}
}

void GodotSharpEditor::_remove_create_sln_menu_option() {

	menu_popup->remove_item(menu_popup->get_item_index(MENU_CREATE_SLN));

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

void GodotSharpEditor::_build_solution_pressed() {

	if (!FileAccess::exists(GodotSharpDirs::get_project_sln_path())) {
		if (!_create_project_solution())
			return; // Failed to create solution
	}

	MonoBottomPanel::get_singleton()->call("_build_project_pressed");
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

	ClassDB::bind_method(D_METHOD("_build_solution_pressed"), &GodotSharpEditor::_build_solution_pressed);
	ClassDB::bind_method(D_METHOD("_create_project_solution"), &GodotSharpEditor::_create_project_solution);
	ClassDB::bind_method(D_METHOD("_make_api_solutions_if_needed"), &GodotSharpEditor::_make_api_solutions_if_needed);
	ClassDB::bind_method(D_METHOD("_remove_create_sln_menu_option"), &GodotSharpEditor::_remove_create_sln_menu_option);
	ClassDB::bind_method(D_METHOD("_toggle_about_dialog_on_start"), &GodotSharpEditor::_toggle_about_dialog_on_start);
	ClassDB::bind_method(D_METHOD("_menu_option_pressed", "id"), &GodotSharpEditor::_menu_option_pressed);
}

MonoBoolean godot_icall_MonoDevelopInstance_IsApplicationBundleInstalled(MonoString *p_bundle_id) {
#ifdef OSX_ENABLED
	return (MonoBoolean)osx_is_app_bundle_installed(GDMonoMarshal::mono_string_to_godot(p_bundle_id));
#else
	(void)p_bundle_id; // UNUSED
	ERR_FAIL_V(false);
#endif
}

MonoString *godot_icall_Utils_OS_GetPlatformName() {
	return GDMonoMarshal::mono_string_from_godot(OS::get_singleton()->get_name());
}

void GodotSharpEditor::register_internal_calls() {

	static bool registered = false;
	ERR_FAIL_COND(registered);
	registered = true;

	mono_add_internal_call("GodotSharpTools.Editor.MonoDevelopInstance::IsApplicationBundleInstalled", (void *)godot_icall_MonoDevelopInstance_IsApplicationBundleInstalled);
	mono_add_internal_call("GodotSharpTools.Utils.OS::GetPlatformName", (void *)godot_icall_Utils_OS_GetPlatformName);

	GodotSharpBuilds::register_internal_calls();
	GodotSharpExport::register_internal_calls();
}

void GodotSharpEditor::show_error_dialog(const String &p_message, const String &p_title) {

	error_dialog->set_title(p_title);
	error_dialog->set_text(p_message);
	error_dialog->popup_centered_minsize();
}

Error GodotSharpEditor::open_in_external_editor(const Ref<Script> &p_script, int p_line, int p_col) {

	ExternalEditor editor = ExternalEditor(int(EditorSettings::get_singleton()->get("mono/editor/external_editor")));

	switch (editor) {
		case EDITOR_VSCODE: {
			static String vscode_path;

			if (vscode_path.empty() || !FileAccess::exists(vscode_path)) {
				// Try to search it again if it wasn't found last time or if it was removed from its location
				bool found = false;

				// TODO: Use initializer lists once C++11 is allowed

				static Vector<String> vscode_names;
				if (vscode_names.empty()) {
					vscode_names.push_back("code");
					vscode_names.push_back("code-oss");
					vscode_names.push_back("vscode");
					vscode_names.push_back("vscode-oss");
					vscode_names.push_back("visual-studio-code");
					vscode_names.push_back("visual-studio-code-oss");
				}
				for (int i = 0; i < vscode_names.size(); i++) {
					vscode_path = path_which(vscode_names[i]);
					if (!vscode_path.empty()) {
						found = true;
						break;
					}
				}

				if (!found)
					vscode_path.clear(); // Not found, clear so next time the empty() check is enough
			}

			List<String> args;

#ifdef OSX_ENABLED
			// The package path is '/Applications/Visual Studio Code.app'
			static const String vscode_bundle_id = "com.microsoft.VSCode";
			static bool osx_app_bundle_installed = osx_is_app_bundle_installed(vscode_bundle_id);

			if (osx_app_bundle_installed) {
				args.push_back("-b");
				args.push_back(vscode_bundle_id);

				// The reusing of existing windows made by the 'open' command might not choose a wubdiw that is
				// editing our folder. It's better to ask for a new window and let VSCode do the window management.
				args.push_back("-n");

				// The open process must wait until the application finishes (which is instant in VSCode's case)
				args.push_back("--wait-apps");

				args.push_back("--args");
			}
#endif

			args.push_back(ProjectSettings::get_singleton()->get_resource_path());

			String script_path = ProjectSettings::get_singleton()->globalize_path(p_script->get_path());

			if (p_line >= 0) {
				args.push_back("-g");
				args.push_back(script_path + ":" + itos(p_line + 1) + ":" + itos(p_col));
			} else {
				args.push_back(script_path);
			}

#ifdef OSX_ENABLED
			ERR_EXPLAIN("Cannot find code editor: VSCode");
			ERR_FAIL_COND_V(!osx_app_bundle_installed && vscode_path.empty(), ERR_FILE_NOT_FOUND);

			String command = osx_app_bundle_installed ? "/usr/bin/open" : vscode_path;
#else
			ERR_EXPLAIN("Cannot find code editor: VSCode");
			ERR_FAIL_COND_V(vscode_path.empty(), ERR_FILE_NOT_FOUND);

			String command = vscode_path;
#endif

			Error err = OS::get_singleton()->execute(command, args, false);

			if (err != OK) {
				ERR_PRINT("Error when trying to execute code editor: VSCode");
				return err;
			}
		} break;
#ifdef OSX_ENABLED
		case EDITOR_VISUALSTUDIO_MAC:
			// [[fallthrough]];
#endif
		case EDITOR_MONODEVELOP: {
#ifdef OSX_ENABLED
			bool is_visualstudio = editor == EDITOR_VISUALSTUDIO_MAC;

			MonoDevelopInstance **instance = is_visualstudio ?
													 &visualstudio_mac_instance :
													 &monodevelop_instance;

			MonoDevelopInstance::EditorId editor_id = is_visualstudio ?
															  MonoDevelopInstance::VISUALSTUDIO_FOR_MAC :
															  MonoDevelopInstance::MONODEVELOP;
#else
			MonoDevelopInstance **instance = &monodevelop_instance;
			MonoDevelopInstance::EditorId editor_id = MonoDevelopInstance::MONODEVELOP;
#endif

			if (!*instance)
				*instance = memnew(MonoDevelopInstance(GodotSharpDirs::get_project_sln_path(), editor_id));

			String script_path = ProjectSettings::get_singleton()->globalize_path(p_script->get_path());

			if (p_line >= 0) {
				script_path += ";" + itos(p_line + 1) + ";" + itos(p_col);
			}

			(*instance)->execute(script_path);
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

	monodevelop_instance = NULL;
#ifdef OSX_ENABLED
	visualstudio_mac_instance = NULL;
#endif

	editor = p_editor;

	error_dialog = memnew(AcceptDialog);
	editor->get_gui_base()->add_child(error_dialog);

	bottom_panel_btn = editor->add_bottom_panel_item(TTR("Mono"), memnew(MonoBottomPanel(editor)));

	godotsharp_builds = memnew(GodotSharpBuilds);

	editor->add_child(memnew(MonoReloadNode));

	menu_popup = memnew(PopupMenu);
	menu_popup->hide();
	menu_popup->set_as_toplevel(true);
	menu_popup->set_pass_on_modal_close_click(false);

	editor->add_tool_submenu_item("Mono", menu_popup);

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
				String("C# support in Godot Engine is in late alpha stage and, while already usable, ") +
				"it is not meant for use in production.\n\n" +
				"Projects can be exported to Linux, macOS and Windows, but not yet to mobile or web platforms. " +
				"Bugs and usability issues will be addressed gradually over future releases, " +
				"potentially including compatibility breaking changes as new features are implemented for a better overall C# experience.\n\n" +
				"If you experience issues with this Mono build, please report them on Godot's issue tracker with details about your system, MSBuild version, IDE, etc.:\n\n" +
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

	if (FileAccess::exists(sln_path) && FileAccess::exists(csproj_path)) {
		// Defer this task because EditorProgress calls Main::iterarion() and the main loop is not yet initialized.
		call_deferred("_make_api_solutions_if_needed");
	} else {
		bottom_panel_btn->hide();
		menu_popup->add_item(TTR("Create C# solution"), MENU_CREATE_SLN);
	}

	menu_popup->connect("id_pressed", this, "_menu_option_pressed");

	ToolButton *build_button = memnew(ToolButton);
	build_button->set_text("Build");
	build_button->set_tooltip("Build solution");
	build_button->set_focus_mode(Control::FOCUS_NONE);
	build_button->connect("pressed", this, "_build_solution_pressed");
	editor->get_menu_hb()->add_child(build_button);

	// External editor settings
	EditorSettings *ed_settings = EditorSettings::get_singleton();
	EDITOR_DEF("mono/editor/external_editor", EDITOR_NONE);

	String settings_hint_str = "Disabled";

#if defined(WINDOWS_ENABLED)
	settings_hint_str += ",MonoDevelop,Visual Studio Code";
#elif defined(OSX_ENABLED)
	settings_hint_str += ",Visual Studio,MonoDevelop,Visual Studio Code";
#elif defined(UNIX_ENABLED)
	settings_hint_str += ",MonoDevelop,Visual Studio Code";
#endif

	ed_settings->add_property_hint(PropertyInfo(Variant::INT, "mono/editor/external_editor", PROPERTY_HINT_ENUM, settings_hint_str));

	// Export plugin
	Ref<GodotSharpExport> godotsharp_export;
	godotsharp_export.instance();
	EditorExport::get_singleton()->add_export_plugin(godotsharp_export);
}

GodotSharpEditor::~GodotSharpEditor() {

	singleton = NULL;

	memdelete(godotsharp_builds);

	if (monodevelop_instance) {
		memdelete(monodevelop_instance);
		monodevelop_instance = NULL;
	}
}

MonoReloadNode *MonoReloadNode::singleton = NULL;

void MonoReloadNode::_reload_timer_timeout() {

	if (CSharpLanguage::get_singleton()->is_assembly_reloading_needed()) {
		CSharpLanguage::get_singleton()->reload_assemblies(false);
	}
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
			if (CSharpLanguage::get_singleton()->is_assembly_reloading_needed()) {
				CSharpLanguage::get_singleton()->reload_assemblies(false);
			}
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
