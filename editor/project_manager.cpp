/**************************************************************************/
/*  project_manager.cpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "project_manager.h"

#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "core/io/stream_peer_tls.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/translation.h"
#include "core/version.h"
#include "editor/editor_about.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/plugins/asset_library_editor_plugin.h"
#include "editor/project_manager/project_dialog.h"
#include "editor/project_manager/project_list.h"
#include "editor/project_manager/project_tag.h"
#include "editor/themes/editor_icons.h"
#include "editor/themes/editor_scale.h"
#include "editor/themes/editor_theme_manager.h"
#include "main/main.h"
#include "scene/gui/center_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/window.h"
#include "servers/display_server.h"
#include "servers/navigation_server_3d.h"
#include "servers/physics_server_2d.h"

constexpr int GODOT4_CONFIG_VERSION = 5;

ProjectManager *ProjectManager::singleton = nullptr;

// Notifications.

void ProjectManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			settings_hb->set_anchors_and_offsets_preset(Control::PRESET_TOP_RIGHT);
			queue_redraw();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			Engine::get_singleton()->set_editor_hint(false);
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			background_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("Background"), EditorStringName(EditorStyles)));
			loading_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), EditorStringName(EditorFonts)));
			search_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("search_panel"), SNAME("ProjectManager")));

			// Top bar.
			search_box->set_right_icon(get_editor_theme_icon(SNAME("Search")));
			language_btn->set_icon(get_editor_theme_icon(SNAME("Environment")));

			// Sidebar.
			create_btn->set_icon(get_editor_theme_icon(SNAME("Add")));
			import_btn->set_icon(get_editor_theme_icon(SNAME("Load")));
			scan_btn->set_icon(get_editor_theme_icon(SNAME("Search")));
			open_btn->set_icon(get_editor_theme_icon(SNAME("Edit")));
			run_btn->set_icon(get_editor_theme_icon(SNAME("Play")));
			rename_btn->set_icon(get_editor_theme_icon(SNAME("Rename")));
			manage_tags_btn->set_icon(get_editor_theme_icon("Script"));
			erase_btn->set_icon(get_editor_theme_icon(SNAME("Remove")));
			erase_missing_btn->set_icon(get_editor_theme_icon(SNAME("Clear")));
			create_tag_btn->set_icon(get_editor_theme_icon("Add"));

			tag_error->add_theme_color_override("font_color", get_theme_color("error_color", EditorStringName(Editor)));
			tag_edit_error->add_theme_color_override("font_color", get_theme_color("error_color", EditorStringName(Editor)));

			create_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			import_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			scan_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			open_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			run_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			rename_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			manage_tags_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			erase_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			erase_missing_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));

			// Asset library popup.
			if (asset_library) {
				// Removes extra border margins.
				asset_library->add_theme_style_override("panel", memnew(StyleBoxEmpty));
			}
		} break;

		case NOTIFICATION_RESIZED: {
			if (open_templates && open_templates->is_visible()) {
				open_templates->popup_centered();
			}
			if (asset_library) {
				real_t size = get_size().x / EDSCALE;
				// Adjust names of tabs to fit the new size.
				if (size < 650) {
					local_projects_vb->set_name(TTR("Local"));
					asset_library->set_name(TTR("Asset Library"));
				} else {
					local_projects_vb->set_name(TTR("Local Projects"));
					asset_library->set_name(TTR("Asset Library Projects"));
				}
			}
		} break;

		case NOTIFICATION_READY: {
			int default_sorting = (int)EDITOR_GET("project_manager/sorting_order");
			filter_option->select(default_sorting);
			_project_list->set_order_option(default_sorting);

#ifndef ANDROID_ENABLED
			if (_project_list->get_project_count() >= 1) {
				// Focus on the search box immediately to allow the user
				// to search without having to reach for their mouse
				search_box->grab_focus();
			}
#endif

			// Suggest browsing asset library to get templates/demos.
			if (asset_library && open_templates && _project_list->get_project_count() == 0) {
				open_templates->popup_centered();
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process_shortcut_input(is_visible_in_tree());
		} break;

		case NOTIFICATION_WM_CLOSE_REQUEST: {
			_dim_window();
		} break;

		case NOTIFICATION_WM_ABOUT: {
			_show_about();
		} break;
	}
}

// Utility data.

Ref<Texture2D> ProjectManager::_file_dialog_get_icon(const String &p_path) {
	if (p_path.get_extension().to_lower() == "godot") {
		return singleton->icon_type_cache["GodotMonochrome"];
	}

	return singleton->icon_type_cache["Object"];
}

Ref<Texture2D> ProjectManager::_file_dialog_get_thumbnail(const String &p_path) {
	if (p_path.get_extension().to_lower() == "godot") {
		return singleton->icon_type_cache["GodotFile"];
	}

	return Ref<Texture2D>();
}

void ProjectManager::_build_icon_type_cache(Ref<Theme> p_theme) {
	if (p_theme.is_null()) {
		return;
	}
	List<StringName> tl;
	p_theme->get_icon_list(EditorStringName(EditorIcons), &tl);
	for (List<StringName>::Element *E = tl.front(); E; E = E->next()) {
		icon_type_cache[E->get()] = p_theme->get_icon(E->get(), EditorStringName(EditorIcons));
	}
}

// Main layout.

void ProjectManager::_update_size_limits() {
	const Size2 minimum_size = Size2(680, 450) * EDSCALE;
	const Size2 default_size = Size2(1024, 600) * EDSCALE;

	// Define a minimum window size to prevent UI elements from overlapping or being cut off.
	Window *w = Object::cast_to<Window>(SceneTree::get_singleton()->get_root());
	if (w) {
		// Calling Window methods this early doesn't sync properties with DS.
		w->set_min_size(minimum_size);
		DisplayServer::get_singleton()->window_set_min_size(minimum_size);
		w->set_size(default_size);
		DisplayServer::get_singleton()->window_set_size(default_size);
	}

	Rect2i screen_rect = DisplayServer::get_singleton()->screen_get_usable_rect(DisplayServer::get_singleton()->window_get_current_screen());
	if (screen_rect.size != Vector2i()) {
		// Center the window on the screen.
		Vector2i window_position;
		window_position.x = screen_rect.position.x + (screen_rect.size.x - default_size.x) / 2;
		window_position.y = screen_rect.position.y + (screen_rect.size.y - default_size.y) / 2;
		DisplayServer::get_singleton()->window_set_position(window_position);

		// Limit popup menus to prevent unusably long lists.
		// We try to set it to half the screen resolution, but no smaller than the minimum window size.
		Size2 half_screen_rect = (screen_rect.size * EDSCALE) / 2;
		Size2 maximum_popup_size = MAX(half_screen_rect, minimum_size);
		language_btn->get_popup()->set_max_size(maximum_popup_size);
	}
}

void ProjectManager::_show_about() {
	about->popup_centered(Size2(780, 500) * EDSCALE);
}

void ProjectManager::_version_button_pressed() {
	DisplayServer::get_singleton()->clipboard_set(version_btn->get_text());
}

void ProjectManager::_on_tab_changed(int p_tab) {
#ifndef ANDROID_ENABLED
	if (p_tab == 0) { // Projects
		// Automatically grab focus when the user moves from the Templates tab
		// back to the Projects tab.
		search_box->grab_focus();
	}

	// The Templates tab's search field is focused on display in the asset
	// library editor plugin code.
#endif
}

void ProjectManager::_open_asset_library() {
	asset_library->disable_community_support();
	tabs->set_current_tab(1);
}

// Quick settings.

void ProjectManager::_language_selected(int p_id) {
	String lang = language_btn->get_item_metadata(p_id);
	EditorSettings::get_singleton()->set("interface/editor/editor_language", lang);

	restart_required_dialog->popup_centered();
}

void ProjectManager::_restart_confirm() {
	List<String> args = OS::get_singleton()->get_cmdline_args();
	Error err = OS::get_singleton()->create_instance(args);
	ERR_FAIL_COND(err);

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_dim_window() {
	// This method must be called before calling `get_tree()->quit()`.
	// Otherwise, its effect won't be visible

	// Dim the project manager window while it's quitting to make it clearer that it's busy.
	// No transition is applied, as the effect needs to be visible immediately
	float c = 0.5f;
	Color dim_color = Color(c, c, c);
	set_modulate(dim_color);
}

// Project list.

void ProjectManager::_scan_projects() {
	scan_dir->popup_file_dialog();
}

void ProjectManager::_run_project() {
	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() < 1) {
		return;
	}

	if (selected_list.size() > 1) {
		multi_run_ask->set_text(vformat(TTR("Are you sure to run %d projects at once?"), selected_list.size()));
		multi_run_ask->popup_centered();
	} else {
		_run_project_confirm();
	}
}

void ProjectManager::_run_project_confirm() {
	Vector<ProjectList::Item> selected_list = _project_list->get_selected_projects();

	for (int i = 0; i < selected_list.size(); ++i) {
		const String &selected_main = selected_list[i].main_scene;
		if (selected_main.is_empty()) {
			run_error_diag->set_text(TTR("Can't run project: no main scene defined.\nPlease edit the project and set the main scene in the Project Settings under the \"Application\" category."));
			run_error_diag->popup_centered();
			continue;
		}

		const String &path = selected_list[i].path;

		// `.substr(6)` on `ProjectSettings::get_singleton()->get_imported_files_path()` strips away the leading "res://".
		if (!DirAccess::exists(path.path_join(ProjectSettings::get_singleton()->get_imported_files_path().substr(6)))) {
			run_error_diag->set_text(TTR("Can't run project: Assets need to be imported.\nPlease edit the project to trigger the initial import."));
			run_error_diag->popup_centered();
			continue;
		}

		print_line("Running project: " + path);

		List<String> args;

		for (const String &a : Main::get_forwardable_cli_arguments(Main::CLI_SCOPE_PROJECT)) {
			args.push_back(a);
		}

		args.push_back("--path");
		args.push_back(path);

		Error err = OS::get_singleton()->create_instance(args);
		ERR_FAIL_COND(err);
	}
}

void ProjectManager::_open_selected_projects() {
	// Show loading text to tell the user that the project manager is busy loading.
	// This is especially important for the Web project manager.
	loading_label->show();

	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	for (const String &path : selected_list) {
		String conf = path.path_join("project.godot");

		if (!FileAccess::exists(conf)) {
			dialog_error->set_text(vformat(TTR("Can't open project at '%s'."), path));
			dialog_error->popup_centered();
			return;
		}

		print_line("Editing project: " + path);

		List<String> args;

		for (const String &a : Main::get_forwardable_cli_arguments(Main::CLI_SCOPE_TOOL)) {
			args.push_back(a);
		}

		args.push_back("--path");
		args.push_back(path);

		args.push_back("--editor");

		Error err = OS::get_singleton()->create_instance(args);
		ERR_FAIL_COND(err);
	}

	_project_list->project_opening_initiated = true;

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_open_selected_projects_ask() {
	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() < 1) {
		return;
	}

	const Size2i popup_min_size = Size2i(600.0 * EDSCALE, 0);

	if (selected_list.size() > 1) {
		multi_open_ask->set_text(vformat(TTR("You requested to open %d projects in parallel. Do you confirm?\nNote that usual checks for engine version compatibility will be bypassed."), selected_list.size()));
		multi_open_ask->popup_centered(popup_min_size);
		return;
	}

	ProjectList::Item project = _project_list->get_selected_projects()[0];
	if (project.missing) {
		return;
	}

	// Update the project settings or don't open.
	const int config_version = project.version;
	PackedStringArray unsupported_features = project.unsupported_features;

	Label *ask_update_label = ask_update_settings->get_label();
	ask_update_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_LEFT); // Reset in case of previous center align.
	full_convert_button->hide();

	ask_update_settings->get_ok_button()->set_text("OK");

	// Check if the config_version property was empty or 0.
	if (config_version == 0) {
		ask_update_settings->set_text(vformat(TTR("The selected project \"%s\" does not specify its supported Godot version in its configuration file (\"project.godot\").\n\nProject path: %s\n\nIf you proceed with opening it, it will be converted to Godot's current configuration file format.\n\nWarning: You won't be able to open the project with previous versions of the engine anymore."), project.project_name, project.path));
		ask_update_settings->popup_centered(popup_min_size);
		return;
	}
	// Check if we need to convert project settings from an earlier engine version.
	if (config_version < ProjectSettings::CONFIG_VERSION) {
		if (config_version == GODOT4_CONFIG_VERSION - 1 && ProjectSettings::CONFIG_VERSION == GODOT4_CONFIG_VERSION) { // Conversion from Godot 3 to 4.
			full_convert_button->show();
			ask_update_settings->set_text(vformat(TTR("The selected project \"%s\" was generated by Godot 3.x, and needs to be converted for Godot 4.x.\n\nProject path: %s\n\nYou have three options:\n- Convert only the configuration file (\"project.godot\"). Use this to open the project without attempting to convert its scenes, resources and scripts.\n- Convert the entire project including its scenes, resources and scripts (recommended if you are upgrading).\n- Do nothing and go back.\n\nWarning: If you select a conversion option, you won't be able to open the project with previous versions of the engine anymore."), project.project_name, project.path));
			ask_update_settings->get_ok_button()->set_text(TTR("Convert project.godot Only"));
		} else {
			ask_update_settings->set_text(vformat(TTR("The selected project \"%s\" was generated by an older engine version, and needs to be converted for this version.\n\nProject path: %s\n\nDo you want to convert it?\n\nWarning: You won't be able to open the project with previous versions of the engine anymore."), project.project_name, project.path));
			ask_update_settings->get_ok_button()->set_text(TTR("Convert project.godot"));
		}
		ask_update_settings->popup_centered(popup_min_size);
		ask_update_settings->get_cancel_button()->grab_focus(); // To prevent accidents.
		return;
	}
	// Check if the file was generated by a newer, incompatible engine version.
	if (config_version > ProjectSettings::CONFIG_VERSION) {
		dialog_error->set_text(vformat(TTR("Can't open project \"%s\" at the following path:\n\n%s\n\nThe project settings were created by a newer engine version, whose settings are not compatible with this version."), project.project_name, project.path));
		dialog_error->popup_centered(popup_min_size);
		return;
	}
	// Check if the project is using features not supported by this build of Godot.
	if (!unsupported_features.is_empty()) {
		String warning_message = "";
		for (int i = 0; i < unsupported_features.size(); i++) {
			String feature = unsupported_features[i];
			if (feature == "Double Precision") {
				warning_message += TTR("Warning: This project uses double precision floats, but this version of\nGodot uses single precision floats. Opening this project may cause data loss.\n\n");
				unsupported_features.remove_at(i);
				i--;
			} else if (feature == "C#") {
				warning_message += TTR("Warning: This project uses C#, but this build of Godot does not have\nthe Mono module. If you proceed you will not be able to use any C# scripts.\n\n");
				unsupported_features.remove_at(i);
				i--;
			} else if (ProjectList::project_feature_looks_like_version(feature)) {
				warning_message += vformat(TTR("Warning: This project was last edited in Godot %s. Opening will change it to Godot %s.\n\n"), Variant(feature), Variant(VERSION_BRANCH));
				unsupported_features.remove_at(i);
				i--;
			}
		}
		if (!unsupported_features.is_empty()) {
			String unsupported_features_str = String(", ").join(unsupported_features);
			warning_message += vformat(TTR("Warning: This project uses the following features not supported by this build of Godot:\n\n%s\n\n"), unsupported_features_str);
		}
		warning_message += TTR("Open anyway? Project will be modified.");
		ask_update_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		ask_update_settings->set_text(warning_message);
		ask_update_settings->popup_centered(popup_min_size);
		return;
	}

	// Open if the project is up-to-date.
	_open_selected_projects();
}

void ProjectManager::_install_project(const String &p_zip_path, const String &p_title) {
	npdialog->set_mode(ProjectDialog::MODE_INSTALL);
	npdialog->set_zip_path(p_zip_path);
	npdialog->set_zip_title(p_title);
	npdialog->show_dialog();
}

void ProjectManager::_import_project() {
	npdialog->set_mode(ProjectDialog::MODE_IMPORT);
	npdialog->ask_for_path_and_show();
}

void ProjectManager::_new_project() {
	npdialog->set_mode(ProjectDialog::MODE_NEW);
	npdialog->show_dialog();
}

void ProjectManager::_rename_project() {
	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() == 0) {
		return;
	}

	for (const String &E : selected_list) {
		npdialog->set_project_path(E);
		npdialog->set_mode(ProjectDialog::MODE_RENAME);
		npdialog->show_dialog();
	}
}

void ProjectManager::_erase_project() {
	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() == 0) {
		return;
	}

	String confirm_message;
	if (selected_list.size() >= 2) {
		confirm_message = vformat(TTR("Remove %d projects from the list?"), selected_list.size());
	} else {
		confirm_message = TTR("Remove this project from the list?");
	}

	erase_ask_label->set_text(confirm_message);
	//delete_project_contents->set_pressed(false);
	erase_ask->popup_centered();
}

void ProjectManager::_erase_missing_projects() {
	erase_missing_ask->set_text(TTR("Remove all missing projects from the list?\nThe project folders' contents won't be modified."));
	erase_missing_ask->popup_centered();
}

void ProjectManager::_erase_project_confirm() {
	_project_list->erase_selected_projects(false);
	_update_project_buttons();
}

void ProjectManager::_erase_missing_projects_confirm() {
	_project_list->erase_missing_projects();
	_update_project_buttons();
}

void ProjectManager::_update_project_buttons() {
	Vector<ProjectList::Item> selected_projects = _project_list->get_selected_projects();
	bool empty_selection = selected_projects.is_empty();

	bool is_missing_project_selected = false;
	for (int i = 0; i < selected_projects.size(); ++i) {
		if (selected_projects[i].missing) {
			is_missing_project_selected = true;
			break;
		}
	}

	erase_btn->set_disabled(empty_selection);
	open_btn->set_disabled(empty_selection || is_missing_project_selected);
	rename_btn->set_disabled(empty_selection || is_missing_project_selected);
	manage_tags_btn->set_disabled(empty_selection || is_missing_project_selected || selected_projects.size() > 1);
	run_btn->set_disabled(empty_selection || is_missing_project_selected);

	erase_missing_btn->set_disabled(!_project_list->is_any_project_missing());
}

void ProjectManager::_on_projects_updated() {
	Vector<ProjectList::Item> selected_projects = _project_list->get_selected_projects();
	int index = 0;
	for (int i = 0; i < selected_projects.size(); ++i) {
		index = _project_list->refresh_project(selected_projects[i].path);
	}
	if (index != -1) {
		_project_list->ensure_project_visible(index);
	}

	_project_list->update_dock_menu();
}

void ProjectManager::_on_project_created(const String &dir) {
	_project_list->add_project(dir, false);
	_project_list->save_config();
	search_box->clear();
	int i = _project_list->refresh_project(dir);
	_project_list->select_project(i);
	_project_list->ensure_project_visible(i);
	_open_selected_projects_ask();

	_project_list->update_dock_menu();
}

void ProjectManager::_on_order_option_changed(int p_idx) {
	if (is_inside_tree()) {
		_project_list->set_order_option(p_idx);
	}
}

void ProjectManager::_on_search_term_changed(const String &p_term) {
	_project_list->set_search_term(p_term);
	_project_list->sort_projects();

	// Select the first visible project in the list.
	// This makes it possible to open a project without ever touching the mouse,
	// as the search field is automatically focused on startup.
	_project_list->select_first_visible_project();
	_update_project_buttons();
}

void ProjectManager::_on_search_term_submitted(const String &p_text) {
	if (tabs->get_current_tab() != 0) {
		return;
	}

	_open_selected_projects_ask();
}

LineEdit *ProjectManager::get_search_box() {
	return search_box;
}

// Project tag management.

void ProjectManager::_manage_project_tags() {
	for (int i = 0; i < project_tags->get_child_count(); i++) {
		project_tags->get_child(i)->queue_free();
	}

	const ProjectList::Item item = _project_list->get_selected_projects()[0];
	current_project_tags = item.tags;
	for (const String &tag : current_project_tags) {
		ProjectTag *tag_control = memnew(ProjectTag(tag, true));
		project_tags->add_child(tag_control);
		tag_control->connect_button_to(callable_mp(this, &ProjectManager::_delete_project_tag).bind(tag));
	}

	tag_edit_error->hide();
	tag_manage_dialog->popup_centered(Vector2i(500, 0) * EDSCALE);
}

void ProjectManager::_add_project_tag(const String &p_tag) {
	if (current_project_tags.has(p_tag)) {
		return;
	}
	current_project_tags.append(p_tag);

	ProjectTag *tag_control = memnew(ProjectTag(p_tag, true));
	project_tags->add_child(tag_control);
	tag_control->connect_button_to(callable_mp(this, &ProjectManager::_delete_project_tag).bind(p_tag));
}

void ProjectManager::_delete_project_tag(const String &p_tag) {
	current_project_tags.erase(p_tag);
	for (int i = 0; i < project_tags->get_child_count(); i++) {
		ProjectTag *tag_control = Object::cast_to<ProjectTag>(project_tags->get_child(i));
		if (tag_control && tag_control->get_tag() == p_tag) {
			memdelete(tag_control);
			break;
		}
	}
}

void ProjectManager::_apply_project_tags() {
	PackedStringArray tags;
	for (int i = 0; i < project_tags->get_child_count(); i++) {
		ProjectTag *tag_control = Object::cast_to<ProjectTag>(project_tags->get_child(i));
		if (tag_control) {
			tags.append(tag_control->get_tag());
		}
	}

	ConfigFile cfg;
	const String project_godot = _project_list->get_selected_projects()[0].path.path_join("project.godot");
	Error err = cfg.load(project_godot);
	if (err != OK) {
		tag_edit_error->set_text(vformat(TTR("Couldn't load project at '%s' (error %d). It may be missing or corrupted."), project_godot, err));
		tag_edit_error->show();
		callable_mp((Window *)tag_manage_dialog, &Window::show).call_deferred(); // Make sure the dialog does not disappear.
		return;
	} else {
		tags.sort();
		cfg.set_value("application", "config/tags", tags);
		err = cfg.save(project_godot);
		if (err != OK) {
			tag_edit_error->set_text(vformat(TTR("Couldn't save project at '%s' (error %d)."), project_godot, err));
			tag_edit_error->show();
			callable_mp((Window *)tag_manage_dialog, &Window::show).call_deferred();
			return;
		}
	}

	_on_projects_updated();
}

void ProjectManager::_set_new_tag_name(const String p_name) {
	create_tag_dialog->get_ok_button()->set_disabled(true);
	if (p_name.is_empty()) {
		tag_error->set_text(TTR("Tag name can't be empty."));
		return;
	}

	if (p_name.contains(" ")) {
		tag_error->set_text(TTR("Tag name can't contain spaces."));
		return;
	}

	for (const String &c : forbidden_tag_characters) {
		if (p_name.contains(c)) {
			tag_error->set_text(vformat(TTR("These characters are not allowed in tags: %s."), String(" ").join(forbidden_tag_characters)));
			return;
		}
	}

	if (p_name.to_lower() != p_name) {
		tag_error->set_text(TTR("Tag name must be lowercase."));
		return;
	}

	tag_error->set_text("");
	create_tag_dialog->get_ok_button()->set_disabled(false);
}

void ProjectManager::_create_new_tag() {
	if (!tag_error->get_text().is_empty()) {
		return;
	}
	create_tag_dialog->hide(); // When using text_submitted, need to hide manually.
	add_new_tag(new_tag_name->get_text());
	_add_project_tag(new_tag_name->get_text());
}

void ProjectManager::add_new_tag(const String &p_tag) {
	if (!tag_set.has(p_tag)) {
		tag_set.insert(p_tag);
		ProjectTag *tag_control = memnew(ProjectTag(p_tag));
		all_tags->add_child(tag_control);
		all_tags->move_child(tag_control, -2);
		tag_control->connect_button_to(callable_mp(this, &ProjectManager::_add_project_tag).bind(p_tag));
	}
}

// Project converter/migration tool.

void ProjectManager::_full_convert_button_pressed() {
	ask_update_settings->hide();
	ask_full_convert_dialog->popup_centered(Size2i(600.0 * EDSCALE, 0));
	ask_full_convert_dialog->get_cancel_button()->grab_focus();
}

void ProjectManager::_perform_full_project_conversion() {
	Vector<ProjectList::Item> selected_list = _project_list->get_selected_projects();
	if (selected_list.is_empty()) {
		return;
	}

	const String &path = selected_list[0].path;

	print_line("Converting project: " + path);
	List<String> args;
	args.push_back("--path");
	args.push_back(path);
	args.push_back("--convert-3to4");
	args.push_back("--rendering-driver");
	args.push_back(Main::get_rendering_driver_name());

	Error err = OS::get_singleton()->create_instance(args);
	ERR_FAIL_COND(err);

	_project_list->set_project_version(path, GODOT4_CONFIG_VERSION);
}

// Input and I/O.

void ProjectManager::shortcut_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	Ref<InputEventKey> k = p_ev;

	if (k.is_valid()) {
		if (!k->is_pressed()) {
			return;
		}

		// Pressing Command + Q quits the Project Manager
		// This is handled by the platform implementation on macOS,
		// so only define the shortcut on other platforms
#ifndef MACOS_ENABLED
		if (k->get_keycode_with_modifiers() == (KeyModifierMask::META | Key::Q)) {
			_dim_window();
			get_tree()->quit();
		}
#endif

		if (tabs->get_current_tab() != 0) {
			return;
		}

		bool keycode_handled = true;

		switch (k->get_keycode()) {
			case Key::ENTER: {
				_open_selected_projects_ask();
			} break;
			case Key::HOME: {
				if (_project_list->get_project_count() > 0) {
					_project_list->select_project(0);
					_update_project_buttons();
				}

			} break;
			case Key::END: {
				if (_project_list->get_project_count() > 0) {
					_project_list->select_project(_project_list->get_project_count() - 1);
					_update_project_buttons();
				}

			} break;
			case Key::UP: {
				if (k->is_shift_pressed()) {
					break;
				}

				int index = _project_list->get_single_selected_index();
				if (index > 0) {
					_project_list->select_project(index - 1);
					_project_list->ensure_project_visible(index - 1);
					_update_project_buttons();
				}

				break;
			}
			case Key::DOWN: {
				if (k->is_shift_pressed()) {
					break;
				}

				int index = _project_list->get_single_selected_index();
				if (index + 1 < _project_list->get_project_count()) {
					_project_list->select_project(index + 1);
					_project_list->ensure_project_visible(index + 1);
					_update_project_buttons();
				}

			} break;
			case Key::F: {
				if (k->is_command_or_control_pressed()) {
					this->search_box->grab_focus();
				} else {
					keycode_handled = false;
				}
			} break;
			default: {
				keycode_handled = false;
			} break;
		}

		if (keycode_handled) {
			accept_event();
		}
	}
}

void ProjectManager::_files_dropped(PackedStringArray p_files) {
	// TODO: Support installing multiple ZIPs at the same time?
	if (p_files.size() == 1 && p_files[0].ends_with(".zip")) {
		const String &file = p_files[0];
		_install_project(file, file.get_file().get_basename().capitalize());
		return;
	}

	HashSet<String> folders_set;
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	for (int i = 0; i < p_files.size(); i++) {
		const String &file = p_files[i];
		folders_set.insert(da->dir_exists(file) ? file : file.get_base_dir());
	}
	ERR_FAIL_COND(folders_set.size() == 0); // This can't really happen, we consume every dropped file path above.

	PackedStringArray folders;
	for (const String &E : folders_set) {
		folders.push_back(E);
	}
	_project_list->find_projects_multiple(folders);
}

// Object methods.

ProjectManager::ProjectManager() {
	singleton = this;

	// Turn off some servers we aren't going to be using in the Project Manager.
	NavigationServer3D::get_singleton()->set_active(false);
	PhysicsServer3D::get_singleton()->set_active(false);
	PhysicsServer2D::get_singleton()->set_active(false);

	// Initialize settings.
	{
		if (!EditorSettings::get_singleton()) {
			EditorSettings::create();
		}
		EditorSettings::get_singleton()->set_optimize_save(false); // Just write settings as they come.

		int display_scale = EDITOR_GET("interface/editor/display_scale");

		switch (display_scale) {
			case 0:
				// Try applying a suitable display scale automatically.
				EditorScale::set_scale(EditorSettings::get_singleton()->get_auto_display_scale());
				break;
			case 1:
				EditorScale::set_scale(0.75);
				break;
			case 2:
				EditorScale::set_scale(1.0);
				break;
			case 3:
				EditorScale::set_scale(1.25);
				break;
			case 4:
				EditorScale::set_scale(1.5);
				break;
			case 5:
				EditorScale::set_scale(1.75);
				break;
			case 6:
				EditorScale::set_scale(2.0);
				break;
			default:
				EditorScale::set_scale(EDITOR_GET("interface/editor/custom_display_scale"));
				break;
		}
		EditorFileDialog::get_icon_func = &ProjectManager::_file_dialog_get_icon;
		EditorFileDialog::get_thumbnail_func = &ProjectManager::_file_dialog_get_thumbnail;

		EditorFileDialog::set_default_show_hidden_files(EDITOR_GET("filesystem/file_dialog/show_hidden_files"));
		EditorFileDialog::set_default_display_mode((EditorFileDialog::DisplayMode)EDITOR_GET("filesystem/file_dialog/display_mode").operator int());

		int swap_cancel_ok = EDITOR_GET("interface/editor/accept_dialog_cancel_ok_buttons");
		if (swap_cancel_ok != 0) { // 0 is auto, set in register_scene based on DisplayServer.
			// Swap on means OK first.
			AcceptDialog::set_swap_cancel_ok(swap_cancel_ok == 2);
		}

		OS::get_singleton()->set_low_processor_usage_mode(true);
	}

	// TRANSLATORS: This refers to the application where users manage their Godot projects.
	DisplayServer::get_singleton()->window_set_title(VERSION_NAME + String(" - ") + TTR("Project Manager", "Application"));

	SceneTree::get_singleton()->get_root()->connect("files_dropped", callable_mp(this, &ProjectManager::_files_dropped));

	// Initialize UI.
	{
		int pm_root_dir = EDITOR_GET("interface/editor/ui_layout_direction");
		Control::set_root_layout_direction(pm_root_dir);
		Window::set_root_layout_direction(pm_root_dir);

		EditorThemeManager::initialize();
		Ref<Theme> theme = EditorThemeManager::generate_theme();
		DisplayServer::set_early_window_clear_color_override(true, theme->get_color(SNAME("background"), EditorStringName(Editor)));

		set_theme(theme);
		set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

		_build_icon_type_cache(theme);
	}

	// Project manager layout.

	background_panel = memnew(Panel);
	add_child(background_panel);
	background_panel->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	VBoxContainer *vb = memnew(VBoxContainer);
	background_panel->add_child(vb);
	vb->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 8 * EDSCALE);

	Control *center_box = memnew(Control);
	center_box->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(center_box);

	tabs = memnew(TabContainer);
	tabs->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	center_box->add_child(tabs);
	tabs->connect("tab_changed", callable_mp(this, &ProjectManager::_on_tab_changed));

	// Quick settings.
	{
		settings_hb = memnew(HBoxContainer);
		settings_hb->set_alignment(BoxContainer::ALIGNMENT_END);
		settings_hb->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
		settings_hb->set_anchors_and_offsets_preset(Control::PRESET_TOP_RIGHT);

		// A VBoxContainer that contains a dummy Control node to adjust the LinkButton's vertical position.
		VBoxContainer *spacer_vb = memnew(VBoxContainer);
		settings_hb->add_child(spacer_vb);

		Control *v_spacer = memnew(Control);
		spacer_vb->add_child(v_spacer);

		version_btn = memnew(LinkButton);
		String hash = String(VERSION_HASH);
		if (hash.length() != 0) {
			hash = " " + vformat("[%s]", hash.left(9));
		}
		version_btn->set_text("v" VERSION_FULL_BUILD + hash);
		// Fade the version label to be less prominent, but still readable.
		version_btn->set_self_modulate(Color(1, 1, 1, 0.6));
		version_btn->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
		version_btn->set_tooltip_text(TTR("Click to copy."));
		version_btn->connect("pressed", callable_mp(this, &ProjectManager::_version_button_pressed));
		spacer_vb->add_child(version_btn);

		// Add a small horizontal spacer between the version and language buttons
		// to distinguish them.
		Control *h_spacer = memnew(Control);
		settings_hb->add_child(h_spacer);

		language_btn = memnew(OptionButton);
		language_btn->set_focus_mode(Control::FOCUS_NONE);
		language_btn->set_fit_to_longest_item(false);
		language_btn->set_flat(true);
		language_btn->connect("item_selected", callable_mp(this, &ProjectManager::_language_selected));
#ifdef ANDROID_ENABLED
		// The language selection dropdown doesn't work on Android (as the setting isn't saved), see GH-60353.
		// Also, the dropdown it spawns is very tall and can't be scrolled without a hardware mouse.
		// Hiding the language selection dropdown also leaves more space for the version label to display.
		language_btn->hide();
#endif

		Vector<String> editor_languages;
		List<PropertyInfo> editor_settings_properties;
		EditorSettings::get_singleton()->get_property_list(&editor_settings_properties);
		for (const PropertyInfo &pi : editor_settings_properties) {
			if (pi.name == "interface/editor/editor_language") {
				editor_languages = pi.hint_string.split(",");
				break;
			}
		}

		String current_lang = EDITOR_GET("interface/editor/editor_language");
		language_btn->set_text(current_lang);

		for (int i = 0; i < editor_languages.size(); i++) {
			const String &lang = editor_languages[i];
			String lang_name = TranslationServer::get_singleton()->get_locale_name(lang);
			language_btn->add_item(vformat("[%s] %s", lang, lang_name), i);
			language_btn->set_item_metadata(i, lang);
			if (current_lang == lang) {
				language_btn->select(i);
			}
		}

		settings_hb->add_child(language_btn);
		center_box->add_child(settings_hb);
	}

	// Project list view.
	{
		local_projects_vb = memnew(VBoxContainer);
		local_projects_vb->set_name(TTR("Local Projects"));
		tabs->add_child(local_projects_vb);

		// Project list's top bar.
		{
			HBoxContainer *hb = memnew(HBoxContainer);
			hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			local_projects_vb->add_child(hb);

			create_btn = memnew(Button);
			create_btn->set_text(TTR("New"));
			create_btn->set_shortcut(ED_SHORTCUT("project_manager/new_project", TTR("New Project"), KeyModifierMask::CMD_OR_CTRL | Key::N));
			create_btn->connect("pressed", callable_mp(this, &ProjectManager::_new_project));
			hb->add_child(create_btn);

			import_btn = memnew(Button);
			import_btn->set_text(TTR("Import"));
			import_btn->set_shortcut(ED_SHORTCUT("project_manager/import_project", TTR("Import Project"), KeyModifierMask::CMD_OR_CTRL | Key::I));
			import_btn->connect("pressed", callable_mp(this, &ProjectManager::_import_project));
			hb->add_child(import_btn);

			scan_btn = memnew(Button);
			scan_btn->set_text(TTR("Scan"));
			scan_btn->set_shortcut(ED_SHORTCUT("project_manager/scan_projects", TTR("Scan Projects"), KeyModifierMask::CMD_OR_CTRL | Key::S));
			scan_btn->connect("pressed", callable_mp(this, &ProjectManager::_scan_projects));
			hb->add_child(scan_btn);

			loading_label = memnew(Label(TTR("Loading, please wait...")));
			loading_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			hb->add_child(loading_label);
			// The loading label is shown later.
			loading_label->hide();

			search_box = memnew(LineEdit);
			search_box->set_placeholder(TTR("Filter Projects"));
			search_box->set_tooltip_text(TTR("This field filters projects by name and last path component.\nTo filter projects by name and full path, the query must contain at least one `/` character."));
			search_box->set_clear_button_enabled(true);
			search_box->connect("text_changed", callable_mp(this, &ProjectManager::_on_search_term_changed));
			search_box->connect("text_submitted", callable_mp(this, &ProjectManager::_on_search_term_submitted));
			search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			hb->add_child(search_box);

			Label *sort_label = memnew(Label);
			sort_label->set_text(TTR("Sort:"));
			hb->add_child(sort_label);

			filter_option = memnew(OptionButton);
			filter_option->set_clip_text(true);
			filter_option->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			filter_option->set_stretch_ratio(0.3);
			filter_option->connect("item_selected", callable_mp(this, &ProjectManager::_on_order_option_changed));
			hb->add_child(filter_option);

			Vector<String> sort_filter_titles;
			sort_filter_titles.push_back(TTR("Last Edited"));
			sort_filter_titles.push_back(TTR("Name"));
			sort_filter_titles.push_back(TTR("Path"));
			sort_filter_titles.push_back(TTR("Tags"));

			for (int i = 0; i < sort_filter_titles.size(); i++) {
				filter_option->add_item(sort_filter_titles[i]);
			}
		}

		// Project list and its sidebar.
		{
			HBoxContainer *search_tree_hb = memnew(HBoxContainer);
			local_projects_vb->add_child(search_tree_hb);
			search_tree_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

			search_panel = memnew(PanelContainer);
			search_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			search_tree_hb->add_child(search_panel);

			_project_list = memnew(ProjectList);
			_project_list->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
			search_panel->add_child(_project_list);
			_project_list->connect(ProjectList::SIGNAL_LIST_CHANGED, callable_mp(this, &ProjectManager::_update_project_buttons));
			_project_list->connect(ProjectList::SIGNAL_SELECTION_CHANGED, callable_mp(this, &ProjectManager::_update_project_buttons));
			_project_list->connect(ProjectList::SIGNAL_PROJECT_ASK_OPEN, callable_mp(this, &ProjectManager::_open_selected_projects_ask));

			// The side bar with the edit, run, rename, etc. buttons.
			VBoxContainer *tree_vb = memnew(VBoxContainer);
			tree_vb->set_custom_minimum_size(Size2(120, 120));
			search_tree_hb->add_child(tree_vb);

			tree_vb->add_child(memnew(HSeparator));

			open_btn = memnew(Button);
			open_btn->set_text(TTR("Edit"));
			open_btn->set_shortcut(ED_SHORTCUT("project_manager/edit_project", TTR("Edit Project"), KeyModifierMask::CMD_OR_CTRL | Key::E));
			open_btn->connect("pressed", callable_mp(this, &ProjectManager::_open_selected_projects_ask));
			tree_vb->add_child(open_btn);

			run_btn = memnew(Button);
			run_btn->set_text(TTR("Run"));
			run_btn->set_shortcut(ED_SHORTCUT("project_manager/run_project", TTR("Run Project"), KeyModifierMask::CMD_OR_CTRL | Key::R));
			run_btn->connect("pressed", callable_mp(this, &ProjectManager::_run_project));
			tree_vb->add_child(run_btn);

			rename_btn = memnew(Button);
			rename_btn->set_text(TTR("Rename"));
			// The F2 shortcut isn't overridden with Enter on macOS as Enter is already used to edit a project.
			rename_btn->set_shortcut(ED_SHORTCUT("project_manager/rename_project", TTR("Rename Project"), Key::F2));
			rename_btn->connect("pressed", callable_mp(this, &ProjectManager::_rename_project));
			tree_vb->add_child(rename_btn);

			manage_tags_btn = memnew(Button);
			manage_tags_btn->set_text(TTR("Manage Tags"));
			tree_vb->add_child(manage_tags_btn);

			erase_btn = memnew(Button);
			erase_btn->set_text(TTR("Remove"));
			erase_btn->set_shortcut(ED_SHORTCUT("project_manager/remove_project", TTR("Remove Project"), Key::KEY_DELETE));
			erase_btn->connect("pressed", callable_mp(this, &ProjectManager::_erase_project));
			tree_vb->add_child(erase_btn);

			erase_missing_btn = memnew(Button);
			erase_missing_btn->set_text(TTR("Remove Missing"));
			erase_missing_btn->connect("pressed", callable_mp(this, &ProjectManager::_erase_missing_projects));
			tree_vb->add_child(erase_missing_btn);

			tree_vb->add_spacer();

			about_btn = memnew(Button);
			about_btn->set_text(TTR("About"));
			about_btn->connect("pressed", callable_mp(this, &ProjectManager::_show_about));
			tree_vb->add_child(about_btn);
		}
	}

	if (AssetLibraryEditorPlugin::is_available()) {
		asset_library = memnew(EditorAssetLibrary(true));
		asset_library->set_name(TTR("Asset Library Projects"));
		tabs->add_child(asset_library);
		asset_library->connect("install_asset", callable_mp(this, &ProjectManager::_install_project));
	} else {
		print_verbose("Asset Library not available (due to using Web editor, or SSL support disabled).");
	}

	// Dialogs.
	{
		restart_required_dialog = memnew(ConfirmationDialog);
		restart_required_dialog->set_ok_button_text(TTR("Restart Now"));
		restart_required_dialog->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_restart_confirm));
		restart_required_dialog->set_cancel_button_text(TTR("Continue"));
		restart_required_dialog->set_text(TTR("Settings changed!\nThe project manager must be restarted for changes to take effect."));
		add_child(restart_required_dialog);

		scan_dir = memnew(EditorFileDialog);
		scan_dir->set_previews_enabled(false);
		scan_dir->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		scan_dir->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
		scan_dir->set_title(TTR("Select a Folder to Scan")); // must be after mode or it's overridden
		scan_dir->set_current_dir(EDITOR_GET("filesystem/directories/default_project_path"));
		add_child(scan_dir);
		scan_dir->connect("dir_selected", callable_mp(_project_list, &ProjectList::find_projects));

		erase_missing_ask = memnew(ConfirmationDialog);
		erase_missing_ask->set_ok_button_text(TTR("Remove All"));
		erase_missing_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_erase_missing_projects_confirm));
		add_child(erase_missing_ask);

		erase_ask = memnew(ConfirmationDialog);
		erase_ask->set_ok_button_text(TTR("Remove"));
		erase_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_erase_project_confirm));
		add_child(erase_ask);

		VBoxContainer *erase_ask_vb = memnew(VBoxContainer);
		erase_ask->add_child(erase_ask_vb);

		erase_ask_label = memnew(Label);
		erase_ask_vb->add_child(erase_ask_label);

		// Comment out for now until we have a better warning system to
		// ensure users delete their project only.
		//delete_project_contents = memnew(CheckBox);
		//delete_project_contents->set_text(TTR("Also delete project contents (no undo!)"));
		//erase_ask_vb->add_child(delete_project_contents);

		multi_open_ask = memnew(ConfirmationDialog);
		multi_open_ask->set_ok_button_text(TTR("Edit"));
		multi_open_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_open_selected_projects));
		add_child(multi_open_ask);

		multi_run_ask = memnew(ConfirmationDialog);
		multi_run_ask->set_ok_button_text(TTR("Run"));
		multi_run_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_run_project_confirm));
		add_child(multi_run_ask);

		ask_update_settings = memnew(ConfirmationDialog);
		ask_update_settings->set_autowrap(true);
		ask_update_settings->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_open_selected_projects));
		full_convert_button = ask_update_settings->add_button(TTR("Convert Full Project"), !GLOBAL_GET("gui/common/swap_cancel_ok"));
		full_convert_button->connect("pressed", callable_mp(this, &ProjectManager::_full_convert_button_pressed));
		add_child(ask_update_settings);

		ask_full_convert_dialog = memnew(ConfirmationDialog);
		ask_full_convert_dialog->set_autowrap(true);
		ask_full_convert_dialog->set_text(TTR("This option will perform full project conversion, updating scenes, resources and scripts from Godot 3 to work in Godot 4.\n\nNote that this is a best-effort conversion, i.e. it makes upgrading the project easier, but it will not open out-of-the-box and will still require manual adjustments.\n\nIMPORTANT: Make sure to backup your project before converting, as this operation makes it impossible to open it in older versions of Godot."));
		ask_full_convert_dialog->connect("confirmed", callable_mp(this, &ProjectManager::_perform_full_project_conversion));
		add_child(ask_full_convert_dialog);

		npdialog = memnew(ProjectDialog);
		npdialog->connect("projects_updated", callable_mp(this, &ProjectManager::_on_projects_updated));
		npdialog->connect("project_created", callable_mp(this, &ProjectManager::_on_project_created));
		add_child(npdialog);

		run_error_diag = memnew(AcceptDialog);
		run_error_diag->set_title(TTR("Can't run project"));
		add_child(run_error_diag);

		dialog_error = memnew(AcceptDialog);
		add_child(dialog_error);

		if (asset_library) {
			open_templates = memnew(ConfirmationDialog);
			open_templates->set_text(TTR("You currently don't have any projects.\nWould you like to explore official example projects in the Asset Library?"));
			open_templates->set_ok_button_text(TTR("Open Asset Library"));
			open_templates->connect("confirmed", callable_mp(this, &ProjectManager::_open_asset_library));
			add_child(open_templates);
		}

		about = memnew(EditorAbout);
		add_child(about);
	}

	// Tag management.
	{
		tag_manage_dialog = memnew(ConfirmationDialog);
		add_child(tag_manage_dialog);
		tag_manage_dialog->set_title(TTR("Manage Project Tags"));
		tag_manage_dialog->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_apply_project_tags));
		manage_tags_btn->connect("pressed", callable_mp(this, &ProjectManager::_manage_project_tags));

		VBoxContainer *tag_vb = memnew(VBoxContainer);
		tag_manage_dialog->add_child(tag_vb);

		Label *label = memnew(Label(TTR("Project Tags")));
		tag_vb->add_child(label);
		label->set_theme_type_variation("HeaderMedium");
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		label = memnew(Label(TTR("Click tag to remove it from the project.")));
		tag_vb->add_child(label);
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		project_tags = memnew(HFlowContainer);
		tag_vb->add_child(project_tags);
		project_tags->set_custom_minimum_size(Vector2(0, 100) * EDSCALE);

		tag_vb->add_child(memnew(HSeparator));

		label = memnew(Label(TTR("All Tags")));
		tag_vb->add_child(label);
		label->set_theme_type_variation("HeaderMedium");
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		label = memnew(Label(TTR("Click tag to add it to the project.")));
		tag_vb->add_child(label);
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		all_tags = memnew(HFlowContainer);
		tag_vb->add_child(all_tags);
		all_tags->set_custom_minimum_size(Vector2(0, 100) * EDSCALE);

		tag_edit_error = memnew(Label);
		tag_vb->add_child(tag_edit_error);
		tag_edit_error->set_autowrap_mode(TextServer::AUTOWRAP_WORD);

		create_tag_dialog = memnew(ConfirmationDialog);
		tag_manage_dialog->add_child(create_tag_dialog);
		create_tag_dialog->set_title(TTR("Create New Tag"));
		create_tag_dialog->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_create_new_tag));

		tag_vb = memnew(VBoxContainer);
		create_tag_dialog->add_child(tag_vb);

		Label *info = memnew(Label(TTR("Tags are capitalized automatically when displayed.")));
		tag_vb->add_child(info);

		new_tag_name = memnew(LineEdit);
		tag_vb->add_child(new_tag_name);
		new_tag_name->connect("text_changed", callable_mp(this, &ProjectManager::_set_new_tag_name));
		new_tag_name->connect("text_submitted", callable_mp(this, &ProjectManager::_create_new_tag).unbind(1));
		create_tag_dialog->connect("about_to_popup", callable_mp(new_tag_name, &LineEdit::clear));
		create_tag_dialog->connect("about_to_popup", callable_mp((Control *)new_tag_name, &Control::grab_focus), CONNECT_DEFERRED);

		tag_error = memnew(Label);
		tag_vb->add_child(tag_error);

		create_tag_btn = memnew(Button);
		all_tags->add_child(create_tag_btn);
		create_tag_btn->connect("pressed", callable_mp((Window *)create_tag_dialog, &Window::popup_centered).bind(Vector2i(500, 0) * EDSCALE));
	}

	// Initialize project list.
	{
		Ref<DirAccess> dir_access = DirAccess::create(DirAccess::AccessType::ACCESS_FILESYSTEM);

		String default_project_path = EDITOR_GET("filesystem/directories/default_project_path");
		if (!default_project_path.is_empty() && !dir_access->dir_exists(default_project_path)) {
			Error error = dir_access->make_dir_recursive(default_project_path);
			if (error != OK) {
				ERR_PRINT("Could not create default project directory at: " + default_project_path);
			}
		}

		bool scanned_for_projects = false; // Scanning will update the list automatically.

		String autoscan_path = EDITOR_GET("filesystem/directories/autoscan_project_path");
		if (!autoscan_path.is_empty()) {
			if (dir_access->dir_exists(autoscan_path)) {
				_project_list->find_projects(autoscan_path);
				scanned_for_projects = true;
			} else {
				Error error = dir_access->make_dir_recursive(autoscan_path);
				if (error != OK) {
					ERR_PRINT("Could not create project autoscan directory at: " + autoscan_path);
				}
			}
		}

		if (!scanned_for_projects) {
			_project_list->update_project_list();
		}
	}

	_update_size_limits();
}

ProjectManager::~ProjectManager() {
	singleton = nullptr;
	if (EditorSettings::get_singleton()) {
		EditorSettings::destroy();
	}

	EditorThemeManager::finalize();
}
