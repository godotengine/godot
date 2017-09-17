/*************************************************************************/
/*  editor_node.cpp                                                      */
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
#include "editor_node.h"

#include "core/bind/core_bind.h"
#include "core/class_db.h"
#include "core/io/config_file.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/io/stream_peer_ssl.h"
#include "core/io/zip_io.h"
#include "core/message_queue.h"
#include "core/os/file_access.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/path_remap.h"
#include "core/print_string.h"
#include "core/project_settings.h"
#include "core/translation.h"
#include "core/version.h"
#include "main/input_default.h"
#include "scene/resources/packed_scene.h"
#include "servers/physics_2d_server.h"

#include "editor/animation_editor.h"
#include "editor/editor_audio_buses.h"
#include "editor/editor_file_system.h"
#include "editor/editor_help.h"
#include "editor/editor_initialize_ssl.h"
#include "editor/editor_settings.h"
#include "editor/editor_themes.h"
#include "editor/import/editor_import_collada.h"
#include "editor/import/editor_scene_importer_gltf.h"
#include "editor/import/resource_importer_csv_translation.h"
#include "editor/import/resource_importer_obj.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/import/resource_importer_texture.h"
#include "editor/import/resource_importer_wav.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/animation_tree_editor_plugin.h"
#include "editor/plugins/asset_library_editor_plugin.h"
#include "editor/plugins/camera_editor_plugin.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#include "editor/plugins/collision_polygon_2d_editor_plugin.h"
#include "editor/plugins/collision_polygon_editor_plugin.h"
#include "editor/plugins/collision_shape_2d_editor_plugin.h"
#include "editor/plugins/cube_grid_theme_editor_plugin.h"
#include "editor/plugins/curve_editor_plugin.h"
#include "editor/plugins/editor_preview_plugins.h"
#include "editor/plugins/gi_probe_editor_plugin.h"
#include "editor/plugins/gradient_editor_plugin.h"
#include "editor/plugins/item_list_editor_plugin.h"
#include "editor/plugins/light_occluder_2d_editor_plugin.h"
#include "editor/plugins/line_2d_editor_plugin.h"
#include "editor/plugins/material_editor_plugin.h"
#include "editor/plugins/mesh_editor_plugin.h"
#include "editor/plugins/mesh_instance_editor_plugin.h"
#include "editor/plugins/multimesh_editor_plugin.h"
#include "editor/plugins/navigation_mesh_editor_plugin.h"
#include "editor/plugins/navigation_polygon_editor_plugin.h"
#include "editor/plugins/particles_2d_editor_plugin.h"
#include "editor/plugins/particles_editor_plugin.h"
#include "editor/plugins/path_2d_editor_plugin.h"
#include "editor/plugins/path_editor_plugin.h"
#include "editor/plugins/polygon_2d_editor_plugin.h"
#include "editor/plugins/resource_preloader_editor_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/plugins/script_text_editor.h"
#include "editor/plugins/shader_editor_plugin.h"
#include "editor/plugins/shader_graph_editor_plugin.h"
#include "editor/plugins/spatial_editor_plugin.h"
#include "editor/plugins/sprite_frames_editor_plugin.h"
#include "editor/plugins/style_box_editor_plugin.h"
#include "editor/plugins/texture_editor_plugin.h"
#include "editor/plugins/texture_region_editor_plugin.h"
#include "editor/plugins/theme_editor_plugin.h"
#include "editor/plugins/tile_map_editor_plugin.h"
#include "editor/plugins/tile_set_editor_plugin.h"
#include "editor/pvrtc_compress.h"
#include "editor/register_exporters.h"
#include "editor/script_editor_debugger.h"

#include <stdio.h>

EditorNode *EditorNode::singleton = NULL;

void EditorNode::_update_scene_tabs() {

	bool show_rb = EditorSettings::get_singleton()->get("interface/show_script_in_scene_tabs");

	scene_tabs->clear_tabs();
	Ref<Texture> script_icon = gui_base->get_icon("Script", "EditorIcons");
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {

		String type = editor_data.get_scene_type(i);
		Ref<Texture> icon;
		if (type != String()) {

			if (!gui_base->has_icon(type, "EditorIcons")) {
				type = "Node";
			}

			icon = gui_base->get_icon(type, "EditorIcons");
		}

		int current = editor_data.get_edited_scene();
		bool unsaved = (i == current) ? saved_version != editor_data.get_undo_redo().get_version() : editor_data.get_scene_version(i) != 0;
		scene_tabs->add_tab(editor_data.get_scene_title(i) + (unsaved ? "(*)" : ""), icon);

		if (show_rb && editor_data.get_scene_root_script(i).is_valid()) {
			scene_tabs->set_tab_right_button(i, script_icon);
		}
	}

	scene_tabs->set_current_tab(editor_data.get_edited_scene());
	scene_tabs->ensure_tab_visible(editor_data.get_edited_scene());
}

void EditorNode::_update_title() {

	String appname = ProjectSettings::get_singleton()->get("application/config/name");
	String title = appname.empty() ? String(VERSION_FULL_NAME) : String(_MKSTR(VERSION_NAME) + String(" - ") + appname);
	String edited = editor_data.get_edited_scene_root() ? editor_data.get_edited_scene_root()->get_filename() : String();
	if (!edited.empty())
		title += " - " + String(edited.get_file());
	if (unsaved_cache)
		title += " (*)";

	OS::get_singleton()->set_window_title(title);
}

void EditorNode::_unhandled_input(const Ref<InputEvent> &p_event) {

	if (Node::get_viewport()->get_modal_stack_top())
		return; //ignore because of modal window

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed() && !k->is_echo() && !gui_base->get_viewport()->gui_has_modal_stack()) {

		if (ED_IS_SHORTCUT("editor/next_tab", p_event)) {
			int next_tab = editor_data.get_edited_scene() + 1;
			next_tab %= editor_data.get_edited_scene_count();
			_scene_tab_changed(next_tab);
		}
		if (ED_IS_SHORTCUT("editor/prev_tab", p_event)) {
			int next_tab = editor_data.get_edited_scene() - 1;
			next_tab = next_tab >= 0 ? next_tab : editor_data.get_edited_scene_count() - 1;
			_scene_tab_changed(next_tab);
		}
		if (ED_IS_SHORTCUT("editor/filter_files", p_event)) {
			filesystem_dock->focus_on_filter();
		}

		if (ED_IS_SHORTCUT("editor/editor_2d", p_event)) {
			_editor_select(EDITOR_2D);
		} else if (ED_IS_SHORTCUT("editor/editor_3d", p_event)) {
			_editor_select(EDITOR_3D);
		} else if (ED_IS_SHORTCUT("editor/editor_script", p_event)) {
			_editor_select(EDITOR_SCRIPT);
		} else if (ED_IS_SHORTCUT("editor/editor_help", p_event)) {
			emit_signal("request_help_search", "");
		} else if (ED_IS_SHORTCUT("editor/editor_assetlib", p_event)) {
			_editor_select(EDITOR_ASSETLIB);
		} else if (ED_IS_SHORTCUT("editor/editor_next", p_event)) {
			_editor_select_next();
		} else if (ED_IS_SHORTCUT("editor/editor_prev", p_event)) {
			_editor_select_prev();
		}
	}
}

void EditorNode::_notification(int p_what) {

	if (p_what == NOTIFICATION_EXIT_TREE) {

		editor_data.save_editor_external_data();
		FileAccess::set_file_close_fail_notify_callback(NULL);
		log->deinit(); // do not get messages anymore
	}
	if (p_what == NOTIFICATION_PROCESS) {

		if (opening_prev && !confirmation->is_visible())
			opening_prev = false;

		if (unsaved_cache != (saved_version != editor_data.get_undo_redo().get_version())) {

			unsaved_cache = (saved_version != editor_data.get_undo_redo().get_version());
			_update_title();
		}

		if (last_checked_version != editor_data.get_undo_redo().get_version()) {
			_update_scene_tabs();
			last_checked_version = editor_data.get_undo_redo().get_version();
		}

		//update the circle
		uint64_t frame = Engine::get_singleton()->get_frames_drawn();
		uint32_t tick = OS::get_singleton()->get_ticks_msec();

		if (frame != circle_step_frame && (tick - circle_step_msec) > (1000 / 8)) {

			circle_step++;
			if (circle_step >= 8)
				circle_step = 0;

			circle_step_msec = tick;
			circle_step_frame = frame + 1;

			// update the circle itself only when its enabled
			if (!update_menu->get_popup()->is_item_checked(3)) {
				update_menu->set_icon(gui_base->get_icon("Progress" + itos(circle_step + 1), "EditorIcons"));
			}
		}
		editor_selection->update();

		scene_root->set_size_override(true, Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height")));

		ResourceImporterTexture::get_singleton()->update_imports();
	}
	if (p_what == NOTIFICATION_ENTER_TREE) {

		Engine::get_singleton()->set_editor_hint(true);

		get_tree()->get_root()->set_disable_3d(true);
		get_tree()->get_root()->set_as_audio_listener(false);
		get_tree()->get_root()->set_as_audio_listener_2d(false);
		get_tree()->set_auto_accept_quit(false);
		get_tree()->connect("files_dropped", this, "_dropped_files");
		property_editable_warning->set_icon(gui_base->get_icon("NodeWarning", "EditorIcons"));
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		editor_data.clear_edited_scenes();
	}
	if (p_what == NOTIFICATION_READY) {

		VisualServer::get_singleton()->viewport_set_hide_scenario(get_scene_root()->get_viewport_rid(), true);
		VisualServer::get_singleton()->viewport_set_hide_canvas(get_scene_root()->get_viewport_rid(), true);
		VisualServer::get_singleton()->viewport_set_disable_environment(get_viewport()->get_viewport_rid(), true);

		_editor_select(EDITOR_3D);
		_update_debug_options();
	}

	if (p_what == MainLoop::NOTIFICATION_WM_FOCUS_IN) {

		EditorFileSystem::get_singleton()->scan_changes();
	}

	if (p_what == MainLoop::NOTIFICATION_WM_QUIT_REQUEST) {

		_menu_option_confirm(FILE_QUIT, false);
	}

	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		scene_tabs->set_tab_close_display_policy((bool(EDITOR_DEF("interface/always_show_close_button_in_scene_tabs", false)) ? Tabs::CLOSE_BUTTON_SHOW_ALWAYS : Tabs::CLOSE_BUTTON_SHOW_ACTIVE_ONLY));
		property_editor->set_enable_capitalize_paths(bool(EDITOR_DEF("interface/capitalize_properties", true)));
		Ref<Theme> theme = create_editor_theme(theme_base->get_theme());

		theme_base->set_theme(theme);

		gui_base->add_style_override("panel", gui_base->get_stylebox("Background", "EditorStyles"));
		play_button_panel->add_style_override("panel", gui_base->get_stylebox("PlayButtonPanel", "EditorStyles"));
		scene_root_parent->add_style_override("panel", gui_base->get_stylebox("Content", "EditorStyles"));
		bottom_panel->add_style_override("panel", gui_base->get_stylebox("panel", "TabContainer"));
		scene_tabs->add_style_override("tab_fg", gui_base->get_stylebox("SceneTabFG", "EditorStyles"));
		scene_tabs->add_style_override("tab_bg", gui_base->get_stylebox("SceneTabBG", "EditorStyles"));

		file_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
		project_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
		debug_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
		settings_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
		help_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));

		if (bool(EDITOR_DEF("interface/scene_tabs/resize_if_many_tabs", true))) {
			scene_tabs->set_min_width(int(EDITOR_DEF("interface/scene_tabs/minimum_width", 50)) * EDSCALE);
		} else {
			scene_tabs->set_min_width(0);
		}
		_update_scene_tabs();

		//_update_icons
		for (int i = 0; i < singleton->main_editor_buttons.size(); i++) {
			main_editor_buttons[i]->set_icon(gui_base->get_icon(singleton->main_editor_buttons[i]->get_name(), "EditorIcons"));
		}
		play_button->set_icon(gui_base->get_icon("MainPlay", "EditorIcons"));
		play_scene_button->set_icon(gui_base->get_icon("PlayScene", "EditorIcons"));
		play_custom_scene_button->set_icon(gui_base->get_icon("PlayCustom", "EditorIcons"));
		pause_button->set_icon(gui_base->get_icon("Pause", "EditorIcons"));
		stop_button->set_icon(gui_base->get_icon("Stop", "EditorIcons"));

		prev_scene->set_icon(gui_base->get_icon("PrevScene", "EditorIcons"));
		distraction_free->set_icon(gui_base->get_icon("DistractionFree", "EditorIcons"));

		resource_new_button->set_icon(gui_base->get_icon("New", "EditorIcons"));
		resource_load_button->set_icon(gui_base->get_icon("Load", "EditorIcons"));
		resource_save_button->set_icon(gui_base->get_icon("Save", "EditorIcons"));

		property_back->set_icon(gui_base->get_icon("Back", "EditorIcons"));
		property_forward->set_icon(gui_base->get_icon("Forward", "EditorIcons"));
		editor_history_menu->set_icon(gui_base->get_icon("History", "EditorIcons"));

		search_button->set_icon(gui_base->get_icon("Search", "EditorIcons"));
		object_menu->set_icon(gui_base->get_icon("Tools", "EditorIcons"));
		// clear_button->set_icon(gui_base->get_icon("Close", "EditorIcons")); dont have access to that node. needs to become a class property
		update_menu->set_icon(gui_base->get_icon("Collapse", "EditorIcons"));
		dock_tab_move_left->set_icon(theme->get_icon("Back", "EditorIcons"));
		dock_tab_move_right->set_icon(theme->get_icon("Forward", "EditorIcons"));
		update_menu->set_icon(gui_base->get_icon("Progress1", "EditorIcons"));
	}
}

void EditorNode::_fs_changed() {

	for (Set<FileDialog *>::Element *E = file_dialogs.front(); E; E = E->next()) {

		E->get()->invalidate();
	}

	for (Set<EditorFileDialog *>::Element *E = editor_file_dialogs.front(); E; E = E->next()) {

		E->get()->invalidate();
	}

	if (export_defer.preset != "") {
		Ref<EditorExportPreset> preset;
		for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); ++i) {
			preset = EditorExport::get_singleton()->get_export_preset(i);
			if (preset->get_name() == export_defer.preset) {
				break;
			}
			preset.unref();
		}
		if (preset.is_null()) {
			String err = "Unknown export preset: " + export_defer.preset;
			ERR_PRINT(err.utf8().get_data());
		} else {
			Ref<EditorExportPlatform> platform = preset->get_platform();
			if (platform.is_null()) {
				String err = "Preset \"" + export_defer.preset + "\" doesn't have a platform.";
				ERR_PRINT(err.utf8().get_data());
			} else {
				platform->export_project(preset, export_defer.debug, export_defer.path, /*p_flags*/ 0);
			}
		}

		get_tree()->quit();
	}

	{
		//reload changed resources
		List<Ref<Resource> > changed;

		List<Ref<Resource> > cached;
		ResourceCache::get_cached_resources(&cached);
		// FIXME: This should be done in a thread.
		for (List<Ref<Resource> >::Element *E = cached.front(); E; E = E->next()) {

			if (!E->get()->editor_can_reload_from_file())
				continue;
			if (!E->get()->get_path().is_resource_file() && !E->get()->get_path().is_abs_path())
				continue;
			if (!FileAccess::exists(E->get()->get_path()))
				continue;

			if (E->get()->get_import_path() != String()) {
//this is an imported resource, will be reloaded if reimported via the _resources_reimported() callback
//imported resource
#if 0
				uint64_t mt = FileAccess::get_modified_time(E->get()->get_import_path());

				if (mt != E->get()->get_import_last_modified_time()) {
					print_line("success");
					changed.push_back(E->get());
				}
#endif
				continue;
			}

			uint64_t mt = FileAccess::get_modified_time(E->get()->get_path());

			if (mt != E->get()->get_last_modified_time()) {
				changed.push_back(E->get());
			}
		}

		if (changed.size()) {
			for (List<Ref<Resource> >::Element *E = changed.front(); E; E = E->next()) {
				E->get()->reload_from_file();
			}
		}
	}

	_mark_unsaved_scenes();
}

void EditorNode::_resources_reimported(const Vector<String> &p_resources) {
	print_line("reimporting");
	List<String> scenes; //will load later

	for (int i = 0; i < p_resources.size(); i++) {
		String file_type = ResourceLoader::get_resource_type(p_resources[i]);
		if (file_type == "PackedScene") {
			scenes.push_back(p_resources[i]);
			//reload later if needed, first go with normal resources
			continue;
		}

		if (!ResourceCache::has(p_resources[i])) {
			continue; //not loaded, no need to reload
		}
		//reload normally
		Resource *resource = ResourceCache::get(p_resources[i]);
		if (resource) {
			resource->reload_from_file();
		}
	}

	for (List<String>::Element *E = scenes.front(); E; E = E->next()) {
		reload_scene(E->get());
	}
}

void EditorNode::_sources_changed(bool p_exist) {

	if (waiting_for_first_scan) {

		if (defer_load_scene != "") {

			print_line("loading scene DEFERRED");
			load_scene(defer_load_scene);
			defer_load_scene = "";
		}

		waiting_for_first_scan = false;
	}
}

void EditorNode::_vp_resized() {
}

void EditorNode::_node_renamed() {

	if (property_editor)
		property_editor->update_tree();
}

void EditorNode::_editor_select_next() {

	int editor = _get_current_main_editor();

	if (editor == editor_table.size() - 1) {
		editor = 0;
	} else {
		editor++;
	}
	_editor_select(editor);
}

void EditorNode::_editor_select_prev() {

	int editor = _get_current_main_editor();

	if (editor == 0) {
		editor = editor_table.size() - 1;
	} else {
		editor--;
	}
	_editor_select(editor);
}

Error EditorNode::load_resource(const String &p_scene) {

	RES res = ResourceLoader::load(p_scene);
	ERR_FAIL_COND_V(!res.is_valid(), ERR_CANT_OPEN);

	edit_resource(res);

	return OK;
}

void EditorNode::edit_resource(const Ref<Resource> &p_resource) {

	_resource_selected(p_resource, "");
}

void EditorNode::edit_node(Node *p_node) {

	push_item(p_node);
}

void EditorNode::open_resource(const String &p_type) {

	file->set_mode(EditorFileDialog::MODE_OPEN_FILE);

	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type(p_type, &extensions);

	file->clear_filters();
	for (int i = 0; i < extensions.size(); i++) {

		file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
	}

	file->popup_centered_ratio();
	current_option = RESOURCE_LOAD;
}

void EditorNode::save_resource_in_path(const Ref<Resource> &p_resource, const String &p_path) {

	editor_data.apply_changes_in_editors();
	int flg = 0;
	if (EditorSettings::get_singleton()->get("filesystem/on_save/compress_binary_resources"))
		flg |= ResourceSaver::FLAG_COMPRESS;

	String path = ProjectSettings::get_singleton()->localize_path(p_path);
	Error err = ResourceSaver::save(path, p_resource, flg | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS);

	if (err != OK) {
		accept->set_text(TTR("Error saving resource!"));
		accept->popup_centered_minsize();
		return;
	}

	((Resource *)p_resource.ptr())->set_path(path);
	emit_signal("resource_saved", p_resource);
}

void EditorNode::save_resource(const Ref<Resource> &p_resource) {

	if (p_resource->get_path().is_resource_file()) {
		save_resource_in_path(p_resource, p_resource->get_path());
	} else {
		save_resource_as(p_resource);
	}
}

void EditorNode::save_resource_as(const Ref<Resource> &p_resource, const String &p_at_path) {

	file->set_mode(EditorFileDialog::MODE_SAVE_FILE);

	current_option = RESOURCE_SAVE_AS;
	List<String> extensions;
	Ref<PackedScene> sd = memnew(PackedScene);
	ResourceSaver::get_recognized_extensions(p_resource, &extensions);
	file->clear_filters();

	List<String> preferred;
	for (int i = 0; i < extensions.size(); i++) {

		if (p_resource->is_class("Script") && (extensions[i] == "tres" || extensions[i] == "res" || extensions[i] == "xml")) {
			//this serves no purpose and confused people
			continue;
		}
		file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
		preferred.push_back(extensions[i]);
	}

	if (p_at_path != String()) {

		file->set_current_dir(p_at_path);
		if (p_resource->get_path().is_resource_file()) {
			file->set_current_file(p_resource->get_path().get_file());
		} else {
			if (extensions.size()) {
				file->set_current_file("new_" + p_resource->get_class().to_lower() + "." + preferred.front()->get().to_lower());
			} else {
				file->set_current_file(String());
			}
		}
	} else if (p_resource->get_path() != "") {

		file->set_current_path(p_resource->get_path());
		if (extensions.size()) {
			String ext = p_resource->get_path().get_extension().to_lower();
			if (extensions.find(ext) == NULL) {
				file->set_current_path(p_resource->get_path().replacen("." + ext, "." + extensions.front()->get()));
			}
		}
	} else if (preferred.size()) {

		String existing;
		if (extensions.size()) {
			existing = "new_" + p_resource->get_class().to_lower() + "." + preferred.front()->get().to_lower();
		}
		file->set_current_path(existing);
	}
	file->popup_centered_ratio();
	file->set_title(TTR("Save Resource As.."));
}

void EditorNode::_menu_option(int p_option) {

	_menu_option_confirm(p_option, false);
}

void EditorNode::_menu_confirm_current() {

	_menu_option_confirm(current_option, true);
}

void EditorNode::_dialog_display_save_error(String p_file, Error p_error) {

	if (p_error) {

		current_option = -1;
		accept->get_ok()->set_text(TTR("I see.."));

		switch (p_error) {

			case ERR_FILE_CANT_WRITE: {

				accept->set_text(TTR("Can't open file for writing:") + " " + p_file.get_extension());
			} break;
			case ERR_FILE_UNRECOGNIZED: {

				accept->set_text(TTR("Requested file format unknown:") + " " + p_file.get_extension());
			} break;
			default: {

				accept->set_text(TTR("Error while saving."));
			} break;
		}

		accept->popup_centered_minsize();
	}
}

void EditorNode::_dialog_display_load_error(String p_file, Error p_error) {

	if (p_error) {

		current_option = -1;
		accept->get_ok()->set_text(TTR("I see.."));

		switch (p_error) {

			case ERR_CANT_OPEN: {

				accept->set_text(vformat(TTR("Can't open '%s'."), p_file.get_file()));
			} break;
			case ERR_PARSE_ERROR: {

				accept->set_text(vformat(TTR("Error while parsing '%s'."), p_file.get_file()));
			} break;
			case ERR_FILE_CORRUPT: {

				accept->set_text(vformat(TTR("Unexpected end of file '%s'."), p_file.get_file()));
			} break;
			case ERR_FILE_NOT_FOUND: {

				accept->set_text(vformat(TTR("Missing '%s' or its dependencies."), p_file.get_file()));
			} break;
			default: {

				accept->set_text(vformat(TTR("Error while loading '%s'."), p_file.get_file()));
			} break;
		}

		accept->popup_centered_minsize();
	}
}

void EditorNode::_get_scene_metadata(const String &p_file) {

	Node *scene = editor_data.get_edited_scene_root();

	if (!scene)
		return;

	String path = EditorSettings::get_singleton()->get_project_settings_path().plus_file(p_file.get_file() + "-editstate-" + p_file.md5_text() + ".cfg");

	Ref<ConfigFile> cf;
	cf.instance();

	Error err = cf->load(path);
	if (err != OK || !cf->has_section("editor_states"))
		return; //must not exist

	List<String> esl;
	cf->get_section_keys("editor_states", &esl);

	Dictionary md;
	for (List<String>::Element *E = esl.front(); E; E = E->next()) {

		Variant st = cf->get_value("editor_states", E->get());
		if (st.get_type()) {
			md[E->get()] = st;
		}
	}

	editor_data.set_editor_states(md);
}

void EditorNode::_set_scene_metadata(const String &p_file, int p_idx) {

	Node *scene = editor_data.get_edited_scene_root(p_idx);

	if (!scene)
		return;

	scene->set_meta("__editor_run_settings__", Variant()); //clear it (no point in keeping it)
	scene->set_meta("__editor_plugin_states__", Variant());

	String path = EditorSettings::get_singleton()->get_project_settings_path().plus_file(p_file.get_file() + "-editstate-" + p_file.md5_text() + ".cfg");

	Ref<ConfigFile> cf;
	cf.instance();

	Dictionary md;

	if (p_idx < 0 || editor_data.get_edited_scene() == p_idx) {
		md = editor_data.get_editor_states();
	} else {
		md = editor_data.get_scene_editor_states(p_idx);
	}

	List<Variant> keys;
	md.get_key_list(&keys);

	for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

		cf->set_value("editor_states", E->get(), md[E->get()]);
	}

	Error err = cf->save(path);
	ERR_FAIL_COND(err != OK);
}

bool EditorNode::_find_and_save_resource(RES p_res, Map<RES, bool> &processed, int32_t flags) {

	if (p_res.is_null())
		return false;

	if (processed.has(p_res)) {

		return processed[p_res];
	}

	bool changed = p_res->is_edited();
	p_res->set_edited(false);

	bool subchanged = _find_and_save_edited_subresources(p_res.ptr(), processed, flags);

	if (p_res->get_path().is_resource_file()) {
		if (changed || subchanged) {
			//save
			print_line("Also saving modified external resource: " + p_res->get_path());
			ResourceSaver::save(p_res->get_path(), p_res, flags);
		}
		processed[p_res] = false; //because it's a file
		return false;
	} else {

		processed[p_res] = changed;
		return changed;
	}
}

bool EditorNode::_find_and_save_edited_subresources(Object *obj, Map<RES, bool> &processed, int32_t flags) {

	bool ret_changed = false;
	List<PropertyInfo> pi;
	obj->get_property_list(&pi);
	for (List<PropertyInfo>::Element *E = pi.front(); E; E = E->next()) {

		if (!(E->get().usage & PROPERTY_USAGE_STORAGE))
			continue;

		switch (E->get().type) {
			case Variant::OBJECT: {

				RES res = obj->get(E->get().name);

				if (_find_and_save_resource(res, processed, flags))
					ret_changed = true;

			} break;
			case Variant::ARRAY: {

				Array varray = obj->get(E->get().name);
				int len = varray.size();
				for (int i = 0; i < len; i++) {

					Variant v = varray.get(i);
					RES res = v;
					if (_find_and_save_resource(res, processed, flags))
						ret_changed = true;
				}

			} break;
			case Variant::DICTIONARY: {

				Dictionary d = obj->get(E->get().name);
				List<Variant> keys;
				d.get_key_list(&keys);
				for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

					Variant v = d[E->get()];
					RES res = v;
					if (_find_and_save_resource(res, processed, flags))
						ret_changed = true;
				}
			} break;
			default: {}
		}
	}

	return ret_changed;
}

void EditorNode::_save_edited_subresources(Node *scene, Map<RES, bool> &processed, int32_t flags) {

	_find_and_save_edited_subresources(scene, processed, flags);

	for (int i = 0; i < scene->get_child_count(); i++) {

		Node *n = scene->get_child(i);
		if (n->get_owner() != editor_data.get_edited_scene_root())
			continue;
		_save_edited_subresources(n, processed, flags);
	}
}

void EditorNode::_find_node_types(Node *p_node, int &count_2d, int &count_3d) {

	if (p_node->is_class("Viewport") || (p_node != editor_data.get_edited_scene_root() && p_node->get_owner() != editor_data.get_edited_scene_root()))
		return;

	if (p_node->is_class("CanvasItem"))
		count_2d++;
	else if (p_node->is_class("Spatial"))
		count_3d++;

	for (int i = 0; i < p_node->get_child_count(); i++)
		_find_node_types(p_node->get_child(i), count_2d, count_3d);
}

void EditorNode::_save_scene_with_preview(String p_file) {

	EditorProgress save("save", TTR("Saving Scene"), 4);
	save.step(TTR("Analyzing"), 0);

	int c2d = 0;
	int c3d = 0;
	_find_node_types(editor_data.get_edited_scene_root(), c2d, c3d);

	RID viewport;
	bool is2d;
	if (c3d < c2d) {
		viewport = scene_root->get_viewport_rid();
		is2d = true;
	} else {
		viewport = SpatialEditor::get_singleton()->get_editor_viewport(0)->get_viewport_node()->get_viewport_rid();
		is2d = false;
	}
	save.step(TTR("Creating Thumbnail"), 1);
	//current view?

	Ref<Image> img;
	if (is2d) {
		img = scene_root->get_texture()->get_data();
	} else {
		img = SpatialEditor::get_singleton()->get_editor_viewport(0)->get_viewport_node()->get_texture()->get_data();
	}

	if (img.is_valid()) {
		save.step(TTR("Creating Thumbnail"), 2);
		save.step(TTR("Creating Thumbnail"), 3);

		int preview_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
		preview_size *= EDSCALE;
		int width, height;
		if (img->get_width() > preview_size && img->get_width() >= img->get_height()) {

			width = preview_size;
			height = img->get_height() * preview_size / img->get_width();
		} else if (img->get_height() > preview_size && img->get_height() >= img->get_width()) {

			height = preview_size;
			width = img->get_width() * preview_size / img->get_height();
		} else {

			width = img->get_width();
			height = img->get_height();
		}

		img->convert(Image::FORMAT_RGB8);
		img->resize(width, height);
		img->flip_y();

		//save thumbnail directly, as thumbnailer may not update due to actual scene not changing md5
		String temp_path = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp");
		String cache_base = ProjectSettings::get_singleton()->globalize_path(p_file).md5_text();
		cache_base = temp_path.plus_file("resthumb-" + cache_base);

		//does not have it, try to load a cached thumbnail

		String file = cache_base + ".png";

		img->save_png(file);
	}

	save.step(TTR("Saving Scene"), 4);
	_save_scene(p_file);
	EditorResourcePreview::get_singleton()->check_for_invalidation(p_file);
}

void EditorNode::_save_scene(String p_file, int idx) {

	Node *scene = editor_data.get_edited_scene_root(idx);

	if (!scene) {

		current_option = -1;
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text(TTR("This operation can't be done without a tree root."));
		accept->popup_centered_minsize();
		return;
	}

	editor_data.apply_changes_in_editors();
	_save_default_environment();

	_set_scene_metadata(p_file, idx);

	Ref<PackedScene> sdata;

	if (ResourceCache::has(p_file)) {
		// something may be referencing this resource and we are good with that.
		// we must update it, but also let the previous scene state go, as
		// old version still work for referencing changes in instanced or inherited scenes

		sdata = Ref<PackedScene>(Object::cast_to<PackedScene>(ResourceCache::get(p_file)));
		if (sdata.is_valid())
			sdata->recreate_state();
		else
			sdata.instance();
	} else {
		sdata.instance();
	}
	Error err = sdata->pack(scene);

	if (err != OK) {

		current_option = -1;
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text(TTR("Couldn't save scene. Likely dependencies (instances) couldn't be satisfied."));
		accept->popup_centered_minsize();
		return;
	}

	// force creation of node path cache
	// (hacky but needed for the tree to update properly)
	Node *dummy_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
	memdelete(dummy_scene);

	int flg = 0;
	if (EditorSettings::get_singleton()->get("filesystem/on_save/compress_binary_resources"))
		flg |= ResourceSaver::FLAG_COMPRESS;
	flg |= ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;

	err = ResourceSaver::save(p_file, sdata, flg);
	Map<RES, bool> processed;
	_save_edited_subresources(scene, processed, flg);
	editor_data.save_editor_external_data();
	if (err == OK) {
		scene->set_filename(ProjectSettings::get_singleton()->localize_path(p_file));
		if (idx < 0 || idx == editor_data.get_edited_scene())
			set_current_version(editor_data.get_undo_redo().get_version());
		else
			editor_data.set_edited_scene_version(0, idx);
		_update_title();
		_update_scene_tabs();
	} else {

		_dialog_display_save_error(p_file, err);
	}
}

void EditorNode::_save_all_scenes() {

	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		Node *scene = editor_data.get_edited_scene_root(i);
		if (scene && scene->get_filename() != "") {
			if (i != editor_data.get_edited_scene())
				_save_scene(scene->get_filename(), i);
			else
				_save_scene_with_preview(scene->get_filename());
		} // else: ignore new scenes
	}

	_save_default_environment();
}

void EditorNode::_mark_unsaved_scenes() {

	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {

		Node *node = editor_data.get_edited_scene_root(i);
		if (!node)
			continue;

		String path = node->get_filename();
		if (!(path == String() || FileAccess::exists(path))) {

			node->set_filename("");
			if (i == editor_data.get_edited_scene())
				set_current_version(-1);
			else
				editor_data.set_edited_scene_version(-1, i);
		}
	}

	_update_title();
	_update_scene_tabs();
}

void EditorNode::_dialog_action(String p_file) {

	switch (current_option) {

		case RESOURCE_LOAD: {

			RES res = ResourceLoader::load(p_file);
			if (res.is_null()) {

				current_option = -1;
				accept->get_ok()->set_text("ok :(");
				accept->set_text(TTR("Failed to load resource."));
				return;
			};

			push_item(res.operator->());
		} break;
		case FILE_NEW_INHERITED_SCENE: {

			load_scene(p_file, false, true);
		} break;
		case FILE_OPEN_SCENE: {

			load_scene(p_file);
		} break;
		case SETTINGS_PICK_MAIN_SCENE: {

			ProjectSettings::get_singleton()->set("application/run/main_scene", p_file);
			ProjectSettings::get_singleton()->save();
			//would be nice to show the project manager opened with the highlighted field..
			_run(false, ""); // automatically run the project
		} break;
		case FILE_CLOSE:
		case FILE_CLOSE_ALL_AND_QUIT:
		case FILE_CLOSE_ALL_AND_RUN_PROJECT_MANAGER:
		case SCENE_TAB_CLOSE:
		case FILE_SAVE_SCENE:
		case FILE_SAVE_AS_SCENE: {

			int scene_idx = (current_option == FILE_SAVE_SCENE || current_option == FILE_SAVE_AS_SCENE) ? -1 : tab_closing;

			if (file->get_mode() == EditorFileDialog::MODE_SAVE_FILE) {

				_save_default_environment();
				if (scene_idx != editor_data.get_edited_scene())
					_save_scene(p_file, scene_idx);
				else
					_save_scene_with_preview(p_file);

				if (scene_idx != -1)
					_discard_changes();
			}

		} break;

		case FILE_SAVE_AND_RUN: {
			if (file->get_mode() == EditorFileDialog::MODE_SAVE_FILE) {

				_save_default_environment();
				_save_scene_with_preview(p_file);
				_run(true);
			}
		} break;

		case FILE_EXPORT_MESH_LIBRARY: {

			Ref<MeshLibrary> ml;
			if (file_export_lib_merge->is_pressed() && FileAccess::exists(p_file)) {
				ml = ResourceLoader::load(p_file, "MeshLibrary");

				if (ml.is_null()) {
					current_option = -1;
					accept->get_ok()->set_text(TTR("I see.."));
					accept->set_text(TTR("Can't load MeshLibrary for merging!"));
					accept->popup_centered_minsize();
					return;
				}
			}

			if (ml.is_null()) {
				ml = Ref<MeshLibrary>(memnew(MeshLibrary));
			}

			MeshLibraryEditor::update_library_file(editor_data.get_edited_scene_root(), ml, true);

			Error err = ResourceSaver::save(p_file, ml);
			if (err) {

				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("Error saving MeshLibrary!"));
				accept->popup_centered_minsize();
				return;
			}

		} break;
		case FILE_EXPORT_TILESET: {

			Ref<TileSet> ml;
			if (FileAccess::exists(p_file)) {
				ml = ResourceLoader::load(p_file, "TileSet");

				if (ml.is_null()) {
					if (file_export_lib_merge->is_pressed()) {
						current_option = -1;
						accept->get_ok()->set_text(TTR("I see.."));
						accept->set_text(TTR("Can't load TileSet for merging!"));
						accept->popup_centered_minsize();
						return;
					}
				} else if (!file_export_lib_merge->is_pressed()) {
					ml->clear();
				}

			} else {
				ml = Ref<TileSet>(memnew(TileSet));
			}

			TileSetEditor::update_library_file(editor_data.get_edited_scene_root(), ml, true);

			Error err = ResourceSaver::save(p_file, ml);
			if (err) {

				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("Error saving TileSet!"));
				accept->popup_centered_minsize();
				return;
			}
		} break;

		case RESOURCE_SAVE:
		case RESOURCE_SAVE_AS: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!Object::cast_to<Resource>(current_obj))

			RES current_res = RES(Object::cast_to<Resource>(current_obj));

			save_resource_in_path(current_res, p_file);

		} break;
		case SETTINGS_LAYOUT_SAVE: {

			if (p_file.empty())
				return;

			Ref<ConfigFile> config;
			config.instance();
			Error err = config->load(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts-3.cfg"));

			if (err == ERR_CANT_OPEN) {
				config.instance(); // new config
			} else if (err != OK) {
				show_warning(TTR("Error trying to save layout!"));
				return;
			}

			_save_docks_to_config(config, p_file);

			config->save(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts-3.cfg"));

			layout_dialog->hide();
			_update_layouts_menu();

			if (p_file == "Default") {
				show_warning(TTR("Default editor layout overridden."));
			}

		} break;
		case SETTINGS_LAYOUT_DELETE: {

			if (p_file.empty())
				return;

			Ref<ConfigFile> config;
			config.instance();
			Error err = config->load(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts-3.cfg"));

			if (err != OK || !config->has_section(p_file)) {
				show_warning(TTR("Layout name not found!"));
				return;
			}

			// erase
			List<String> keys;
			config->get_section_keys(p_file, &keys);
			for (List<String>::Element *E = keys.front(); E; E = E->next()) {
				config->set_value(p_file, E->get(), Variant());
			}

			config->save(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts-3.cfg"));

			layout_dialog->hide();
			_update_layouts_menu();

			if (p_file == "Default") {
				show_warning(TTR("Restored default layout to base settings."));
			}

		} break;
		default: { //save scene?

			if (file->get_mode() == EditorFileDialog::MODE_SAVE_FILE) {
				_save_scene_with_preview(p_file);
			}

		} break;
	}
}

void EditorNode::push_item(Object *p_object, const String &p_property) {

	if (!p_object) {
		property_editor->edit(NULL);
		node_dock->set_node(NULL);
		scene_tree_dock->set_selected(NULL);
		return;
	}

	uint32_t id = p_object->get_instance_id();
	if (id != editor_history.get_current()) {

		if (p_property == "")
			editor_history.add_object(id);
		else
			editor_history.add_object(id, p_property);
	}

	_edit_current();
}

void EditorNode::_select_history(int p_idx) {

	//push it to the top, it is not correct, but it's more useful
	ObjectID id = editor_history.get_history_obj(p_idx);
	Object *obj = ObjectDB::get_instance(id);
	if (!obj)
		return;
	push_item(obj);
}

void EditorNode::_prepare_history() {

	int history_to = MAX(0, editor_history.get_history_len() - 25);

	editor_history_menu->get_popup()->clear();

	Ref<Texture> base_icon = gui_base->get_icon("Object", "EditorIcons");
	Set<ObjectID> already;
	for (int i = editor_history.get_history_len() - 1; i >= history_to; i--) {

		ObjectID id = editor_history.get_history_obj(i);
		Object *obj = ObjectDB::get_instance(id);
		if (!obj || already.has(id)) {
			if (history_to > 0) {
				history_to--;
			}
			continue;
		}

		already.insert(id);

		Ref<Texture> icon = gui_base->get_icon("Object", "EditorIcons");
		if (gui_base->has_icon(obj->get_class(), "EditorIcons"))
			icon = gui_base->get_icon(obj->get_class(), "EditorIcons");
		else
			icon = base_icon;

		String text;
		if (Object::cast_to<Resource>(obj)) {
			Resource *r = Object::cast_to<Resource>(obj);
			if (r->get_path().is_resource_file())
				text = r->get_path().get_file();
			else if (r->get_name() != String()) {
				text = r->get_name();
			} else {
				text = r->get_class();
			}
		} else if (Object::cast_to<Node>(obj)) {
			text = Object::cast_to<Node>(obj)->get_name();
		} else {
			text = obj->get_class();
		}

		if (i == editor_history.get_history_pos()) {
			text = "[" + text + "]";
		}
		editor_history_menu->get_popup()->add_icon_item(icon, text, i);
	}
}

void EditorNode::_property_editor_forward() {

	if (editor_history.next())
		_edit_current();
}
void EditorNode::_property_editor_back() {

	if (editor_history.previous())
		_edit_current();
}

void EditorNode::_save_default_environment() {

	Ref<Environment> fallback = get_tree()->get_root()->get_world()->get_fallback_environment();

	if (fallback.is_valid() && fallback->get_path().is_resource_file()) {
		Map<RES, bool> processed;
		_find_and_save_edited_subresources(fallback.ptr(), processed, 0);
		save_resource_in_path(fallback, fallback->get_path());
	}
}

void EditorNode::_hide_top_editors() {

	_display_top_editors(false);

	editor_plugins_over->clear();
}

void EditorNode::_display_top_editors(bool p_display) {
	editor_plugins_over->make_visible(p_display);
}

void EditorNode::_set_top_editors(Vector<EditorPlugin *> p_editor_plugins_over) {
	editor_plugins_over->set_plugins_list(p_editor_plugins_over);
}

void EditorNode::_set_editing_top_editors(Object *p_current_object) {
	editor_plugins_over->edit(p_current_object);
}

static bool overrides_external_editor(Object *p_object) {

	Script *script = Object::cast_to<Script>(p_object);

	if (!script)
		return false;

	return script->get_language()->overrides_external_editor();
}

void EditorNode::_property_editable_warning_pressed() {

	property_editable_warning_dialog->popup_centered_minsize();
}

void EditorNode::_edit_current() {

	uint32_t current = editor_history.get_current();
	Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

	property_back->set_disabled(editor_history.is_at_beginning());
	property_forward->set_disabled(editor_history.is_at_end());

	this->current = current_obj;
	editor_path->update_path();

	String editable_warning; //none by default
	property_editable_warning->hide(); //hide by default

	if (!current_obj) {

		scene_tree_dock->set_selected(NULL);
		property_editor->edit(NULL);
		node_dock->set_node(NULL);
		object_menu->set_disabled(true);

		_display_top_editors(false);

		return;
	}

	object_menu->set_disabled(true);

	bool is_resource = current_obj->is_class("Resource");
	bool is_node = current_obj->is_class("Node");
	resource_save_button->set_disabled(!is_resource);

	if (is_resource) {

		Resource *current_res = Object::cast_to<Resource>(current_obj);
		ERR_FAIL_COND(!current_res);
		scene_tree_dock->set_selected(NULL);
		property_editor->edit(current_res);
		node_dock->set_node(NULL);
		object_menu->set_disabled(false);
		EditorNode::get_singleton()->get_import_dock()->set_edit_path(current_res->get_path());

		int subr_idx = current_res->get_path().find("::");
		if (subr_idx != -1) {
			String base_path = current_res->get_path().substr(0, subr_idx);
			if (FileAccess::exists(base_path + ".import")) {
				editable_warning = TTR("This resource belongs to a scene that was imported, so it's not editable.\nPlease read the documentation relevant to importing scenes to better understand this workflow.");
			} else {
				if (!get_edited_scene() || get_edited_scene()->get_filename() != base_path) {
					editable_warning = TTR("This resource belongs to a scene that was instanced or inherited.\nChanges to it will not be kept when saving the current scene.");
				}
			}
		} else if (current_res->get_path().is_resource_file()) {
			if (FileAccess::exists(current_res->get_path() + ".import")) {
				editable_warning = TTR("This resource was imported, so it's not editable. Change it's settings in the import panel and re-import.");
			}
		}
	} else if (is_node) {

		Node *current_node = Object::cast_to<Node>(current_obj);
		ERR_FAIL_COND(!current_node);

		property_editor->edit(current_node);
		if (current_node->is_inside_tree()) {
			node_dock->set_node(current_node);
			scene_tree_dock->set_selected(current_node);
		} else {
			node_dock->set_node(NULL);
			scene_tree_dock->set_selected(NULL);
		}
		object_menu->get_popup()->clear();

		if (get_edited_scene() && get_edited_scene()->get_filename() != String()) {
			String source_scene = get_edited_scene()->get_filename();
			if (FileAccess::exists(source_scene + ".import")) {
				editable_warning = TTR("This scene was imported, so changes to it will not be kept.\nInstancing it or inheriting will allow making changes to it.\nPlease read the documentation relevant to importing scenes to better understand this workflow.");
			}
		}

	} else {

		property_editor->edit(current_obj);
		node_dock->set_node(NULL);
	}

	if (editable_warning != String()) {
		property_editable_warning->show(); //hide by default
		property_editable_warning_dialog->set_text(editable_warning);
	}

	/* Take care of PLUGIN EDITOR */

	EditorPlugin *main_plugin = editor_data.get_editor(current_obj);

	if (main_plugin) {

		// special case if use of external editor is true
		if (main_plugin->get_name() == "Script" && (bool(EditorSettings::get_singleton()->get("text_editor/external/use_external_editor")) || overrides_external_editor(current_obj))) {
			main_plugin->edit(current_obj);
		}

		else if (main_plugin != editor_plugin_screen && (!ScriptEditor::get_singleton() || !ScriptEditor::get_singleton()->is_visible_in_tree() || ScriptEditor::get_singleton()->can_take_away_focus())) {
			// update screen main_plugin

			if (!changing_scene) {

				if (editor_plugin_screen)
					editor_plugin_screen->make_visible(false);
				editor_plugin_screen = main_plugin;
				editor_plugin_screen->edit(current_obj);

				editor_plugin_screen->make_visible(true);

				int plugin_count = editor_data.get_editor_plugin_count();
				for (int i = 0; i < plugin_count; i++) {
					editor_data.get_editor_plugin(i)->notify_main_screen_changed(editor_plugin_screen->get_name());
				}

				for (int i = 0; i < editor_table.size(); i++) {

					main_editor_buttons[i]->set_pressed(editor_table[i] == main_plugin);
				}
			}

		} else {

			editor_plugin_screen->edit(current_obj);
		}
	}

	Vector<EditorPlugin *> sub_plugins = editor_data.get_subeditors(current_obj);

	if (!sub_plugins.empty()) {
		_display_top_editors(false);

		_set_top_editors(sub_plugins);
		_set_editing_top_editors(current_obj);
		_display_top_editors(true);

	} else if (!editor_plugins_over->get_plugins_list().empty()) {

		_hide_top_editors();
	}

	object_menu->set_disabled(false);

	PopupMenu *p = object_menu->get_popup();

	p->clear();
	p->add_shortcut(ED_SHORTCUT("property_editor/copy_params", TTR("Copy Params")), OBJECT_COPY_PARAMS);
	p->add_shortcut(ED_SHORTCUT("property_editor/paste_params", TTR("Paste Params")), OBJECT_PASTE_PARAMS);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("property_editor/paste_resource", TTR("Paste Resource")), RESOURCE_PASTE);
	if (is_resource) {
		p->add_shortcut(ED_SHORTCUT("property_editor/copy_resource", TTR("Copy Resource")), RESOURCE_COPY);
		p->add_shortcut(ED_SHORTCUT("property_editor/unref_resource", TTR("Make Built-In")), RESOURCE_UNREF);
	}

	if (is_resource || is_node) {
		p->add_separator();
		p->add_shortcut(ED_SHORTCUT("property_editor/make_subresources_unique", TTR("Make Sub-Resources Unique")), OBJECT_UNIQUE_RESOURCES);
		p->add_separator();
		p->add_icon_shortcut(gui_base->get_icon("HelpSearch", "EditorIcons"), ED_SHORTCUT("property_editor/open_help", TTR("Open in Help")), OBJECT_REQUEST_HELP);
	}

	List<MethodInfo> methods;
	current_obj->get_method_list(&methods);

	if (!methods.empty()) {

		bool found = false;
		List<MethodInfo>::Element *I = methods.front();
		int i = 0;
		while (I) {

			if (I->get().flags & METHOD_FLAG_EDITOR) {
				if (!found) {
					p->add_separator();
					found = true;
				}
				p->add_item(I->get().name.capitalize(), OBJECT_METHOD_BASE + i);
			}
			i++;
			I = I->next();
		}
	}

	update_keying();
}

void EditorNode::_resource_created() {

	Object *c = create_dialog->instance_selected();

	ERR_FAIL_COND(!c);
	Resource *r = Object::cast_to<Resource>(c);
	ERR_FAIL_COND(!r);

	REF res(r);

	push_item(c);
}

void EditorNode::_resource_selected(const RES &p_res, const String &p_property) {

	if (p_res.is_null())
		return;

	RES r = p_res;
	push_item(r.operator->(), p_property);
}

void EditorNode::_run(bool p_current, const String &p_custom) {

	if (editor_run.get_status() == EditorRun::STATUS_PLAY) {

		play_button->set_pressed(!_playing_edited);
		play_scene_button->set_pressed(_playing_edited);
		return;
	}

	play_button->set_pressed(false);
	play_button->set_icon(gui_base->get_icon("MainPlay", "EditorIcons"));
	play_scene_button->set_pressed(false);
	play_scene_button->set_icon(gui_base->get_icon("PlayScene", "EditorIcons"));
	play_custom_scene_button->set_pressed(false);
	play_custom_scene_button->set_icon(gui_base->get_icon("PlayCustom", "EditorIcons"));

	String main_scene;
	String run_filename;
	String args;

	if (p_current || (editor_data.get_edited_scene_root() && p_custom == editor_data.get_edited_scene_root()->get_filename())) {

		Node *scene = editor_data.get_edited_scene_root();

		if (!scene) {
			current_option = -1;
			accept->get_ok()->set_text(TTR("I see.."));
			accept->set_text(TTR("There is no defined scene to run."));
			accept->popup_centered_minsize();
			return;
		}

		if (scene->get_filename() == "") {
			current_option = -1;
			_menu_option_confirm(FILE_SAVE_BEFORE_RUN, false);
			return;
		}

		run_filename = scene->get_filename();
	} else if (p_custom != "") {
		run_filename = p_custom;
	}

	if (run_filename == "") {

		//evidently, run the scene
		main_scene = GLOBAL_DEF("application/run/main_scene", "");
		if (main_scene == "") {

			current_option = -1;
			pick_main_scene->set_text(TTR("No main scene has ever been defined, select one?\nYou can change it later in \"Project Settings\" under the 'application' category."));
			pick_main_scene->popup_centered_minsize();
			return;
		}

		if (!FileAccess::exists(main_scene)) {

			current_option = -1;
			pick_main_scene->set_text(vformat(TTR("Selected scene '%s' does not exist, select a valid one?\nYou can change it later in \"Project Settings\" under the 'application' category."), main_scene));
			pick_main_scene->popup_centered_minsize();
			return;
		}

		if (ResourceLoader::get_resource_type(main_scene) != "PackedScene") {

			current_option = -1;
			pick_main_scene->set_text(vformat(TTR("Selected scene '%s' is not a scene file, select a valid one?\nYou can change it later in \"Project Settings\" under the 'application' category."), main_scene));
			pick_main_scene->popup_centered_minsize();
			return;
		}
	}

	if (bool(EDITOR_DEF("run/auto_save/save_before_running", true))) {

		if (unsaved_cache) {

			Node *scene = editor_data.get_edited_scene_root();

			if (scene) { //only autosave if there is a scene obviously

				if (scene->get_filename() == "") {

					current_option = -1;
					accept->get_ok()->set_text(TTR("I see.."));
					accept->set_text(TTR("Current scene was never saved, please save it prior to running."));
					accept->popup_centered_minsize();
					return;
				}

				_save_scene_with_preview(scene->get_filename());
			}
		}
		_menu_option(FILE_SAVE_ALL_SCENES);
		editor_data.save_editor_external_data();
	}

	if (!_call_build())
		return;

	if (bool(EDITOR_DEF("run/output/always_clear_output_on_play", true))) {
		log->clear();
	}

	if (bool(EDITOR_DEF("run/output/always_open_output_on_play", true))) {
		make_bottom_panel_item_visible(log);
	}

	List<String> breakpoints;
	editor_data.get_editor_breakpoints(&breakpoints);

	args = ProjectSettings::get_singleton()->get("editor/main_run_args");

	Error error = editor_run.run(run_filename, args, breakpoints);

	if (error != OK) {

		current_option = -1;
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text(TTR("Could not start subprocess!"));
		accept->popup_centered_minsize();
		return;
	}

	emit_signal("play_pressed");
	if (p_current) {
		play_scene_button->set_pressed(true);
		play_scene_button->set_icon(gui_base->get_icon("Reload", "EditorIcons"));
	} else if (p_custom != "") {
		run_custom_filename = p_custom;
		play_custom_scene_button->set_pressed(true);
		play_custom_scene_button->set_icon(gui_base->get_icon("Reload", "EditorIcons"));
	} else {
		play_button->set_pressed(true);
		play_button->set_icon(gui_base->get_icon("Reload", "EditorIcons"));
	}

	_playing_edited = p_current;
}

void EditorNode::_menu_option_confirm(int p_option, bool p_confirmed) {

	if (!p_confirmed) //this may be a hack..
		current_option = (MenuOptions)p_option;

	switch (p_option) {
		case FILE_NEW_SCENE: {

			int idx = editor_data.add_edited_scene(-1);
			_scene_tab_changed(idx);
			editor_data.clear_editor_states();

		} break;
		case FILE_NEW_INHERITED_SCENE:
		case FILE_OPEN_SCENE: {

			file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {

				file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			Node *scene = editor_data.get_edited_scene_root();
			if (scene) {
				file->set_current_path(scene->get_filename());
			};
			file->set_title(p_option == FILE_OPEN_SCENE ? TTR("Open Scene") : TTR("Open Base Scene"));
			file->popup_centered_ratio();

		} break;
		case FILE_QUICK_OPEN_SCENE: {

			quick_open->popup("PackedScene", true);
			quick_open->set_title(TTR("Quick Open Scene.."));

		} break;
		case FILE_QUICK_OPEN_SCRIPT: {

			quick_open->popup("Script", true);
			quick_open->set_title(TTR("Quick Open Script.."));

		} break;
		case FILE_OPEN_PREV: {

			if (previous_scenes.empty())
				break;
			opening_prev = true;
			open_request(previous_scenes.back()->get());

		} break;
		case FILE_CLOSE_ALL_AND_QUIT:
		case FILE_CLOSE_ALL_AND_RUN_PROJECT_MANAGER:
		case FILE_CLOSE: {

			if (!p_confirmed && (unsaved_cache || p_option == FILE_CLOSE_ALL_AND_QUIT || p_option == FILE_CLOSE_ALL_AND_RUN_PROJECT_MANAGER)) {
				tab_closing = p_option == FILE_CLOSE ? editor_data.get_edited_scene() : _next_unsaved_scene(false);
				String scene_filename = editor_data.get_edited_scene_root(tab_closing)->get_filename();
				save_confirmation->get_ok()->set_text(TTR("Save & Close"));
				save_confirmation->set_text(vformat(TTR("Save changes to '%s' before closing?"), scene_filename != "" ? scene_filename : "unsaved scene"));
				save_confirmation->popup_centered_minsize();
				break;
			}
		} // fallthrough
		case SCENE_TAB_CLOSE:
		case FILE_SAVE_SCENE: {

			int scene_idx = (p_option == FILE_SAVE_SCENE) ? -1 : tab_closing;

			Node *scene = editor_data.get_edited_scene_root(scene_idx);
			if (scene && scene->get_filename() != "") {

				if (scene_idx != editor_data.get_edited_scene())
					_save_scene(scene->get_filename(), scene_idx);
				else
					_save_scene_with_preview(scene->get_filename());

				if (scene_idx != -1)
					_discard_changes();

				break;
			}
			// fallthrough to save_as
		};
		case FILE_SAVE_AS_SCENE: {
			int scene_idx = (p_option == FILE_SAVE_SCENE || p_option == FILE_SAVE_AS_SCENE) ? -1 : tab_closing;

			Node *scene = editor_data.get_edited_scene_root(scene_idx);

			if (!scene) {

				current_option = -1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation can't be done without a tree root."));
				accept->popup_centered_minsize();
				break;
			}

			file->set_mode(EditorFileDialog::MODE_SAVE_FILE);

			List<String> extensions;
			Ref<PackedScene> sd = memnew(PackedScene);
			ResourceSaver::get_recognized_extensions(sd, &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {

				file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			//file->set_current_path(current_path);
			if (scene->get_filename() != "") {
				file->set_current_path(scene->get_filename());
				if (extensions.size()) {
					String ext = scene->get_filename().get_extension().to_lower();
					if (extensions.find(ext) == NULL) {
						file->set_current_path(scene->get_filename().replacen("." + ext, "." + extensions.front()->get()));
					}
				}
			} else {

				String existing;
				if (extensions.size()) {
					String root_name(scene->get_name());
					existing = root_name + "." + extensions.front()->get().to_lower();
				}
				file->set_current_path(existing);
			}
			file->popup_centered_ratio();
			file->set_title(TTR("Save Scene As.."));

		} break;

		case FILE_SAVE_ALL_SCENES: {

			_save_all_scenes();
		} break;
		case FILE_SAVE_BEFORE_RUN: {
			if (!p_confirmed) {
				confirmation->get_cancel()->set_text(TTR("No"));
				confirmation->get_ok()->set_text(TTR("Yes"));
				confirmation->set_text(TTR("This scene has never been saved. Save before running?"));
				confirmation->popup_centered_minsize();
				break;
			}

			_menu_option(FILE_SAVE_AS_SCENE);
			_menu_option_confirm(FILE_SAVE_AND_RUN, false);
		} break;

		case FILE_EXPORT_PROJECT: {

			project_export->popup_export();
		} break;

		case FILE_EXPORT_MESH_LIBRARY: {

			if (!editor_data.get_edited_scene_root()) {

				current_option = -1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation can't be done without a scene."));
				accept->popup_centered_minsize();
				break;
			}

			List<String> extensions;
			Ref<MeshLibrary> ml(memnew(MeshLibrary));
			ResourceSaver::get_recognized_extensions(ml, &extensions);
			file_export_lib->clear_filters();
			for (List<String>::Element *E = extensions.front(); E; E = E->next()) {
				file_export_lib->add_filter("*." + E->get());
			}

			file_export_lib->popup_centered_ratio();
			file_export_lib->set_title(TTR("Export Mesh Library"));

		} break;
		case FILE_EXPORT_TILESET: {

			List<String> extensions;
			Ref<TileSet> ml(memnew(TileSet));
			ResourceSaver::get_recognized_extensions(ml, &extensions);
			file_export_lib->clear_filters();
			for (List<String>::Element *E = extensions.front(); E; E = E->next()) {
				file_export_lib->add_filter("*." + E->get());
			}

			file_export_lib->popup_centered_ratio();
			file_export_lib->set_title(TTR("Export Tile Set"));

		} break;

		case SETTINGS_EXPORT_PREFERENCES: {

			//project_export_settings->popup_centered_ratio();
		} break;
		case FILE_IMPORT_SUBSCENE: {

			//import_subscene->popup_centered_ratio();

			if (!editor_data.get_edited_scene_root()) {

				current_option = -1;
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation can't be done without a selected node."));
				accept->popup_centered_minsize();
				break;
			}

			scene_tree_dock->import_subscene();

		} break;

		case FILE_EXTERNAL_OPEN_SCENE: {

			if (unsaved_cache && !p_confirmed) {

				confirmation->get_ok()->set_text(TTR("Open"));
				//confirmation->get_cancel()->show();
				confirmation->set_text(TTR("Current scene not saved. Open anyway?"));
				confirmation->popup_centered_minsize();
				break;
			}

			bool oprev = opening_prev;
			Error err = load_scene(external_file);
			if (err == OK && oprev) {
				previous_scenes.pop_back();
				opening_prev = false;
			}

		} break;

		case EDIT_UNDO: {

			if (Input::get_singleton()->get_mouse_button_mask() & 0x7) {
				print_line("no because state");
				break; // can't undo while mouse buttons are pressed
			}

			String action = editor_data.get_undo_redo().get_current_action_name();
			if (action != "")
				log->add_message("UNDO: " + action);

			editor_data.get_undo_redo().undo();
		} break;
		case EDIT_REDO: {

			if (Input::get_singleton()->get_mouse_button_mask() & 0x7)
				break; // can't redo while mouse buttons are pressed

			editor_data.get_undo_redo().redo();
			String action = editor_data.get_undo_redo().get_current_action_name();
			if (action != "")
				log->add_message("REDO: " + action);

		} break;
		case TOOLS_ORPHAN_RESOURCES: {

			orphan_resources->show();
		} break;

		case EDIT_REVERT: {

			Node *scene = get_edited_scene();

			if (!scene)
				break;

			String filename = scene->get_filename();

			if (filename == String()) {
				show_warning(TTR("Can't reload a scene that was never saved."));
				break;
			}

			if (unsaved_cache && !p_confirmed) {
				confirmation->get_ok()->set_text(TTR("Revert"));
				confirmation->set_text(TTR("This action cannot be undone. Revert anyway?"));
				confirmation->popup_centered_minsize();
				break;
			}

			int cur_idx = editor_data.get_edited_scene();
			_remove_edited_scene();
			Error err = load_scene(filename);
			if (err != OK)
				ERR_PRINT("Failed to load scene");
			editor_data.move_edited_scene_to_index(cur_idx);
			get_undo_redo()->clear_history();
			scene_tabs->set_current_tab(cur_idx);

		} break;
		case RESOURCE_NEW: {

			create_dialog->popup_create(true);
		} break;
		case RESOURCE_LOAD: {

			open_resource();
		} break;
		case RESOURCE_SAVE: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!Object::cast_to<Resource>(current_obj))

			RES current_res = RES(Object::cast_to<Resource>(current_obj));

			save_resource(current_res);

		} break;
		case RESOURCE_SAVE_AS: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!Object::cast_to<Resource>(current_obj))

			RES current_res = RES(Object::cast_to<Resource>(current_obj));

			save_resource_as(current_res);

		} break;
		case RESOURCE_UNREF: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!Object::cast_to<Resource>(current_obj))

			RES current_res = RES(Object::cast_to<Resource>(current_obj));
			current_res->set_path("");
			_edit_current();
		} break;
		case RESOURCE_COPY: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!Object::cast_to<Resource>(current_obj))

			RES current_res = RES(Object::cast_to<Resource>(current_obj));

			EditorSettings::get_singleton()->set_resource_clipboard(current_res);

		} break;
		case RESOURCE_PASTE: {

			RES r = EditorSettings::get_singleton()->get_resource_clipboard();
			if (r.is_valid()) {
				push_item(EditorSettings::get_singleton()->get_resource_clipboard().ptr(), String());
			}

		} break;
		case OBJECT_REQUEST_HELP: {

			if (current) {
				_editor_select(EDITOR_SCRIPT);
				emit_signal("request_help", current->get_class());
			}

		} break;
		case OBJECT_COPY_PARAMS: {

			editor_data.apply_changes_in_editors();
			if (current)
				editor_data.copy_object_params(current);
		} break;
		case OBJECT_PASTE_PARAMS: {

			editor_data.apply_changes_in_editors();
			if (current)
				editor_data.paste_object_params(current);
			editor_data.get_undo_redo().clear_history();
		} break;
		case OBJECT_UNIQUE_RESOURCES: {

			editor_data.apply_changes_in_editors();
			if (current) {
				List<PropertyInfo> props;
				current->get_property_list(&props);
				Map<RES, RES> duplicates;
				for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

					if (!(E->get().usage & PROPERTY_USAGE_STORAGE))
						continue;

					Variant v = current->get(E->get().name);
					if (v.is_ref()) {
						REF ref = v;
						if (ref.is_valid()) {

							RES res = ref;
							if (res.is_valid()) {

								if (!duplicates.has(res)) {
									duplicates[res] = res->duplicate();
								}
								res = duplicates[res];

								current->set(E->get().name, res);
							}
						}
					}
				}
			}

			editor_data.get_undo_redo().clear_history();

			_set_editing_top_editors(NULL);
			_set_editing_top_editors(current);

		} break;
		case RUN_PLAY: {
			_menu_option_confirm(RUN_STOP, true);
			_run(false);

		} break;
		case RUN_PLAY_CUSTOM_SCENE: {
			if (run_custom_filename.empty() || editor_run.get_status() == EditorRun::STATUS_STOP) {
				_menu_option_confirm(RUN_STOP, true);
				quick_run->popup("PackedScene", true);
				quick_run->set_title(TTR("Quick Run Scene.."));
				play_custom_scene_button->set_pressed(false);
			} else {
				String last_custom_scene = run_custom_filename;
				_menu_option_confirm(RUN_STOP, true);
				_run(false, last_custom_scene);
			}

		} break;
		case RUN_STOP: {

			if (editor_run.get_status() == EditorRun::STATUS_STOP)
				break;

			editor_run.stop();
			run_custom_filename.clear();
			play_button->set_pressed(false);
			play_button->set_icon(gui_base->get_icon("MainPlay", "EditorIcons"));
			play_scene_button->set_pressed(false);
			play_scene_button->set_icon(gui_base->get_icon("PlayScene", "EditorIcons"));
			play_custom_scene_button->set_pressed(false);
			play_custom_scene_button->set_icon(gui_base->get_icon("PlayCustom", "EditorIcons"));
			if (bool(EDITOR_DEF("run/output/always_close_output_on_stop", true))) {
				for (int i = 0; i < bottom_panel_items.size(); i++) {
					if (bottom_panel_items[i].control == log) {
						_bottom_panel_switch(false, i);
						break;
					}
				}
			}
			emit_signal("stop_pressed");

		} break;
		case RUN_PLAY_SCENE: {

			_save_default_environment();
			_menu_option_confirm(RUN_STOP, true);
			_run(true);

		} break;
		case RUN_PLAY_NATIVE: {

			bool autosave = EDITOR_DEF("run/auto_save/save_before_running", true);
			if (autosave) {
				_menu_option_confirm(FILE_SAVE_ALL_SCENES, false);
			}
			if (run_native->is_deploy_debug_remote_enabled()) {
				_menu_option_confirm(RUN_STOP, true);

				if (!_call_build())
					break; // build failed

				emit_signal("play_pressed");
				editor_run.run_native_notify();
			}
		} break;
		case RUN_SCENE_SETTINGS: {

			run_settings_dialog->popup_run_settings();
		} break;
		case RUN_SETTINGS: {

			project_settings->popup_project_settings();
		} break;
		case FILE_QUIT:
		case RUN_PROJECT_MANAGER: {

			if (!p_confirmed) {
				bool save_each = EDITOR_DEF("interface/save_each_scene_on_quit", true);
				if (_next_unsaved_scene(!save_each) == -1) {

					bool confirm = EDITOR_DEF("interface/quit_confirmation", true);
					if (confirm) {

						confirmation->get_ok()->set_text(p_option == FILE_QUIT ? TTR("Quit") : TTR("Yes"));
						confirmation->set_text(p_option == FILE_QUIT ? TTR("Exit the editor?") : TTR("Open Project Manager?"));
						confirmation->popup_centered_minsize();
					} else {
						_discard_changes();
						break;
					}
				} else {

					if (save_each) {

						_menu_option_confirm(p_option == FILE_QUIT ? FILE_CLOSE_ALL_AND_QUIT : FILE_CLOSE_ALL_AND_RUN_PROJECT_MANAGER, false);
					} else {

						String unsaved_scenes;
						int i = _next_unsaved_scene(true, 0);
						while (i != -1) {
							unsaved_scenes += "\n            " + editor_data.get_edited_scene_root(i)->get_filename();
							i = _next_unsaved_scene(true, ++i);
						}

						save_confirmation->get_ok()->set_text(TTR("Save & Quit"));
						save_confirmation->set_text((p_option == FILE_QUIT ? TTR("Save changes to the following scene(s) before quitting?") : TTR("Save changes the following scene(s) before opening Project Manager?")) + unsaved_scenes);
						save_confirmation->popup_centered_minsize();
					}
				}

				OS::get_singleton()->request_attention();
				break;
			}

			if (_next_unsaved_scene(true) != -1) {
				_save_all_scenes();
			}
			_discard_changes();
		} break;
		case RUN_FILE_SERVER: {

			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_FILE_SERVER));

			if (ischecked) {
				file_server->stop();
				run_native->set_deploy_dumb(false);
			} else {
				file_server->start();
				run_native->set_deploy_dumb(true);
			}

			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_FILE_SERVER), !ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_file_server", !ischecked);
		} break;
		case RUN_LIVE_DEBUG: {

			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_LIVE_DEBUG));

			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_LIVE_DEBUG), !ischecked);
			ScriptEditor::get_singleton()->get_debugger()->set_live_debugging(!ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_live_debug", !ischecked);

		} break;
		case RUN_DEPLOY_REMOTE_DEBUG: {

			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEPLOY_REMOTE_DEBUG));
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEPLOY_REMOTE_DEBUG), !ischecked);
			run_native->set_deploy_debug_remote(!ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_deploy_remote_debug", !ischecked);

		} break;
		case RUN_DEBUG_COLLISONS: {

			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_COLLISONS));
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_COLLISONS), !ischecked);
			run_native->set_debug_collisions(!ischecked);
			editor_run.set_debug_collisions(!ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_debug_collisons", !ischecked);

		} break;
		case RUN_DEBUG_NAVIGATION: {

			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_NAVIGATION));
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_DEBUG_NAVIGATION), !ischecked);
			run_native->set_debug_navigation(!ischecked);
			editor_run.set_debug_navigation(!ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_debug_navigation", !ischecked);

		} break;
		case RUN_RELOAD_SCRIPTS: {

			bool ischecked = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(RUN_RELOAD_SCRIPTS));
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(RUN_RELOAD_SCRIPTS), !ischecked);

			ScriptEditor::get_singleton()->set_live_auto_reload_running_scripts(!ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "run_reload_scripts", !ischecked);

		} break;
		case SETTINGS_UPDATE_ALWAYS: {

			update_menu->get_popup()->set_item_checked(0, true);
			update_menu->get_popup()->set_item_checked(1, false);
			OS::get_singleton()->set_low_processor_usage_mode(false);
			EditorSettings::get_singleton()->set_project_metadata("editor_options", "update_always", true);

			current_option = -1;
			accept->get_ok()->set_text(TTR("I see.."));
			accept->set_text(TTR("This option is deprecated. Situations where refresh must be forced are now considered a bug. Please report."));
			accept->popup_centered_minsize();
		} break;
		case SETTINGS_UPDATE_CHANGES: {

			update_menu->get_popup()->set_item_checked(0, false);
			update_menu->get_popup()->set_item_checked(1, true);
			OS::get_singleton()->set_low_processor_usage_mode(true);
			EditorSettings::get_singleton()->set_project_metadata("editor_options", "update_always", false);
		} break;
		case SETTINGS_UPDATE_SPINNER_HIDE: {

			update_menu->set_icon(gui_base->get_icon("Collapse", "EditorIcons"));
			update_menu->get_popup()->toggle_item_checked(3);
			bool checked = update_menu->get_popup()->is_item_checked(3);
			EditorSettings::get_singleton()->set_project_metadata("editor_options", "update_spinner_hide", checked);
		} break;
		case SETTINGS_PREFERENCES: {

			settings_config_dialog->popup_edit_settings();
		} break;
		case SETTINGS_MANAGE_EXPORT_TEMPLATES: {

			export_template_manager->popup_manager();

		} break;
		case SETTINGS_TOGGLE_FULLSCREN: {

			OS::get_singleton()->set_window_fullscreen(!OS::get_singleton()->is_window_fullscreen());

		} break;
		case SETTINGS_PICK_MAIN_SCENE: {

			file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {

				file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			Node *scene = editor_data.get_edited_scene_root();
			if (scene) {
				file->set_current_path(scene->get_filename());
			};
			file->set_title(TTR("Pick a Main Scene"));
			file->popup_centered_ratio();

		} break;
		case HELP_CLASSES: {
			emit_signal("request_help_index", "");
		} break;
		case HELP_SEARCH: {
			emit_signal("request_help_search", "");
		} break;
		case HELP_DOCS: {
			OS::get_singleton()->shell_open("http://docs.godotengine.org/");
		} break;
		case HELP_QA: {
			OS::get_singleton()->shell_open("https://godotengine.org/qa/");
		} break;
		case HELP_ISSUES: {
			OS::get_singleton()->shell_open("https://github.com/godotengine/godot/issues");
		} break;
		case HELP_COMMUNITY: {
			OS::get_singleton()->shell_open("https://godotengine.org/community");
		} break;
		case HELP_ABOUT: {
			about->popup_centered_minsize(Size2(780, 500) * EDSCALE);
		} break;

		default: {

			if (p_option >= OBJECT_METHOD_BASE) {

				ERR_FAIL_COND(!current);

				int idx = p_option - OBJECT_METHOD_BASE;

				List<MethodInfo> methods;
				current->get_method_list(&methods);

				ERR_FAIL_INDEX(idx, methods.size());
				String name = methods[idx].name;

				if (current)
					current->call(name);
			} else if (p_option >= IMPORT_PLUGIN_BASE) {
			}
		}
	}
}

int EditorNode::_next_unsaved_scene(bool p_valid_filename, int p_start) {

	for (int i = p_start; i < editor_data.get_edited_scene_count(); i++) {

		if (!editor_data.get_edited_scene_root(i))
			continue;
		int current = editor_data.get_edited_scene();
		bool unsaved = (i == current) ? saved_version != editor_data.get_undo_redo().get_version() : editor_data.get_scene_version(i) != 0;
		if (unsaved) {
			String scene_filename = editor_data.get_edited_scene_root(i)->get_filename();
			if (p_valid_filename && scene_filename.length() == 0)
				continue;
			return i;
		}
	}
	return -1;
}

void EditorNode::_discard_changes(const String &p_str) {

	switch (current_option) {

		case FILE_CLOSE_ALL_AND_QUIT:
		case FILE_CLOSE_ALL_AND_RUN_PROJECT_MANAGER:
		case FILE_CLOSE:
		case SCENE_TAB_CLOSE: {

			_remove_scene(tab_closing);
			_update_scene_tabs();

			if (current_option == FILE_CLOSE_ALL_AND_QUIT || current_option == FILE_CLOSE_ALL_AND_RUN_PROJECT_MANAGER) {
				if (_next_unsaved_scene(false) == -1) {
					current_option = current_option == FILE_CLOSE_ALL_AND_QUIT ? FILE_QUIT : RUN_PROJECT_MANAGER;
					_discard_changes();
				} else {
					_menu_option_confirm(current_option, false);
				}
			} else {
				current_option = -1;
				save_confirmation->hide();
			}
		} break;
		case FILE_QUIT: {

			_menu_option_confirm(RUN_STOP, true);
			exiting = true;
			get_tree()->quit();
		} break;
		case RUN_PROJECT_MANAGER: {

			_menu_option_confirm(RUN_STOP, true);
			exiting = true;
			get_tree()->quit();
			String exec = OS::get_singleton()->get_executable_path();

			List<String> args;
			args.push_back("--path");
			args.push_back(exec.get_base_dir());
			args.push_back("--project-manager");

			OS::ProcessID pid = 0;
			Error err = OS::get_singleton()->execute(exec, args, false, &pid);
			ERR_FAIL_COND(err);
		} break;
	}
}

void EditorNode::_update_debug_options() {

	bool check_deploy_remote = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_deploy_remote_debug", false);
	bool check_file_server = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_file_server", false);
	bool check_debug_collisons = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_collisons", false);
	bool check_debug_navigation = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_navigation", false);
	bool check_live_debug = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_live_debug", false);
	bool check_reload_scripts = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_reload_scripts", false);

	if (check_deploy_remote) _menu_option_confirm(RUN_DEPLOY_REMOTE_DEBUG, true);
	if (check_file_server) _menu_option_confirm(RUN_FILE_SERVER, true);
	if (check_debug_collisons) _menu_option_confirm(RUN_DEBUG_COLLISONS, true);
	if (check_debug_navigation) _menu_option_confirm(RUN_DEBUG_NAVIGATION, true);
	if (check_live_debug) _menu_option_confirm(RUN_LIVE_DEBUG, true);
	if (check_reload_scripts) _menu_option_confirm(RUN_RELOAD_SCRIPTS, true);
}

Control *EditorNode::get_viewport() {

	return viewport;
}

void EditorNode::_editor_select(int p_which) {

	static bool selecting = false;
	if (selecting || changing_scene)
		return;

	selecting = true;

	ERR_FAIL_INDEX(p_which, editor_table.size());

	for (int i = 0; i < main_editor_buttons.size(); i++) {
		main_editor_buttons[i]->set_pressed(i == p_which);
	}

	selecting = false;

	EditorPlugin *new_editor = editor_table[p_which];
	ERR_FAIL_COND(!new_editor);

	if (editor_plugin_screen == new_editor)
		return;

	if (editor_plugin_screen) {
		editor_plugin_screen->make_visible(false);
	}

	editor_plugin_screen = new_editor;
	editor_plugin_screen->make_visible(true);
	editor_plugin_screen->selected_notify();

	int plugin_count = editor_data.get_editor_plugin_count();
	for (int i = 0; i < plugin_count; i++) {
		editor_data.get_editor_plugin(i)->notify_main_screen_changed(editor_plugin_screen->get_name());
	}

	if (EditorSettings::get_singleton()->get("interface/separate_distraction_mode")) {
		if (p_which == EDITOR_SCRIPT) {
			set_distraction_free_mode(script_distraction);
		} else {
			set_distraction_free_mode(scene_distraction);
		}
	}
}

void EditorNode::add_editor_plugin(EditorPlugin *p_editor) {

	if (p_editor->has_main_screen()) {

		ToolButton *tb = memnew(ToolButton);
		tb->set_toggle_mode(true);
		tb->connect("pressed", singleton, "_editor_select", varray(singleton->main_editor_buttons.size()));
		tb->set_text(p_editor->get_name());
		tb->set_icon(singleton->gui_base->get_icon(p_editor->get_name(), "EditorIcons"));
		tb->set_name(p_editor->get_name());
		singleton->main_editor_buttons.push_back(tb);
		singleton->main_editor_button_vb->add_child(tb);
		singleton->editor_table.push_back(p_editor);

		singleton->distraction_free->raise();
	}
	singleton->editor_data.add_editor_plugin(p_editor);
	singleton->add_child(p_editor);
}

void EditorNode::remove_editor_plugin(EditorPlugin *p_editor) {

	if (p_editor->has_main_screen()) {

		for (int i = 0; i < singleton->main_editor_buttons.size(); i++) {

			if (p_editor->get_name() == singleton->main_editor_buttons[i]->get_text()) {

				if (singleton->main_editor_buttons[i]->is_pressed()) {
					singleton->_editor_select(EDITOR_SCRIPT);
				}

				memdelete(singleton->main_editor_buttons[i]);
				singleton->main_editor_buttons.remove(i);

				break;
			}
		}

		singleton->editor_table.erase(p_editor);
	}
	p_editor->make_visible(false);
	p_editor->clear();
	singleton->editor_plugins_over->get_plugins_list().erase(p_editor);
	singleton->remove_child(p_editor);
	singleton->editor_data.remove_editor_plugin(p_editor);
}

void EditorNode::_update_addon_config() {

	if (_initializing_addons)
		return;

	Vector<String> enabled_addons;

	for (Map<String, EditorPlugin *>::Element *E = plugin_addons.front(); E; E = E->next()) {
		enabled_addons.push_back(E->key());
	}

	if (enabled_addons.size() == 0) {
		ProjectSettings::get_singleton()->set("editor_plugins/enabled", Variant());
	} else {
		ProjectSettings::get_singleton()->set("editor_plugins/enabled", enabled_addons);
	}

	project_settings->queue_save();
}

void EditorNode::set_addon_plugin_enabled(const String &p_addon, bool p_enabled) {

	ERR_FAIL_COND(p_enabled && plugin_addons.has(p_addon));
	ERR_FAIL_COND(!p_enabled && !plugin_addons.has(p_addon));

	if (!p_enabled) {

		EditorPlugin *addon = plugin_addons[p_addon];
		remove_editor_plugin(addon);
		memdelete(addon); //bye
		plugin_addons.erase(p_addon);
		_update_addon_config();
		return;
	}

	Ref<ConfigFile> cf;
	cf.instance();
	String addon_path = "res://addons/" + p_addon + "/plugin.cfg";
	Error err = cf->load(addon_path);
	if (err != OK) {
		show_warning(TTR("Unable to enable addon plugin at: '") + addon_path + TTR("' parsing of config failed."));
		return;
	}

	if (!cf->has_section_key("plugin", "script")) {
		show_warning(TTR("Unable to find script field for addon plugin at: 'res://addons/") + p_addon + "''.");
		return;
	}

	String path = cf->get_value("plugin", "script");
	path = "res://addons/" + p_addon + "/" + path;

	Ref<Script> script = ResourceLoader::load(path);

	if (script.is_null()) {
		show_warning(TTR("Unable to load addon script from path: '") + path + "'.");
		return;
	}

	//could check inheritance..
	if (String(script->get_instance_base_type()) != "EditorPlugin") {
		show_warning(TTR("Unable to load addon script from path: '") + path + "' Base type is not EditorPlugin.");
		return;
	}

	if (!script->is_tool()) {
		show_warning(TTR("Unable to load addon script from path: '") + path + "' Script is not in tool mode.");
		return;
	}

	EditorPlugin *ep = memnew(EditorPlugin);
	ep->set_script(script.get_ref_ptr());
	plugin_addons[p_addon] = ep;
	add_editor_plugin(ep);

	_update_addon_config();
}

bool EditorNode::is_addon_plugin_enabled(const String &p_addon) const {

	return plugin_addons.has(p_addon);
}

void EditorNode::_remove_edited_scene() {
	int new_index = editor_data.get_edited_scene();
	int old_index = new_index;

	if (new_index > 0) {
		new_index = new_index - 1;
	} else if (editor_data.get_edited_scene_count() > 1) {
		new_index = 1;
	} else {
		editor_data.add_edited_scene(-1);
		new_index = 1;
	}

	if (editor_data.get_scene_path(old_index) != String()) {
		ScriptEditor::get_singleton()->close_builtin_scripts_from_scene(editor_data.get_scene_path(old_index));
	}
	_scene_tab_changed(new_index);
	editor_data.remove_scene(old_index);
	editor_data.get_undo_redo().clear_history();
	_update_title();
	_update_scene_tabs();
}

void EditorNode::_remove_scene(int index) {

	if (editor_data.get_edited_scene() == index) {
		//Scene to remove is current scene
		_remove_edited_scene();
	} else {
		// Scene to remove is not active scene
		editor_data.remove_scene(index);
	}
}

void EditorNode::set_edited_scene(Node *p_scene) {

	if (get_editor_data().get_edited_scene_root()) {
		if (get_editor_data().get_edited_scene_root()->get_parent() == scene_root)
			scene_root->remove_child(get_editor_data().get_edited_scene_root());
	}
	get_editor_data().set_edited_scene_root(p_scene);

	if (Object::cast_to<Popup>(p_scene))
		Object::cast_to<Popup>(p_scene)->show(); //show popups
	scene_tree_dock->set_edited_scene(p_scene);
	if (get_tree())
		get_tree()->set_edited_scene_root(p_scene);

	if (p_scene) {
		if (p_scene->get_parent() != scene_root)
			scene_root->add_child(p_scene);
	}
}

int EditorNode::_get_current_main_editor() {

	for (int i = 0; i < editor_table.size(); i++) {
		if (editor_table[i] == editor_plugin_screen)
			return i;
	}

	return 0;
}

Dictionary EditorNode::_get_main_scene_state() {

	Dictionary state;
	state["main_tab"] = _get_current_main_editor();
	state["scene_tree_offset"] = scene_tree_dock->get_tree_editor()->get_scene_tree()->get_vscroll_bar()->get_value();
	state["property_edit_offset"] = get_property_editor()->get_scene_tree()->get_vscroll_bar()->get_value();
	state["saved_version"] = saved_version;
	state["node_filter"] = scene_tree_dock->get_filter();
	return state;
}

void EditorNode::_set_main_scene_state(Dictionary p_state, Node *p_for_scene) {

	if (get_edited_scene() != p_for_scene && p_for_scene != NULL)
		return; //not for this scene

	changing_scene = false;

	int current = -1;
	for (int i = 0; i < editor_table.size(); i++) {
		if (editor_plugin_screen == editor_table[i]) {
			current = i;
			break;
		}
	}

	if (p_state.has("editor_index")) {
		int index = p_state["editor_index"];
		if (current < 2) { //if currently in spatial/2d, only switch to spatial/2d. if curently in script, stay there
			if (index < 2 || !get_edited_scene()) {
				_editor_select(index);
			}
		}
	}

	if (get_edited_scene()) {
		if (current < 2) {
			//use heuristic instead
			int n2d = 0, n3d = 0;
			_find_node_types(get_edited_scene(), n2d, n3d);
			if (n2d > n3d) {
				_editor_select(EDITOR_2D);
			} else if (n3d > n2d) {
				_editor_select(EDITOR_3D);
			}
		}
	}

	if (p_state.has("scene_tree_offset"))
		scene_tree_dock->get_tree_editor()->get_scene_tree()->get_vscroll_bar()->set_value(p_state["scene_tree_offset"]);
	if (p_state.has("property_edit_offset"))
		get_property_editor()->get_scene_tree()->get_vscroll_bar()->set_value(p_state["property_edit_offset"]);

	if (p_state.has("node_filter"))
		scene_tree_dock->set_filter(p_state["node_filter"]);

	//this should only happen at the very end

	ScriptEditor::get_singleton()->get_debugger()->update_live_edit_root();
	ScriptEditor::get_singleton()->set_scene_root_script(editor_data.get_scene_root_script(editor_data.get_edited_scene()));
	editor_data.notify_edited_scene_changed();
}

void EditorNode::set_current_version(uint64_t p_version) {

	saved_version = p_version;
	editor_data.set_edited_scene_version(p_version);
}

bool EditorNode::is_changing_scene() const {
	return changing_scene;
}

void EditorNode::_clear_undo_history() {

	get_undo_redo()->clear_history();
}

void EditorNode::set_current_scene(int p_idx) {

	if (editor_data.check_and_update_scene(p_idx)) {
		call_deferred("_clear_undo_history");
	}

	changing_scene = true;
	editor_data.save_edited_scene_state(editor_selection, &editor_history, _get_main_scene_state());

	if (get_editor_data().get_edited_scene_root()) {
		if (get_editor_data().get_edited_scene_root()->get_parent() == scene_root)
			scene_root->remove_child(get_editor_data().get_edited_scene_root());
	}

	editor_selection->clear();
	editor_data.set_edited_scene(p_idx);

	Node *new_scene = editor_data.get_edited_scene_root();

	if (Object::cast_to<Popup>(new_scene))
		Object::cast_to<Popup>(new_scene)->show(); //show popups

	scene_tree_dock->set_edited_scene(new_scene);
	if (get_tree())
		get_tree()->set_edited_scene_root(new_scene);

	if (new_scene) {
		if (new_scene->get_parent() != scene_root)
			scene_root->add_child(new_scene);
	}

	Dictionary state = editor_data.restore_edited_scene_state(editor_selection, &editor_history);
	_edit_current();

	call_deferred("_set_main_scene_state", state, get_edited_scene()); //do after everything else is done setting up
}

bool EditorNode::is_scene_open(const String &p_path) {

	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		if (editor_data.get_scene_path(i) == p_path)
			return true;
	}

	return false;
}

void EditorNode::fix_dependencies(const String &p_for_file) {
	dependency_fixer->edit(p_for_file);
}

Error EditorNode::load_scene(const String &p_scene, bool p_ignore_broken_deps, bool p_set_inherited, bool p_clear_errors, bool p_force_open_imported) {

	if (!is_inside_tree()) {
		defer_load_scene = p_scene;
		return OK;
	}

	if (!p_set_inherited) {

		for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {

			if (editor_data.get_scene_path(i) == p_scene) {
				_scene_tab_changed(i);
				return OK;
			}
		}

		if (!p_force_open_imported && FileAccess::exists(p_scene + ".import")) {
			open_imported->set_text(vformat(TTR("Scene '%s' was automatically imported, so it can't be modified.\nTo make changes to it, a new inherited scene can be created."), p_scene.get_file()));
			open_imported->popup_centered_minsize();
			new_inherited_button->grab_focus();
			open_import_request = p_scene;
			return OK;
		}
	}

	if (p_clear_errors)
		load_errors->clear();

	String lpath = ProjectSettings::get_singleton()->localize_path(p_scene);

	if (!lpath.begins_with("res://")) {

		current_option = -1;
		accept->get_ok()->set_text(TTR("Ugh"));
		accept->set_text(TTR("Error loading scene, it must be inside the project path. Use 'Import' to open the scene, then save it inside the project path."));
		accept->popup_centered_minsize();
		opening_prev = false;
		return ERR_FILE_NOT_FOUND;
	}

	int prev = editor_data.get_edited_scene();
	int idx = editor_data.add_edited_scene(-1);

	if (!editor_data.get_edited_scene_root() && editor_data.get_edited_scene_count() == 2) {
		_remove_edited_scene();
	} else {
		_scene_tab_changed(idx);
	}

	dependency_errors.clear();

	print_line("actually loading it");
	Error err;
	Ref<PackedScene> sdata = ResourceLoader::load(lpath, "", true, &err);
	if (!sdata.is_valid()) {

		_dialog_display_load_error(lpath, err);
		opening_prev = false;

		if (prev != -1) {
			set_current_scene(prev);
			editor_data.remove_scene(idx);
		}
		return ERR_FILE_NOT_FOUND;
	}

	if (!p_ignore_broken_deps && dependency_errors.has(lpath)) {

		current_option = -1;
		Vector<String> errors;
		for (Set<String>::Element *E = dependency_errors[lpath].front(); E; E = E->next()) {

			errors.push_back(E->get());
		}
		dependency_error->show(lpath, errors);
		opening_prev = false;

		if (prev != -1) {
			set_current_scene(prev);
			editor_data.remove_scene(idx);
		}
		return ERR_FILE_MISSING_DEPENDENCIES;
	}

	dependency_errors.erase(lpath); //at least not self path

	for (Map<String, Set<String> >::Element *E = dependency_errors.front(); E; E = E->next()) {

		String txt = vformat(TTR("Scene '%s' has broken dependencies:"), E->key()) + "\n";
		for (Set<String>::Element *F = E->get().front(); F; F = F->next()) {
			txt += "\t" + F->get() + "\n";
		}
		add_io_error(txt);
	}

	if (ResourceCache::has(lpath)) {
		//used from somewhere else? no problem! update state and replace sdata
		Ref<PackedScene> ps = Ref<PackedScene>(Object::cast_to<PackedScene>(ResourceCache::get(lpath)));
		if (ps.is_valid()) {
			ps->replace_state(sdata->get_state());
			ps->set_last_modified_time(sdata->get_last_modified_time());
			sdata = ps;
		}

	} else {
		sdata->set_path(lpath, true); //take over path
	}

	Node *new_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_MAIN);

	if (!new_scene) {

		sdata.unref();
		_dialog_display_load_error(lpath, ERR_FILE_NOT_FOUND);
		opening_prev = false;
		if (prev != -1) {
			set_current_scene(prev);
			editor_data.remove_scene(idx);
		}
		return ERR_FILE_NOT_FOUND;
	}

	if (p_set_inherited) {
		Ref<SceneState> state = sdata->get_state();
		state->set_path(lpath);
		new_scene->set_scene_inherited_state(state);
		new_scene->set_filename(String());
	}

	new_scene->set_scene_instance_state(Ref<SceneState>());

	set_edited_scene(new_scene);
	_get_scene_metadata(p_scene);

	saved_version = editor_data.get_undo_redo().get_version();
	_update_title();
	_update_scene_tabs();
	_add_to_recent_scenes(lpath);

	prev_scene->set_disabled(previous_scenes.size() == 0);
	opening_prev = false;

	ScriptEditor::get_singleton()->get_debugger()->update_live_edit_root();

	push_item(new_scene);

	return OK;
}

void EditorNode::open_request(const String &p_path) {

	load_scene(p_path); // as it will be opened in separate tab
}

void EditorNode::request_instance_scene(const String &p_path) {

	scene_tree_dock->instance(p_path);
}

void EditorNode::request_instance_scenes(const Vector<String> &p_files) {

	scene_tree_dock->instance_scenes(p_files);
}

ImportDock *EditorNode::get_import_dock() {
	return import_dock;
}

FileSystemDock *EditorNode::get_filesystem_dock() {

	return filesystem_dock;
}
SceneTreeDock *EditorNode::get_scene_tree_dock() {

	return scene_tree_dock;
}

void EditorNode::_instance_request(const Vector<String> &p_files) {

	request_instance_scenes(p_files);
}

void EditorNode::_property_keyed(const String &p_keyed, const Variant &p_value, bool p_advance) {

	AnimationPlayerEditor::singleton->get_key_editor()->insert_value_key(p_keyed, p_value, p_advance);
}

void EditorNode::_transform_keyed(Object *sp, const String &p_sub, const Transform &p_key) {

	Spatial *s = Object::cast_to<Spatial>(sp);
	if (!s)
		return;
	AnimationPlayerEditor::singleton->get_key_editor()->insert_transform_key(s, p_sub, p_key);
}

void EditorNode::update_keying() {

	bool valid = false;

	if (AnimationPlayerEditor::singleton->get_key_editor()->has_keying()) {

		if (editor_history.get_path_size() >= 1) {

			Object *obj = ObjectDB::get_instance(editor_history.get_path_object(0));
			if (Object::cast_to<Node>(obj)) {

				valid = true;
			}
		}
	}

	property_editor->set_keying(valid);

	AnimationPlayerEditor::singleton->get_key_editor()->update_keying();
}

void EditorNode::_close_messages() {

	old_split_ofs = center_split->get_split_offset();
	center_split->set_split_offset(0);
}

void EditorNode::_show_messages() {

	center_split->set_split_offset(old_split_ofs);
}

void EditorNode::_add_to_recent_scenes(const String &p_scene) {

	String base = "_" + ProjectSettings::get_singleton()->get_resource_path().replace("\\", "::").replace("/", "::");
	Vector<String> rc = EDITOR_DEF(base + "/_recent_scenes", Array());
	String name = p_scene;
	name = name.replace("res://", "");
	if (rc.find(name) != -1)
		rc.erase(name);
	rc.insert(0, name);
	if (rc.size() > 10)
		rc.resize(10);

	EditorSettings::get_singleton()->set(base + "/_recent_scenes", rc);
	EditorSettings::get_singleton()->save();
	_update_recent_scenes();
}

void EditorNode::_open_recent_scene(int p_idx) {

	String base = "_" + ProjectSettings::get_singleton()->get_resource_path().replace("\\", "::").replace("/", "::");
	Vector<String> rc = EDITOR_DEF(base + "/_recent_scenes", Array());

	ERR_FAIL_INDEX(p_idx, rc.size());

	String path = "res://" + rc[p_idx];
	load_scene(path);
}

void EditorNode::_update_recent_scenes() {

	String base = "_" + ProjectSettings::get_singleton()->get_resource_path().replace("\\", "::").replace("/", "::");
	Vector<String> rc = EDITOR_DEF(base + "/_recent_scenes", Array());
	recent_scenes->clear();
	for (int i = 0; i < rc.size(); i++) {

		recent_scenes->add_item(rc[i], i);
	}
}

void EditorNode::_quick_opened() {

	Vector<String> files = quick_open->get_selected_files();

	for (int i = 0; i < files.size(); i++) {
		String res_path = files[i];

		if (quick_open->get_base_type() == "PackedScene") {
			open_request(res_path);
		} else {
			load_resource(res_path);
		}
	}
}

void EditorNode::_quick_run() {

	_run(false, quick_run->get_selected());
}

void EditorNode::notify_child_process_exited() {

	_menu_option_confirm(RUN_STOP, false);
	stop_button->set_pressed(false);
	editor_run.stop();
}

void EditorNode::add_io_error(const String &p_error) {
	_load_error_notify(singleton, p_error);
}

void EditorNode::_load_error_notify(void *p_ud, const String &p_text) {

	EditorNode *en = (EditorNode *)p_ud;
	en->load_errors->add_image(en->gui_base->get_icon("Error", "EditorIcons"));
	en->load_errors->add_text(p_text + "\n");
	en->load_error_dialog->popup_centered_ratio(0.5);
}

bool EditorNode::_find_scene_in_use(Node *p_node, const String &p_path) const {

	if (p_node->get_filename() == p_path) {
		return true;
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {

		if (_find_scene_in_use(p_node->get_child(i), p_path)) {
			return true;
		}
	}

	return false;
}

bool EditorNode::is_scene_in_use(const String &p_path) {

	Node *es = get_edited_scene();
	if (es)
		return _find_scene_in_use(es, p_path);
	return false;
}

void EditorNode::register_editor_types() {

	ClassDB::register_class<EditorPlugin>();
	ClassDB::register_class<EditorImportPlugin>();
	ClassDB::register_class<EditorScript>();
	ClassDB::register_class<EditorSelection>();
	ClassDB::register_class<EditorFileDialog>();
	ClassDB::register_class<EditorSettings>();
	ClassDB::register_class<EditorSpatialGizmo>();
	ClassDB::register_class<EditorResourcePreview>();
	ClassDB::register_class<EditorResourcePreviewGenerator>();
	ClassDB::register_class<EditorFileSystem>();
	ClassDB::register_class<EditorFileSystemDirectory>();
	ClassDB::register_virtual_class<ScriptEditor>();
	ClassDB::register_virtual_class<EditorInterface>();
	ClassDB::register_class<EditorExportPlugin>();

	// FIXME: Is this stuff obsolete, or should it be ported to new APIs?
	//ClassDB::register_class<EditorScenePostImport>();
	//ClassDB::register_type<EditorImportExport>();
}

void EditorNode::unregister_editor_types() {

	_init_callbacks.clear();
}

void EditorNode::stop_child_process() {

	_menu_option_confirm(RUN_STOP, false);
}

void EditorNode::progress_add_task(const String &p_task, const String &p_label, int p_steps) {

	singleton->progress_dialog->add_task(p_task, p_label, p_steps);
}

void EditorNode::progress_task_step(const String &p_task, const String &p_state, int p_step, bool p_force_refresh) {

	singleton->progress_dialog->task_step(p_task, p_state, p_step, p_force_refresh);
}

void EditorNode::progress_end_task(const String &p_task) {

	singleton->progress_dialog->end_task(p_task);
}

void EditorNode::progress_add_task_bg(const String &p_task, const String &p_label, int p_steps) {

	singleton->progress_hb->add_task(p_task, p_label, p_steps);
}

void EditorNode::progress_task_step_bg(const String &p_task, int p_step) {

	singleton->progress_hb->task_step(p_task, p_step);
}

void EditorNode::progress_end_task_bg(const String &p_task) {

	singleton->progress_hb->end_task(p_task);
}

Ref<Texture> EditorNode::_file_dialog_get_icon(const String &p_path) {

	EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_filesystem_path(p_path.get_base_dir());
	if (efsd) {

		String file = p_path.get_file();
		for (int i = 0; i < efsd->get_file_count(); i++) {
			if (efsd->get_file(i) == file) {

				String type = efsd->get_file_type(i);

				if (singleton->icon_type_cache.has(type)) {
					return singleton->icon_type_cache[type];
				} else {
					return singleton->icon_type_cache["Object"];
				}
			}
		}
	}

	return singleton->icon_type_cache["Object"];
}

void EditorNode::_file_dialog_register(FileDialog *p_dialog) {

	singleton->file_dialogs.insert(p_dialog);
}

void EditorNode::_file_dialog_unregister(FileDialog *p_dialog) {

	singleton->file_dialogs.erase(p_dialog);
}

void EditorNode::_editor_file_dialog_register(EditorFileDialog *p_dialog) {

	singleton->editor_file_dialogs.insert(p_dialog);
}

void EditorNode::_editor_file_dialog_unregister(EditorFileDialog *p_dialog) {

	singleton->editor_file_dialogs.erase(p_dialog);
}

Vector<EditorNodeInitCallback> EditorNode::_init_callbacks;

Error EditorNode::export_preset(const String &p_preset, const String &p_path, bool p_debug, const String &p_password, bool p_quit_after) {

	export_defer.preset = p_preset;
	export_defer.path = p_path;
	export_defer.debug = p_debug;
	export_defer.password = p_password;

	return OK;
}

void EditorNode::show_warning(const String &p_text, const String &p_title) {

	warning->set_text(p_text);
	warning->set_title(p_title);
	warning->popup_centered_minsize();
}

void EditorNode::_dock_select_input(const Ref<InputEvent> &p_input) {

	Ref<InputEventMouse> me = p_input;

	if (me.is_valid()) {

		Vector2 point = me->get_position();

		int nrect = -1;
		for (int i = 0; i < DOCK_SLOT_MAX; i++) {
			if (dock_select_rect[i].has_point(point)) {
				nrect = i;
				break;
			}
		}

		if (nrect != dock_select_rect_over) {
			dock_select->update();
			dock_select_rect_over = nrect;
		}

		if (nrect == -1)
			return;

		Ref<InputEventMouseButton> mb = me;

		if (mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed() && dock_popup_selected != nrect) {
			Control *dock = dock_slot[dock_popup_selected]->get_current_tab_control();
			if (dock) {
				dock_slot[dock_popup_selected]->remove_child(dock);
			}
			if (dock_slot[dock_popup_selected]->get_tab_count() == 0) {
				dock_slot[dock_popup_selected]->hide();

			} else {

				dock_slot[dock_popup_selected]->set_current_tab(0);
			}

			dock_slot[nrect]->add_child(dock);
			dock_popup_selected = nrect;
			dock_slot[nrect]->set_current_tab(dock_slot[nrect]->get_tab_count() - 1);
			dock_slot[nrect]->show();
			dock_select->update();

			VSplitContainer *splits[DOCK_SLOT_MAX / 2] = {
				left_l_vsplit,
				left_r_vsplit,
				right_l_vsplit,
				right_r_vsplit,
			};

			for (int i = 0; i < 4; i++) {
				bool in_use = dock_slot[i * 2 + 0]->get_tab_count() || dock_slot[i * 2 + 1]->get_tab_count();
				if (in_use)
					splits[i]->show();
				else
					splits[i]->hide();
			}

			_edit_current();
			_save_docks();
		}
	}
}

void EditorNode::_dock_popup_exit() {

	dock_select_rect_over = -1;
	dock_select->update();
}

void EditorNode::_dock_pre_popup(int p_which) {

	dock_popup_selected = p_which;
}

void EditorNode::_dock_move_left() {

	if (dock_popup_selected < 0 || dock_popup_selected >= DOCK_SLOT_MAX)
		return;
	Control *current = dock_slot[dock_popup_selected]->get_tab_control(dock_slot[dock_popup_selected]->get_current_tab());
	Control *prev = dock_slot[dock_popup_selected]->get_tab_control(dock_slot[dock_popup_selected]->get_current_tab() - 1);
	if (!current || !prev)
		return;
	dock_slot[dock_popup_selected]->move_child(current, prev->get_index());
	dock_slot[dock_popup_selected]->set_current_tab(dock_slot[dock_popup_selected]->get_current_tab() - 1);
	dock_select->update();
	_edit_current();
	_save_docks();
}

void EditorNode::_dock_move_right() {

	Control *current = dock_slot[dock_popup_selected]->get_tab_control(dock_slot[dock_popup_selected]->get_current_tab());
	Control *next = dock_slot[dock_popup_selected]->get_tab_control(dock_slot[dock_popup_selected]->get_current_tab() + 1);
	if (!current || !next)
		return;
	dock_slot[dock_popup_selected]->move_child(next, current->get_index());
	dock_slot[dock_popup_selected]->set_current_tab(dock_slot[dock_popup_selected]->get_current_tab() + 1);
	dock_select->update();
	_edit_current();
	_save_docks();
}

void EditorNode::_dock_select_draw() {
	Size2 s = dock_select->get_size();
	s.y /= 2.0;
	s.x /= 6.0;

	Color used = Color(0.6, 0.6, 0.6, 0.8);
	Color used_selected = Color(0.8, 0.8, 0.8, 0.8);
	Color tab_selected = Color(1, 1, 1, 1);
	Color unused = used;
	unused.a = 0.4;
	Color unusable = unused;
	unusable.a = 0.1;

	Rect2 unr(s.x * 2, 0, s.x * 2, s.y * 2);
	unr.position += Vector2(2, 5);
	unr.size -= Vector2(4, 7);

	dock_select->draw_rect(unr, unusable);

	dock_tab_move_left->set_disabled(true);
	dock_tab_move_right->set_disabled(true);

	if (dock_popup_selected != -1 && dock_slot[dock_popup_selected]->get_tab_count()) {

		dock_tab_move_left->set_disabled(dock_slot[dock_popup_selected]->get_current_tab() == 0);
		dock_tab_move_right->set_disabled(dock_slot[dock_popup_selected]->get_current_tab() >= dock_slot[dock_popup_selected]->get_tab_count() - 1);
	}

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {

		Vector2 ofs;

		switch (i) {
			case DOCK_SLOT_LEFT_UL: {

			} break;
			case DOCK_SLOT_LEFT_BL: {
				ofs.y += s.y;
			} break;
			case DOCK_SLOT_LEFT_UR: {
				ofs.x += s.x;
			} break;
			case DOCK_SLOT_LEFT_BR: {
				ofs += s;
			} break;
			case DOCK_SLOT_RIGHT_UL: {
				ofs.x += s.x * 4;
			} break;
			case DOCK_SLOT_RIGHT_BL: {
				ofs.x += s.x * 4;
				ofs.y += s.y;

			} break;
			case DOCK_SLOT_RIGHT_UR: {
				ofs.x += s.x * 4;
				ofs.x += s.x;

			} break;
			case DOCK_SLOT_RIGHT_BR: {
				ofs.x += s.x * 4;
				ofs += s;

			} break;
		}

		Rect2 r(ofs, s);
		dock_select_rect[i] = r;
		r.position += Vector2(2, 5);
		r.size -= Vector2(4, 7);

		if (i == dock_select_rect_over) {
			dock_select->draw_rect(r, used_selected);
		} else if (dock_slot[i]->get_child_count() == 0) {
			dock_select->draw_rect(r, unused);
		} else {

			dock_select->draw_rect(r, used);
		}

		for (int j = 0; j < MIN(3, dock_slot[i]->get_child_count()); j++) {
			int xofs = (r.size.width / 3) * j;
			Color c = used;
			if (i == dock_popup_selected && (dock_slot[i]->get_current_tab() > 3 || dock_slot[i]->get_current_tab() == j))
				c = tab_selected;
			dock_select->draw_rect(Rect2(2 + ofs.x + xofs, ofs.y, r.size.width / 3 - 1, 3), c);
		}
	}
}

void EditorNode::_save_docks() {

	Ref<ConfigFile> config;
	config.instance();

	_save_docks_to_config(config, "docks");
	editor_data.get_plugin_window_layout(config);

	config->save(EditorSettings::get_singleton()->get_project_settings_path().plus_file("editor_layout.cfg"));
}

void EditorNode::_save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section) {

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		String names;
		for (int j = 0; j < dock_slot[i]->get_tab_count(); j++) {
			String name = dock_slot[i]->get_tab_control(j)->get_name();
			if (names != "")
				names += ",";
			names += name;
		}

		if (names != "") {
			p_layout->set_value(p_section, "dock_" + itos(i + 1), names);
		}
	}

	p_layout->set_value(p_section, "dock_filesystem_split", filesystem_dock->get_split_offset());

	VSplitContainer *splits[DOCK_SLOT_MAX / 2] = {
		left_l_vsplit,
		left_r_vsplit,
		right_l_vsplit,
		right_r_vsplit,
	};

	for (int i = 0; i < DOCK_SLOT_MAX / 2; i++) {

		if (splits[i]->is_visible_in_tree()) {
			p_layout->set_value(p_section, "dock_split_" + itos(i + 1), splits[i]->get_split_offset());
		}
	}

	HSplitContainer *h_splits[4] = {
		left_l_hsplit,
		left_r_hsplit,
		main_hsplit,
		right_hsplit,
	};

	for (int i = 0; i < 4; i++) {

		p_layout->set_value(p_section, "dock_hsplit_" + itos(i + 1), h_splits[i]->get_split_offset());
	}
}

void EditorNode::save_layout() {

	dock_drag_timer->start();
}

void EditorNode::_dock_split_dragged(int ofs) {

	dock_drag_timer->start();
}

void EditorNode::_load_docks() {

	Ref<ConfigFile> config;
	config.instance();
	Error err = config->load(EditorSettings::get_singleton()->get_project_settings_path().plus_file("editor_layout.cfg"));
	if (err != OK) {
		//no config
		if (overridden_default_layout >= 0) {
			_layout_menu_option(overridden_default_layout);
		}
		return;
	}

	_load_docks_from_config(config, "docks");
	editor_data.set_plugin_window_layout(config);
}

void EditorNode::_update_dock_slots_visibility() {

	VSplitContainer *splits[DOCK_SLOT_MAX / 2] = {
		left_l_vsplit,
		left_r_vsplit,
		right_l_vsplit,
		right_r_vsplit,
	};

	if (!docks_visible) {

		for (int i = 0; i < DOCK_SLOT_MAX; i++) {
			dock_slot[i]->hide();
		}

		for (int i = 0; i < DOCK_SLOT_MAX / 2; i++) {
			splits[i]->hide();
		}

		right_hsplit->hide();
		bottom_panel->hide();
	} else {
		for (int i = 0; i < DOCK_SLOT_MAX; i++) {

			if (dock_slot[i]->get_tab_count())
				dock_slot[i]->show();
			else
				dock_slot[i]->hide();
		}

		for (int i = 0; i < DOCK_SLOT_MAX / 2; i++) {
			bool in_use = dock_slot[i * 2 + 0]->get_tab_count() || dock_slot[i * 2 + 1]->get_tab_count();
			if (in_use)
				splits[i]->show();
			else
				splits[i]->hide();
		}

		for (int i = 0; i < DOCK_SLOT_MAX; i++) {

			if (dock_slot[i]->is_visible() && dock_slot[i]->get_tab_count()) {
				dock_slot[i]->set_current_tab(0);
			}
		}
		bottom_panel->show();
		right_hsplit->show();
	}
}

void EditorNode::_load_docks_from_config(Ref<ConfigFile> p_layout, const String &p_section) {

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {

		if (!p_layout->has_section_key(p_section, "dock_" + itos(i + 1)))
			continue;

		Vector<String> names = String(p_layout->get_value(p_section, "dock_" + itos(i + 1))).split(",");

		for (int j = 0; j < names.size(); j++) {

			String name = names[j];
			//find it, in a horribly inefficient way
			int atidx = -1;
			Control *node = NULL;
			for (int k = 0; k < DOCK_SLOT_MAX; k++) {
				if (!dock_slot[k]->has_node(name))
					continue;
				node = Object::cast_to<Control>(dock_slot[k]->get_node(name));
				if (!node)
					continue;
				atidx = k;
				break;
			}
			if (atidx == -1) //well, it's not anywhere
				continue;

			if (atidx == i) {
				node->raise();
				continue;
			}

			dock_slot[atidx]->remove_child(node);

			if (dock_slot[atidx]->get_tab_count() == 0) {
				dock_slot[atidx]->hide();
			}
			dock_slot[i]->add_child(node);
			dock_slot[i]->show();
		}
	}

	int fs_split_ofs = 0;
	if (p_layout->has_section_key(p_section, "dock_filesystem_split")) {
		fs_split_ofs = p_layout->get_value(p_section, "dock_filesystem_split");
	}
	filesystem_dock->set_split_offset(fs_split_ofs);

	VSplitContainer *splits[DOCK_SLOT_MAX / 2] = {
		left_l_vsplit,
		left_r_vsplit,
		right_l_vsplit,
		right_r_vsplit,
	};

	for (int i = 0; i < DOCK_SLOT_MAX / 2; i++) {

		if (!p_layout->has_section_key(p_section, "dock_split_" + itos(i + 1)))
			continue;

		int ofs = p_layout->get_value(p_section, "dock_split_" + itos(i + 1));
		splits[i]->set_split_offset(ofs);
	}

	HSplitContainer *h_splits[4] = {
		left_l_hsplit,
		left_r_hsplit,
		main_hsplit,
		right_hsplit,
	};

	for (int i = 0; i < 4; i++) {
		if (!p_layout->has_section_key(p_section, "dock_hsplit_" + itos(i + 1)))
			continue;
		int ofs = p_layout->get_value(p_section, "dock_hsplit_" + itos(i + 1));
		h_splits[i]->set_split_offset(ofs);
	}

	for (int i = 0; i < DOCK_SLOT_MAX / 2; i++) {
		bool in_use = dock_slot[i * 2 + 0]->get_tab_count() || dock_slot[i * 2 + 1]->get_tab_count();
		if (in_use)
			splits[i]->show();
		else
			splits[i]->hide();
	}

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {

		if (dock_slot[i]->is_visible() && dock_slot[i]->get_tab_count()) {
			dock_slot[i]->set_current_tab(0);
		}
	}
}

void EditorNode::_update_layouts_menu() {

	editor_layouts->clear();
	overridden_default_layout = -1;

	editor_layouts->set_size(Vector2());
	editor_layouts->add_shortcut(ED_SHORTCUT("layout/save", TTR("Save Layout")), SETTINGS_LAYOUT_SAVE);
	editor_layouts->add_shortcut(ED_SHORTCUT("layout/delete", TTR("Delete Layout")), SETTINGS_LAYOUT_DELETE);
	editor_layouts->add_separator();
	editor_layouts->add_shortcut(ED_SHORTCUT("layout/default", TTR("Default")), SETTINGS_LAYOUT_DEFAULT);

	Ref<ConfigFile> config;
	config.instance();
	Error err = config->load(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts-3.cfg"));
	if (err != OK) {
		return; //no config
	}

	List<String> layouts;
	config.ptr()->get_sections(&layouts);

	for (List<String>::Element *E = layouts.front(); E; E = E->next()) {

		String layout = E->get();

		if (layout == TTR("Default")) {
			editor_layouts->remove_item(editor_layouts->get_item_index(SETTINGS_LAYOUT_DEFAULT));
			overridden_default_layout = editor_layouts->get_item_count();
		}

		editor_layouts->add_item(layout);
	}
}

void EditorNode::_layout_menu_option(int p_id) {

	switch (p_id) {

		case SETTINGS_LAYOUT_SAVE: {

			current_option = p_id;
			layout_dialog->set_title(TTR("Save Layout"));
			layout_dialog->get_ok()->set_text(TTR("Save"));
			layout_dialog->popup_centered();
		} break;
		case SETTINGS_LAYOUT_DELETE: {

			current_option = p_id;
			layout_dialog->set_title(TTR("Delete Layout"));
			layout_dialog->get_ok()->set_text(TTR("Delete"));
			layout_dialog->popup_centered();
		} break;
		case SETTINGS_LAYOUT_DEFAULT: {

			_load_docks_from_config(default_layout, "docks");
			_save_docks();
		} break;
		default: {

			Ref<ConfigFile> config;
			config.instance();
			Error err = config->load(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts-3.cfg"));
			if (err != OK) {
				return; //no config
			}

			_load_docks_from_config(config, editor_layouts->get_item_text(p_id));
			_save_docks();
		}
	}
}

void EditorNode::_scene_tab_script_edited(int p_tab) {

	Ref<Script> script = editor_data.get_scene_root_script(p_tab);
	if (script.is_valid())
		edit_resource(script);
}

void EditorNode::_scene_tab_closed(int p_tab) {
	current_option = SCENE_TAB_CLOSE;
	tab_closing = p_tab;
	Node *scene = editor_data.get_edited_scene_root(p_tab);
	if (!scene) {
		_discard_changes();
		return;
	}

	bool unsaved = (p_tab == editor_data.get_edited_scene()) ?
						   saved_version != editor_data.get_undo_redo().get_version() :
						   editor_data.get_scene_version(p_tab) != 0;
	if (unsaved) {
		save_confirmation->get_ok()->set_text(TTR("Save & Close"));
		save_confirmation->set_text(vformat(TTR("Save changes to '%s' before closing?"), scene->get_filename() != "" ? scene->get_filename() : "unsaved scene"));
		save_confirmation->popup_centered_minsize();
	} else {
		_discard_changes();
	}
}

void EditorNode::_scene_tab_hover(int p_tab) {
	if (bool(EDITOR_DEF("interface/scene_tabs/show_thumbnail_on_hover", true)) == false) {
		return;
	}
	int current_tab = scene_tabs->get_current_tab();

	if (p_tab == current_tab || p_tab < 0) {
		tab_preview_panel->hide();
	} else {
		String path = editor_data.get_scene_path(p_tab);
		EditorResourcePreview::get_singleton()->queue_resource_preview(path, this, "_thumbnail_done", p_tab);
	}
}

void EditorNode::_scene_tab_exit() {
	tab_preview_panel->hide();
}

void EditorNode::_scene_tab_input(const Ref<InputEvent> &p_input) {
	Ref<InputEventMouseButton> mb = p_input;

	if (mb.is_valid()) {
		if (scene_tabs->get_hovered_tab() >= 0) {
			if (mb->get_button_index() == BUTTON_MIDDLE && mb->is_pressed()) {
				_scene_tab_closed(scene_tabs->get_hovered_tab());
			}
		} else {
			if ((mb->get_button_index() == BUTTON_LEFT && mb->is_doubleclick()) || (mb->get_button_index() == BUTTON_MIDDLE && mb->is_pressed())) {
				_menu_option_confirm(FILE_NEW_SCENE, true);
			}
		}
	}
}

void EditorNode::_reposition_active_tab(int idx_to) {
	editor_data.move_edited_scene_to_index(idx_to);
	_update_scene_tabs();
}

void EditorNode::_thumbnail_done(const String &p_path, const Ref<Texture> &p_preview, const Variant &p_udata) {
	int p_tab = p_udata.operator signed int();
	if (p_preview.is_valid()) {
		Rect2 rect = scene_tabs->get_tab_rect(p_tab);
		rect.position += scene_tabs->get_global_position();
		tab_preview->set_texture(p_preview);
		tab_preview_panel->set_position(rect.position + Vector2(0, rect.size.height));
		tab_preview_panel->show();
	}
}

void EditorNode::_scene_tab_changed(int p_tab) {
	tab_preview_panel->hide();

	bool unsaved = (saved_version != editor_data.get_undo_redo().get_version());

	if (p_tab == editor_data.get_edited_scene())
		return; //pointless

	uint64_t next_scene_version = editor_data.get_scene_version(p_tab);

	editor_data.get_undo_redo().create_action(TTR("Switch Scene Tab"));
	editor_data.get_undo_redo().add_do_method(this, "set_current_version", unsaved ? saved_version : 0);
	editor_data.get_undo_redo().add_do_method(this, "set_current_scene", p_tab);
	editor_data.get_undo_redo().add_do_method(this, "set_current_version", next_scene_version == 0 ? editor_data.get_undo_redo().get_version() + 1 : next_scene_version);

	editor_data.get_undo_redo().add_undo_method(this, "set_current_version", next_scene_version);
	editor_data.get_undo_redo().add_undo_method(this, "set_current_scene", editor_data.get_edited_scene());
	editor_data.get_undo_redo().add_undo_method(this, "set_current_version", saved_version);
	editor_data.get_undo_redo().commit_action();
}

void EditorNode::_toggle_search_bar(bool p_pressed) {

	property_editor->set_use_filter(p_pressed);

	if (p_pressed) {

		search_bar->show();
		search_box->grab_focus();
		search_box->select_all();
	} else {

		search_bar->hide();
	}
}

void EditorNode::_clear_search_box() {

	if (search_box->get_text() == "")
		return;

	search_box->clear();
	property_editor->update_tree();
}

ToolButton *EditorNode::add_bottom_panel_item(String p_text, Control *p_item) {

	ToolButton *tb = memnew(ToolButton);
	tb->connect("toggled", this, "_bottom_panel_switch", varray(bottom_panel_items.size()));
	tb->set_text(p_text);
	tb->set_toggle_mode(true);
	tb->set_focus_mode(Control::FOCUS_NONE);
	bottom_panel_vb->add_child(p_item);
	bottom_panel_hb->raise();
	bottom_panel_hb->add_child(tb);
	p_item->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	p_item->hide();
	BottomPanelItem bpi;
	bpi.button = tb;
	bpi.control = p_item;
	bpi.name = p_text;
	bottom_panel_items.push_back(bpi);

	return tb;
}

bool EditorNode::are_bottom_panels_hidden() const {

	for (int i = 0; i < bottom_panel_items.size(); i++) {
		if (bottom_panel_items[i].button->is_pressed())
			return false;
	}

	return true;
}

void EditorNode::hide_bottom_panel() {

	_bottom_panel_switch(false, 0);
}

void EditorNode::make_bottom_panel_item_visible(Control *p_item) {

	for (int i = 0; i < bottom_panel_items.size(); i++) {

		if (bottom_panel_items[i].control == p_item) {
			_bottom_panel_switch(true, i);
			break;
		}
	}
}

void EditorNode::raise_bottom_panel_item(Control *p_item) {

	for (int i = 0; i < bottom_panel_items.size(); i++) {

		if (bottom_panel_items[i].control == p_item) {
			bottom_panel_items[i].button->raise();
			SWAP(bottom_panel_items[i], bottom_panel_items[bottom_panel_items.size() - 1]);
			break;
		}
	}

	for (int i = 0; i < bottom_panel_items.size(); i++) {
		bottom_panel_items[i].button->disconnect("toggled", this, "_bottom_panel_switch");
		bottom_panel_items[i].button->connect("toggled", this, "_bottom_panel_switch", varray(i));
	}
}

void EditorNode::remove_bottom_panel_item(Control *p_item) {

	for (int i = 0; i < bottom_panel_items.size(); i++) {

		if (bottom_panel_items[i].control == p_item) {
			if (p_item->is_visible_in_tree()) {
				_bottom_panel_switch(false, 0);
			}
			bottom_panel_vb->remove_child(bottom_panel_items[i].control);
			bottom_panel_hb->remove_child(bottom_panel_items[i].button);
			memdelete(bottom_panel_items[i].button);
			bottom_panel_items.remove(i);
			break;
		}
	}

	for (int i = 0; i < bottom_panel_items.size(); i++) {
		bottom_panel_items[i].button->disconnect("toggled", this, "_bottom_panel_switch");
		bottom_panel_items[i].button->connect("toggled", this, "_bottom_panel_switch", varray(i));
	}
}

void EditorNode::_bottom_panel_switch(bool p_enable, int p_idx) {

	ERR_FAIL_INDEX(p_idx, bottom_panel_items.size());

	if (p_enable) {
		// this is the debug panel wich uses tabs, so the top section should be smaller
		if (ScriptEditor::get_singleton()->get_debugger() == bottom_panel_items[p_idx].control) {
			Ref<StyleBoxFlat> style_panel_invisible_top = gui_base->get_stylebox("debugger_panel", "EditorStyles");
			bottom_panel->add_style_override("panel", style_panel_invisible_top);
		} else {
			bottom_panel->add_style_override("panel", gui_base->get_stylebox("panel", "TabContainer"));
		}
		for (int i = 0; i < bottom_panel_items.size(); i++) {

			bottom_panel_items[i].button->set_pressed(i == p_idx);
			bottom_panel_items[i].control->set_visible(i == p_idx);
		}
		center_split->set_dragger_visibility(SplitContainer::DRAGGER_VISIBLE);
		center_split->set_collapsed(false);
	} else {
		bottom_panel->add_style_override("panel", gui_base->get_stylebox("panel", "TabContainer"));

		for (int i = 0; i < bottom_panel_items.size(); i++) {

			bottom_panel_items[i].button->set_pressed(false);
			bottom_panel_items[i].control->set_visible(false);
		}

		center_split->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN);
		center_split->set_collapsed(true);
	}
}

void EditorNode::set_docks_visible(bool p_show) {
	docks_visible = p_show;
	_update_dock_slots_visibility();
}

bool EditorNode::get_docks_visible() const {
	return docks_visible;
}

void EditorNode::_toggle_distraction_free_mode() {

	if (EditorSettings::get_singleton()->get("interface/separate_distraction_mode")) {
		int screen = -1;
		for (int i = 0; i < editor_table.size(); i++) {
			if (editor_plugin_screen == editor_table[i]) {
				screen = i;
				break;
			}
		}

		if (screen == EDITOR_SCRIPT) {
			script_distraction = !script_distraction;
			set_distraction_free_mode(script_distraction);
		} else {
			scene_distraction = !scene_distraction;
			set_distraction_free_mode(scene_distraction);
		}
	} else {
		set_distraction_free_mode(distraction_free->is_pressed());
	}
}

void EditorNode::set_distraction_free_mode(bool p_enter) {

	distraction_free->set_pressed(p_enter);

	if (p_enter) {
		if (docks_visible) {
			set_docks_visible(false);
		}
	} else {
		set_docks_visible(true);
	}
}

bool EditorNode::get_distraction_free_mode() const {
	return distraction_free->is_pressed();
}

void EditorNode::add_control_to_dock(DockSlot p_slot, Control *p_control) {
	ERR_FAIL_INDEX(p_slot, DOCK_SLOT_MAX);
	dock_slot[p_slot]->add_child(p_control);
	_update_dock_slots_visibility();
}

void EditorNode::remove_control_from_dock(Control *p_control) {

	Control *dock = NULL;
	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		if (p_control->get_parent() == dock_slot[i]) {
			dock = dock_slot[i];
			break;
		}
	}

	ERR_EXPLAIN("Control was not in dock");
	ERR_FAIL_COND(!dock);

	dock->remove_child(p_control);
	_update_dock_slots_visibility();
}

Variant EditorNode::drag_resource(const Ref<Resource> &p_res, Control *p_from) {

	Control *drag_control = memnew(Control);
	TextureRect *drag_preview = memnew(TextureRect);
	Label *label = memnew(Label);

	Ref<Texture> preview;

	{
		//todo make proper previews
		Ref<ImageTexture> pic = gui_base->get_icon("FileBigThumb", "EditorIcons");
		Ref<Image> img = pic->get_data();
		img = img->duplicate();
		img->resize(48, 48); //meh
		Ref<ImageTexture> resized_pic = Ref<ImageTexture>(memnew(ImageTexture));
		resized_pic->create_from_image(img);
		preview = resized_pic;
	}

	drag_preview->set_texture(preview);
	drag_control->add_child(drag_preview);
	if (p_res->get_path().is_resource_file()) {
		label->set_text(p_res->get_path().get_file());
	} else if (p_res->get_name() != "") {
		label->set_text(p_res->get_name());
	} else {
		label->set_text(p_res->get_class());
	}

	drag_control->add_child(label);

	p_from->set_drag_preview(drag_control); //wait until it enters scene

	label->set_position(Point2((preview->get_width() - label->get_minimum_size().width) / 2, preview->get_height()));

	Dictionary drag_data;
	drag_data["type"] = "resource";
	drag_data["resource"] = p_res;
	drag_data["from"] = p_from;

	return drag_data;
}

Variant EditorNode::drag_files(const Vector<String> &p_files, Control *p_from) {

	VBoxContainer *files = memnew(VBoxContainer);

	int max_files = 6;

	for (int i = 0; i < MIN(max_files, p_files.size()); i++) {

		Label *label = memnew(Label);
		label->set_text(p_files[i].get_file());
		files->add_child(label);
	}

	if (p_files.size() > max_files) {

		Label *label = memnew(Label);
		label->set_text(vformat(TTR("%d more file(s)"), p_files.size() - max_files));
		files->add_child(label);
	}
	Dictionary drag_data;
	drag_data["type"] = "files";
	drag_data["files"] = p_files;
	drag_data["from"] = p_from;

	p_from->set_drag_preview(files); //wait until it enters scene

	return drag_data;
}

Variant EditorNode::drag_files_and_dirs(const Vector<String> &p_files, Control *p_from) {

	VBoxContainer *files = memnew(VBoxContainer);

	int max_files = 6;

	for (int i = 0; i < MIN(max_files, p_files.size()); i++) {

		Label *label = memnew(Label);
		label->set_text(p_files[i].get_file());
		files->add_child(label);
	}

	if (p_files.size() > max_files) {

		Label *label = memnew(Label);
		label->set_text(vformat(TTR("%d more file(s) or folder(s)"), p_files.size() - max_files));
		files->add_child(label);
	}
	Dictionary drag_data;
	drag_data["type"] = "files_and_dirs";
	drag_data["files"] = p_files;
	drag_data["from"] = p_from;

	p_from->set_drag_preview(files); //wait until it enters scene

	return drag_data;
}

void EditorNode::_dropped_files(const Vector<String> &p_files, int p_screen) {

	/*
	String cur_path = filesystem_dock->get_current_path();
	for(int i=0;i<EditorImportExport::get_singleton()->get_import_plugin_count();i++) {
		EditorImportExport::get_singleton()->get_import_plugin(i)->import_from_drop(p_files,cur_path);
	}
	*/
}

void EditorNode::_file_access_close_error_notify(const String &p_str) {

	add_io_error("Unable to write to file '" + p_str + "', file in use, locked or lacking permissions.");
}

void EditorNode::reload_scene(const String &p_path) {

	//first of all, reload internal textures, materials, meshes, etc. as they might have changed on disk

	print_line("reloading: " + p_path);
	List<Ref<Resource> > cached;
	ResourceCache::get_cached_resources(&cached);
	List<Ref<Resource> > to_clear; //clear internal resources from previous scene from being used
	for (List<Ref<Resource> >::Element *E = cached.front(); E; E = E->next()) {

		if (E->get()->get_path().find("::") != -1) {
			print_line(E->get()->get_path());
		}
		if (E->get()->get_path().begins_with(p_path + "::")) { //subresources of existing scene
			to_clear.push_back(E->get());
		}
	}

	//so reload reloads everything, clear subresources of previous scene
	while (to_clear.front()) {
		print_line("bye bye: " + to_clear.front()->get()->get_path());
		to_clear.front()->get()->set_path("");
		to_clear.pop_front();
	}

	int scene_idx = -1;
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {

		if (editor_data.get_scene_path(i) == p_path) {
			scene_idx = i;
			break;
		}
	}

	int current_tab = editor_data.get_edited_scene();

	if (scene_idx == -1) {
		if (get_edited_scene()) {
			//scene is not open, so at it might be instanced, just refresh, set tab to itself and it will reload
			set_current_scene(current_tab);
			editor_data.get_undo_redo().clear_history();
		}
		return;
	}

	if (current_tab == scene_idx) {
		editor_data.apply_changes_in_editors();
		_set_scene_metadata(p_path);
	}
	//remove scene
	_remove_scene(scene_idx);
	//reload scene

	load_scene(p_path, true, false, true, true);
	//adjust index so tab is back a the previous position
	editor_data.move_edited_scene_to_index(scene_idx);
	get_undo_redo()->clear_history();
	//recover the tab
	scene_tabs->set_current_tab(current_tab);
	_scene_tab_changed(current_tab);
}

int EditorNode::plugin_init_callback_count = 0;

void EditorNode::add_plugin_init_callback(EditorPluginInitializeCallback p_callback) {

	ERR_FAIL_COND(plugin_init_callback_count == MAX_INIT_CALLBACKS);

	plugin_init_callbacks[plugin_init_callback_count++] = p_callback;
}

EditorPluginInitializeCallback EditorNode::plugin_init_callbacks[EditorNode::MAX_INIT_CALLBACKS];

int EditorNode::build_callback_count = 0;

void EditorNode::add_build_callback(EditorBuildCallback p_callback) {

	ERR_FAIL_COND(build_callback_count == MAX_INIT_CALLBACKS);

	build_callbacks[build_callback_count++] = p_callback;
}

EditorBuildCallback EditorNode::build_callbacks[EditorNode::MAX_BUILD_CALLBACKS];

bool EditorNode::_call_build() {

	for (int i = 0; i < build_callback_count; i++) {
		if (!build_callbacks[i]())
			return false;
	}

	return true;
}

void EditorNode::_inherit_imported(const String &p_action) {

	open_imported->hide();
	load_scene(open_import_request, true, true);
}

void EditorNode::_open_imported() {

	load_scene(open_import_request, true, false, true, true);
}

void EditorNode::dim_editor(bool p_dimming) {
	static int dim_count = 0;
	bool dim_ui = EditorSettings::get_singleton()->get("interface/dim_editor_on_dialog_popup");
	if (p_dimming) {
		if (dim_ui) {
			if (dim_count == 0) {
				_start_dimming(true);
			}
			dim_count++;
		}
	} else {
		if (dim_count == 1) {
			_start_dimming(false);
			dim_count = 0;
		} else if (dim_ui && dim_count > 0) {
			dim_count--;
		}
	}
}

void EditorNode::_start_dimming(bool p_dimming) {
	_dimming = p_dimming;
	_dim_time = 0.0f;
	_dim_timer->start();
}

void EditorNode::_dim_timeout() {

	_dim_time += _dim_timer->get_wait_time();
	float wait_time = EditorSettings::get_singleton()->get("interface/dim_transition_time");

	float c = 1.0f - (float)EditorSettings::get_singleton()->get("interface/dim_amount");

	Color base = _dimming ? Color(1, 1, 1) : Color(c, c, c);
	Color final = _dimming ? Color(c, c, c) : Color(1, 1, 1);

	if (_dim_time + _dim_timer->get_wait_time() >= wait_time) {
		gui_base->set_modulate(final);
		_dim_timer->stop();
	} else {
		gui_base->set_modulate(base.linear_interpolate(final, _dim_time / wait_time));
	}
}

void EditorNode::_check_gui_base_size() {
	if (gui_base->get_size().width > 1200 * EDSCALE) {
		for (int i = 0; i < singleton->main_editor_button_vb->get_child_count(); i++) {
			ToolButton *btn = Object::cast_to<ToolButton>(singleton->main_editor_button_vb->get_child(i));
			if (btn == singleton->distraction_free) continue;
			btn->set_text(btn->get_name());
		}
	} else {
		for (int i = 0; i < singleton->main_editor_button_vb->get_child_count(); i++) {
			ToolButton *btn = Object::cast_to<ToolButton>(singleton->main_editor_button_vb->get_child(i));
			if (btn == singleton->distraction_free) continue;
			btn->set_text("");
		}
	}
}

void EditorNode::open_export_template_manager() {

	export_template_manager->popup_manager();
}

void EditorNode::_bind_methods() {

	ClassDB::bind_method("_menu_option", &EditorNode::_menu_option);
	ClassDB::bind_method("_menu_confirm_current", &EditorNode::_menu_confirm_current);
	ClassDB::bind_method("_dialog_action", &EditorNode::_dialog_action);
	ClassDB::bind_method("_resource_selected", &EditorNode::_resource_selected, DEFVAL(""));
	ClassDB::bind_method("_property_editor_forward", &EditorNode::_property_editor_forward);
	ClassDB::bind_method("_property_editor_back", &EditorNode::_property_editor_back);
	ClassDB::bind_method("_editor_select", &EditorNode::_editor_select);
	ClassDB::bind_method("_node_renamed", &EditorNode::_node_renamed);
	ClassDB::bind_method("edit_node", &EditorNode::edit_node);
	ClassDB::bind_method("_unhandled_input", &EditorNode::_unhandled_input);

	ClassDB::bind_method("_get_scene_metadata", &EditorNode::_get_scene_metadata);
	ClassDB::bind_method("set_edited_scene", &EditorNode::set_edited_scene);
	ClassDB::bind_method("open_request", &EditorNode::open_request);
	ClassDB::bind_method("_instance_request", &EditorNode::_instance_request);
	ClassDB::bind_method("update_keying", &EditorNode::update_keying);
	ClassDB::bind_method("_property_keyed", &EditorNode::_property_keyed);
	ClassDB::bind_method("_transform_keyed", &EditorNode::_transform_keyed);
	ClassDB::bind_method("_close_messages", &EditorNode::_close_messages);
	ClassDB::bind_method("_show_messages", &EditorNode::_show_messages);
	ClassDB::bind_method("_vp_resized", &EditorNode::_vp_resized);
	ClassDB::bind_method("_quick_opened", &EditorNode::_quick_opened);
	ClassDB::bind_method("_quick_run", &EditorNode::_quick_run);

	ClassDB::bind_method("_resource_created", &EditorNode::_resource_created);
	ClassDB::bind_method("_open_recent_scene", &EditorNode::_open_recent_scene);

	ClassDB::bind_method("stop_child_process", &EditorNode::stop_child_process);

	ClassDB::bind_method("_sources_changed", &EditorNode::_sources_changed);
	ClassDB::bind_method("_fs_changed", &EditorNode::_fs_changed);
	ClassDB::bind_method("_dock_select_draw", &EditorNode::_dock_select_draw);
	ClassDB::bind_method("_dock_select_input", &EditorNode::_dock_select_input);
	ClassDB::bind_method("_dock_pre_popup", &EditorNode::_dock_pre_popup);
	ClassDB::bind_method("_dock_split_dragged", &EditorNode::_dock_split_dragged);
	ClassDB::bind_method("_save_docks", &EditorNode::_save_docks);
	ClassDB::bind_method("_dock_popup_exit", &EditorNode::_dock_popup_exit);
	ClassDB::bind_method("_dock_move_left", &EditorNode::_dock_move_left);
	ClassDB::bind_method("_dock_move_right", &EditorNode::_dock_move_right);

	ClassDB::bind_method("_layout_menu_option", &EditorNode::_layout_menu_option);

	ClassDB::bind_method("set_current_scene", &EditorNode::set_current_scene);
	ClassDB::bind_method("set_current_version", &EditorNode::set_current_version);
	ClassDB::bind_method("_scene_tab_changed", &EditorNode::_scene_tab_changed);
	ClassDB::bind_method("_scene_tab_closed", &EditorNode::_scene_tab_closed);
	ClassDB::bind_method("_scene_tab_hover", &EditorNode::_scene_tab_hover);
	ClassDB::bind_method("_scene_tab_exit", &EditorNode::_scene_tab_exit);
	ClassDB::bind_method("_scene_tab_input", &EditorNode::_scene_tab_input);
	ClassDB::bind_method("_reposition_active_tab", &EditorNode::_reposition_active_tab);
	ClassDB::bind_method("_thumbnail_done", &EditorNode::_thumbnail_done);
	ClassDB::bind_method("_scene_tab_script_edited", &EditorNode::_scene_tab_script_edited);
	ClassDB::bind_method("_set_main_scene_state", &EditorNode::_set_main_scene_state);
	ClassDB::bind_method("_update_scene_tabs", &EditorNode::_update_scene_tabs);
	ClassDB::bind_method("_discard_changes", &EditorNode::_discard_changes);

	ClassDB::bind_method("_prepare_history", &EditorNode::_prepare_history);
	ClassDB::bind_method("_select_history", &EditorNode::_select_history);

	ClassDB::bind_method("_toggle_search_bar", &EditorNode::_toggle_search_bar);
	ClassDB::bind_method("_clear_search_box", &EditorNode::_clear_search_box);
	ClassDB::bind_method("_clear_undo_history", &EditorNode::_clear_undo_history);
	ClassDB::bind_method("_dropped_files", &EditorNode::_dropped_files);
	ClassDB::bind_method("_toggle_distraction_free_mode", &EditorNode::_toggle_distraction_free_mode);
	ClassDB::bind_method("_property_editable_warning_pressed", &EditorNode::_property_editable_warning_pressed);

	ClassDB::bind_method(D_METHOD("get_gui_base"), &EditorNode::get_gui_base);
	ClassDB::bind_method(D_METHOD("_bottom_panel_switch"), &EditorNode::_bottom_panel_switch);

	ClassDB::bind_method(D_METHOD("_open_imported"), &EditorNode::_open_imported);
	ClassDB::bind_method(D_METHOD("_inherit_imported"), &EditorNode::_inherit_imported);
	ClassDB::bind_method(D_METHOD("_dim_timeout"), &EditorNode::_dim_timeout);
	ClassDB::bind_method(D_METHOD("_check_gui_base_size"), &EditorNode::_check_gui_base_size);

	ClassDB::bind_method(D_METHOD("_resources_reimported"), &EditorNode::_resources_reimported);

	ADD_SIGNAL(MethodInfo("play_pressed"));
	ADD_SIGNAL(MethodInfo("pause_pressed"));
	ADD_SIGNAL(MethodInfo("stop_pressed"));
	ADD_SIGNAL(MethodInfo("request_help"));
	ADD_SIGNAL(MethodInfo("request_help_search"));
	ADD_SIGNAL(MethodInfo("request_help_index"));
	ADD_SIGNAL(MethodInfo("script_add_function_request", PropertyInfo(Variant::OBJECT, "obj"), PropertyInfo(Variant::STRING, "function"), PropertyInfo(Variant::POOL_STRING_ARRAY, "args")));
	ADD_SIGNAL(MethodInfo("resource_saved", PropertyInfo(Variant::OBJECT, "obj")));
}

static Node *_resource_get_edited_scene() {

	return EditorNode::get_singleton()->get_edited_scene();
}

EditorNode::EditorNode() {

	Resource::_get_local_scene_func = _resource_get_edited_scene;

	VisualServer::get_singleton()->textures_keep_original(true);
	VisualServer::get_singleton()->set_debug_generate_wireframes(true);

	EditorHelp::generate_doc(); //before any editor classes are crated
	SceneState::set_disable_placeholders(true);
	ResourceLoader::clear_translation_remaps(); //no remaps using during editor
	editor_initialize_certificates(); //for asset sharing

	InputDefault *id = Object::cast_to<InputDefault>(Input::get_singleton());

	if (id) {

		if (!OS::get_singleton()->has_touchscreen_ui_hint() && Input::get_singleton()) {
			//only if no touchscreen ui hint, set emulation
			id->set_emulate_touch(false); //just disable just in case
		}
		id->set_custom_mouse_cursor(RES());
	}

	singleton = this;
	exiting = false;
	last_checked_version = 0;
	changing_scene = false;
	_initializing_addons = false;
	docks_visible = true;

	scene_distraction = false;
	script_distraction = false;

	FileAccess::set_backup_save(true);

	TranslationServer::get_singleton()->set_enabled(false);
	// load settings
	if (!EditorSettings::get_singleton())
		EditorSettings::create();

	{
		int dpi_mode = EditorSettings::get_singleton()->get("interface/hidpi_mode");
		if (dpi_mode == 0) {
			editor_set_scale(OS::get_singleton()->get_screen_dpi(0) >= 192 && OS::get_singleton()->get_screen_size(OS::get_singleton()->get_current_screen()).x > 2000 ? 2.0 : 1.0);
		} else if (dpi_mode == 1) {
			editor_set_scale(0.75);
		} else if (dpi_mode == 2) {
			editor_set_scale(1.0);
		} else if (dpi_mode == 3) {
			editor_set_scale(1.5);
		} else if (dpi_mode == 4) {
			editor_set_scale(2.0);
		}
	}

	ResourceLoader::set_abort_on_missing_resources(false);
	FileDialog::set_default_show_hidden_files(EditorSettings::get_singleton()->get("filesystem/file_dialog/show_hidden_files"));
	EditorFileDialog::set_default_show_hidden_files(EditorSettings::get_singleton()->get("filesystem/file_dialog/show_hidden_files"));
	EditorFileDialog::set_default_display_mode((EditorFileDialog::DisplayMode)EditorSettings::get_singleton()->get("filesystem/file_dialog/display_mode").operator int());
	ResourceLoader::set_error_notify_func(this, _load_error_notify);
	ResourceLoader::set_dependency_error_notify_func(this, _dependency_error_report);

	ResourceLoader::set_timestamp_on_load(true);
	ResourceSaver::set_timestamp_on_save(true);

	{ //register importers at the beginning, so dialogs are created with the right extensions
		Ref<ResourceImporterTexture> import_texture;
		import_texture.instance();
		ResourceFormatImporter::get_singleton()->add_importer(import_texture);

		Ref<ResourceImporterCSVTranslation> import_csv_translation;
		import_csv_translation.instance();
		ResourceFormatImporter::get_singleton()->add_importer(import_csv_translation);

		Ref<ResourceImporterWAV> import_wav;
		import_wav.instance();
		ResourceFormatImporter::get_singleton()->add_importer(import_wav);

		Ref<ResourceImporterScene> import_scene;
		import_scene.instance();
		ResourceFormatImporter::get_singleton()->add_importer(import_scene);

		{
			Ref<EditorSceneImporterCollada> import_collada;
			import_collada.instance();
			import_scene->add_importer(import_collada);

			Ref<EditorOBJImporter> import_obj;
			import_obj.instance();
			import_scene->add_importer(import_obj);

			Ref<EditorSceneImporterGLTF> import_gltf;
			import_gltf.instance();
			import_scene->add_importer(import_gltf);
		}
	}

	_pvrtc_register_compressors();

	editor_selection = memnew(EditorSelection);

	EditorFileSystem *efs = memnew(EditorFileSystem);
	add_child(efs);

	//used for previews
	FileDialog::get_icon_func = _file_dialog_get_icon;
	FileDialog::register_func = _file_dialog_register;
	FileDialog::unregister_func = _file_dialog_unregister;

	EditorFileDialog::get_icon_func = _file_dialog_get_icon;
	EditorFileDialog::register_func = _editor_file_dialog_register;
	EditorFileDialog::unregister_func = _editor_file_dialog_unregister;

	editor_export = memnew(EditorExport);
	add_child(editor_export);

	register_exporters();

	GLOBAL_DEF("editor/main_run_args", "");

	ClassDB::set_class_enabled("CollisionShape", true);
	ClassDB::set_class_enabled("CollisionShape2D", true);
	ClassDB::set_class_enabled("CollisionPolygon2D", true);

	theme_base = memnew(Control);
	add_child(theme_base);
	theme_base->set_area_as_parent_rect();

	gui_base = memnew(Panel);
	theme_base->add_child(gui_base);
	gui_base->set_area_as_parent_rect();
	gui_base->connect("item_rect_changed", this, "_check_gui_base_size");

	Ref<Theme> theme = create_editor_theme();
	theme_base->set_theme(theme);
	gui_base->set_theme(create_custom_theme());
	gui_base->add_style_override("panel", gui_base->get_stylebox("Background", "EditorStyles"));

	resource_preview = memnew(EditorResourcePreview);
	add_child(resource_preview);
	progress_dialog = memnew(ProgressDialog);
	gui_base->add_child(progress_dialog);

	// take up all screen
	gui_base->set_anchor(MARGIN_RIGHT, Control::ANCHOR_END);
	gui_base->set_anchor(MARGIN_BOTTOM, Control::ANCHOR_END);
	gui_base->set_end(Point2(0, 0));

	main_vbox = memnew(VBoxContainer);
	gui_base->add_child(main_vbox);
	main_vbox->set_area_as_parent_rect(8);
	main_vbox->set_margin(MARGIN_TOP, 5 * EDSCALE);

	menu_hb = memnew(HBoxContainer);
	main_vbox->add_child(menu_hb);

	left_l_hsplit = memnew(HSplitContainer);
	main_vbox->add_child(left_l_hsplit);

	left_l_hsplit->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	left_l_vsplit = memnew(VSplitContainer);
	left_l_hsplit->add_child(left_l_vsplit);
	dock_slot[DOCK_SLOT_LEFT_UL] = memnew(TabContainer);
	left_l_vsplit->add_child(dock_slot[DOCK_SLOT_LEFT_UL]);
	dock_slot[DOCK_SLOT_LEFT_BL] = memnew(TabContainer);
	left_l_vsplit->add_child(dock_slot[DOCK_SLOT_LEFT_BL]);
	left_l_vsplit->hide();
	dock_slot[DOCK_SLOT_LEFT_UL]->hide();
	dock_slot[DOCK_SLOT_LEFT_BL]->hide();

	left_r_hsplit = memnew(HSplitContainer);
	left_l_hsplit->add_child(left_r_hsplit);
	left_r_vsplit = memnew(VSplitContainer);
	left_r_hsplit->add_child(left_r_vsplit);
	dock_slot[DOCK_SLOT_LEFT_UR] = memnew(TabContainer);
	left_r_vsplit->add_child(dock_slot[DOCK_SLOT_LEFT_UR]);
	dock_slot[DOCK_SLOT_LEFT_BR] = memnew(TabContainer);
	left_r_vsplit->add_child(dock_slot[DOCK_SLOT_LEFT_BR]);

	main_hsplit = memnew(HSplitContainer);
	left_r_hsplit->add_child(main_hsplit);
	VBoxContainer *center_vb = memnew(VBoxContainer);
	main_hsplit->add_child(center_vb);
	center_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	center_split = memnew(VSplitContainer);
	center_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	center_split->set_collapsed(false);
	center_vb->add_child(center_split);

	right_hsplit = memnew(HSplitContainer);
	main_hsplit->add_child(right_hsplit);

	right_l_vsplit = memnew(VSplitContainer);
	right_hsplit->add_child(right_l_vsplit);
	dock_slot[DOCK_SLOT_RIGHT_UL] = memnew(TabContainer);
	right_l_vsplit->add_child(dock_slot[DOCK_SLOT_RIGHT_UL]);
	dock_slot[DOCK_SLOT_RIGHT_BL] = memnew(TabContainer);
	right_l_vsplit->add_child(dock_slot[DOCK_SLOT_RIGHT_BL]);

	right_r_vsplit = memnew(VSplitContainer);
	right_hsplit->add_child(right_r_vsplit);
	dock_slot[DOCK_SLOT_RIGHT_UR] = memnew(TabContainer);
	right_r_vsplit->add_child(dock_slot[DOCK_SLOT_RIGHT_UR]);
	dock_slot[DOCK_SLOT_RIGHT_BR] = memnew(TabContainer);
	right_r_vsplit->add_child(dock_slot[DOCK_SLOT_RIGHT_BR]);
	right_r_vsplit->hide();
	dock_slot[DOCK_SLOT_RIGHT_UR]->hide();
	dock_slot[DOCK_SLOT_RIGHT_BR]->hide();

	left_l_vsplit->connect("dragged", this, "_dock_split_dragged");
	left_r_vsplit->connect("dragged", this, "_dock_split_dragged");
	right_l_vsplit->connect("dragged", this, "_dock_split_dragged");
	right_r_vsplit->connect("dragged", this, "_dock_split_dragged");

	left_l_hsplit->connect("dragged", this, "_dock_split_dragged");
	left_r_hsplit->connect("dragged", this, "_dock_split_dragged");
	main_hsplit->connect("dragged", this, "_dock_split_dragged");
	right_hsplit->connect("dragged", this, "_dock_split_dragged");

	dock_select_popup = memnew(PopupPanel);
	gui_base->add_child(dock_select_popup);
	VBoxContainer *dock_vb = memnew(VBoxContainer);
	dock_select_popup->add_child(dock_vb);

	HBoxContainer *dock_hb = memnew(HBoxContainer);
	dock_tab_move_left = memnew(ToolButton);
	dock_tab_move_left->set_icon(theme->get_icon("Back", "EditorIcons"));
	dock_tab_move_left->set_focus_mode(Control::FOCUS_NONE);
	dock_tab_move_left->connect("pressed", this, "_dock_move_left");
	dock_hb->add_child(dock_tab_move_left);
	dock_hb->add_spacer();
	dock_tab_move_right = memnew(ToolButton);
	dock_tab_move_right->set_icon(theme->get_icon("Forward", "EditorIcons"));
	dock_tab_move_right->set_focus_mode(Control::FOCUS_NONE);
	dock_tab_move_right->connect("pressed", this, "_dock_move_right");

	dock_hb->add_child(dock_tab_move_right);
	dock_vb->add_child(dock_hb);

	dock_select = memnew(Control);
	dock_select->set_custom_minimum_size(Size2(128, 64) * EDSCALE);
	dock_select->connect("gui_input", this, "_dock_select_input");
	dock_select->connect("draw", this, "_dock_select_draw");
	dock_select->connect("mouse_exited", this, "_dock_popup_exit");
	dock_select->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	dock_vb->add_child(dock_select);

	dock_select_popup->set_as_minsize();
	dock_select_rect_over = -1;
	dock_popup_selected = -1;
	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		dock_slot[i]->set_custom_minimum_size(Size2(230, 220) * EDSCALE);
		dock_slot[i]->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		dock_slot[i]->set_popup(dock_select_popup);
		dock_slot[i]->connect("pre_popup_pressed", this, "_dock_pre_popup", varray(i));
		dock_slot[i]->set_tab_align(TabContainer::ALIGN_LEFT);
	}

	dock_drag_timer = memnew(Timer);
	add_child(dock_drag_timer);
	dock_drag_timer->set_wait_time(0.5);
	dock_drag_timer->set_one_shot(true);
	dock_drag_timer->connect("timeout", this, "_save_docks");

	top_split = memnew(VSplitContainer);
	center_split->add_child(top_split);
	top_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	top_split->set_collapsed(true);

	VBoxContainer *srt = memnew(VBoxContainer);
	srt->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	top_split->add_child(srt);
	srt->add_constant_override("separation", 0);

	tab_preview_panel = memnew(Panel);
	tab_preview_panel->set_size(Size2(100, 100) * EDSCALE);
	tab_preview_panel->hide();
	tab_preview_panel->set_self_modulate(Color(1, 1, 1, 0.7));
	gui_base->add_child(tab_preview_panel);

	tab_preview = memnew(TextureRect);
	tab_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	tab_preview->set_size(Size2(96, 96) * EDSCALE);
	tab_preview->set_position(Point2(2, 2) * EDSCALE);
	tab_preview_panel->add_child(tab_preview);

	scene_tabs = memnew(Tabs);
	scene_tabs->add_style_override("tab_fg", gui_base->get_stylebox("SceneTabFG", "EditorStyles"));
	scene_tabs->add_style_override("tab_bg", gui_base->get_stylebox("SceneTabBG", "EditorStyles"));
	scene_tabs->add_tab("unsaved");
	scene_tabs->set_tab_align(Tabs::ALIGN_LEFT);
	scene_tabs->set_tab_close_display_policy((bool(EDITOR_DEF("interface/always_show_close_button_in_scene_tabs", false)) ? Tabs::CLOSE_BUTTON_SHOW_ALWAYS : Tabs::CLOSE_BUTTON_SHOW_ACTIVE_ONLY));
	scene_tabs->set_min_width(int(EDITOR_DEF("interface/scene_tabs/minimum_width", 50)) * EDSCALE);
	scene_tabs->connect("tab_changed", this, "_scene_tab_changed");
	scene_tabs->connect("right_button_pressed", this, "_scene_tab_script_edited");
	scene_tabs->connect("tab_close", this, "_scene_tab_closed");
	scene_tabs->connect("tab_hover", this, "_scene_tab_hover");
	scene_tabs->connect("mouse_exited", this, "_scene_tab_exit");
	scene_tabs->connect("gui_input", this, "_scene_tab_input");
	scene_tabs->connect("reposition_active_tab_request", this, "_reposition_active_tab");

	HBoxContainer *tabbar_container = memnew(HBoxContainer);
	scene_tabs->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	srt->add_child(tabbar_container);
	tabbar_container->add_child(scene_tabs);
	distraction_free = memnew(ToolButton);
	tabbar_container->add_child(distraction_free);
	distraction_free->set_shortcut(ED_SHORTCUT("editor/distraction_free_mode", TTR("Distraction Free Mode"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_F11));
	distraction_free->set_tooltip(TTR("Toggle distraction-free mode."));
	distraction_free->connect("pressed", this, "_toggle_distraction_free_mode");
	distraction_free->set_icon(gui_base->get_icon("DistractionFree", "EditorIcons"));
	distraction_free->set_toggle_mode(true);

	scene_root_parent = memnew(PanelContainer);
	scene_root_parent->set_custom_minimum_size(Size2(0, 80) * EDSCALE);
	scene_root_parent->add_style_override("panel", gui_base->get_stylebox("Content", "EditorStyles"));
	scene_root_parent->set_draw_behind_parent(true);
	srt->add_child(scene_root_parent);
	scene_root_parent->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	scene_root = memnew(Viewport);
	scene_root->set_disable_3d(true);

	VisualServer::get_singleton()->viewport_set_hide_scenario(scene_root->get_viewport_rid(), true);
	scene_root->set_disable_input(true);
	scene_root->set_as_audio_listener_2d(true);

	viewport = memnew(VBoxContainer);
	viewport->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewport->add_constant_override("separation", 0);
	scene_root_parent->add_child(viewport);

	PanelContainer *top_region = memnew(PanelContainer);
	top_region->add_style_override("panel", gui_base->get_stylebox("MenuPanel", "EditorStyles"));
	HBoxContainer *left_menu_hb = memnew(HBoxContainer);
	top_region->add_child(left_menu_hb);
	menu_hb->add_child(top_region);

	{
		Control *sp = memnew(Control);
		sp->set_custom_minimum_size(Size2(30, 0) * EDSCALE);
		menu_hb->add_child(sp);
	}

	file_menu = memnew(MenuButton);
	file_menu->set_flat(false);
	file_menu->set_text(TTR("Scene"));
	file_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
	left_menu_hb->add_child(file_menu);

	prev_scene = memnew(ToolButton);
	prev_scene->set_icon(gui_base->get_icon("PrevScene", "EditorIcons"));
	prev_scene->set_tooltip(TTR("Go to previously opened scene."));
	prev_scene->set_disabled(true);
	prev_scene->connect("pressed", this, "_menu_option", make_binds(FILE_OPEN_PREV));
	gui_base->add_child(prev_scene);
	prev_scene->set_position(Point2(3, 24));
	prev_scene->hide();

	accept = memnew(AcceptDialog);
	gui_base->add_child(accept);
	accept->connect("confirmed", this, "_menu_confirm_current");

	project_export = memnew(ProjectExportDialog);
	gui_base->add_child(project_export);

	dependency_error = memnew(DependencyErrorDialog);
	gui_base->add_child(dependency_error);

	dependency_fixer = memnew(DependencyEditor);
	gui_base->add_child(dependency_fixer);

	settings_config_dialog = memnew(EditorSettingsDialog);
	// settings_config_dialog->add_style_override("panel", gui_base->get_stylebox("EditorSettingsDialog", "EditorStyles"));
	gui_base->add_child(settings_config_dialog);

	project_settings = memnew(ProjectSettingsEditor(&editor_data));
	gui_base->add_child(project_settings);

	run_settings_dialog = memnew(RunSettingsDialog);
	gui_base->add_child(run_settings_dialog);

	export_template_manager = memnew(ExportTemplateManager);
	gui_base->add_child(export_template_manager);

	about = memnew(EditorAbout);
	about->get_logo()->set_texture(gui_base->get_icon("Logo", "EditorIcons"));
	gui_base->add_child(about);

	warning = memnew(AcceptDialog);
	gui_base->add_child(warning);

	ED_SHORTCUT("editor/next_tab", TTR("Next tab"), KEY_MASK_CMD + KEY_TAB);
	ED_SHORTCUT("editor/prev_tab", TTR("Previous tab"), KEY_MASK_CMD + KEY_MASK_SHIFT + KEY_TAB);
	ED_SHORTCUT("editor/filter_files", TTR("Filter Files.."), KEY_MASK_ALT + KEY_MASK_CMD + KEY_P);
	PopupMenu *p;

	file_menu->set_tooltip(TTR("Operations with scene files."));
	p = file_menu->get_popup();
	p->add_shortcut(ED_SHORTCUT("editor/new_scene", TTR("New Scene")), FILE_NEW_SCENE);
	p->add_shortcut(ED_SHORTCUT("editor/new_inherited_scene", TTR("New Inherited Scene..")), FILE_NEW_INHERITED_SCENE);
	p->add_shortcut(ED_SHORTCUT("editor/open_scene", TTR("Open Scene.."), KEY_MASK_CMD + KEY_O), FILE_OPEN_SCENE);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("editor/save_scene", TTR("Save Scene"), KEY_MASK_CMD + KEY_S), FILE_SAVE_SCENE);
	p->add_shortcut(ED_SHORTCUT("editor/save_scene_as", TTR("Save Scene As.."), KEY_MASK_SHIFT + KEY_MASK_CMD + KEY_S), FILE_SAVE_AS_SCENE);
	p->add_shortcut(ED_SHORTCUT("editor/save_all_scenes", TTR("Save all Scenes"), KEY_MASK_ALT + KEY_MASK_SHIFT + KEY_MASK_CMD + KEY_S), FILE_SAVE_ALL_SCENES);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("editor/close_scene", TTR("Close Scene"), KEY_MASK_SHIFT + KEY_MASK_CTRL + KEY_W), FILE_CLOSE);
	p->add_separator();
	p->add_submenu_item(TTR("Open Recent"), "RecentScenes", FILE_OPEN_RECENT);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("editor/quick_open_scene", TTR("Quick Open Scene.."), KEY_MASK_SHIFT + KEY_MASK_CMD + KEY_O), FILE_QUICK_OPEN_SCENE);
	p->add_shortcut(ED_SHORTCUT("editor/quick_open_script", TTR("Quick Open Script.."), KEY_MASK_ALT + KEY_MASK_CMD + KEY_O), FILE_QUICK_OPEN_SCRIPT);
	p->add_separator();
	PopupMenu *pm_export = memnew(PopupMenu);
	pm_export->set_name("Export");
	p->add_child(pm_export);
	p->add_submenu_item(TTR("Convert To.."), "Export");
	pm_export->add_shortcut(ED_SHORTCUT("editor/convert_to_MeshLibrary", TTR("MeshLibrary..")), FILE_EXPORT_MESH_LIBRARY);
	pm_export->add_shortcut(ED_SHORTCUT("editor/convert_to_TileSet", TTR("TileSet..")), FILE_EXPORT_TILESET);
	pm_export->connect("id_pressed", this, "_menu_option");

	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("editor/undo", TTR("Undo"), KEY_MASK_CMD + KEY_Z), EDIT_UNDO, true);
	p->add_shortcut(ED_SHORTCUT("editor/redo", TTR("Redo"), KEY_MASK_SHIFT + KEY_MASK_CMD + KEY_Z), EDIT_REDO, true);
	p->add_separator();
	p->add_item(TTR("Revert Scene"), EDIT_REVERT);

	recent_scenes = memnew(PopupMenu);
	recent_scenes->set_name("RecentScenes");
	p->add_child(recent_scenes);
	recent_scenes->connect("id_pressed", this, "_open_recent_scene");

	p->add_separator();
	p->add_item(TTR("Quit"), FILE_QUIT, KEY_MASK_CMD + KEY_Q);

	project_menu = memnew(MenuButton);
	project_menu->set_flat(false);
	project_menu->set_tooltip(TTR("Miscellaneous project or scene-wide tools."));
	project_menu->set_text(TTR("Project"));
	project_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
	left_menu_hb->add_child(project_menu);

	p = project_menu->get_popup();
	p->add_item(TTR("Project Settings"), RUN_SETTINGS);
	p->add_separator();
	p->connect("id_pressed", this, "_menu_option");
	//p->add_item(TTR("Run Script"), FILE_RUN_SCRIPT, KEY_MASK_SHIFT + KEY_MASK_CMD + KEY_R);
	p->add_item(TTR("Export"), FILE_EXPORT_PROJECT);

	PopupMenu *tool_menu = memnew(PopupMenu);
	tool_menu->set_name("Tools");
	tool_menu->connect("id_pressed", this, "_menu_option");
	p->add_child(tool_menu);
	p->add_submenu_item(TTR("Tools"), "Tools");
	tool_menu->add_item(TTR("Orphan Resource Explorer"), TOOLS_ORPHAN_RESOURCES);
	p->add_separator();

#ifdef OSX_ENABLED
	p->add_item(TTR("Quit to Project List"), RUN_PROJECT_MANAGER, KEY_MASK_SHIFT + KEY_MASK_ALT + KEY_Q);
#else
	p->add_item(TTR("Quit to Project List"), RUN_PROJECT_MANAGER, KEY_MASK_SHIFT + KEY_MASK_CTRL + KEY_Q);
#endif

	PanelContainer *editor_region = memnew(PanelContainer);
	main_editor_button_vb = memnew(HBoxContainer);
	editor_region->add_child(main_editor_button_vb);
	menu_hb->add_spacer();
	menu_hb->add_child(editor_region);

	debug_menu = memnew(MenuButton);
	debug_menu->set_flat(false);
	debug_menu->set_text(TTR("Debug"));
	debug_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
	left_menu_hb->add_child(debug_menu);

	p = debug_menu->get_popup();
	p->set_hide_on_item_selection(false);
	p->add_check_item(TTR("Deploy with Remote Debug"), RUN_DEPLOY_REMOTE_DEBUG);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("When exporting or deploying, the resulting executable will attempt to connect to the IP of this computer in order to be debugged."));
	p->add_check_item(TTR("Small Deploy with Network FS"), RUN_FILE_SERVER);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("When this option is enabled, export or deploy will produce a minimal executable.\nThe filesystem will be provided from the project by the editor over the network.\nOn Android, deploy will use the USB cable for faster performance. This option speeds up testing for games with a large footprint."));
	p->add_separator();
	p->add_check_item(TTR("Visible Collision Shapes"), RUN_DEBUG_COLLISONS);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("Collision shapes and raycast nodes (for 2D and 3D) will be visible on the running game if this option is turned on."));
	p->add_check_item(TTR("Visible Navigation"), RUN_DEBUG_NAVIGATION);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("Navigation meshes and polygons will be visible on the running game if this option is turned on."));
	p->add_separator();
	p->add_check_item(TTR("Sync Scene Changes"), RUN_LIVE_DEBUG);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("When this option is turned on, any changes made to the scene in the editor will be replicated in the running game.\nWhen used remotely on a device, this is more efficient with network filesystem."));
	p->add_check_item(TTR("Sync Script Changes"), RUN_RELOAD_SCRIPTS);
	p->set_item_tooltip(p->get_item_count() - 1, TTR("When this option is turned on, any script that is saved will be reloaded on the running game.\nWhen used remotely on a device, this is more efficient with network filesystem."));
	p->connect("id_pressed", this, "_menu_option");

	menu_hb->add_spacer();

	settings_menu = memnew(MenuButton);
	settings_menu->set_flat(false);
	settings_menu->set_text(TTR("Editor"));
	settings_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
	left_menu_hb->add_child(settings_menu);
	p = settings_menu->get_popup();

	p->add_item(TTR("Editor Settings"), SETTINGS_PREFERENCES);
	p->add_separator();
	editor_layouts = memnew(PopupMenu);
	editor_layouts->set_name("Layouts");
	p->add_child(editor_layouts);
	editor_layouts->connect("id_pressed", this, "_layout_menu_option");
	p->add_submenu_item(TTR("Editor Layout"), "Layouts");
	p->add_shortcut(ED_SHORTCUT("editor/fullscreen_mode", TTR("Toggle Fullscreen"), KEY_MASK_SHIFT | KEY_F11), SETTINGS_TOGGLE_FULLSCREN);
	p->add_separator();
	p->add_item(TTR("Manage Export Templates"), SETTINGS_MANAGE_EXPORT_TEMPLATES);

	// Help Menu
	help_menu = memnew(MenuButton);
	help_menu->set_flat(false);
	help_menu->set_text(TTR("Help"));
	help_menu->add_style_override("hover", gui_base->get_stylebox("MenuHover", "EditorStyles"));
	left_menu_hb->add_child(help_menu);

	p = help_menu->get_popup();
	p->connect("id_pressed", this, "_menu_option");
	p->add_icon_item(gui_base->get_icon("ClassList", "EditorIcons"), TTR("Classes"), HELP_CLASSES);
	p->add_icon_item(gui_base->get_icon("HelpSearch", "EditorIcons"), TTR("Search"), HELP_SEARCH);
	p->add_separator();
	p->add_icon_item(gui_base->get_icon("Instance", "EditorIcons"), TTR("Online Docs"), HELP_DOCS);
	p->add_icon_item(gui_base->get_icon("Instance", "EditorIcons"), TTR("Q&A"), HELP_QA);
	p->add_icon_item(gui_base->get_icon("Instance", "EditorIcons"), TTR("Issue Tracker"), HELP_ISSUES);
	p->add_icon_item(gui_base->get_icon("Instance", "EditorIcons"), TTR("Community"), HELP_COMMUNITY);
	p->add_separator();
	p->add_icon_item(gui_base->get_icon("Godot", "EditorIcons"), TTR("About"), HELP_ABOUT);

	play_cc = memnew(CenterContainer);
	play_cc->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	menu_hb->add_child(play_cc);
	play_cc->set_area_as_parent_rect();
	play_cc->set_anchor_and_margin(MARGIN_BOTTOM, Control::ANCHOR_BEGIN, 10);
	play_cc->set_margin(MARGIN_TOP, 5);

	play_button_panel = memnew(PanelContainer);
	// play_button_panel->add_style_override("panel", gui_base->get_stylebox("PlayButtonPanel", "EditorStyles"));
	play_cc->add_child(play_button_panel);

	HBoxContainer *play_hb = memnew(HBoxContainer);
	play_button_panel->add_child(play_hb);

	play_button = memnew(ToolButton);
	play_hb->add_child(play_button);
	play_button->set_toggle_mode(true);
	play_button->set_icon(gui_base->get_icon("MainPlay", "EditorIcons"));
	play_button->set_focus_mode(Control::FOCUS_NONE);
	play_button->connect("pressed", this, "_menu_option", make_binds(RUN_PLAY));
	play_button->set_tooltip(TTR("Play the project."));
	play_button->set_shortcut(ED_SHORTCUT("editor/play", TTR("Play"), KEY_F5));

	pause_button = memnew(ToolButton);
	pause_button->set_toggle_mode(true);
	pause_button->set_icon(gui_base->get_icon("Pause", "EditorIcons"));
	pause_button->set_focus_mode(Control::FOCUS_NONE);
	pause_button->set_tooltip(TTR("Pause the scene"));
	pause_button->set_disabled(true);
	play_hb->add_child(pause_button);
	pause_button->set_shortcut(ED_SHORTCUT("editor/pause_scene", TTR("Pause Scene"), KEY_F7));

	stop_button = memnew(ToolButton);
	play_hb->add_child(stop_button);
	stop_button->set_focus_mode(Control::FOCUS_NONE);
	stop_button->set_icon(gui_base->get_icon("Stop", "EditorIcons"));
	stop_button->connect("pressed", this, "_menu_option", make_binds(RUN_STOP));
	stop_button->set_tooltip(TTR("Stop the scene."));
	stop_button->set_shortcut(ED_SHORTCUT("editor/stop", TTR("Stop"), KEY_F8));

	run_native = memnew(EditorRunNative);
	play_hb->add_child(run_native);
	native_play_button = memnew(MenuButton);
	native_play_button->set_text("NTV");
	menu_hb->add_child(native_play_button);
	native_play_button->hide();
	native_play_button->get_popup()->connect("id_pressed", this, "_run_in_device");
	run_native->connect("native_run", this, "_menu_option", varray(RUN_PLAY_NATIVE));

	play_scene_button = memnew(ToolButton);
	play_hb->add_child(play_scene_button);
	play_scene_button->set_toggle_mode(true);
	play_scene_button->set_focus_mode(Control::FOCUS_NONE);
	play_scene_button->set_icon(gui_base->get_icon("PlayScene", "EditorIcons"));
	play_scene_button->connect("pressed", this, "_menu_option", make_binds(RUN_PLAY_SCENE));
	play_scene_button->set_tooltip(TTR("Play the edited scene."));
	play_scene_button->set_shortcut(ED_SHORTCUT("editor/play_scene", TTR("Play Scene"), KEY_F6));

	play_custom_scene_button = memnew(ToolButton);
	play_hb->add_child(play_custom_scene_button);
	play_custom_scene_button->set_toggle_mode(true);
	play_custom_scene_button->set_focus_mode(Control::FOCUS_NONE);
	play_custom_scene_button->set_icon(gui_base->get_icon("PlayCustom", "EditorIcons"));
	play_custom_scene_button->connect("pressed", this, "_menu_option", make_binds(RUN_PLAY_CUSTOM_SCENE));
	play_custom_scene_button->set_tooltip(TTR("Play custom scene"));
	play_custom_scene_button->set_shortcut(ED_SHORTCUT("editor/play_custom_scene", TTR("Play Custom Scene"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_F5));

	progress_hb = memnew(BackgroundProgress);

	HBoxContainer *right_menu_hb = memnew(HBoxContainer);
	menu_hb->add_child(right_menu_hb);

	layout_dialog = memnew(EditorNameDialog);
	gui_base->add_child(layout_dialog);
	layout_dialog->set_hide_on_ok(false);
	layout_dialog->set_size(Size2(175, 70) * EDSCALE);
	layout_dialog->connect("name_confirmed", this, "_dialog_action");

	update_menu = memnew(MenuButton);
	update_menu->set_tooltip(TTR("Spins when the editor window repaints!"));
	right_menu_hb->add_child(update_menu);
	update_menu->set_icon(gui_base->get_icon("Progress1", "EditorIcons"));
	update_menu->get_popup()->connect("id_pressed", this, "_menu_option");
	p = update_menu->get_popup();
	p->add_check_item(TTR("Update Always"), SETTINGS_UPDATE_ALWAYS);
	p->add_check_item(TTR("Update Changes"), SETTINGS_UPDATE_CHANGES);
	p->add_separator();
	p->add_check_item(TTR("Disable Update Spinner"), SETTINGS_UPDATE_SPINNER_HIDE);
	int update_always = EditorSettings::get_singleton()->get_project_metadata("editor_options", "update_always", false);
	int hide_spinner = EditorSettings::get_singleton()->get_project_metadata("editor_options", "update_spinner_hide", false);
	_menu_option(update_always ? SETTINGS_UPDATE_ALWAYS : SETTINGS_UPDATE_CHANGES);
	if (hide_spinner) {
		_menu_option(SETTINGS_UPDATE_SPINNER_HIDE);
	}

	scene_tree_dock = memnew(SceneTreeDock(this, scene_root, editor_selection, editor_data));
	scene_tree_dock->set_name(TTR("Scene"));
	dock_slot[DOCK_SLOT_RIGHT_UL]->add_child(scene_tree_dock);
	dock_slot[DOCK_SLOT_LEFT_BR]->hide();

	VBoxContainer *prop_editor_base = memnew(VBoxContainer);
	prop_editor_base->set_name(TTR("Inspector")); // Properties?
	dock_slot[DOCK_SLOT_RIGHT_BL]->add_child(prop_editor_base);

	HBoxContainer *prop_editor_hb = memnew(HBoxContainer);

	prop_editor_base->add_child(prop_editor_hb);
	prop_editor_vb = prop_editor_base;

	resource_new_button = memnew(ToolButton);
	resource_new_button->set_tooltip(TTR("Create a new resource in memory and edit it."));
	resource_new_button->set_icon(gui_base->get_icon("New", "EditorIcons"));
	prop_editor_hb->add_child(resource_new_button);
	resource_new_button->connect("pressed", this, "_menu_option", varray(RESOURCE_NEW));
	resource_new_button->set_focus_mode(Control::FOCUS_NONE);

	resource_load_button = memnew(ToolButton);
	resource_load_button->set_tooltip(TTR("Load an existing resource from disk and edit it."));
	resource_load_button->set_icon(gui_base->get_icon("Load", "EditorIcons"));
	prop_editor_hb->add_child(resource_load_button);
	resource_load_button->connect("pressed", this, "_menu_option", varray(RESOURCE_LOAD));
	resource_load_button->set_focus_mode(Control::FOCUS_NONE);

	resource_save_button = memnew(MenuButton);
	resource_save_button->set_tooltip(TTR("Save the currently edited resource."));
	resource_save_button->set_icon(gui_base->get_icon("Save", "EditorIcons"));
	prop_editor_hb->add_child(resource_save_button);
	resource_save_button->get_popup()->add_item(TTR("Save"), RESOURCE_SAVE);
	resource_save_button->get_popup()->add_item(TTR("Save As.."), RESOURCE_SAVE_AS);
	resource_save_button->get_popup()->connect("id_pressed", this, "_menu_option");
	resource_save_button->set_focus_mode(Control::FOCUS_NONE);
	resource_save_button->set_disabled(true);

	prop_editor_hb->add_spacer();

	property_back = memnew(ToolButton);
	property_back->set_icon(gui_base->get_icon("Back", "EditorIcons"));
	property_back->set_flat(true);
	property_back->set_tooltip(TTR("Go to the previous edited object in history."));
	property_back->set_disabled(true);

	prop_editor_hb->add_child(property_back);

	property_forward = memnew(ToolButton);
	property_forward->set_icon(gui_base->get_icon("Forward", "EditorIcons"));
	property_forward->set_flat(true);
	property_forward->set_tooltip(TTR("Go to the next edited object in history."));
	property_forward->set_disabled(true);

	prop_editor_hb->add_child(property_forward);

	editor_history_menu = memnew(MenuButton);
	editor_history_menu->set_tooltip(TTR("History of recently edited objects."));
	editor_history_menu->set_icon(gui_base->get_icon("History", "EditorIcons"));
	prop_editor_hb->add_child(editor_history_menu);
	editor_history_menu->connect("about_to_show", this, "_prepare_history");
	editor_history_menu->get_popup()->connect("id_pressed", this, "_select_history");

	prop_editor_hb = memnew(HBoxContainer); //again...
	prop_editor_base->add_child(prop_editor_hb);

	editor_path = memnew(EditorPath(&editor_history));
	editor_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	prop_editor_hb->add_child(editor_path);

	search_button = memnew(ToolButton);
	search_button->set_toggle_mode(true);
	search_button->set_pressed(false);
	search_button->set_icon(gui_base->get_icon("Search", "EditorIcons"));
	prop_editor_hb->add_child(search_button);
	search_button->connect("toggled", this, "_toggle_search_bar");

	object_menu = memnew(MenuButton);
	object_menu->set_icon(gui_base->get_icon("Tools", "EditorIcons"));
	prop_editor_hb->add_child(object_menu);
	object_menu->set_tooltip(TTR("Object properties."));

	create_dialog = memnew(CreateDialog);
	gui_base->add_child(create_dialog);
	create_dialog->set_base_type("Resource");
	create_dialog->connect("create", this, "_resource_created");

	search_bar = memnew(HBoxContainer);
	search_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	prop_editor_base->add_child(search_bar);
	search_bar->hide();

	Label *l = memnew(Label(TTR("Search:") + " "));
	search_bar->add_child(l);

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_bar->add_child(search_box);

	ToolButton *clear_button = memnew(ToolButton);
	clear_button->set_icon(gui_base->get_icon("Close", "EditorIcons"));
	search_bar->add_child(clear_button);
	clear_button->connect("pressed", this, "_clear_search_box");

	property_editable_warning = memnew(Button);
	property_editable_warning->set_text(TTR("Changes may be lost!"));
	prop_editor_base->add_child(property_editable_warning);
	property_editable_warning_dialog = memnew(AcceptDialog);
	gui_base->add_child(property_editable_warning_dialog);
	property_editable_warning->hide();
	property_editable_warning->connect("pressed", this, "_property_editable_warning_pressed");

	property_editor = memnew(PropertyEditor);
	property_editor->set_autoclear(true);
	property_editor->set_show_categories(true);
	property_editor->set_use_folding(true);
	property_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	property_editor->set_use_doc_hints(true);
	property_editor->set_hide_script(false);
	property_editor->set_enable_capitalize_paths(bool(EDITOR_DEF("interface/capitalize_properties", true)));

	property_editor->hide_top_label();
	property_editor->register_text_enter(search_box);

	Button *property_editable_warning;
	prop_editor_base->add_child(property_editor);
	property_editor->set_undo_redo(&editor_data.get_undo_redo());

	import_dock = memnew(ImportDock);
	dock_slot[DOCK_SLOT_RIGHT_UL]->add_child(import_dock);
	import_dock->set_name(TTR("Import"));

	bool use_single_dock_column = (OS::get_singleton()->get_screen_size(OS::get_singleton()->get_current_screen()).x < 1200);

	node_dock = memnew(NodeDock);
	if (use_single_dock_column) {
		dock_slot[DOCK_SLOT_RIGHT_UL]->add_child(node_dock);
	} else {
		dock_slot[DOCK_SLOT_RIGHT_BL]->add_child(node_dock);
	}

	filesystem_dock = memnew(FileSystemDock(this));
	filesystem_dock->set_name(TTR("FileSystem"));
	filesystem_dock->set_display_mode(int(EditorSettings::get_singleton()->get("docks/filesystem/display_mode")));

	if (use_single_dock_column) {
		dock_slot[DOCK_SLOT_RIGHT_BL]->add_child(filesystem_dock);
		left_r_vsplit->hide();
		dock_slot[DOCK_SLOT_LEFT_UR]->hide();
		dock_slot[DOCK_SLOT_LEFT_BR]->hide();
	} else {
		dock_slot[DOCK_SLOT_LEFT_UR]->add_child(filesystem_dock);
	}
	filesystem_dock->connect("open", this, "open_request");
	filesystem_dock->connect("instance", this, "_instance_request");

	const String docks_section = "docks";

	overridden_default_layout = -1;
	default_layout.instance();
	default_layout->set_value(docks_section, "dock_3", TTR("FileSystem"));
	default_layout->set_value(docks_section, "dock_5", TTR("Scene") + "," + TTR("Import"));
	default_layout->set_value(docks_section, "dock_6", TTR("Inspector") + "," + TTR("Node"));

	for (int i = 0; i < DOCK_SLOT_MAX / 2; i++)
		default_layout->set_value(docks_section, "dock_hsplit_" + itos(i + 1), 0);
	for (int i = 0; i < DOCK_SLOT_MAX / 2; i++)
		default_layout->set_value(docks_section, "dock_split_" + itos(i + 1), 0);

	_update_layouts_menu();

	bottom_panel = memnew(PanelContainer);
	bottom_panel->add_style_override("panel", gui_base->get_stylebox("panel", "TabContainer"));
	center_split->add_child(bottom_panel);
	center_split->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN);

	bottom_panel_vb = memnew(VBoxContainer);
	bottom_panel->add_child(bottom_panel_vb);

	bottom_panel_hb = memnew(HBoxContainer);
	bottom_panel_vb->add_child(bottom_panel_hb);

	log = memnew(EditorLog);
	add_bottom_panel_item(TTR("Output"), log);

	old_split_ofs = 0;

	center_split->connect("resized", this, "_vp_resized");

	orphan_resources = memnew(OrphanResourcesDialog);
	gui_base->add_child(orphan_resources);

	confirmation = memnew(ConfirmationDialog);
	gui_base->add_child(confirmation);
	confirmation->connect("confirmed", this, "_menu_confirm_current");

	save_confirmation = memnew(ConfirmationDialog);
	save_confirmation->add_button(TTR("Don't Save"), OS::get_singleton()->get_swap_ok_cancel(), "discard");
	gui_base->add_child(save_confirmation);
	save_confirmation->connect("confirmed", this, "_menu_confirm_current");
	save_confirmation->connect("custom_action", this, "_discard_changes");

	file_templates = memnew(FileDialog);
	file_templates->set_title(TTR("Import Templates From ZIP File"));

	gui_base->add_child(file_templates);
	file_templates->set_mode(FileDialog::MODE_OPEN_FILE);
	file_templates->set_access(FileDialog::ACCESS_FILESYSTEM);
	file_templates->clear_filters();
	file_templates->add_filter("*.tpz ; Template Package");

	file = memnew(EditorFileDialog);
	gui_base->add_child(file);
	file->set_current_dir("res://");

	file_export = memnew(FileDialog);
	file_export->set_access(FileDialog::ACCESS_FILESYSTEM);
	gui_base->add_child(file_export);
	file_export->set_title(TTR("Export Project"));
	file_export->connect("file_selected", this, "_dialog_action");

	file_export_lib = memnew(FileDialog);
	file_export_lib->set_title(TTR("Export Library"));
	file_export_lib->set_mode(FileDialog::MODE_SAVE_FILE);
	file_export_lib->connect("file_selected", this, "_dialog_action");
	file_export_lib_merge = memnew(CheckButton);
	file_export_lib_merge->set_text(TTR("Merge With Existing"));
	file_export_lib_merge->set_pressed(true);
	file_export_lib->get_vbox()->add_child(file_export_lib_merge);
	gui_base->add_child(file_export_lib);

	file_export_password = memnew(LineEdit);
	file_export_password->set_secret(true);
	file_export_password->set_editable(false);
	file_export->get_vbox()->add_margin_child(TTR("Password:"), file_export_password);

	file_script = memnew(FileDialog);
	file_script->set_title(TTR("Open & Run a Script"));
	file_script->set_access(FileDialog::ACCESS_FILESYSTEM);
	file_script->set_mode(FileDialog::MODE_OPEN_FILE);
	List<String> sexts;
	ResourceLoader::get_recognized_extensions_for_type("Script", &sexts);
	for (List<String>::Element *E = sexts.front(); E; E = E->next()) {
		file_script->add_filter("*." + E->get());
	}
	gui_base->add_child(file_script);
	file_script->connect("file_selected", this, "_dialog_action");

	property_forward->connect("pressed", this, "_property_editor_forward");
	property_back->connect("pressed", this, "_property_editor_back");

	file_menu->get_popup()->connect("id_pressed", this, "_menu_option");
	object_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	settings_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	file->connect("file_selected", this, "_dialog_action");
	file_templates->connect("file_selected", this, "_dialog_action");
	property_editor->connect("resource_selected", this, "_resource_selected");
	property_editor->connect("property_keyed", this, "_property_keyed");

	//plugin stuff

	file_server = memnew(EditorFileServer);

	add_editor_plugin(memnew(AnimationPlayerEditorPlugin(this)));
	add_editor_plugin(memnew(CanvasItemEditorPlugin(this)));
	add_editor_plugin(memnew(SpatialEditorPlugin(this)));
	add_editor_plugin(memnew(ScriptEditorPlugin(this)));

	EditorAudioBuses *audio_bus_editor = EditorAudioBuses::register_editor();

	ScriptTextEditor::register_editor(); //register one for text scripts

	if (StreamPeerSSL::is_available()) {
		add_editor_plugin(memnew(AssetLibraryEditorPlugin(this)));
	} else {
		WARN_PRINT("Asset Library not available, as it requires SSL to work.");
	}

	//add interface before adding plugins

	editor_interface = memnew(EditorInterface);
	add_child(editor_interface);

	//more visually meaningful to have this later
	raise_bottom_panel_item(AnimationPlayerEditor::singleton);

	add_editor_plugin(memnew(ShaderEditorPlugin(this)));
	// FIXME: Disabled for Godot 3.0 as made incompatible, it needs to be ported to the new API.
	//add_editor_plugin(memnew(ShaderGraphEditorPlugin(this)));

	add_editor_plugin(memnew(CameraEditorPlugin(this)));
	add_editor_plugin(memnew(ThemeEditorPlugin(this)));
	add_editor_plugin(memnew(MultiMeshEditorPlugin(this)));
	add_editor_plugin(memnew(MeshInstanceEditorPlugin(this)));
	add_editor_plugin(memnew(AnimationTreeEditorPlugin(this)));
	add_editor_plugin(memnew(MeshLibraryEditorPlugin(this)));
	add_editor_plugin(memnew(StyleBoxEditorPlugin(this)));
	add_editor_plugin(memnew(ParticlesEditorPlugin(this)));
	add_editor_plugin(memnew(ResourcePreloaderEditorPlugin(this)));
	add_editor_plugin(memnew(ItemListEditorPlugin(this)));
	add_editor_plugin(memnew(CollisionPolygonEditorPlugin(this)));
	add_editor_plugin(memnew(CollisionPolygon2DEditorPlugin(this)));
	add_editor_plugin(memnew(TileSetEditorPlugin(this)));
	add_editor_plugin(memnew(TileMapEditorPlugin(this)));
	add_editor_plugin(memnew(SpriteFramesEditorPlugin(this)));
	add_editor_plugin(memnew(TextureRegionEditorPlugin(this)));
	add_editor_plugin(memnew(Particles2DEditorPlugin(this)));
	add_editor_plugin(memnew(GIProbeEditorPlugin(this)));
	add_editor_plugin(memnew(Path2DEditorPlugin(this)));
	add_editor_plugin(memnew(PathEditorPlugin(this)));
	add_editor_plugin(memnew(Line2DEditorPlugin(this)));
	add_editor_plugin(memnew(Polygon2DEditorPlugin(this)));
	add_editor_plugin(memnew(LightOccluder2DEditorPlugin(this)));
	add_editor_plugin(memnew(NavigationPolygonEditorPlugin(this)));
	add_editor_plugin(memnew(GradientEditorPlugin(this)));
	add_editor_plugin(memnew(CollisionShape2DEditorPlugin(this)));
	add_editor_plugin(memnew(CurveEditorPlugin(this)));
	add_editor_plugin(memnew(TextureEditorPlugin(this)));
	add_editor_plugin(memnew(MeshEditorPlugin(this)));
	add_editor_plugin(memnew(AudioBusesEditorPlugin(audio_bus_editor)));
	add_editor_plugin(memnew(NavigationMeshEditorPlugin(this)));

	// FIXME: Disabled as (according to reduz) users were complaining that it gets in the way
	// Waiting for PropertyEditor rewrite (planned for 3.1) to be refactored.
	//add_editor_plugin(memnew(MaterialEditorPlugin(this)));

	for (int i = 0; i < EditorPlugins::get_plugin_count(); i++)
		add_editor_plugin(EditorPlugins::create(i, this));

	for (int i = 0; i < plugin_init_callback_count; i++) {
		plugin_init_callbacks[i]();
	}

	resource_preview->add_preview_generator(Ref<EditorTexturePreviewPlugin>(memnew(EditorTexturePreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorPackedScenePreviewPlugin>(memnew(EditorPackedScenePreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorMaterialPreviewPlugin>(memnew(EditorMaterialPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorScriptPreviewPlugin>(memnew(EditorScriptPreviewPlugin)));
	// FIXME: Needs to be rewritten for AudioStream in Godot 3.0+
	//resource_preview->add_preview_generator( Ref<EditorSamplePreviewPlugin>( memnew(EditorSamplePreviewPlugin )));
	resource_preview->add_preview_generator(Ref<EditorMeshPreviewPlugin>(memnew(EditorMeshPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorBitmapPreviewPlugin>(memnew(EditorBitmapPreviewPlugin)));

	circle_step_msec = OS::get_singleton()->get_ticks_msec();
	circle_step_frame = Engine::get_singleton()->get_frames_drawn();
	circle_step = 0;

	editor_plugin_screen = NULL;
	editor_plugins_over = memnew(EditorPluginList);
	editor_plugins_force_input_forwarding = memnew(EditorPluginList);

	_edit_current();
	current = NULL;

	PhysicsServer::get_singleton()->set_active(false); // no physics by default if editor
	Physics2DServer::get_singleton()->set_active(false); // no physics by default if editor
	ScriptServer::set_scripting_enabled(false); // no scripting by default if editor

	reference_resource_mem = true;
	save_external_resources_mem = true;

	set_process(true);

	open_imported = memnew(ConfirmationDialog);
	open_imported->get_ok()->set_text(TTR("Open Anyway"));
	new_inherited_button = open_imported->add_button(TTR("New Inherited"), !OS::get_singleton()->get_swap_ok_cancel(), "inherit");
	open_imported->connect("confirmed", this, "_open_imported");
	open_imported->connect("custom_action", this, "_inherit_imported");
	gui_base->add_child(open_imported);

	saved_version = 1;
	unsaved_cache = true;
	_last_instanced_scene = NULL;

	quick_open = memnew(EditorQuickOpen);
	gui_base->add_child(quick_open);
	quick_open->connect("quick_open", this, "_quick_opened");

	quick_run = memnew(EditorQuickOpen);
	gui_base->add_child(quick_run);
	quick_run->connect("quick_open", this, "_quick_run");

	_update_recent_scenes();

	editor_data.restore_editor_global_states();
	convert_old = false;
	opening_prev = false;
	set_process_unhandled_input(true);
	_playing_edited = false;

	load_errors = memnew(RichTextLabel);
	load_error_dialog = memnew(AcceptDialog);
	load_error_dialog->add_child(load_errors);
	load_error_dialog->set_title(TTR("Load Errors"));
	gui_base->add_child(load_error_dialog);

	EditorFileSystem::get_singleton()->connect("sources_changed", this, "_sources_changed");
	EditorFileSystem::get_singleton()->connect("filesystem_changed", this, "_fs_changed");
	EditorFileSystem::get_singleton()->connect("resources_reimported", this, "_resources_reimported");

	{
		List<StringName> tl;
		StringName ei = "EditorIcons";
		theme_base->get_theme()->get_icon_list(ei, &tl);
		for (List<StringName>::Element *E = tl.front(); E; E = E->next()) {

			if (!ClassDB::class_exists(E->get()))
				continue;
			icon_type_cache[E->get()] = theme_base->get_theme()->get_icon(E->get(), ei);
		}
	}

	Node::set_human_readable_collision_renaming(true);

	pick_main_scene = memnew(ConfirmationDialog);
	gui_base->add_child(pick_main_scene);
	pick_main_scene->get_ok()->set_text(TTR("Select"));
	pick_main_scene->connect("confirmed", this, "_menu_option", varray(SETTINGS_PICK_MAIN_SCENE));

	for (int i = 0; i < _init_callbacks.size(); i++)
		_init_callbacks[i]();

	editor_data.add_edited_scene(-1);
	editor_data.set_edited_scene(0);
	_update_scene_tabs();

	import_dock->initialize_import_options();

	{
		_initializing_addons = true;
		Vector<String> addons;
		if (ProjectSettings::get_singleton()->has("editor_plugins/enabled")) {
			addons = ProjectSettings::get_singleton()->get("editor_plugins/enabled");
		}

		for (int i = 0; i < addons.size(); i++) {
			set_addon_plugin_enabled(addons[i], true);
		}
		_initializing_addons = false;
	}

	_load_docks();

	FileAccess::set_file_close_fail_notify_callback(_file_access_close_error_notify);

	waiting_for_first_scan = true;

	_dimming = false;
	_dim_time = 0.0f;
	_dim_timer = memnew(Timer);
	_dim_timer->set_wait_time(0.01666f);
	_dim_timer->connect("timeout", this, "_dim_timeout");
	add_child(_dim_timer);

	ED_SHORTCUT("editor/editor_2d", TTR("Open 2D Editor"), KEY_F1);
	ED_SHORTCUT("editor/editor_3d", TTR("Open 3D Editor"), KEY_F2);
	ED_SHORTCUT("editor/editor_script", TTR("Open Script Editor"), KEY_F3); //hack neded for script editor F3 search to work :) Assign like this or don't use F3
	ED_SHORTCUT("editor/editor_help", TTR("Search Help"), KEY_F4);
	ED_SHORTCUT("editor/editor_assetlib", TTR("Open Asset Library"));
	ED_SHORTCUT("editor/editor_next", TTR("Open the next Editor"));
	ED_SHORTCUT("editor/editor_prev", TTR("Open the previous Editor"));
}

EditorNode::~EditorNode() {

	memdelete(EditorHelp::get_doc_data());
	memdelete(editor_selection);
	memdelete(editor_plugins_over);
	memdelete(editor_plugins_force_input_forwarding);
	memdelete(file_server);
	memdelete(progress_hb);
	EditorSettings::destroy();
}

/*
 * EDITOR PLUGIN LIST
 */

void EditorPluginList::make_visible(bool p_visible) {

	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->make_visible(p_visible);
	}
}

void EditorPluginList::edit(Object *p_object) {

	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->edit(p_object);
	}
}

bool EditorPluginList::forward_gui_input(const Transform2D &p_canvas_xform, const Ref<InputEvent> &p_event) {

	bool discard = false;

	for (int i = 0; i < plugins_list.size(); i++) {
		if (plugins_list[i]->forward_canvas_gui_input(p_canvas_xform, p_event)) {
			discard = true;
		}
	}

	return discard;
}

bool EditorPluginList::forward_spatial_gui_input(Camera *p_camera, const Ref<InputEvent> &p_event, bool serve_when_force_input_enabled) {
	bool discard = false;

	for (int i = 0; i < plugins_list.size(); i++) {
		if ((!serve_when_force_input_enabled) && plugins_list[i]->is_input_event_forwarding_always_enabled()) {
			continue;
		}

		if (plugins_list[i]->forward_spatial_gui_input(p_camera, p_event)) {
			discard = true;
		}
	}

	return discard;
}

void EditorPluginList::forward_draw_over_canvas(const Transform2D &p_canvas_xform, Control *p_canvas) {

	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->forward_draw_over_canvas(p_canvas_xform, p_canvas);
	}
}

void EditorPluginList::add_plugin(EditorPlugin *p_plugin) {
	plugins_list.push_back(p_plugin);
}

bool EditorPluginList::empty() {
	return plugins_list.empty();
}

void EditorPluginList::clear() {
	plugins_list.clear();
}

EditorPluginList::EditorPluginList() {
}

EditorPluginList::~EditorPluginList() {
}
