/*************************************************************************/
/*  editor_node.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "editor_themes.h"
#include "print_string.h"
#include "version.h"

#include "animation_editor.h"
#include "authors.gen.h"
#include "bind/core_bind.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "editor_file_system.h"
#include "editor_help.h"
#include "editor_settings.h"
#include "globals.h"
#include "io/config_file.h"
#include "io/stream_peer_ssl.h"
#include "io/zip_io.h"
#include "io_plugins/editor_import_collada.h"
#include "io_plugins/editor_scene_importer_fbxconv.h"
#include "message_queue.h"
#include "object_type_db.h"
#include "os/file_access.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "path_remap.h"
#include "pvrtc_compress.h"
#include "register_exporters.h"
#include "scene/resources/packed_scene.h"
#include "servers/physics_2d_server.h"
#include "translation.h"
#include <stdio.h>
// plugins
#include "asset_library_editor_plugin.h"
#include "plugins/camera_editor_plugin.h"
#include "plugins/canvas_item_editor_plugin.h"
#include "plugins/item_list_editor_plugin.h"
#include "plugins/material_editor_plugin.h"
#include "plugins/mesh_editor_plugin.h"
#include "plugins/mesh_instance_editor_plugin.h"
#include "plugins/multimesh_editor_plugin.h"
#include "plugins/resource_preloader_editor_plugin.h"
#include "plugins/sample_editor_plugin.h"
#include "plugins/sample_library_editor_plugin.h"
#include "plugins/sample_player_editor_plugin.h"
#include "plugins/spatial_editor_plugin.h"
#include "plugins/sprite_frames_editor_plugin.h"
#include "plugins/stream_editor_plugin.h"
#include "plugins/style_box_editor_plugin.h"
#include "plugins/texture_editor_plugin.h"
#include "plugins/texture_region_editor_plugin.h"
#include "plugins/theme_editor_plugin.h"

#include "main/input_default.h"
#include "plugins/animation_player_editor_plugin.h"
#include "plugins/animation_tree_editor_plugin.h"
#include "plugins/baked_light_editor_plugin.h"
#include "plugins/collision_polygon_2d_editor_plugin.h"
#include "plugins/collision_polygon_editor_plugin.h"
#include "plugins/collision_shape_2d_editor_plugin.h"
#include "plugins/color_ramp_editor_plugin.h"
#include "plugins/cube_grid_theme_editor_plugin.h"
#include "plugins/light_occluder_2d_editor_plugin.h"
#include "plugins/navigation_polygon_editor_plugin.h"
#include "plugins/particles_2d_editor_plugin.h"
#include "plugins/particles_editor_plugin.h"
#include "plugins/path_2d_editor_plugin.h"
#include "plugins/path_editor_plugin.h"
#include "plugins/polygon_2d_editor_plugin.h"
#include "plugins/rich_text_editor_plugin.h"
#include "plugins/script_editor_plugin.h"
#include "plugins/shader_editor_plugin.h"
#include "plugins/shader_graph_editor_plugin.h"
#include "plugins/tile_map_editor_plugin.h"
#include "plugins/tile_set_editor_plugin.h"
// end
#include "editor/io_plugins/editor_bitmask_import_plugin.h"
#include "editor/io_plugins/editor_export_scene.h"
#include "editor/io_plugins/editor_font_import_plugin.h"
#include "editor/io_plugins/editor_mesh_import_plugin.h"
#include "editor/io_plugins/editor_sample_import_plugin.h"
#include "editor/io_plugins/editor_scene_import_plugin.h"
#include "editor/io_plugins/editor_texture_import_plugin.h"
#include "editor/io_plugins/editor_translation_import_plugin.h"

#include "editor_initialize_ssl.h"
#include "plugins/editor_preview_plugins.h"
#include "script_editor_debugger.h"

EditorNode *EditorNode::singleton = NULL;

void EditorNode::_update_scene_tabs() {

	bool show_rb = EditorSettings::get_singleton()->get("global/show_script_in_scene_tabs");

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

	String appname = Globals::get_singleton()->get("application/name");
	String title = appname.empty() ? String(VERSION_FULL_NAME) : String(_MKSTR(VERSION_NAME) + String(" - ") + appname);
	String edited = editor_data.get_edited_scene_root() ? editor_data.get_edited_scene_root()->get_filename() : String();
	if (!edited.empty())
		title += " - " + String(edited.get_file());
	if (unsaved_cache)
		title += " (*)";

	OS::get_singleton()->set_window_title(title);
}

void EditorNode::_unhandled_input(const InputEvent &p_event) {

	if (Node::get_viewport()->get_modal_stack_top())
		return; //ignore because of modal window

	if (p_event.type == InputEvent::KEY && p_event.key.pressed && !p_event.key.echo && !gui_base->get_viewport()->gui_has_modal_stack()) {

		if (ED_IS_SHORTCUT("editor/fullscreen_mode", p_event)) {
			if (distraction_free_mode) {
				distraction_free_mode = false;
				_update_top_menu_visibility();
			} else {
				set_docks_visible(!get_docks_visible());
			}
		}
		if (ED_IS_SHORTCUT("editor/distraction_free_mode", p_event)) {
			set_distraction_free_mode(!get_distraction_free_mode());
		}
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

//force the whole tree viewport
#if 0
		{
			Rect2 grect = scene_root_base->get_global_rect();
			Rect2 grectsrp = scene_root_parent->get_global_rect();
			if (grect!=grectsrp) {
				scene_root_parent->set_pos(grect.pos);
				scene_root_parent->set_size(grect.size);
			}
		}

#endif
		if (opening_prev && confirmation->is_hidden())
			opening_prev = false;

		if (unsaved_cache != (saved_version != editor_data.get_undo_redo().get_version())) {

			unsaved_cache = (saved_version != editor_data.get_undo_redo().get_version());
			_update_title();
		}

		if (last_checked_version != editor_data.get_undo_redo().get_version()) {
			_update_scene_tabs();
			last_checked_version = editor_data.get_undo_redo().get_version();
		}

		//get_root_node()->set_rect(viewport->get_global_rect());

		//update the circle
		uint64_t frame = OS::get_singleton()->get_frames_drawn();
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

		scene_root->set_size_override(true, Size2(Globals::get_singleton()->get("display/width"), Globals::get_singleton()->get("display/height")));

		editor_selection->update();

		{
			uint32_t p32 = AudioServer::get_singleton()->read_output_peak() >> 8;

			float peak = p32 == 0 ? -80 : Math::linear2db(p32 / 65535.0);

			if (peak < -80)
				peak = -80;
			float vu = audio_vu->get_val();

			if (peak > vu) {
				audio_vu->set_val(peak);
			} else {
				float new_vu = vu - get_process_delta_time() * 70.0;
				if (new_vu < -80)
					new_vu = -80;
				if (new_vu != -80 && vu != -80)
					audio_vu->set_val(new_vu);
			}
		}
	}
	if (p_what == NOTIFICATION_ENTER_TREE) {

		//MessageQueue::get_singleton()->push_call(this,"_get_scene_metadata");
		get_tree()->set_editor_hint(true);
		get_tree()->get_root()->set_as_audio_listener(false);
		get_tree()->get_root()->set_as_audio_listener_2d(false);
		get_tree()->set_auto_accept_quit(false);
		get_tree()->connect("files_dropped", this, "_dropped_files");
		//VisualServer::get_singleton()->viewport_set_hide_canvas(editor->get_scene_root()->get_viewport(),false);

		//import_monitor->scan_changes();
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		editor_data.clear_edited_scenes();
	}
	if (p_what == NOTIFICATION_READY) {

		VisualServer::get_singleton()->viewport_set_hide_scenario(get_scene_root()->get_viewport(), true);
		VisualServer::get_singleton()->viewport_set_hide_canvas(get_scene_root()->get_viewport(), true);
		VisualServer::get_singleton()->viewport_set_disable_environment(get_viewport()->get_viewport_rid(), true);

		_editor_select(EDITOR_3D);

		if (defer_load_scene != "") {

			load_scene(defer_load_scene);
			defer_load_scene = "";
		}

		if (defer_translatable != "") {

			Error ok = save_translatable_strings(defer_translatable);
			if (ok != OK)
				OS::get_singleton()->set_exit_code(255);
			defer_translatable = "";
			get_tree()->quit();
		}

		/*
		if (defer_optimize!="") {
			Error ok = save_optimized_copy(defer_optimize,defer_optimize_preset);
			defer_optimize_preset="";
			if (ok!=OK)
				OS::get_singleton()->set_exit_code(255);
			get_scene()->quit();
		}
*/

		/*  // moved to "_sources_changed"
		if (export_defer.platform!="") {

			project_export_settings->export_platform(export_defer.platform,export_defer.path,export_defer.debug,export_defer.password,true);
			export_defer.platform="";
		}

		*/
	}

	if (p_what == MainLoop::NOTIFICATION_WM_FOCUS_IN) {

		/*
		List<Ref<Resource> > cached;
		ResourceCache::get_cached_resources(&cached);

		bool changes=false;
		for(List<Ref<Resource> >::Element *E=cached.front();E;E=E->next()) {

			if (!E->get()->can_reload_from_file())
				continue;
			if (E->get()->get_path().find("::")!=-1)
				continue;
			uint64_t mt = FileAccess::get_modified_time(E->get()->get_path());
			if (mt!=E->get()->get_last_modified_time()) {
				changes=true;
				break;
			}
		}



		sources_button->get_popup()->set_item_disabled(sources_button->get_popup()->get_item_index(DEPENDENCY_UPDATE_LOCAL),!changes);
		if (changes && sources_button->get_popup()->is_item_disabled(sources_button->get_popup()->get_item_index(DEPENDENCY_UPDATE_IMPORTED))) {
			sources_button->set_icon(gui_base->get_icon("DependencyLocalChanged","EditorIcons"));
		}
*/

		if (bool(EDITOR_DEF("resources/auto_reload_modified_images", true))) {

			_menu_option_confirm(DEPENDENCY_LOAD_CHANGED_IMAGES, true);
		}

		waiting_for_sources_changed = true;
		EditorFileSystem::get_singleton()->scan_sources();
	}

	if (p_what == MainLoop::NOTIFICATION_WM_QUIT_REQUEST) {

		_menu_option_confirm(FILE_QUIT, false);
	};

	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		scene_tabs->set_tab_close_display_policy((bool(EDITOR_DEF("global/always_show_close_button_in_scene_tabs", false)) ? Tabs::CLOSE_BUTTON_SHOW_ALWAYS : Tabs::CLOSE_BUTTON_SHOW_ACTIVE_ONLY));
		property_editor->set_enable_capitalize_paths(bool(EDITOR_DEF("inspector/capitalize_properties", true)));
	}
}

void EditorNode::_fs_changed() {

	for (Set<FileDialog *>::Element *E = file_dialogs.front(); E; E = E->next()) {

		E->get()->invalidate();
	}

	for (Set<EditorFileDialog *>::Element *E = editor_file_dialogs.front(); E; E = E->next()) {

		E->get()->invalidate();
	}

	if (export_defer.platform != "") {

		project_export_settings->export_platform(export_defer.platform, export_defer.path, export_defer.debug, export_defer.password, true);
		export_defer.platform = "";
	}
}

void EditorNode::_sources_changed(bool p_exist) {

	if (p_exist && bool(EditorSettings::get_singleton()->get("import/automatic_reimport_on_sources_changed"))) {
		p_exist = false;

		List<String> changed_sources;
		EditorFileSystem::get_singleton()->get_changed_sources(&changed_sources);

		EditorProgress ep("reimport", TTR("Re-Importing"), changed_sources.size());
		int step_idx = 0;

		for (List<String>::Element *E = changed_sources.front(); E; E = E->next()) {

			ep.step(TTR("Importing:") + " " + E->get(), step_idx++);

			Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(E->get());
			ERR_CONTINUE(rimd.is_null());
			String editor = rimd->get_editor();
			if (editor.begins_with("texture_")) {
				editor = "texture"; //compatibility fix for old versions
			}
			Ref<EditorImportPlugin> eip = EditorImportExport::get_singleton()->get_import_plugin_by_name(editor);
			ERR_CONTINUE(eip.is_null());
			Error err = eip->import(E->get(), rimd);
			if (err != OK) {
				EditorNode::add_io_error("Error Re Importing:\n  " + E->get());
			}
		}

		EditorFileSystem::get_singleton()->scan_sources();
		waiting_for_sources_changed = false;

		return;
	}

	if (p_exist) {

		sources_button->set_icon(gui_base->get_icon("DependencyChanged", "EditorIcons"));
		sources_button->set_disabled(false);

	} else {

		sources_button->set_icon(gui_base->get_icon("DependencyOk", "EditorIcons"));
		sources_button->set_disabled(true);
	}

	waiting_for_sources_changed = false;
}

void EditorNode::_vp_resized() {
}

void EditorNode::_rebuild_import_menu() {
	PopupMenu *p = import_menu->get_popup();
	p->clear();
	//p->add_item(TTR("Node From Scene"), FILE_IMPORT_SUBSCENE);
	//p->add_separator();
	for (int i = 0; i < editor_import_export->get_import_plugin_count(); i++) {
		p->add_item(editor_import_export->get_import_plugin(i)->get_visible_name(), IMPORT_PLUGIN_BASE + i);
	}
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

	//file->set_current_path(current_path);

	file->popup_centered_ratio();
	current_option = RESOURCE_LOAD;
}

void EditorNode::save_resource_in_path(const Ref<Resource> &p_resource, const String &p_path) {

	editor_data.apply_changes_in_editors();
	int flg = 0;
	if (EditorSettings::get_singleton()->get("on_save/compress_binary_resources"))
		flg |= ResourceSaver::FLAG_COMPRESS;
	//if (EditorSettings::get_singleton()->get("on_save/save_paths_as_relative"))
	//	flg|=ResourceSaver::FLAG_RELATIVE_PATHS;

	String path = Globals::get_singleton()->localize_path(p_path);
	Error err = ResourceSaver::save(path, p_resource, flg | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS);

	if (err != OK) {
		accept->set_text(TTR("Error saving resource!"));
		accept->popup_centered_minsize();
		return;
	}
	//	EditorFileSystem::get_singleton()->update_file(path,p_resource->get_type());

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

		if (p_resource->is_type("Script") && (extensions[i] == "tres" || extensions[i] == "res" || extensions[i] == "xml")) {
			//this serves no purpose and confused people
			continue;
		}
		file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
		preferred.push_back(extensions[i]);
	}

	//file->set_current_path(current_path);

	if (p_at_path != String()) {

		file->set_current_dir(p_at_path);
		if (p_resource->get_path().is_resource_file()) {
			file->set_current_file(p_resource->get_path().get_file());
		} else {
			if (extensions.size()) {
				file->set_current_file("new_" + p_resource->get_type().to_lower() + "." + preferred.front()->get().to_lower());
			} else {
				file->set_current_file(String());
			}
		}
	} else if (p_resource->get_path() != "") {

		file->set_current_path(p_resource->get_path());
		if (extensions.size()) {
			String ext = p_resource->get_path().extension().to_lower();
			if (extensions.find(ext) == NULL) {
				file->set_current_path(p_resource->get_path().replacen("." + ext, "." + extensions.front()->get()));
			}
		}
	} else if (preferred.size()) {

		String existing;
		if (extensions.size()) {
			existing = "new_" + p_resource->get_type().to_lower() + "." + preferred.front()->get().to_lower();
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

void EditorNode::_dialog_display_file_error(String p_file, Error p_error) {

	if (p_error) {

		current_option = -1;
		//accept->"()->hide();
		accept->get_ok()->set_text(TTR("I see.."));

		switch (p_error) {

			case ERR_FILE_CANT_WRITE: {

				accept->set_text(TTR("Can't open file for writing:") + " " + p_file.extension());
			} break;
			case ERR_FILE_UNRECOGNIZED: {

				accept->set_text(TTR("Requested file format unknown:") + " " + p_file.extension());
			} break;
			default: {

				accept->set_text(TTR("Error while saving."));
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

bool EditorNode::_find_and_save_resource(RES res, Map<RES, bool> &processed, int32_t flags) {

	if (res.is_null())
		return false;

	if (processed.has(res)) {

		return processed[res];
	}

	bool changed = res->is_edited();
	res->set_edited(false);

	bool subchanged = _find_and_save_edited_subresources(res.ptr(), processed, flags);

	//	print_line("checking if edited: "+res->get_type()+" :: "+res->get_name()+" :: "+res->get_path()+" :: "+itos(changed)+" :: SR "+itos(subchanged));

	if (res->get_path().is_resource_file()) {
		if (changed || subchanged) {
			//save
			print_line("Also saving modified external resource: " + res->get_path());
			Error err = ResourceSaver::save(res->get_path(), res, flags);
		}
		processed[res] = false; //because it's a file
		return false;
	} else {

		processed[res] = changed;
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

					//_find_resources(v);
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

	if (p_node->is_type("Viewport") || (p_node != editor_data.get_edited_scene_root() && p_node->get_owner() != editor_data.get_edited_scene_root()))
		return;

	if (p_node->is_type("CanvasItem"))
		count_2d++;
	else if (p_node->is_type("Spatial"))
		count_3d++;

	for (int i = 0; i < p_node->get_child_count(); i++)
		_find_node_types(p_node->get_child(i), count_2d, count_3d);
}

void EditorNode::_save_scene_with_preview(String p_file) {

	int c2d = 0;
	int c3d = 0;

	EditorProgress save("save", TTR("Saving Scene"), 4);
	save.step(TTR("Analyzing"), 0);
	_find_node_types(editor_data.get_edited_scene_root(), c2d, c3d);

	RID viewport;
	bool is2d;
	if (c3d < c2d) {
		viewport = scene_root->get_viewport();
		is2d = true;
	} else {
		viewport = SpatialEditor::get_singleton()->get_editor_viewport(0)->get_viewport_node()->get_viewport();
		is2d = false;
	}
	save.step(TTR("Creating Thumbnail"), 1);
	//current view?
	int screen = -1;
	for (int i = 0; i < editor_table.size(); i++) {
		if (editor_plugin_screen == editor_table[i]) {
			screen = i;
			break;
		}
	}

	_editor_select(is2d ? EDITOR_2D : EDITOR_3D);

	VS::get_singleton()->viewport_queue_screen_capture(viewport);
	save.step(TTR("Creating Thumbnail"), 2);
	save.step(TTR("Creating Thumbnail"), 3);
	Image img = VS::get_singleton()->viewport_get_screen_capture(viewport);
	int preview_size = EditorSettings::get_singleton()->get("file_dialog/thumbnail_size");
	preview_size *= EDSCALE;
	int width, height;
	if (img.get_width() > preview_size && img.get_width() >= img.get_height()) {

		width = preview_size;
		height = img.get_height() * preview_size / img.get_width();
	} else if (img.get_height() > preview_size && img.get_height() >= img.get_width()) {

		height = preview_size;
		width = img.get_width() * preview_size / img.get_height();
	} else {

		width = img.get_width();
		height = img.get_height();
	}

	img.convert(Image::FORMAT_RGB);
	img.resize(width, height);

	String pfile = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp/last_scene_preview.png");
	img.save_png(pfile);
	Vector<uint8_t> imgdata = FileAccess::get_file_as_array(pfile);

	//print_line("img data is "+itos(imgdata.size()));

	if (editor_data.get_edited_scene_import_metadata().is_null())
		editor_data.set_edited_scene_import_metadata(Ref<ResourceImportMetadata>(memnew(ResourceImportMetadata)));
	editor_data.get_edited_scene_import_metadata()->set_option("thumbnail", imgdata);

	//tamanio tel thumbnail
	if (screen != -1) {
		_editor_select(screen);
	}
	save.step(TTR("Saving Scene"), 4);
	_save_scene(p_file);
}

void EditorNode::_save_scene(String p_file, int idx) {

	Node *scene = editor_data.get_edited_scene_root(idx);

	if (!scene) {

		current_option = -1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text("This operation can't be done without a tree root.");
		accept->popup_centered_minsize();
		return;
	}

	editor_data.apply_changes_in_editors();

	_set_scene_metadata(p_file, idx);

	Ref<PackedScene> sdata;

	if (ResourceCache::has(p_file)) {
		// something may be referencing this resource and we are good with that.
		// we must update it, but also let the previous scene state go, as
		// old version still work for referencing changes in instanced or inherited scenes

		sdata = Ref<PackedScene>(ResourceCache::get(p_file)->cast_to<PackedScene>());
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
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text(TTR("Couldn't save scene. Likely dependencies (instances) couldn't be satisfied."));
		accept->popup_centered_minsize();
		return;
	}

	// force creation of node path cache
	// (hacky but needed for the tree to update properly)
	Node *dummy_scene = sdata->instance(true);
	memdelete(dummy_scene);

	sdata->set_import_metadata(editor_data.get_edited_scene_import_metadata(idx));
	int flg = 0;
	if (EditorSettings::get_singleton()->get("on_save/compress_binary_resources"))
		flg |= ResourceSaver::FLAG_COMPRESS;
	//if (EditorSettings::get_singleton()->get("on_save/save_paths_as_relative"))
	//	flg|=ResourceSaver::FLAG_RELATIVE_PATHS;
	flg |= ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;

	err = ResourceSaver::save(p_file, sdata, flg);

	Map<RES, bool> processed;
	_save_edited_subresources(scene, processed, flg);
	editor_data.save_editor_external_data();
	if (err == OK) {
		scene->set_filename(Globals::get_singleton()->localize_path(p_file));
		//EditorFileSystem::get_singleton()->update_file(p_file,sdata->get_type());
		if (idx < 0 || idx == editor_data.get_edited_scene())
			set_current_version(editor_data.get_undo_redo().get_version());
		else
			editor_data.set_edited_scene_version(0, idx);
		_update_title();
		_update_scene_tabs();
	} else {

		_dialog_display_file_error(p_file, err);
	}
};

void EditorNode::_import_action(const String &p_action) {
#if 0
	import_confirmation->hide();

	if (p_action=="re-import") {
		_import(_tmp_import_path);
	}
	if (p_action=="update") {

		Node *src = EditorImport::import_scene(_tmp_import_path);

		if (!src) {

			current_option=-1;
			//accept->get_cancel()->hide();
			accept->get_ok()->set_text("Ugh");
			accept->set_text("Error importing scene.");
			accept->popup_centered(Size2(300,70));
			return;
		}

		//as soon as the scene is imported, version hashes must be generated for comparison against saved scene
		EditorImport::generate_version_hashes(src);


		Node *dst = SceneLoader::load(editor_data.get_imported_scene(Globals::get_singleton()->localize_path(_tmp_import_path)));

		if (!dst) {

			memdelete(src);
			//accept->get_cancel()->hide();
			accept->get_ok()->set_text("Ugh");
			accept->set_text("Error load scene to update.");
			accept->popup_centered(Size2(300,70));
			return;
		}

		List<EditorImport::Conflict> conflicts;
		EditorImport::check_conflicts(src,dst,&conflicts);

		bool conflicted=false;
		for (List<EditorImport::Conflict>::Element *E=conflicts.front();E;E=E->next()) {


			if (E->get().status==EditorImport::Conflict::STATUS_CONFLICT) {

				conflicted=true;
				break;
			}
		}

		if (conflicted) {
			import_conflicts_dialog->popup(src,dst,conflicts);
			return;
		}

		_import_with_conflicts(src,dst,conflicts);
		//not conflicted, just reimport!

	}
#endif
}

void EditorNode::_import(const String &p_file) {

#if 0
	Node *new_scene = EditorImport::import_scene(p_file);

	if (!new_scene) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("Ugh");
		accept->set_text("Error importing scene.");
		accept->popup_centered(Size2(300,70));
		return;
	}

	//as soon as the scene is imported, version hashes must be generated for comparison against saved scene
	EditorImport::generate_version_hashes(new_scene);

	Node *old_scene = edited_scene;
	_hide_top_editors();
	set_edited_scene(NULL);
	editor_data.clear_editor_states();
	if (old_scene) {
		memdelete(old_scene);
	}

	set_edited_scene(new_scene);
	scene_tree_dock->set_selected(new_scene);
	//_get_scene_metadata();

	editor_data.get_undo_redo().clear_history();
	saved_version=editor_data.get_undo_redo().get_version();
	_update_title();

#endif
}

void EditorNode::_dialog_action(String p_file) {

	switch (current_option) {

		case RESOURCE_LOAD: {

			RES res = ResourceLoader::load(p_file);
			if (res.is_null()) {

				current_option = -1;
				//accept->get_cancel()->hide();
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

			Globals::get_singleton()->set("application/main_scene", p_file);
			Globals::get_singleton()->set_persisting("application/main_scene", true);
			Globals::get_singleton()->save();
			//would be nice to show the project manager opened with the highlighted field..
			_run(false, ""); // automatically run the project
		} break;
		case FILE_SAVE_OPTIMIZED: {

		} break;
		case FILE_RUN_SCRIPT: {

			Ref<Script> scr = ResourceLoader::load(p_file, "Script", true);
			if (scr.is_null()) {
				add_io_error("Script Failed to Load:\n" + p_file);
				return;
			}
			if (!scr->is_tool()) {

				add_io_error("Script is not tool, will not be able to run:\n" + p_file);
				return;
			}

			Ref<EditorScript> es = memnew(EditorScript);
			es->set_script(scr.get_ref_ptr());
			es->set_editor(this);
			es->_run();

			get_undo_redo()->clear_history();
		} break;
		case FILE_DUMP_STRINGS: {

			save_translatable_strings(p_file);

		} break;

		case FILE_SAVE_SCENE:
		case FILE_SAVE_AS_SCENE: {

			if (file->get_mode() == EditorFileDialog::MODE_SAVE_FILE) {

				//_save_scene(p_file);
				_save_scene_with_preview(p_file);
			}

		} break;

		case FILE_SAVE_AND_RUN: {
			if (file->get_mode() == EditorFileDialog::MODE_SAVE_FILE) {

				//_save_scene(p_file);
				_save_scene_with_preview(p_file);
				_run(false);
			}
		} break;

		case FILE_EXPORT_MESH_LIBRARY: {

			Ref<MeshLibrary> ml;
			if (file_export_lib_merge->is_pressed() && FileAccess::exists(p_file)) {
				ml = ResourceLoader::load(p_file, "MeshLibrary");

				if (ml.is_null()) {
					current_option = -1;
					//accept->get_cancel()->hide();
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
						//accept->get_cancel()->hide();
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

		case SETTINGS_LOAD_EXPORT_TEMPLATES: {

			FileAccess *fa = NULL;
			zlib_filefunc_def io = zipio_create_io_from_file(&fa);

			unzFile pkg = unzOpen2(p_file.utf8().get_data(), &io);
			if (!pkg) {

				current_option = -1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("Can't open export templates zip."));
				accept->popup_centered_minsize();
				return;
			}
			int ret = unzGoToFirstFile(pkg);

			int fc = 0; //count them

			while (ret == UNZ_OK) {
				fc++;
				ret = unzGoToNextFile(pkg);
			}

			ret = unzGoToFirstFile(pkg);

			EditorProgress p("ltask", TTR("Loading Export Templates"), fc);

			fc = 0;

			while (ret == UNZ_OK) {

				//get filename
				unz_file_info info;
				char fname[16384];
				ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

				String file = fname;

				Vector<uint8_t> data;
				data.resize(info.uncompressed_size);

				//read
				ret = unzOpenCurrentFile(pkg);
				ret = unzReadCurrentFile(pkg, data.ptr(), data.size());
				unzCloseCurrentFile(pkg);

				print_line(fname);
				//for(int i=0;i<512;i++) {
				//	print_line(itos(data[i]));
				//}

				file = file.get_file();

				p.step(TTR("Importing:") + " " + file, fc);

				FileAccess *f = FileAccess::open(EditorSettings::get_singleton()->get_settings_path() + "/templates/" + file, FileAccess::WRITE);

				ERR_CONTINUE(!f);
				f->store_buffer(data.ptr(), data.size());

				memdelete(f);

				ret = unzGoToNextFile(pkg);
				fc++;
			}

			unzClose(pkg);

		} break;
		case RESOURCE_SAVE:
		case RESOURCE_SAVE_AS: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!current_obj->cast_to<Resource>())

			RES current_res = RES(current_obj->cast_to<Resource>());

			save_resource_in_path(current_res, p_file);

		} break;
		case SETTINGS_LAYOUT_SAVE: {

			if (p_file.empty())
				return;

			Ref<ConfigFile> config;
			config.instance();
			Error err = config->load(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts.cfg"));

			if (err == ERR_CANT_OPEN) {
				config.instance(); // new config
			} else if (err != OK) {
				show_warning(TTR("Error trying to save layout!"));
				return;
			}

			_save_docks_to_config(config, p_file);

			config->save(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts.cfg"));

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
			Error err = config->load(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts.cfg"));

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

			config->save(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts.cfg"));

			layout_dialog->hide();
			_update_layouts_menu();

			if (p_file == "Default") {
				show_warning(TTR("Restored default layout to base settings."));
			}

		} break;
		default: { //save scene?

			if (file->get_mode() == EditorFileDialog::MODE_SAVE_FILE) {

				//_save_scene(p_file);
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

	uint32_t id = p_object->get_instance_ID();
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
		if (gui_base->has_icon(obj->get_type(), "EditorIcons"))
			icon = gui_base->get_icon(obj->get_type(), "EditorIcons");
		else
			icon = base_icon;

		String text;
		if (obj->cast_to<Resource>()) {
			Resource *r = obj->cast_to<Resource>();
			if (r->get_path().is_resource_file())
				text = r->get_path().get_file();
			else if (r->get_name() != String()) {
				text = r->get_name();
			} else {
				text = r->get_type();
			}
		} else if (obj->cast_to<Node>()) {
			text = obj->cast_to<Node>()->get_name();
		} else {
			text = obj->get_type();
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

void EditorNode::_imported(Node *p_node) {

	/*
	Node *scene = editor_data.get_edited_scene_root();
	add_edited_scene(p_node);

	if (scene) {
		String path = scene->get_filename();
		p_node->set_filename(path);
		memdelete(scene);
	}
*/
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

void EditorNode::_edit_current() {

	uint32_t current = editor_history.get_current();
	Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

	property_back->set_disabled(editor_history.is_at_begining());
	property_forward->set_disabled(editor_history.is_at_end());

	this->current = current_obj;
	editor_path->update_path();

	if (!current_obj) {

		scene_tree_dock->set_selected(NULL);
		property_editor->edit(NULL);
		node_dock->set_node(NULL);
		object_menu->set_disabled(true);

		_display_top_editors(false);

		return;
	}

	object_menu->set_disabled(true);

	bool is_resource = current_obj->is_type("Resource");
	bool is_node = current_obj->is_type("Node");
	resource_save_button->set_disabled(!is_resource);

	if (is_resource) {

		Resource *current_res = current_obj->cast_to<Resource>();
		ERR_FAIL_COND(!current_res);
		scene_tree_dock->set_selected(NULL);
		property_editor->edit(current_res);
		node_dock->set_node(NULL);
		object_menu->set_disabled(false);

		//resources_dock->add_resource(Ref<Resource>(current_res));

		//top_pallete->set_current_tab(1);
	} else if (is_node) {

		Node *current_node = current_obj->cast_to<Node>();
		ERR_FAIL_COND(!current_node);
		ERR_FAIL_COND(!current_node->is_inside_tree());

		property_editor->edit(current_node);
		node_dock->set_node(current_node);
		scene_tree_dock->set_selected(current_node);
		object_menu->get_popup()->clear();

		//top_pallete->set_current_tab(0);

	} else {

		property_editor->edit(current_obj);
		node_dock->set_node(NULL);
		//scene_tree_dock->set_selected(current_node);
		//object_menu->get_popup()->clear();
	}

	/* Take care of PLUGIN EDITOR */

	EditorPlugin *main_plugin = editor_data.get_editor(current_obj);

	if (main_plugin) {

		if (main_plugin != editor_plugin_screen) {

			// update screen main_plugin

			if (!changing_scene) {

				if (editor_plugin_screen)
					editor_plugin_screen->make_visible(false);
				editor_plugin_screen = main_plugin;
				editor_plugin_screen->edit(current_obj);

				editor_plugin_screen->make_visible(true);

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
	/*
	if (!plugin || plugin->has_main_screen()) {
		// remove the OVER plugin if exists
		if (editor_plugin_over)
			editor_plugin_over->make_visible(false);
		editor_plugin_over=NULL;
	}
*/
	/* Take care of OBJECT MENU */

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
		p->add_icon_shortcut(gui_base->get_icon("Help", "EditorIcons"), ED_SHORTCUT("property_editor/open_help", TTR("Open in Help")), OBJECT_REQUEST_HELP);
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

	//p->add_separator();
	//p->add_item("All Methods",OBJECT_CALL_METHOD);

	update_keying();
}

void EditorNode::_resource_created() {

	Object *c = create_dialog->instance_selected();

	ERR_FAIL_COND(!c);
	Resource *r = c->cast_to<Resource>();
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
	//pause_button->set_pressed(false);
	play_scene_button->set_pressed(false);
	play_scene_button->set_icon(gui_base->get_icon("PlayScene", "EditorIcons"));
	play_custom_scene_button->set_pressed(false);
	play_custom_scene_button->set_icon(gui_base->get_icon("PlayCustom", "EditorIcons"));

	String current_filename;
	String run_filename;
	String args;

	if (p_current || (editor_data.get_edited_scene_root() && p_custom == editor_data.get_edited_scene_root()->get_filename())) {

		Node *scene = editor_data.get_edited_scene_root();

		if (!scene) {
			current_option = -1;
			//accept->get_cancel()->hide();
			accept->get_ok()->set_text(TTR("I see.."));
			accept->set_text(TTR("There is no defined scene to run."));
			accept->popup_centered_minsize();
			return;
		}

		if (scene->get_filename() == "") {
			current_option = -1;
			//accept->get_cancel()->hide();
			/**/
			_menu_option_confirm(FILE_SAVE_BEFORE_RUN, false);
			return;
		}

		if (run_settings_dialog->get_run_mode() == RunSettingsDialog::RUN_LOCAL_SCENE) {

			run_filename = scene->get_filename();
		} else {
			current_filename = scene->get_filename();
		}

	} else if (p_custom != "") {

		run_filename = p_custom;
	}

	if (run_filename == "") {

		//evidently, run the scene
		run_filename = GLOBAL_DEF("application/main_scene", "");
		if (run_filename == "") {

			current_option = -1;
			//accept->get_cancel()->hide();
			pick_main_scene->set_text(TTR("No main scene has ever been defined, select one?\nYou can change it later in later in \"Project Settings\" under the 'application' category."));
			pick_main_scene->popup_centered_minsize();
			return;
		}

		if (!FileAccess::exists(run_filename)) {

			current_option = -1;
			//accept->get_cancel()->hide();
			pick_main_scene->set_text(vformat(TTR("Selected scene '%s' does not exist, select a valid one?\nYou can change it later in \"Project Settings\" under the 'application' category."), run_filename));
			pick_main_scene->popup_centered_minsize();
			return;
		}

		if (ResourceLoader::get_resource_type(run_filename) != "PackedScene") {

			current_option = -1;
			//accept->get_cancel()->hide();
			pick_main_scene->set_text(vformat(TTR("Selected scene '%s' is not a scene file, select a valid one?\nYou can change it later in \"Project Settings\" under the 'application' category."), run_filename));
			pick_main_scene->popup_centered_minsize();
			return;
		}
	}

	if (bool(EDITOR_DEF("run/auto_save_before_running", true))) {

		if (unsaved_cache) {

			Node *scene = editor_data.get_edited_scene_root();

			if (scene) { //only autosave if there is a scene obviously

				if (scene->get_filename() == "") {

					current_option = -1;
					//accept->get_cancel()->hide();
					accept->get_ok()->set_text(TTR("I see.."));
					accept->set_text(TTR("Current scene was never saved, please save it prior to running."));
					accept->popup_centered_minsize();
					return;
				}

				//_save_scene(scene->get_filename());
				_save_scene_with_preview(scene->get_filename());
			}
		}
		_menu_option(FILE_SAVE_ALL_SCENES);
		editor_data.save_editor_external_data();
	}

	if (bool(EDITOR_DEF("run/always_clear_output_on_play", true))) {
		log->clear();
	}

	if (bool(EDITOR_DEF("run/always_open_output_on_play", true))) {
		make_bottom_panel_item_visible(log);
	}

	List<String> breakpoints;
	editor_data.get_editor_breakpoints(&breakpoints);

	args = Globals::get_singleton()->get("editor/main_run_args");

	Error error = editor_run.run(run_filename, args, breakpoints, current_filename);

	if (error != OK) {

		current_option = -1;
		//confirmation->get_cancel()->hide();
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
		run_custom_filename = run_filename;
		play_custom_scene_button->set_pressed(true);
		play_custom_scene_button->set_icon(gui_base->get_icon("Reload", "EditorIcons"));
	} else {
		play_button->set_pressed(true);
		play_button->set_icon(gui_base->get_icon("Reload", "EditorIcons"));
	}

	_playing_edited = p_current;
}

void EditorNode::_cleanup_scene() {

#if 0
	Node *scene = editor_data.get_edited_scene_root();
	editor_selection->clear();
	editor_data.clear_editor_states();
	editor_history.clear();
	_hide_top_editors();
	animation_editor->cleanup();
	property_editor->edit(NULL);
	resources_dock->cleanup();
	scene_import_metadata.unref();
	//set_edited_scene(NULL);
	if (scene) {
		if (scene->get_filename()!="") {
			previous_scenes.push_back(scene->get_filename());
		}

		memdelete(scene);
	}
	editor_data.get_undo_redo().clear_history();
	saved_version=editor_data.get_undo_redo().get_version();
	run_settings_dialog->set_run_mode(0);
	run_settings_dialog->set_custom_arguments("-l $scene");

	List<Ref<Resource> > cached;
	ResourceCache::get_cached_resources(&cached);

	for(List<Ref<Resource> >::Element *E=cached.front();E;E=E->next()) {

		String path = E->get()->get_path();
		if (path.is_resource_file()) {
			ERR_PRINT(("Stray resource not cleaned:"+path).utf8().get_data());
		}

	}

	_update_title();
#endif
}

void EditorNode::_menu_option_confirm(int p_option, bool p_confirmed) {

	//print_line("option "+itos(p_option)+" confirm "+itos(p_confirmed));
	if (!p_confirmed) //this may be a hack..
		current_option = (MenuOptions)p_option;

	switch (p_option) {
		case FILE_NEW_SCENE: {

			// TODO: Drop such obsolete commented code
			/*
			if (!p_confirmed) {
				confirmation->get_ok()->set_text("Yes");
				//confirmation->get_cancel()->show();
				confirmation->set_text("Start a New Scene? (Current will be lost)");
				confirmation->popup_centered_minsize();
				break;
			}*/

			int idx = editor_data.add_edited_scene(-1);
			_scene_tab_changed(idx);
			editor_data.clear_editor_states();

			//_cleanup_scene();

		} break;
		case FILE_NEW_INHERITED_SCENE:
		case FILE_OPEN_SCENE: {

			//print_tree();
			file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
			//not for now?
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {

				file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			//file->set_current_path(current_path);
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
		case FILE_RUN_SCRIPT: {

			file_script->popup_centered_ratio();
		} break;
		case FILE_OPEN_PREV: {

			if (previous_scenes.empty())
				break;
			opening_prev = true;
			open_request(previous_scenes.back()->get());

		} break;
		case FILE_CLOSE: {

			if (!p_confirmed && unsaved_cache) {
				confirmation->get_ok()->set_text(TTR("Yes"));
				//confirmation->get_cancel()->show();
				confirmation->set_text(TTR("Close scene? (Unsaved changes will be lost)"));
				confirmation->popup_centered_minsize();
				break;
			}

			_remove_edited_scene();

		} break;
		case SCENE_TAB_CLOSE: {
			_remove_scene(tab_closing);
			_update_scene_tabs();
			current_option = -1;
		} break;
		case FILE_SAVE_SCENE: {

			Node *scene = editor_data.get_edited_scene_root();
			if (scene && scene->get_filename() != "") {

				// save in background if in the script editor
				if (_get_current_main_editor() == EDITOR_SCRIPT) {
					_save_scene(scene->get_filename());
				} else {
					_save_scene_with_preview(scene->get_filename());
				}
				return;
			};
			// fallthrough to save_as
		};
		case FILE_SAVE_AS_SCENE: {

			Node *scene = editor_data.get_edited_scene_root();

			if (!scene) {

				current_option = -1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text("This operation can't be done without a tree root.");
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
					String ext = scene->get_filename().extension().to_lower();
					if (extensions.find(ext) == NULL) {
						file->set_current_path(scene->get_filename().replacen("." + ext, "." + extensions.front()->get()));
					}
				}
			} else {

				String existing;
				if (extensions.size()) {
					String root_name(get_edited_scene()->get_name());
					existing = root_name + "." + extensions.front()->get().to_lower();
				}
				file->set_current_path(existing);
			}
			file->popup_centered_ratio();
			file->set_title(TTR("Save Scene As.."));

		} break;

		case FILE_SAVE_ALL_SCENES: {
			for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
				Node *scene = editor_data.get_edited_scene_root(i);
				if (scene && scene->get_filename() != "") {
					// save in background if in the script editor
					if (i != editor_data.get_edited_scene() || _get_current_main_editor() == EDITOR_SCRIPT) {
						_save_scene(scene->get_filename(), i);
					} else {
						_save_scene_with_preview(scene->get_filename());
					}
				} // else: ignore new scenes
			}
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

		case FILE_DUMP_STRINGS: {

			Node *scene = editor_data.get_edited_scene_root();

			if (!scene) {

				current_option = -1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text("This operation can't be done without a tree root.");
				accept->popup_centered_minsize();
				break;
			}

			String cpath;
			if (scene->get_filename() != "") {
				cpath = scene->get_filename();

				String fn = cpath.substr(0, cpath.length() - cpath.extension().size());
				String ext = cpath.extension();
				cpath = fn + ".pot";

			} else {
				current_option = -1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("Please save the scene first."));
				accept->popup_centered_minsize();
				break;
			}

			file->set_mode(EditorFileDialog::MODE_SAVE_FILE);

			file->set_current_path(cpath);
			file->set_title(TTR("Save Translatable Strings"));
			file->popup_centered_ratio();

		} break;
		case FILE_SAVE_OPTIMIZED: {
#if 0
			Node *scene = editor_data.get_edited_scene_root();
			if (!scene) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a tree root.");
				accept->popup_centered(Size2(300,70));
				break;
			}



			//file->set_current_path(current_path);

			String cpath;
			if (scene->get_filename()!="") {
				cpath = scene->get_filename();

				String fn = cpath.substr(0,cpath.length() - cpath.extension().size());
				String ext=cpath.extension();
				cpath=fn+".optimized.scn";
				optimized_save->set_optimized_scene(cpath);
				optimized_save->popup_centered(Size2(500,143));
			} else {
				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("Please save the scene first.");
				accept->popup_centered(Size2(300,70));
				break;

			}
#endif
		} break;

		case FILE_EXPORT_PROJECT: {

			project_export_settings->popup_export();
			/*
			String target = export_db->get_current_platform();
			Ref<EditorExporter> exporter = export_db->get_exporter(target);
			if (exporter.is_null()) {
				accept->set_text("No exporter for platform '"+target+"' yet.");
				accept->popup_centered(Size2(300,70));
				return;
			}

			String extension = exporter->get_binary_extension();
			print_line("for target: "+target+" extension: "+extension);
			file_export_password->set_editable( exporter->requieres_password(file_export_check->is_pressed()));

			file_export->clear_filters();
			if (extension!="") {
				file_export->add_filter("*."+extension);
			}
			file_export->popup_centered_ratio();*/
		} break;

		case FILE_EXPORT_MESH_LIBRARY: {

			if (!editor_data.get_edited_scene_root()) {

				current_option = -1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text("This operation can't be done without a scene.");
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
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text("This operation can't be done without a selected node.");
				accept->popup_centered_minsize();
				break;
			}

			scene_tree_dock->import_subscene();

		} break;

		case FILE_QUIT: {

			if (!p_confirmed) {

				confirmation->get_ok()->set_text(TTR("Quit"));
				//confirmation->get_cancel()->show();
				confirmation->set_text(TTR("Exit the editor?"));
				confirmation->popup_centered(Size2(180, 70) * EDSCALE);
				break;
			}

			_menu_option_confirm(RUN_STOP, true);
			exiting = true;
			get_tree()->quit();

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

			if (OS::get_singleton()->get_mouse_button_state())
				break; // can't undo while mouse buttons are pressed

			String action = editor_data.get_undo_redo().get_current_action_name();
			if (action != "")
				log->add_message("UNDO: " + action);

			editor_data.get_undo_redo().undo();
		} break;
		case EDIT_REDO: {

			if (OS::get_singleton()->get_mouse_button_state())
				break; // can't redo while mouse buttons are pressed

			editor_data.get_undo_redo().redo();
			String action = editor_data.get_undo_redo().get_current_action_name();
			if (action != "")
				log->add_message("REDO: " + action);

		} break;
		case TOOLS_ORPHAN_RESOURCES: {

			orphan_resources->show();
		} break;
		case TOOLS_EXPORT_GODOT3: {

			export_godot3_dialog->popup_centered_ratio();
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
			editor_data.move_edited_scene_to_index(cur_idx);
			get_undo_redo()->clear_history();
			scene_tabs->set_current_tab(cur_idx);

		} break;

#if 0
		case NODE_EXTERNAL_INSTANCE: {


			if (!edited_scene) {

				current_option=-1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a selected node.");
				accept->popup_centered(Size2(300,70));
				break;
			}

			Node *parent = scene_tree_editor->get_selected();

			if (!parent) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a selected node.");
				accept->popup_centered(Size2(300,70));
				break;
			}

			Node*instanced_scene=SceneLoader::load(external_file,true);

			if (!instanced_scene) {

				current_option=-1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text("Ugh");
				accept->set_text("Error loading scene from "+external_file);
				accept->popup_centered(Size2(300,70));
				return;
			}

			instanced_scene->generate_instance_state();
			instanced_scene->set_filename( Globals::get_singleton()->localize_path(external_file) );

			editor_data.get_undo_redo().create_action("Instance Scene");
			editor_data.get_undo_redo().add_do_method(parent,"add_child",instanced_scene);
			editor_data.get_undo_redo().add_do_method(instanced_scene,"set_owner",edited_scene);
			editor_data.get_undo_redo().add_do_reference(instanced_scene);
			editor_data.get_undo_redo().add_undo_method(parent,"remove_child",instanced_scene);
			editor_data.get_undo_redo().commit_action();

//			parent->add_child(instanced_scene);
//			instanced_scene->set_owner(edited_scene);
			_last_instanced_scene=instanced_scene;

		} break;
#endif
		case RESOURCE_NEW: {

			create_dialog->popup(true);
		} break;
		case RESOURCE_LOAD: {

			open_resource();
		} break;
		case RESOURCE_SAVE: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!current_obj->cast_to<Resource>())

			RES current_res = RES(current_obj->cast_to<Resource>());

			save_resource(current_res);

		} break;
		case RESOURCE_SAVE_AS: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!current_obj->cast_to<Resource>())

			RES current_res = RES(current_obj->cast_to<Resource>());

			save_resource_as(current_res);

		} break;
		case RESOURCE_UNREF: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!current_obj->cast_to<Resource>())

			RES current_res = RES(current_obj->cast_to<Resource>());
			current_res->set_path("");
			_edit_current();
		} break;
		case RESOURCE_COPY: {

			uint32_t current = editor_history.get_current();
			Object *current_obj = current > 0 ? ObjectDB::get_instance(current) : NULL;

			ERR_FAIL_COND(!current_obj->cast_to<Resource>())

			RES current_res = RES(current_obj->cast_to<Resource>());

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
				emit_signal("request_help", current->get_type());
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
		case OBJECT_CALL_METHOD: {

			editor_data.apply_changes_in_editors();
			;
			call_dialog->set_object(current);
			call_dialog->popup_centered_ratio();
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
			//pause_button->set_pressed(false);
			if (bool(EDITOR_DEF("run/always_close_output_on_stop", false))) {
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
			_menu_option_confirm(RUN_STOP, true);
			_run(true);

		} break;
		case RUN_PLAY_NATIVE: {

			bool autosave = EDITOR_DEF("run/auto_save_before_running", true);
			if (autosave) {
				_menu_option_confirm(FILE_SAVE_ALL_SCENES, false);
			}
			if (run_native->is_deploy_debug_remote_enabled()) {
				_menu_option_confirm(RUN_STOP, true);
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
		case RUN_PROJECT_MANAGER: {

			if (!p_confirmed) {
				confirmation->get_ok()->set_text(TTR("Yes"));
				confirmation->set_text(TTR("Open Project Manager? \n(Unsaved changes will be lost)"));
				confirmation->popup_centered_minsize();
				break;
			}

			_menu_option_confirm(RUN_STOP, true);
			exiting = true;
			get_tree()->quit();
			String exec = OS::get_singleton()->get_executable_path();

			List<String> args;
			args.push_back("-path");
			args.push_back(exec.get_base_dir());
			args.push_back("-pm");

			OS::ProcessID pid = 0;
			Error err = OS::get_singleton()->execute(exec, args, false, &pid);
			ERR_FAIL_COND(err);
		} break;
		case RUN_FILE_SERVER: {

			//file_server
			bool ischecked = debug_button->get_popup()->is_item_checked(debug_button->get_popup()->get_item_index(RUN_FILE_SERVER));

			if (ischecked) {
				file_server->stop();
				run_native->set_deploy_dumb(false);
				//debug_button->set_icon(gui_base->get_icon("FileServer","EditorIcons"));
				//debug_button->get_popup()->set_item_text( debug_button->get_popup()->get_item_index(RUN_FILE_SERVER),"Enable File Server");
			} else {
				file_server->start();
				run_native->set_deploy_dumb(true);
				//debug_button->set_icon(gui_base->get_icon("FileServerActive","EditorIcons"));
				//debug_button->get_popup()->set_item_text( debug_button->get_popup()->get_item_index(RUN_FILE_SERVER),"Disable File Server");
			}

			debug_button->get_popup()->set_item_checked(debug_button->get_popup()->get_item_index(RUN_FILE_SERVER), !ischecked);

		} break;
		case RUN_LIVE_DEBUG: {

			bool ischecked = debug_button->get_popup()->is_item_checked(debug_button->get_popup()->get_item_index(RUN_LIVE_DEBUG));

			debug_button->get_popup()->set_item_checked(debug_button->get_popup()->get_item_index(RUN_LIVE_DEBUG), !ischecked);
			ScriptEditor::get_singleton()->get_debugger()->set_live_debugging(!ischecked);
		} break;

		/*case RUN_DEPLOY_DUMB_CLIENTS: {

			bool ischecked = debug_button->get_popup()->is_item_checked( debug_button->get_popup()->get_item_index(RUN_DEPLOY_DUMB_CLIENTS));
			debug_button->get_popup()->set_item_checked( debug_button->get_popup()->get_item_index(RUN_DEPLOY_DUMB_CLIENTS),!ischecked);
			run_native->set_deploy_dumb(!ischecked);

		} break;*/
		case RUN_DEPLOY_REMOTE_DEBUG: {

			bool ischecked = debug_button->get_popup()->is_item_checked(debug_button->get_popup()->get_item_index(RUN_DEPLOY_REMOTE_DEBUG));
			debug_button->get_popup()->set_item_checked(debug_button->get_popup()->get_item_index(RUN_DEPLOY_REMOTE_DEBUG), !ischecked);
			run_native->set_deploy_debug_remote(!ischecked);

		} break;
		case RUN_DEBUG_COLLISONS: {

			bool ischecked = debug_button->get_popup()->is_item_checked(debug_button->get_popup()->get_item_index(RUN_DEBUG_COLLISONS));
			debug_button->get_popup()->set_item_checked(debug_button->get_popup()->get_item_index(RUN_DEBUG_COLLISONS), !ischecked);
			run_native->set_debug_collisions(!ischecked);
			editor_run.set_debug_collisions(!ischecked);
		} break;
		case RUN_DEBUG_NAVIGATION: {

			bool ischecked = debug_button->get_popup()->is_item_checked(debug_button->get_popup()->get_item_index(RUN_DEBUG_NAVIGATION));
			debug_button->get_popup()->set_item_checked(debug_button->get_popup()->get_item_index(RUN_DEBUG_NAVIGATION), !ischecked);
			run_native->set_debug_navigation(!ischecked);
			editor_run.set_debug_navigation(!ischecked);
		} break;
		case RUN_RELOAD_SCRIPTS: {

			bool ischecked = debug_button->get_popup()->is_item_checked(debug_button->get_popup()->get_item_index(RUN_RELOAD_SCRIPTS));
			debug_button->get_popup()->set_item_checked(debug_button->get_popup()->get_item_index(RUN_RELOAD_SCRIPTS), !ischecked);

			ScriptEditor::get_singleton()->set_live_auto_reload_running_scripts(!ischecked);
		} break;
		case SETTINGS_UPDATE_ALWAYS: {

			update_menu->get_popup()->set_item_checked(0, true);
			update_menu->get_popup()->set_item_checked(1, false);
			OS::get_singleton()->set_low_processor_usage_mode(false);
		} break;
		case SETTINGS_UPDATE_CHANGES: {

			update_menu->get_popup()->set_item_checked(0, false);
			update_menu->get_popup()->set_item_checked(1, true);
			OS::get_singleton()->set_low_processor_usage_mode(true);
		} break;
		case SETTINGS_UPDATE_SPINNER_HIDE: {
			update_menu->set_icon(gui_base->get_icon("Collapse", "EditorIcons"));
			update_menu->get_popup()->toggle_item_checked(3);
		} break;
		case SETTINGS_PREFERENCES: {

			settings_config_dialog->popup_edit_settings();
		} break;
		case SETTINGS_OPTIMIZED_PRESETS: {

			//optimized_presets->popup_centered_ratio();
		} break;
		case SETTINGS_LOAD_EXPORT_TEMPLATES: {

			file_templates->popup_centered_ratio();

		} break;
		case SETTINGS_PICK_MAIN_SCENE: {

			//print_tree();
			file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
			//not for now?
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {

				file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			//file->set_current_path(current_path);
			Node *scene = editor_data.get_edited_scene_root();
			if (scene) {
				file->set_current_path(scene->get_filename());
			};
			file->set_title(TTR("Pick a Main Scene"));
			file->popup_centered_ratio();

		} break;
		case SETTINGS_ABOUT: {

			about->popup_centered_minsize(Size2(500, 130) * EDSCALE);
		} break;
		case SOURCES_REIMPORT: {

			reimport_dialog->popup_reimport();
		} break;
		case DEPENDENCY_LOAD_CHANGED_IMAGES: {

			List<Ref<Resource> > cached;
			ResourceCache::get_cached_resources(&cached);
			//this should probably be done in a thread..
			for (List<Ref<Resource> >::Element *E = cached.front(); E; E = E->next()) {

				if (!E->get()->editor_can_reload_from_file())
					continue;
				if (!E->get()->get_path().is_resource_file() && !E->get()->get_path().is_abs_path())
					continue;
				if (!FileAccess::exists(E->get()->get_path()))
					continue;
				uint64_t mt = FileAccess::get_modified_time(E->get()->get_path());
				if (mt != E->get()->get_last_modified_time()) {
					E->get()->reload_from_file();
				}
			}

		} break;
		case DEPENDENCY_UPDATE_IMPORTED: {

			/*
			bool editing_changed = _find_editing_changed_scene(get_edited_scene());

			import_reload_fn="";

			if (editing_changed) {
				if (unsaved_cache && !bool(EDITOR_DEF("import/ask_save_before_reimport",false))) {
					if (!p_confirmed) {


						confirmation->get_ok()->set_text("Open");
						//confirmation->get_cancel()->show();
						confirmation->set_text("Current scene changed, save and re-import ?");
						confirmation->popup_centered(Size2(300,70));
						break;

					}
				}

				Node *scene = get_edited_scene();

				if (scene->get_filename()=="") {

					current_option=-1;
					//accept->get_cancel()->hide();
					accept->get_ok()->set_text("I see..");
					accept->set_text("Can't import if edited scene was not saved."); //i dont think this code will ever run
					accept->popup_centered(Size2(300,70));
					break;

				}


				import_reload_fn = scene->get_filename();
				_save_scene(import_reload_fn);
				_cleanup_scene();


			}

*/

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

				Ref<EditorImportPlugin> p = editor_import_export->get_import_plugin(p_option - IMPORT_PLUGIN_BASE);
				if (p.is_valid()) {
					p->import_dialog();
				}
			}
		}
	}
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
}

void EditorNode::add_editor_plugin(EditorPlugin *p_editor) {

	if (p_editor->has_main_screen()) {

		ToolButton *tb = memnew(ToolButton);
		tb->set_toggle_mode(true);
		tb->connect("pressed", singleton, "_editor_select", varray(singleton->main_editor_buttons.size()));
		tb->set_text(p_editor->get_name());
		singleton->main_editor_buttons.push_back(tb);
		singleton->main_editor_button_vb->add_child(tb);
		singleton->editor_table.push_back(p_editor);
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

		//singleton->main_editor_tabs->add_tab(p_editor->get_name());
		singleton->editor_table.erase(p_editor);
	}
	p_editor->make_visible(false);
	p_editor->clear();
	singleton->editor_plugins_over->get_plugins_list().erase(p_editor);
	singleton->remove_child(p_editor);
	singleton->editor_data.remove_editor_plugin(p_editor);
}

void EditorNode::add_editor_import_plugin(const Ref<EditorImportPlugin> &p_editor_import) {

	ERR_FAIL_COND(p_editor_import.is_null());
	editor_import_export->add_import_plugin(p_editor_import);
	_rebuild_import_menu();
}

void EditorNode::remove_editor_import_plugin(const Ref<EditorImportPlugin> &p_editor_import) {
	ERR_FAIL_COND(p_editor_import.is_null());

	editor_import_export->remove_import_plugin(p_editor_import);
	_rebuild_import_menu();
}

void EditorNode::_update_addon_config() {

	if (_initializing_addons)
		return;

	Vector<String> enabled_addons;

	for (Map<String, EditorPlugin *>::Element *E = plugin_addons.front(); E; E = E->next()) {
		enabled_addons.push_back(E->key());
	}

	if (enabled_addons.size() == 0) {
		Globals::get_singleton()->set("editor_plugins/enabled", Variant());
		Globals::get_singleton()->set_persisting("editor_plugins/enabled", false);
	} else {
		Globals::get_singleton()->set("editor_plugins/enabled", enabled_addons);
		Globals::get_singleton()->set_persisting("editor_plugins/enabled", true);
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
		show_warning("Unable to enable addon plugin at: '" + addon_path + "' parsing of config failed.");
		return;
	}

	if (!cf->has_section_key("plugin", "script")) {
		show_warning("Unable to find script field for addon plugin at: 'res://addons/" + p_addon + "''.");
		return;
	}

	String path = cf->get_value("plugin", "script");
	path = "res://addons/" + p_addon + "/" + path;

	Ref<Script> script = ResourceLoader::load(path);

	if (script.is_null()) {
		show_warning("Unable to load addon script from path: '" + path + "'.");
		return;
	}

	//could check inheritance..
	if (String(script->get_instance_base_type()) != "EditorPlugin") {
		show_warning("Unable to load addon script from path: '" + path + "' Base type is not EditorPlugin.");
		return;
	}

	if (!script->is_tool()) {
		show_warning("Unable to load addon script from path: '" + path + "' Script is not in tool mode.");
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

	//	if (editor_data.get_edited_scene_count()==1) {
	//		//make new scene appear saved
	//		set_current_version(editor_data.get_undo_redo().get_version());
	//		unsaved_cache=false;
	//	}
}

void EditorNode::_remove_scene(int index) {
	//	printf("Attempting to remove scene %d (current is %d)\n", index, editor_data.get_edited_scene());

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

	if (p_scene && p_scene->cast_to<Popup>())
		p_scene->cast_to<Popup>()->show(); //show popups
	scene_tree_dock->set_edited_scene(p_scene);
	if (get_tree())
		get_tree()->set_edited_scene_root(p_scene);

	if (p_scene) {
		if (p_scene->get_parent() != scene_root)
			scene_root->add_child(p_scene);
	}
}

void EditorNode::_fetch_translatable_strings(const Object *p_object, Set<StringName> &strings) {

	List<String> tstrings;
	p_object->get_translatable_strings(&tstrings);
	for (List<String>::Element *E = tstrings.front(); E; E = E->next())
		strings.insert(E->get());

	const Node *node = p_object->cast_to<Node>();

	if (!node)
		return;

	Ref<Script> script = node->get_script();
	if (script.is_valid())
		_fetch_translatable_strings(script.ptr(), strings);

	for (int i = 0; i < node->get_child_count(); i++) {

		Node *c = node->get_child(i);
		if (c->get_owner() != get_edited_scene())
			continue;

		_fetch_translatable_strings(c, strings);
	}
}

Error EditorNode::save_translatable_strings(const String &p_to_file) {

	if (!is_inside_tree()) {
		defer_translatable = p_to_file;
		return OK;
	}

	ERR_FAIL_COND_V(!get_edited_scene(), ERR_INVALID_DATA);

	Set<StringName> strings;
	_fetch_translatable_strings(get_edited_scene(), strings);

	Error err;
	FileAccess *f = FileAccess::open(p_to_file, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V(err, err);

	OS::Date date = OS::get_singleton()->get_date();
	OS::Time time = OS::get_singleton()->get_time();
	f->store_line("# Translation Strings Dump.");
	f->store_line("# Created By.");
	f->store_line("# \t" VERSION_FULL_NAME " (c) 2008-2017 Juan Linietsky, Ariel Manzur.");
	f->store_line("# From Scene: ");
	f->store_line("# \t" + get_edited_scene()->get_filename());
	f->store_line("");
	f->store_line("msgid \"\"");
	f->store_line("msgstr \"\"");
	f->store_line("\"Report-Msgid-Bugs-To: <define>\\n\"");
	f->store_line("\"POT-Creation-Date: " + itos(date.year) + "-" + itos(date.month) + "-" + itos(date.day) + " " + itos(time.hour) + ":" + itos(time.min) + "0000\\n\"");
	//	f->store_line("\"PO-Revision-Date: 2006-08-30 13:56-0700\\n\"");
	//	f->store_line("\"Last-Translator: Rubén C. Díaz Alonso <outime@gmail.com>\\n\"");
	f->store_line("\"Language-Team: <define>\\n\"");
	f->store_line("\"MIME-Version: 1.0\\n\"");
	f->store_line("\"Content-Type: text/plain; charset=UTF-8\\n\"");
	f->store_line("\"Content-Transfer-Encoding: 8bit\\n\"");
	f->store_line("");

	for (Set<StringName>::Element *E = strings.front(); E; E = E->next()) {

		String s = E->get();
		if (s == "" || s.strip_edges() == "")
			continue;
		Vector<String> substr = s.split("\n");
		ERR_CONTINUE(substr.size() == 0);

		f->store_line("");

		if (substr.size() == 1) {

			f->store_line("msgid \"" + substr[0].c_escape() + "\"");
		} else {

			f->store_line("msgid \"\"");
			for (int i = 0; i < substr.size(); i++) {

				String s = substr[i];
				if (i != substr.size() - 1)
					s += "\n";
				f->store_line("\"" + s.c_escape() + "\"");
			}
		}

		f->store_line("msgstr \"\"");
	}

	f->close();
	memdelete(f);

	return OK;
}

Error EditorNode::save_optimized_copy(const String &p_scene, const String &p_preset) {

#if 0

	if (!is_inside_scene()) {
		defer_optimize=p_scene;
		defer_optimize_preset=p_preset;
		return OK;
	}


	if (!get_edited_scene()) {

		get_scene()->quit();
		ERR_EXPLAIN("No scene to optimize (loading failed?)");
		ERR_FAIL_V(ERR_FILE_NOT_FOUND);
	}


	String src_scene=Globals::get_singleton()->localize_path(get_edited_scene()->get_filename());


	String path=p_scene;
	print_line("p_path: "+p_scene);
	print_line("src_scene: "+p_scene);

	if (path.is_rel_path()) {
		print_line("rel path!?");
		path=src_scene.get_base_dir()+"/"+path;
	}
	path = Globals::get_singleton()->localize_path(path);

	print_line("path: "+path);


	String preset = "optimizer_presets/"+p_preset;
	if (!Globals::get_singleton()->has(preset)) {

		//accept->"()->hide();
		accept->get_ok()->set_text("I see..");
		accept->set_text("Optimizer preset not found: "+p_preset);
		accept->popup_centered(Size2(300,70));
		ERR_EXPLAIN("Optimizer preset not found: "+p_preset);
		ERR_FAIL_V(ERR_INVALID_PARAMETER);

	}

	Dictionary d = Globals::get_singleton()->get(preset);

	ERR_FAIL_COND_V(!d.has("__type__"),ERR_INVALID_DATA);
	String type=d["__type__"];

	Ref<EditorOptimizedSaver> saver;

	for(int i=0;i<editor_data.get_optimized_saver_count();i++) {

		print_line(type+" vs "+editor_data.get_optimized_saver(i)->get_target_name());
		if (editor_data.get_optimized_saver(i)->get_target_name()==type) {
			saver=editor_data.get_optimized_saver(i);
		}
	}

	ERR_EXPLAIN("Preset '"+p_preset+"' references nonexistent saver: "+type);
	ERR_FAIL_COND_V(saver.is_null(),ERR_INVALID_DATA);

	List<Variant> keys;
	d.get_key_list(&keys);

	saver->clear();

	for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

		saver->set(E->get(),d[E->get()]);
	}

	uint32_t flags=0;

//	if (saver->is_bundle_scenes_enabled())
//		flags|=ResourceSaver::FLAG_BUNDLE_INSTANCED_SCENES;
	if (saver->is_bundle_resources_enabled())
		flags|=ResourceSaver::FLAG_BUNDLE_RESOURCES;
	if (saver->is_remove_editor_data_enabled())
		flags|=ResourceSaver::FLAG_OMIT_EDITOR_PROPERTIES;
	if (saver->is_big_endian_data_enabled())
		flags|=ResourceSaver::FLAG_SAVE_BIG_ENDIAN;

	String platform=saver->get_target_platform();
	if (platform=="")
		platform="all";

	Ref<PackedScene> sdata = memnew( PackedScene );
	Error err = sdata->pack(get_edited_scene());

	if (err) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("I see..");
		accept->set_text("Couldn't save scene. Likely dependencies (instances) couldn't be satisfied.");
		accept->popup_centered(Size2(300,70));
		return ERR_INVALID_DATA;

	}
	err = ResourceSaver::save(path,sdata,flags); //todo, saverSceneSaver::save(path,get_edited_scene(),flags,saver);

	if (err) {

		//accept->"()->hide();
		accept->get_ok()->set_text("I see..");
		accept->set_text("Error saving optimized scene: "+path);
		accept->popup_centered(Size2(300,70));

		ERR_FAIL_COND_V(err,err);

	}

	project_settings->add_remapped_path(src_scene,path,platform);
#endif
	return OK;
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
	state["scene_tree_offset"] = scene_tree_dock->get_tree_editor()->get_scene_tree()->get_vscroll_bar()->get_val();
	state["property_edit_offset"] = get_property_editor()->get_scene_tree()->get_vscroll_bar()->get_val();
	state["saved_version"] = saved_version;
	state["node_filter"] = scene_tree_dock->get_filter();
	//print_line(" getting main tab: "+itos(state["main_tab"]));
	return state;
}

void EditorNode::_set_main_scene_state(Dictionary p_state, Node *p_for_scene) {

	if (get_edited_scene() != p_for_scene && p_for_scene != NULL)
		return; //not for this scene

	//print_line("set current 7 ");
	changing_scene = false;

#if 0
	if (p_state.has("main_tab")) {
		int idx = p_state["main_tab"];


		print_line("comes with tab: "+itos(idx));
		int current=-1;
		for(int i=0;i<editor_table.size();i++) {
			if (editor_plugin_screen==editor_table[i]) {
				current=i;
				break;
			}
		}


		if (idx<2 && current<2) {
			//only set tab for 2D and 3D
			_editor_select(idx);
			//print_line(" setting main tab: "+itos(p_state["main_tab"]));
		}
	}
#else

	if (get_edited_scene()) {

		int current = -1;
		for (int i = 0; i < editor_table.size(); i++) {
			if (editor_plugin_screen == editor_table[i]) {
				current = i;
				break;
			}
		}

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
#endif

	if (p_state.has("scene_tree_offset"))
		scene_tree_dock->get_tree_editor()->get_scene_tree()->get_vscroll_bar()->set_val(p_state["scene_tree_offset"]);
	if (p_state.has("property_edit_offset"))
		get_property_editor()->get_scene_tree()->get_vscroll_bar()->set_val(p_state["property_edit_offset"]);

	if (p_state.has("node_filter"))
		scene_tree_dock->set_filter(p_state["node_filter"]);
	//print_line("set current 8 ");

	//this should only happen at the very end

	//changing_scene=true; //avoid script change from opening editor
	ScriptEditor::get_singleton()->get_debugger()->update_live_edit_root();
	ScriptEditor::get_singleton()->set_scene_root_script(editor_data.get_scene_root_script(editor_data.get_edited_scene()));
	editor_data.notify_edited_scene_changed();

	//changing_scene=false;
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

	//print_line("set current 2 ");

	editor_selection->clear();
	editor_data.set_edited_scene(p_idx);

	Node *new_scene = editor_data.get_edited_scene_root();

	if (new_scene && new_scene->cast_to<Popup>())
		new_scene->cast_to<Popup>()->show(); //show popups

	//print_line("set current 3 ");

	scene_tree_dock->set_edited_scene(new_scene);
	if (get_tree())
		get_tree()->set_edited_scene_root(new_scene);

	if (new_scene) {
		if (new_scene->get_parent() != scene_root)
			scene_root->add_child(new_scene);
	}
	//print_line("set current 4 ");

	Dictionary state = editor_data.restore_edited_scene_state(editor_selection, &editor_history);
	_edit_current();

	/*if (!unsaved) {
		saved_version=editor_data.get_undo_redo().get_version();
		if (p_backwards)
			saved_version--;
		else
			saved_version++;
		print_line("was saved, updating version");
	} else {
		saved_version=state["saved_version"];
	}*/
	//_set_main_scene_state(state);

	call_deferred("_set_main_scene_state", state, get_edited_scene()); //do after everything else is done setting up
	//print_line("set current 6 ");
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

Error EditorNode::load_scene(const String &p_scene, bool p_ignore_broken_deps, bool p_set_inherited, bool p_clear_errors) {

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
	}

	if (p_clear_errors)
		load_errors->clear();

	String lpath = Globals::get_singleton()->localize_path(p_scene);

	if (!lpath.begins_with("res://")) {

		current_option = -1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text(TTR("Ugh"));
		accept->set_text(TTR("Error loading scene, it must be inside the project path. Use 'Import' to open the scene, then save it inside the project path."));
		accept->popup_centered_minsize();
		opening_prev = false;
		return ERR_FILE_NOT_FOUND;
	}

	int prev = editor_data.get_edited_scene();
	int idx = editor_data.add_edited_scene(-1);
	//print_line("load scene callback");
	//set_current_scene(idx);

	if (!editor_data.get_edited_scene_root() && editor_data.get_edited_scene_count() == 2) {
		_remove_edited_scene();
	} else {
		_scene_tab_changed(idx);
	}

	//_cleanup_scene(); // i'm sorry but this MUST happen to avoid modified resources to not be reloaded.

	dependency_errors.clear();

	Ref<PackedScene> sdata = ResourceLoader::load(lpath, "", true);
	if (!sdata.is_valid()) {

		current_option = -1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text(TTR("Ugh"));
		accept->set_text(TTR("Error loading scene."));
		accept->popup_centered_minsize();
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
		Ref<PackedScene> ps = Ref<PackedScene>(ResourceCache::get(lpath)->cast_to<PackedScene>());
		if (ps.is_valid()) {
			ps->replace_state(sdata->get_state());
			ps->set_last_modified_time(sdata->get_last_modified_time());
			sdata = ps;
		}

	} else {
		sdata->set_path(lpath, true); //take over path
	}

	Node *new_scene = sdata->instance(true);

	if (!new_scene) {

		sdata.unref();
		current_option = -1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text(TTR("Ugh"));
		accept->set_text(TTR("Error loading scene."));
		accept->popup_centered_minsize();
		opening_prev = false;
		if (prev != -1) {
			set_current_scene(prev);
			editor_data.remove_scene(idx);
		}
		return ERR_FILE_NOT_FOUND;
	}

	//guess not needed in the end?
	//new_scene->clear_internal_tree_resource_paths(); //make sure no internal tree paths to internal resources exist

	/*
	Node *old_scene = edited_scene;
	_hide_top_editors();
	set_edited_scene(NULL);
	editor_data.clear_editor_states();
	if (old_scene) {
		if (!opening_prev && old_scene->get_filename()!="") {
			previous_scenes.push_back(old_scene->get_filename());
		}
		memdelete(old_scene);
	}
*/

	if (p_set_inherited) {
		Ref<SceneState> state = sdata->get_state();
		state->set_path(lpath);
		new_scene->set_scene_inherited_state(state);
		new_scene->set_filename(String());
		//if (new_scene->get_scene_instance_state().is_valid())
		//	new_scene->get_scene_instance_state()->set_path(String());
	}

	new_scene->set_scene_instance_state(Ref<SceneState>());

	set_edited_scene(new_scene);
	_get_scene_metadata(p_scene);
	/*
	editor_data.set_edited_scene_root(new_scene);

	scene_tree_dock->set_selected(new_scene, true);
	property_editor->edit(new_scene);
	editor_data.set_edited_scene_root(new_scene);
*/
	editor_data.set_edited_scene_import_metadata(sdata->get_import_metadata());

	//	editor_data.get_undo_redo().clear_history();
	saved_version = editor_data.get_undo_redo().get_version();
	_update_title();
	_update_scene_tabs();
	_add_to_recent_scenes(lpath);

	prev_scene->set_disabled(previous_scenes.size() == 0);
	opening_prev = false;

	ScriptEditor::get_singleton()->get_debugger()->update_live_edit_root();

	//top_pallete->set_current_tab(0); //always go to scene

	push_item(new_scene);

	return OK;
}

void EditorNode::open_request(const String &p_path) {

	load_scene(p_path); // as it will be opened in separate tab
	//external_file=p_path;
	//_menu_option_confirm(FILE_EXTERNAL_OPEN_SCENE,false);
}

void EditorNode::request_instance_scene(const String &p_path) {

	scene_tree_dock->instance(p_path);
}

void EditorNode::request_instance_scenes(const Vector<String> &p_files) {

	scene_tree_dock->instance_scenes(p_files);
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

	Spatial *s = sp->cast_to<Spatial>();
	if (!s)
		return;
	AnimationPlayerEditor::singleton->get_key_editor()->insert_transform_key(s, p_sub, p_key);
}

void EditorNode::update_keying() {

	//print_line("KR: "+itos(p_enabled));

	bool valid = false;

	if (AnimationPlayerEditor::singleton->get_key_editor()->has_keying()) {

		if (editor_history.get_path_size() >= 1) {

			Object *obj = ObjectDB::get_instance(editor_history.get_path_object(0));
			if (obj && obj->cast_to<Node>()) {

				valid = true;
			}
		}
	}

	property_editor->set_keying(valid);

	AnimationPlayerEditor::singleton->get_key_editor()->update_keying();
}

void EditorNode::_close_messages() {

	//	left_split->set_dragger_visible(false);
	old_split_ofs = center_split->get_split_offset();
	center_split->set_split_offset(0);
	//	scene_root_parent->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_END,0);
}

void EditorNode::_show_messages() {

	//	left_split->set_dragger_visible(true);
	center_split->set_split_offset(old_split_ofs);
	//	scene_root_parent->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_END,log->get_margin(MARGIN_TOP));
}

#if 0
void EditorNode::animation_panel_make_visible(bool p_visible) {

	if (!p_visible) {
		animation_panel->hide();
	} else {
		animation_panel->show();
	}

	int idx = settings_menu->get_popup()->get_item_index(SETTINGS_SHOW_ANIMATION);
	settings_menu->get_popup()->set_item_checked(idx,p_visible);
}


void EditorNode::animation_editor_make_visible(bool p_visible) {

	if (p_visible) {

		animation_editor->show();
		animation_vb->get_parent_control()->minimum_size_changed();
		//pd_anim->show();
		top_split->set_collapsed(false);

		//scene_root_parent->set_margin(MARGIN_TOP,animation_editor->get_margin(MARGIN_BOTTOM));
	} else {
		//pd_anim->hide();
		animation_editor->hide();
//		scene_root_parent->set_margin(MARGIN_TOP,0);
		if (!animation_vb->get_parent_control())
			return;
		animation_vb->get_parent_control()->minimum_size_changed();
		top_split->set_collapsed(true);
	}

	animation_editor->set_keying(p_visible);

}
#endif
void EditorNode::_add_to_recent_scenes(const String &p_scene) {

	String base = "_" + Globals::get_singleton()->get_resource_path().replace("\\", "::").replace("/", "::");
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

	String base = "_" + Globals::get_singleton()->get_resource_path().replace("\\", "::").replace("/", "::");
	Vector<String> rc = EDITOR_DEF(base + "/_recent_scenes", Array());

	ERR_FAIL_INDEX(p_idx, rc.size());

	String path = "res://" + rc[p_idx];

	/*if (unsaved_cache) {
		_recent_scene=rc[p_idx];
		open_recent_confirmation->set_text("Discard current scene and open:\n'"+rc[p_idx]+"'");
		open_recent_confirmation->get_label()->set_align(Label::ALIGN_CENTER);
		open_recent_confirmation->popup_centered(Size2(400,100));
		return;
	}*/

	load_scene(path);
}

void EditorNode::_save_optimized() {

//	save_optimized_copy(optimized_save->get_optimized_scene(),optimized_save->get_preset());
#if 0
	String path = optimized_save->get_optimized_scene();

	uint32_t flags=0;

	String platform="all";
	Ref<EditorOptimizedSaver> saver=editor_data.get_optimized_saver(optimized_save->get_preset());

	if (saver->is_bundle_scenes_enabled())
		flags|=SceneSaver::FLAG_BUNDLE_INSTANCED_SCENES;
	if (saver->is_bundle_resources_enabled())
		flags|=SceneSaver::FLAG_BUNDLE_RESOURCES;
	if (saver->is_remove_editor_data_enabled())
		flags|=SceneSaver::FLAG_OMIT_EDITOR_PROPERTIES;
	if (saver->is_big_endian_data_enabled())
		flags|=SceneSaver::FLAG_SAVE_BIG_ENDIAN;

	platform=saver->get_target_platform();

	Error err = SceneSaver::save(path,get_edited_scene(),flags,saver);

	if (err) {

		//accept->"()->hide();
		accept->get_ok()->set_text("I see..");
		accept->set_text("Error saving optimized scene: "+path);
		accept->popup_centered(Size2(300,70));
		return;

	}

	project_settings->add_remapped_path(Globals::get_singleton()->localize_path(get_edited_scene()->get_filename()),Globals::get_singleton()->localize_path(path),platform);
#endif
}

void EditorNode::_update_recent_scenes() {

	String base = "_" + Globals::get_singleton()->get_resource_path().replace("\\", "::").replace("/", "::");
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

bool EditorNode::_find_editing_changed_scene(Node *p_from) {
	/*
	if (!p_from)
		return false;

	if (p_from->get_filename()!="") {

		StringName fn = p_from->get_filename();
		for(int i=0;i<import_monitor->get_changes().size();i++) {

			if (fn==import_monitor->get_changes()[i])
				return true;
		}
	}

	for(int i=0;i<p_from->get_child_count();i++) {

		if (_find_editing_changed_scene(p_from->get_child(i)))
			return true;
	}
*/
	return false;
}

void EditorNode::add_io_error(const String &p_error) {
	//CharString err_ut = p_error.utf8();
	//ERR_PRINT(!err_ut.get_data());
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

	ObjectTypeDB::register_type<EditorPlugin>();
	ObjectTypeDB::register_type<EditorImportPlugin>();
	ObjectTypeDB::register_type<EditorExportPlugin>();
	ObjectTypeDB::register_type<EditorScenePostImport>();
	ObjectTypeDB::register_type<EditorScript>();
	ObjectTypeDB::register_type<EditorSelection>();
	ObjectTypeDB::register_type<EditorFileDialog>();
	//ObjectTypeDB::register_type<EditorImportExport>();
	ObjectTypeDB::register_type<EditorSettings>();
	ObjectTypeDB::register_type<EditorSpatialGizmo>();

	//ObjectTypeDB::register_type<EditorImporter>();
	//	ObjectTypeDB::register_type<EditorPostImport>();
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

void EditorNode::progress_task_step(const String &p_task, const String &p_state, int p_step, bool p_force_redraw) {

	singleton->progress_dialog->task_step(p_task, p_state, p_step, p_force_redraw);
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

	EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_path(p_path.get_base_dir());
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

Error EditorNode::export_platform(const String &p_platform, const String &p_path, bool p_debug, const String &p_password, bool p_quit_after) {

	export_defer.platform = p_platform;
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

void EditorNode::_dock_select_input(const InputEvent &p_input) {

	if (p_input.type == InputEvent::MOUSE_BUTTON || p_input.type == InputEvent::MOUSE_MOTION) {

		Vector2 point(p_input.mouse_motion.x, p_input.mouse_motion.y);

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

		if (p_input.type == InputEvent::MOUSE_BUTTON && p_input.mouse_button.button_index == 1 && p_input.mouse_button.pressed && dock_popup_selected != nrect) {
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
	unr.pos += Vector2(2, 5);
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
		r.pos += Vector2(2, 5);
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

		if (splits[i]->is_visible()) {
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

	HSplitContainer *h_splits[4] = {
		left_l_hsplit,
		left_r_hsplit,
		main_hsplit,
		right_hsplit,
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

			if (!dock_slot[i]->is_hidden() && dock_slot[i]->get_tab_count()) {
				dock_slot[i]->set_current_tab(0);
			}
		}
		bottom_panel->show();
		right_hsplit->show();
	}
}

void EditorNode::_update_top_menu_visibility() {
	if (distraction_free_mode) {
		play_cc->hide();
		menu_hb->hide();
		scene_tabs->hide();
	} else {
		play_cc->show();
		menu_hb->show();
		scene_tabs->show();
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
				node = dock_slot[k]->get_node(name)->cast_to<Control>();
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

		if (!dock_slot[i]->is_hidden() && dock_slot[i]->get_tab_count()) {
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
	Error err = config->load(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts.cfg"));
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
			Error err = config->load(EditorSettings::get_singleton()->get_settings_path().plus_file("editor_layouts.cfg"));
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

	bool unsaved = (p_tab == editor_data.get_edited_scene()) ?
						   saved_version != editor_data.get_undo_redo().get_version() :
						   editor_data.get_scene_version(p_tab) != 0;
	if (unsaved) {
		confirmation->get_ok()->set_text(TTR("Yes"));

		//confirmation->get_cancel()->show();
		confirmation->set_text(TTR("Close scene? (Unsaved changes will be lost)"));
		confirmation->popup_centered_minsize();
	} else {
		_remove_scene(p_tab);
		_update_scene_tabs();
	}
}

void EditorNode::_scene_tab_changed(int p_tab) {

	//print_line("set current 1 ");
	bool unsaved = (saved_version != editor_data.get_undo_redo().get_version());
	//print_line("version: "+itos(editor_data.get_undo_redo().get_version())+", saved "+itos(saved_version));

	if (p_tab == editor_data.get_edited_scene())
		return; //pointless

	uint64_t next_scene_version = editor_data.get_scene_version(p_tab);

	//print_line("scene tab changed???");
	editor_data.get_undo_redo().create_action(TTR("Switch Scene Tab"));
	editor_data.get_undo_redo().add_do_method(this, "set_current_version", unsaved ? saved_version : 0);
	editor_data.get_undo_redo().add_do_method(this, "set_current_scene", p_tab);
	//editor_data.get_undo_redo().add_do_method(scene_tabs,"set_current_tab",p_tab);
	//editor_data.get_undo_redo().add_do_method(scene_tabs,"ensure_tab_visible",p_tab);
	editor_data.get_undo_redo().add_do_method(this, "set_current_version", next_scene_version == 0 ? editor_data.get_undo_redo().get_version() + 1 : next_scene_version);

	editor_data.get_undo_redo().add_undo_method(this, "set_current_version", next_scene_version);
	editor_data.get_undo_redo().add_undo_method(this, "set_current_scene", editor_data.get_edited_scene());
	//editor_data.get_undo_redo().add_undo_method(scene_tabs,"set_current_tab",editor_data.get_edited_scene());
	//editor_data.get_undo_redo().add_undo_method(scene_tabs,"ensure_tab_visible",p_tab,editor_data.get_edited_scene());
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
			if (p_item->is_visible()) {
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
		for (int i = 0; i < bottom_panel_items.size(); i++) {

			bottom_panel_items[i].button->set_pressed(i == p_idx);
			bottom_panel_items[i].control->set_hidden(i != p_idx);
		}
		center_split->set_dragger_visibility(SplitContainer::DRAGGER_VISIBLE);
		center_split->set_collapsed(false);
	} else {
		for (int i = 0; i < bottom_panel_items.size(); i++) {

			bottom_panel_items[i].button->set_pressed(false);
			bottom_panel_items[i].control->set_hidden(true);
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

void EditorNode::set_distraction_free_mode(bool p_enter) {
	distraction_free_mode = p_enter;

	if (p_enter) {
		if (docks_visible) {
			set_docks_visible(false);
		}
	} else {
		set_docks_visible(true);
	}
	_update_top_menu_visibility();
}

bool EditorNode::get_distraction_free_mode() const {
	return distraction_free_mode;
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
	TextureFrame *drag_preview = memnew(TextureFrame);
	Label *label = memnew(Label);

	waiting_for_sources_changed = true; //
	Ref<Texture> preview;

	{
		//todo make proper previews
		Ref<ImageTexture> pic = gui_base->get_icon("FileBig", "EditorIcons");
		Image img = pic->get_data();
		img.resize(48, 48); //meh
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
		label->set_text(p_res->get_type());
	}

	drag_control->add_child(label);

	p_from->set_drag_preview(drag_control); //wait until it enters scene

	label->set_pos(Point2((preview->get_width() - label->get_minimum_size().width) / 2, preview->get_height()));

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

	String cur_path = filesystem_dock->get_current_path();
	for (int i = 0; i < EditorImportExport::get_singleton()->get_import_plugin_count(); i++) {
		EditorImportExport::get_singleton()->get_import_plugin(i)->import_from_drop(p_files, cur_path);
	}
}
void EditorNode::_file_access_close_error_notify(const String &p_str) {

	add_io_error("Unable to write to file '" + p_str + "', file in use, locked or lacking permissions.");
}

void EditorNode::reload_scene(const String &p_path) {

	//first of all, reload textures as they might have changed on disk

	List<Ref<Resource> > cached;
	ResourceCache::get_cached_resources(&cached);
	List<Ref<Resource> > to_clear; //clear internal resources from previous scene from being used
	for (List<Ref<Resource> >::Element *E = cached.front(); E; E = E->next()) {

		if (E->get()->get_path().begins_with(p_path + "::")) //subresources of existing scene
			to_clear.push_back(E->get());

		if (!E->get()->cast_to<Texture>())
			continue;
		if (!E->get()->get_path().is_resource_file() && !E->get()->get_path().is_abs_path())
			continue;
		if (!FileAccess::exists(E->get()->get_path()))
			continue;
		uint64_t mt = FileAccess::get_modified_time(E->get()->get_path());
		if (mt != E->get()->get_last_modified_time()) {
			E->get()->reload_from_file();
		}
	}

	//so reload reloads everything, clear subresources of previous scene
	while (to_clear.front()) {
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
	load_scene(p_path);
	//adjust index so tab is back a the previous position
	editor_data.move_edited_scene_to_index(scene_idx);
	get_undo_redo()->clear_history();
	//recover the tab
	scene_tabs->set_current_tab(current_tab);
	_scene_tab_changed(current_tab);
}

void EditorNode::_bind_methods() {

	ObjectTypeDB::bind_method("_menu_option", &EditorNode::_menu_option);
	ObjectTypeDB::bind_method("_menu_confirm_current", &EditorNode::_menu_confirm_current);
	ObjectTypeDB::bind_method("_dialog_action", &EditorNode::_dialog_action);
	ObjectTypeDB::bind_method("_resource_selected", &EditorNode::_resource_selected, DEFVAL(""));
	ObjectTypeDB::bind_method("_property_editor_forward", &EditorNode::_property_editor_forward);
	ObjectTypeDB::bind_method("_property_editor_back", &EditorNode::_property_editor_back);
	ObjectTypeDB::bind_method("_editor_select", &EditorNode::_editor_select);
	ObjectTypeDB::bind_method("_node_renamed", &EditorNode::_node_renamed);
	ObjectTypeDB::bind_method("edit_node", &EditorNode::edit_node);
	ObjectTypeDB::bind_method("_imported", &EditorNode::_imported);
	ObjectTypeDB::bind_method("_unhandled_input", &EditorNode::_unhandled_input);

	ObjectTypeDB::bind_method("_get_scene_metadata", &EditorNode::_get_scene_metadata);
	ObjectTypeDB::bind_method("set_edited_scene", &EditorNode::set_edited_scene);
	ObjectTypeDB::bind_method("open_request", &EditorNode::open_request);
	ObjectTypeDB::bind_method("_instance_request", &EditorNode::_instance_request);
	ObjectTypeDB::bind_method("update_keying", &EditorNode::update_keying);
	ObjectTypeDB::bind_method("_property_keyed", &EditorNode::_property_keyed);
	ObjectTypeDB::bind_method("_transform_keyed", &EditorNode::_transform_keyed);
	ObjectTypeDB::bind_method("_close_messages", &EditorNode::_close_messages);
	ObjectTypeDB::bind_method("_show_messages", &EditorNode::_show_messages);
	ObjectTypeDB::bind_method("_vp_resized", &EditorNode::_vp_resized);
	ObjectTypeDB::bind_method("_quick_opened", &EditorNode::_quick_opened);
	ObjectTypeDB::bind_method("_quick_run", &EditorNode::_quick_run);

	ObjectTypeDB::bind_method("_resource_created", &EditorNode::_resource_created);

	ObjectTypeDB::bind_method("_import_action", &EditorNode::_import_action);
	//ObjectTypeDB::bind_method("_import",&EditorNode::_import);
	//	ObjectTypeDB::bind_method("_import_conflicts_solved",&EditorNode::_import_conflicts_solved);
	ObjectTypeDB::bind_method("_open_recent_scene", &EditorNode::_open_recent_scene);
	//	ObjectTypeDB::bind_method("_open_recent_scene_confirm",&EditorNode::_open_recent_scene_confirm);

	ObjectTypeDB::bind_method("_save_optimized", &EditorNode::_save_optimized);

	ObjectTypeDB::bind_method("stop_child_process", &EditorNode::stop_child_process);

	ObjectTypeDB::bind_method("_sources_changed", &EditorNode::_sources_changed);
	ObjectTypeDB::bind_method("_fs_changed", &EditorNode::_fs_changed);
	ObjectTypeDB::bind_method("_dock_select_draw", &EditorNode::_dock_select_draw);
	ObjectTypeDB::bind_method("_dock_select_input", &EditorNode::_dock_select_input);
	ObjectTypeDB::bind_method("_dock_pre_popup", &EditorNode::_dock_pre_popup);
	ObjectTypeDB::bind_method("_dock_split_dragged", &EditorNode::_dock_split_dragged);
	ObjectTypeDB::bind_method("_save_docks", &EditorNode::_save_docks);
	ObjectTypeDB::bind_method("_dock_popup_exit", &EditorNode::_dock_popup_exit);
	ObjectTypeDB::bind_method("_dock_move_left", &EditorNode::_dock_move_left);
	ObjectTypeDB::bind_method("_dock_move_right", &EditorNode::_dock_move_right);

	ObjectTypeDB::bind_method("_layout_menu_option", &EditorNode::_layout_menu_option);

	ObjectTypeDB::bind_method("set_current_scene", &EditorNode::set_current_scene);
	ObjectTypeDB::bind_method("set_current_version", &EditorNode::set_current_version);
	ObjectTypeDB::bind_method("_scene_tab_changed", &EditorNode::_scene_tab_changed);
	ObjectTypeDB::bind_method("_scene_tab_closed", &EditorNode::_scene_tab_closed);
	ObjectTypeDB::bind_method("_scene_tab_script_edited", &EditorNode::_scene_tab_script_edited);
	ObjectTypeDB::bind_method("_set_main_scene_state", &EditorNode::_set_main_scene_state);
	ObjectTypeDB::bind_method("_update_scene_tabs", &EditorNode::_update_scene_tabs);

	ObjectTypeDB::bind_method("_prepare_history", &EditorNode::_prepare_history);
	ObjectTypeDB::bind_method("_select_history", &EditorNode::_select_history);

	ObjectTypeDB::bind_method("_toggle_search_bar", &EditorNode::_toggle_search_bar);
	ObjectTypeDB::bind_method("_clear_search_box", &EditorNode::_clear_search_box);
	ObjectTypeDB::bind_method("_clear_undo_history", &EditorNode::_clear_undo_history);
	ObjectTypeDB::bind_method("_dropped_files", &EditorNode::_dropped_files);
	ObjectTypeDB::bind_method("_export_godot3_path", &EditorNode::_export_godot3_path);

	ObjectTypeDB::bind_method(_MD("add_editor_import_plugin", "plugin"), &EditorNode::add_editor_import_plugin);
	ObjectTypeDB::bind_method(_MD("remove_editor_import_plugin", "plugin"), &EditorNode::remove_editor_import_plugin);
	ObjectTypeDB::bind_method(_MD("get_gui_base"), &EditorNode::get_gui_base);
	ObjectTypeDB::bind_method(_MD("_bottom_panel_switch"), &EditorNode::_bottom_panel_switch);

	ADD_SIGNAL(MethodInfo("play_pressed"));
	ADD_SIGNAL(MethodInfo("pause_pressed"));
	ADD_SIGNAL(MethodInfo("stop_pressed"));
	ADD_SIGNAL(MethodInfo("request_help"));
	ADD_SIGNAL(MethodInfo("request_help_search"));
	ADD_SIGNAL(MethodInfo("script_add_function_request", PropertyInfo(Variant::OBJECT, "obj"), PropertyInfo(Variant::STRING, "function"), PropertyInfo(Variant::STRING_ARRAY, "args")));
	ADD_SIGNAL(MethodInfo("resource_saved", PropertyInfo(Variant::OBJECT, "obj")));
}

void EditorNode::_export_godot3_path(const String &p_path) {

	Error err = export_godot3.export_godot3(p_path);
	if (err != OK) {
		show_warning("Error exporting to Godot 3.0");
	}
}

EditorNode::EditorNode() {

	EditorHelp::generate_doc(); //before any editor classes are crated
	SceneState::set_disable_placeholders(true);
	editor_initialize_certificates(); //for asset sharing

	InputDefault *id = Input::get_singleton()->cast_to<InputDefault>();

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
	distraction_free_mode = false;

	FileAccess::set_backup_save(true);

	PathRemap::get_singleton()->clear_remaps(); //editor uses no remaps
	TranslationServer::get_singleton()->set_enabled(false);
	// load settings
	if (!EditorSettings::get_singleton())
		EditorSettings::create();
	{
		int dpi_mode = EditorSettings::get_singleton()->get("global/hidpi_mode");
		if (dpi_mode == 0) {
			editor_set_hidpi(OS::get_singleton()->get_screen_dpi(0) >= 192 && OS::get_singleton()->get_screen_size(OS::get_singleton()->get_current_screen()).x > 2000);
		} else if (dpi_mode == 2) {
			editor_set_hidpi(true);
		} else {
			editor_set_hidpi(false);
		}
	}

	ResourceLoader::set_abort_on_missing_resources(false);
	FileDialog::set_default_show_hidden_files(EditorSettings::get_singleton()->get("file_dialog/show_hidden_files"));
	EditorFileDialog::set_default_show_hidden_files(EditorSettings::get_singleton()->get("file_dialog/show_hidden_files"));
	EditorFileDialog::set_default_display_mode((EditorFileDialog::DisplayMode)EditorSettings::get_singleton()->get("file_dialog/display_mode").operator int());
	ResourceLoader::set_error_notify_func(this, _load_error_notify);
	ResourceLoader::set_dependency_error_notify_func(this, _dependency_error_report);

	ResourceLoader::set_timestamp_on_load(true);
	ResourceSaver::set_timestamp_on_save(true);

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

	editor_import_export = memnew(EditorImportExport);
	add_child(editor_import_export);

	register_exporters();

	editor_import_export->load_config();

	GLOBAL_DEF("editor/main_run_args", "$scene");

	ObjectTypeDB::set_type_enabled("CollisionShape", true);
	ObjectTypeDB::set_type_enabled("CollisionShape2D", true);
	ObjectTypeDB::set_type_enabled("CollisionPolygon2D", true);

	Control *theme_base = memnew(Control);
	add_child(theme_base);
	theme_base->set_area_as_parent_rect();

	gui_base = memnew(Panel);
	theme_base->add_child(gui_base);
	gui_base->set_area_as_parent_rect();

	theme_base->set_theme(create_default_theme());
	theme = create_editor_theme();
	gui_base->set_theme(theme);

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

#if 0
	PanelContainer *top_dark_panel = memnew( PanelContainer );
	Ref<StyleBoxTexture> top_dark_sb;
	top_dark_sb.instance();
	top_dark_sb->set_texture(theme->get_icon("PanelTop","EditorIcons"));
	for(int i=0;i<4;i++) {
		top_dark_sb->set_margin_size(Margin(i),3);
		top_dark_sb->set_default_margin(Margin(i),0);
	}
	top_dark_sb->set_expand_margin_size(MARGIN_LEFT,20);
	top_dark_sb->set_expand_margin_size(MARGIN_RIGHT,20);

	top_dark_panel->add_style_override("panel",top_dark_sb);
	VBoxContainer *top_dark_vb = memnew( VBoxContainer );
	main_vbox->add_child(top_dark_panel);
	top_dark_panel->add_child(top_dark_vb);
#endif

	menu_hb = memnew(HBoxContainer);
	main_vbox->add_child(menu_hb);

	//	top_dark_vb->add_child(scene_tabs);
	//left
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
	//left_r_vsplit->hide();
	//dock_slot[DOCK_SLOT_LEFT_UR]->hide();
	//dock_slot[DOCK_SLOT_LEFT_BR]->hide();

	main_hsplit = memnew(HSplitContainer);
	left_r_hsplit->add_child(main_hsplit);
	//main_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	VBoxContainer *center_vb = memnew(VBoxContainer);
	main_hsplit->add_child(center_vb);
	center_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	center_split = memnew(VSplitContainer);
	//main_hsplit->add_child(center_split);
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
	//right_l_vsplit->hide();
	//dock_slot[DOCK_SLOT_RIGHT_UL]->hide();
	//dock_slot[DOCK_SLOT_RIGHT_BL]->hide();

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

	dock_select_popoup = memnew(PopupPanel);
	gui_base->add_child(dock_select_popoup);
	VBoxContainer *dock_vb = memnew(VBoxContainer);
	dock_select_popoup->add_child(dock_vb);

	HBoxContainer *dock_hb = memnew(HBoxContainer);
	dock_tab_move_left = memnew(ToolButton);
	dock_tab_move_left->set_icon(theme->get_icon("Back", "EditorIcons"));
	dock_tab_move_left->set_focus_mode(Control::FOCUS_NONE);
	dock_tab_move_left->connect("pressed", this, "_dock_move_left");
	//dock_tab_move_left->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dock_hb->add_child(dock_tab_move_left);
	dock_hb->add_spacer();
	dock_tab_move_right = memnew(ToolButton);
	dock_tab_move_right->set_icon(theme->get_icon("Forward", "EditorIcons"));
	dock_tab_move_right->set_focus_mode(Control::FOCUS_NONE);
	dock_tab_move_right->connect("pressed", this, "_dock_move_right");

	//dock_tab_move_right->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dock_hb->add_child(dock_tab_move_right);
	dock_vb->add_child(dock_hb);

	dock_select = memnew(Control);
	dock_select->set_custom_minimum_size(Size2(128, 64) * EDSCALE);
	dock_select->connect("input_event", this, "_dock_select_input");
	dock_select->connect("draw", this, "_dock_select_draw");
	dock_select->connect("mouse_exit", this, "_dock_popup_exit");
	dock_select->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	dock_vb->add_child(dock_select);

	dock_select_popoup->set_child_rect(dock_vb);
	dock_select_popoup->set_as_minsize();
	dock_select_rect_over = -1;
	dock_popup_selected = -1;
	//dock_select_popoup->set_(Size2(20,20));

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		dock_slot[i]->set_custom_minimum_size(Size2(230, 220) * EDSCALE);
		dock_slot[i]->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		dock_slot[i]->set_popup(dock_select_popoup);
		dock_slot[i]->connect("pre_popup_pressed", this, "_dock_pre_popup", varray(i));

		//dock_slot[i]->set_tab_align(TabContainer::ALIGN_LEFT);
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

	/*	main_editor_tabs  = memnew( Tabs );
	main_editor_tabs->connect("tab_changed",this,"_editor_select");
	main_editor_tabs->set_tab_close_display_policy(Tabs::SHOW_NEVER);
*/
	scene_tabs = memnew(Tabs);
	scene_tabs->add_tab("unsaved");
	scene_tabs->set_tab_align(Tabs::ALIGN_CENTER);
	scene_tabs->set_tab_close_display_policy((bool(EDITOR_DEF("global/always_show_close_button_in_scene_tabs", false)) ? Tabs::CLOSE_BUTTON_SHOW_ALWAYS : Tabs::CLOSE_BUTTON_SHOW_ACTIVE_ONLY));
	scene_tabs->connect("tab_changed", this, "_scene_tab_changed");
	scene_tabs->connect("right_button_pressed", this, "_scene_tab_script_edited");
	scene_tabs->connect("tab_close", this, "_scene_tab_closed");

	srt->add_child(scene_tabs);

	scene_root_parent = memnew(PanelContainer);
	scene_root_parent->set_custom_minimum_size(Size2(0, 80) * EDSCALE);

	//Ref<StyleBox> sp = scene_root_parent->get_stylebox("panel","TabContainer");
	//scene_root_parent->add_style_override("panel",sp);

	/*scene_root_parent->set_anchor( MARGIN_RIGHT, Control::ANCHOR_END );
	scene_root_parent->set_anchor( MARGIN_BOTTOM, Control::ANCHOR_END );
	scene_root_parent->set_begin( Point2( 0, 0) );
	scene_root_parent->set_end( Point2( 0,80 ) );*/
	srt->add_child(scene_root_parent);
	scene_root_parent->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	scene_root = memnew(Viewport);

	//scene_root_base->add_child(scene_root);
	//scene_root->set_meta("_editor_disable_input",true);
	VisualServer::get_singleton()->viewport_set_hide_scenario(scene_root->get_viewport(), true);
	scene_root->set_disable_input(true);
	scene_root->set_as_audio_listener_2d(true);
	scene_root->set_size_override(true, Size2(Globals::get_singleton()->get("display/width"), Globals::get_singleton()->get("display/height")));

	//	scene_root->set_world_2d( Ref<World2D>( memnew( World2D )) );

	viewport = memnew(VBoxContainer);
	viewport->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	/*for(int i=0;i<4;i++) {
		viewport->set_margin(Margin(i),sp->get_margin(Margin(i)));
	}*/
	scene_root_parent->add_child(viewport);

	PanelContainer *top_region = memnew(PanelContainer);
	top_region->add_style_override("panel", gui_base->get_stylebox("hover", "Button"));
	HBoxContainer *left_menu_hb = memnew(HBoxContainer);
	top_region->add_child(left_menu_hb);
	menu_hb->add_child(top_region);

	PopupMenu *p;

	file_menu = memnew(MenuButton);
	file_menu->set_text(TTR("Scene"));
	//file_menu->set_icon(gui_base->get_icon("Save","EditorIcons"));
	left_menu_hb->add_child(file_menu);

	prev_scene = memnew(ToolButton);
	prev_scene->set_icon(gui_base->get_icon("PrevScene", "EditorIcons"));
	prev_scene->set_tooltip(TTR("Go to previously opened scene."));
	prev_scene->set_disabled(true);
	//left_menu_hb->add_child( prev_scene );
	prev_scene->connect("pressed", this, "_menu_option", make_binds(FILE_OPEN_PREV));
	gui_base->add_child(prev_scene);
	prev_scene->set_pos(Point2(3, 24));
	prev_scene->hide();

	ED_SHORTCUT("editor/fullscreen_mode", TTR("Fullscreen Mode"), KEY_MASK_SHIFT | KEY_F11);
	ED_SHORTCUT("editor/distraction_free_mode", TTR("Distraction Free Mode"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_F11);

	ED_SHORTCUT("editor/next_tab", TTR("Next tab"), KEY_MASK_CMD + KEY_TAB);
	ED_SHORTCUT("editor/prev_tab", TTR("Previous tab"), KEY_MASK_CMD + KEY_MASK_SHIFT + KEY_TAB);
	ED_SHORTCUT("editor/filter_files", TTR("Filter Files.."), KEY_MASK_ALT + KEY_MASK_CMD + KEY_P);

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
	//p->add_shortcut(ED_SHORTCUT("editor/save_scene",TTR("Close Goto Prev. Scene")),FILE_OPEN_PREV,KEY_MASK_SHIFT+KEY_MASK_CMD+KEY_P);
	p->add_submenu_item(TTR("Open Recent"), "RecentScenes", FILE_OPEN_RECENT);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("editor/quick_open_scene", TTR("Quick Open Scene.."), KEY_MASK_SHIFT + KEY_MASK_CMD + KEY_O), FILE_QUICK_OPEN_SCENE);
	p->add_shortcut(ED_SHORTCUT("editor/quick_open_script", TTR("Quick Open Script.."), KEY_MASK_ALT + KEY_MASK_CMD + KEY_O), FILE_QUICK_OPEN_SCRIPT);
	p->add_separator();

	PopupMenu *pm_export = memnew(PopupMenu);
	pm_export->set_name("Export");
	p->add_child(pm_export);
	p->add_submenu_item(TTR("Convert To.."), "Export");
	pm_export->add_item(TTR("Translatable Strings.."), FILE_DUMP_STRINGS);
	pm_export->add_separator();
	pm_export->add_shortcut(ED_SHORTCUT("editor/convert_to_MeshLibrary", TTR("MeshLibrary..")), FILE_EXPORT_MESH_LIBRARY);
	pm_export->add_shortcut(ED_SHORTCUT("editor/convert_to_TileSet", TTR("TileSet..")), FILE_EXPORT_TILESET);
	pm_export->connect("item_pressed", this, "_menu_option");

	p->add_separator();
	p->add_item(TTR("Undo"), EDIT_UNDO, KEY_MASK_CMD + KEY_Z);
	p->add_item(TTR("Redo"), EDIT_REDO, KEY_MASK_CMD + KEY_MASK_SHIFT + KEY_Z);
	p->add_separator();
	p->add_item(TTR("Run Script"), FILE_RUN_SCRIPT, KEY_MASK_SHIFT + KEY_MASK_CMD + KEY_R);
	p->add_separator();
	p->add_item(TTR("Project Settings"), RUN_SETTINGS);
	p->add_separator();
	p->add_item(TTR("Revert Scene"), EDIT_REVERT);
	p->add_separator();
#ifdef OSX_ENABLED
	p->add_item(TTR("Quit to Project List"), RUN_PROJECT_MANAGER, KEY_MASK_SHIFT + KEY_MASK_ALT + KEY_Q);
#else
	p->add_item(TTR("Quit to Project List"), RUN_PROJECT_MANAGER, KEY_MASK_SHIFT + KEY_MASK_CTRL + KEY_Q);
#endif
	p->add_item(TTR("Quit"), FILE_QUIT, KEY_MASK_CMD + KEY_Q);

	recent_scenes = memnew(PopupMenu);
	recent_scenes->set_name("RecentScenes");
	p->add_child(recent_scenes);
	recent_scenes->connect("item_pressed", this, "_open_recent_scene");

	{
		Control *sp = memnew(Control);
		sp->set_custom_minimum_size(Size2(30, 0) * EDSCALE);
		menu_hb->add_child(sp);
	}

	PanelContainer *editor_region = memnew(PanelContainer);
	editor_region->add_style_override("panel", gui_base->get_stylebox("hover", "Button"));
	main_editor_button_vb = memnew(HBoxContainer);
	editor_region->add_child(main_editor_button_vb);
	menu_hb->add_child(editor_region);

//menu_hb->add_spacer();
#if 0
	node_menu = memnew( MenuButton );
	node_menu->set_text("Node");
	node_menu->set_pos( Point2( 50,0) );
	menu_panel->add_child( node_menu );

	p=node_menu->get_popup();
	p->add_item("Create",NODE_CREATE);
	p->add_item("Instance",NODE_INSTANCE);
	p->add_separator();
	p->add_item("Reparent",NODE_REPARENT);
	p->add_item("Move Up",NODE_MOVE_UP);
	p->add_item("Move Down",NODE_MOVE_DOWN);
	p->add_separator();
	p->add_item("Duplicate",NODE_DUPLICATE);
	p->add_separator();
	p->add_item("Remove (Branch)",NODE_REMOVE_BRANCH);
	p->add_item("Remove (Element)",NODE_REMOVE_ELEMENT);
	p->add_separator();
	p->add_item("Edit Subscriptions..",NODE_CONNECTIONS);
	p->add_item("Edit Groups..",NODE_GROUPS);

	resource_menu = memnew( MenuButton );
	resource_menu->set_text("Resource");
	resource_menu->set_pos( Point2( 90,0) );
	menu_panel->add_child( resource_menu );
#endif

	import_menu = memnew(MenuButton);
	import_menu->set_tooltip(TTR("Import assets to the project."));
	import_menu->set_text(TTR("Import"));
	//import_menu->set_icon(gui_base->get_icon("Save","EditorIcons"));
	left_menu_hb->add_child(import_menu);

	p = import_menu->get_popup();
	p->connect("item_pressed", this, "_menu_option");

	tool_menu = memnew(MenuButton);
	tool_menu->set_tooltip(TTR("Miscellaneous project or scene-wide tools."));
	tool_menu->set_text(TTR("Tools"));

	//tool_menu->set_icon(gui_base->get_icon("Save","EditorIcons"));
	left_menu_hb->add_child(tool_menu);

	p = tool_menu->get_popup();
	p->connect("item_pressed", this, "_menu_option");
	p->add_item(TTR("Orphan Resource Explorer"), TOOLS_ORPHAN_RESOURCES);
	p->add_item(TTR("Export to Godot 3.0 (WIP)"), TOOLS_EXPORT_GODOT3);

	export_button = memnew(ToolButton);
	export_button->set_tooltip(TTR("Export the project to many platforms."));
	export_button->set_text(TTR("Export"));
	export_button->connect("pressed", this, "_menu_option", varray(FILE_EXPORT_PROJECT));
	export_button->set_focus_mode(Control::FOCUS_NONE);
	left_menu_hb->add_child(export_button);

	menu_hb->add_spacer();

	//Separator *s1 = memnew( VSeparator );
	//menu_panel->add_child(s1);
	//s1->set_pos(Point2(210,4));
	//s1->set_size(Point2(10,15));

	play_cc = memnew(CenterContainer);
	play_cc->set_ignore_mouse(true);
	gui_base->add_child(play_cc);
	play_cc->set_area_as_parent_rect();
	play_cc->set_anchor_and_margin(MARGIN_BOTTOM, Control::ANCHOR_BEGIN, 10);
	play_cc->set_margin(MARGIN_TOP, 5);

	top_region = memnew(PanelContainer);
	top_region->add_style_override("panel", gui_base->get_stylebox("hover", "Button"));
	play_cc->add_child(top_region);

	HBoxContainer *play_hb = memnew(HBoxContainer);
	top_region->add_child(play_hb);

	play_button = memnew(ToolButton);
	play_hb->add_child(play_button);
	play_button->set_toggle_mode(true);
	play_button->set_icon(gui_base->get_icon("MainPlay", "EditorIcons"));
	play_button->set_focus_mode(Control::FOCUS_NONE);
	play_button->connect("pressed", this, "_menu_option", make_binds(RUN_PLAY));
	play_button->set_tooltip(TTR("Play the project."));
	play_button->set_shortcut(ED_SHORTCUT("editor/play", TTR("Play"), KEY_F5));

	pause_button = memnew(ToolButton);
	//menu_panel->add_child(pause_button); - not needed for now?
	pause_button->set_toggle_mode(true);
	pause_button->set_icon(gui_base->get_icon("Pause", "EditorIcons"));
	pause_button->set_focus_mode(Control::FOCUS_NONE);
	//pause_button->connect("pressed", this,"_menu_option",make_binds(RUN_PAUSE));
	pause_button->set_tooltip(TTR("Pause the scene"));
	pause_button->set_disabled(true);
	play_hb->add_child(pause_button);
	pause_button->set_shortcut(ED_SHORTCUT("editor/pause_scene", TTR("Pause Scene"), KEY_F7));

	stop_button = memnew(ToolButton);
	play_hb->add_child(stop_button);
	//stop_button->set_toggle_mode(true);
	stop_button->set_focus_mode(Control::FOCUS_NONE);
	stop_button->set_icon(gui_base->get_icon("MainStop", "EditorIcons"));
	stop_button->connect("pressed", this, "_menu_option", make_binds(RUN_STOP));
	stop_button->set_tooltip(TTR("Stop the scene."));
	stop_button->set_shortcut(ED_SHORTCUT("editor/stop", TTR("Stop"), KEY_F8));

	run_native = memnew(EditorRunNative);
	play_hb->add_child(run_native);
	native_play_button = memnew(MenuButton);
	native_play_button->set_text("NTV");
	menu_hb->add_child(native_play_button);
	native_play_button->hide();
	native_play_button->get_popup()->connect("item_pressed", this, "_run_in_device");
	run_native->connect("native_run", this, "_menu_option", varray(RUN_PLAY_NATIVE));

	//	VSeparator *s1 = memnew( VSeparator );
	//	play_hb->add_child(s1);

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

	debug_button = memnew(MenuButton);
	debug_button->set_flat(true);
	play_hb->add_child(debug_button);
	//debug_button->set_toggle_mode(true);
	debug_button->set_focus_mode(Control::FOCUS_NONE);
	debug_button->set_icon(gui_base->get_icon("Remote", "EditorIcons"));
	//debug_button->connect("pressed", this,"_menu_option",make_binds(RUN_LIVE_DEBUG));
	debug_button->set_tooltip(TTR("Debug options"));

	p = debug_button->get_popup();
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
	p->connect("item_pressed", this, "_menu_option");

	/*
	run_settings_button = memnew( ToolButton );
	//menu_hb->add_child(run_settings_button);
	//run_settings_button->set_toggle_mode(true);
	run_settings_button->set_focus_mode(Control::FOCUS_NONE);
	run_settings_button->set_icon(gui_base->get_icon("Run","EditorIcons"));
	run_settings_button->connect("pressed", this,"_menu_option",make_binds(RUN_SCENE_SETTINGS));
*/

	/*
	run_settings_button = memnew( ToolButton );
	menu_panel->add_child(run_settings_button);
	run_settings_button->set_pos(Point2(305,0));
	run_settings_button->set_focus_mode(Control::FOCUS_NONE);
	run_settings_button->set_icon(gui_base->get_icon("Run","EditorIcons"));
	run_settings_button->connect("pressed", this,"_menu_option",make_binds(RUN_SETTINGS));
*/

	progress_hb = memnew(BackgroundProgress);
	menu_hb->add_child(progress_hb);

	{
		Control *sp = memnew(Control);
		sp->set_custom_minimum_size(Size2(30, 0) * EDSCALE);
		menu_hb->add_child(sp);
	}

	PanelContainer *vu_cont = memnew(PanelContainer);
	vu_cont->add_style_override("panel", gui_base->get_stylebox("hover", "Button"));
	menu_hb->add_child(vu_cont);

	audio_vu = memnew(TextureProgress);
	CenterContainer *vu_cc = memnew(CenterContainer);
	vu_cc->add_child(audio_vu);
	vu_cont->add_child(vu_cc);
	audio_vu->set_under_texture(gui_base->get_icon("VuEmpty", "EditorIcons"));
	audio_vu->set_progress_texture(gui_base->get_icon("VuFull", "EditorIcons"));
	audio_vu->set_max(24);
	audio_vu->set_min(-80);
	audio_vu->set_step(0.01);
	audio_vu->set_val(0);

	{
		Control *sp = memnew(Control);
		sp->set_custom_minimum_size(Size2(30, 0) * EDSCALE);
		menu_hb->add_child(sp);
	}

	top_region = memnew(PanelContainer);
	top_region->add_style_override("panel", gui_base->get_stylebox("hover", "Button"));
	HBoxContainer *right_menu_hb = memnew(HBoxContainer);
	top_region->add_child(right_menu_hb);
	menu_hb->add_child(top_region);

	settings_menu = memnew(MenuButton);
	settings_menu->set_text(TTR("Settings"));
	//settings_menu->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	right_menu_hb->add_child(settings_menu);
	p = settings_menu->get_popup();

	//p->add_item("Export Settings",SETTINGS_EXPORT_PREFERENCES);
	p->add_item(TTR("Editor Settings"), SETTINGS_PREFERENCES);
	//p->add_item("Optimization Presets",SETTINGS_OPTIMIZED_PRESETS);
	p->add_separator();
	editor_layouts = memnew(PopupMenu);
	editor_layouts->set_name("Layouts");
	p->add_child(editor_layouts);
	editor_layouts->connect("item_pressed", this, "_layout_menu_option");
	p->add_submenu_item(TTR("Editor Layout"), "Layouts");
	p->add_separator();
	p->add_item(TTR("Install Export Templates"), SETTINGS_LOAD_EXPORT_TEMPLATES);
	p->add_separator();
	p->add_item(TTR("About"), SETTINGS_ABOUT);

	layout_dialog = memnew(EditorNameDialog);
	gui_base->add_child(layout_dialog);
	layout_dialog->set_hide_on_ok(false);
	layout_dialog->set_size(Size2(175, 70) * EDSCALE);
	layout_dialog->connect("name_confirmed", this, "_dialog_action");

	sources_button = memnew(ToolButton);
	right_menu_hb->add_child(sources_button);
	sources_button->set_icon(gui_base->get_icon("DependencyOk", "EditorIcons"));
	sources_button->connect("pressed", this, "_menu_option", varray(SOURCES_REIMPORT));
	sources_button->set_tooltip(TTR("Alerts when an external resource has changed."));

	update_menu = memnew(MenuButton);
	update_menu->set_tooltip(TTR("Spins when the editor window repaints!"));
	right_menu_hb->add_child(update_menu);
	update_menu->set_icon(gui_base->get_icon("Progress1", "EditorIcons"));
	p = update_menu->get_popup();
	p->add_check_item(TTR("Update Always"), SETTINGS_UPDATE_ALWAYS);
	p->add_check_item(TTR("Update Changes"), SETTINGS_UPDATE_CHANGES);
	p->add_separator();
	p->add_check_item(TTR("Disable Update Spinner"), SETTINGS_UPDATE_SPINNER_HIDE);
	p->set_item_checked(1, true);

	//sources_button->connect();

	/*
	Separator *s2 = memnew( VSeparator );
	menu_panel->add_child(s2);
	s2->set_pos(Point2(338,4));
	s2->set_size(Point2(10,15));
*/

	//editor_hsplit = memnew( HSplitContainer );
	//main_split->add_child(editor_hsplit);
	//editor_hsplit->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	//editor_vsplit = memnew( VSplitContainer );
	//editor_hsplit->add_child(editor_vsplit);

	//top_pallete = memnew( TabContainer );
	scene_tree_dock = memnew(SceneTreeDock(this, scene_root, editor_selection, editor_data));
	scene_tree_dock->set_name(TTR("Scene"));
	//top_pallete->add_child(scene_tree_dock);
	dock_slot[DOCK_SLOT_RIGHT_UL]->add_child(scene_tree_dock);
#if 0
	resources_dock = memnew( ResourcesDock(this) );
	resources_dock->set_name("Resources");
	//top_pallete->add_child(resources_dock);
	dock_slot[DOCK_SLOT_RIGHT_BL]->add_child(resources_dock);
	//top_pallete->set_v_size_flags(Control::SIZE_EXPAND_FILL);
#endif
	dock_slot[DOCK_SLOT_LEFT_BR]->hide();
	/*Control *editor_spacer = memnew( Control );
	editor_spacer->set_custom_minimum_size(Size2(260,200));
	editor_spacer->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor_vsplit->add_child( editor_spacer );
	editor_spacer->add_child( top_pallete );
	top_pallete->set_area_as_parent_rect();*/

	//prop_pallete = memnew( TabContainer );

	//prop_pallete->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	/*editor_spacer = memnew( Control );
	editor_spacer->set_custom_minimum_size(Size2(260,200));
	editor_spacer->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor_vsplit->add_child( editor_spacer );
	editor_spacer->add_child( prop_pallete );
	prop_pallete->set_area_as_parent_rect();*/

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
	resource_save_button->get_popup()->connect("item_pressed", this, "_menu_option");
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
	editor_history_menu->get_popup()->connect("item_pressed", this, "_select_history");

	prop_editor_hb = memnew(HBoxContainer); //again...

	prop_editor_base->add_child(prop_editor_hb);
	editor_path = memnew(EditorPath(&editor_history));
	editor_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	prop_editor_hb->add_child(editor_path);

	search_button = memnew(ToolButton);
	search_button->set_toggle_mode(true);
	search_button->set_pressed(false);
	search_button->set_icon(gui_base->get_icon("Zoom", "EditorIcons"));
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

	property_editor = memnew(PropertyEditor);
	property_editor->set_autoclear(true);
	property_editor->set_show_categories(true);
	property_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	property_editor->set_use_doc_hints(true);
	property_editor->set_enable_capitalize_paths(bool(EDITOR_DEF("inspector/capitalize_properties", true)));

	property_editor->hide_top_label();
	property_editor->register_text_enter(search_box);

	prop_editor_base->add_child(property_editor);
	property_editor->set_undo_redo(&editor_data.get_undo_redo());

	bool use_single_dock_column = OS::get_singleton()->get_screen_size(OS::get_singleton()->get_current_screen()).x < 1200;

	node_dock = memnew(NodeDock);
	//node_dock->set_undoredo(&editor_data.get_undo_redo());
	if (use_single_dock_column) {
		dock_slot[DOCK_SLOT_RIGHT_UL]->add_child(node_dock);
	} else {
		dock_slot[DOCK_SLOT_RIGHT_BL]->add_child(node_dock);
	}

	filesystem_dock = memnew(FileSystemDock(this));
	filesystem_dock->set_name(TTR("FileSystem"));
	filesystem_dock->set_display_mode(int(EditorSettings::get_singleton()->get("filesystem_dock/display_mode")));

	if (use_single_dock_column) {
		dock_slot[DOCK_SLOT_RIGHT_BL]->add_child(filesystem_dock);
		left_r_vsplit->hide();
		dock_slot[DOCK_SLOT_LEFT_UR]->hide();
		dock_slot[DOCK_SLOT_LEFT_BR]->hide();
	} else {
		dock_slot[DOCK_SLOT_LEFT_UR]->add_child(filesystem_dock);
	}

	//prop_pallete->add_child(filesystem_dock);
	filesystem_dock->connect("open", this, "open_request");
	filesystem_dock->connect("instance", this, "_instance_request");

	const String docks_section = "docks";

	overridden_default_layout = -1;
	default_layout.instance();
	default_layout->set_value(docks_section, "dock_3", TTR("FileSystem"));
	default_layout->set_value(docks_section, "dock_5", TTR("Scene"));
	default_layout->set_value(docks_section, "dock_6", TTR("Inspector") + "," + TTR("Node"));

	for (int i = 0; i < DOCK_SLOT_MAX / 2; i++)
		default_layout->set_value(docks_section, "dock_hsplit_" + itos(i + 1), 0);
	for (int i = 0; i < DOCK_SLOT_MAX / 2; i++)
		default_layout->set_value(docks_section, "dock_split_" + itos(i + 1), 0);

	_update_layouts_menu();

	bottom_panel = memnew(PanelContainer);
	bottom_panel->add_style_override("panel", gui_base->get_stylebox("panelf", "Panel"));
	center_split->add_child(bottom_panel);
	center_split->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN);

	bottom_panel_vb = memnew(VBoxContainer);
	bottom_panel->add_child(bottom_panel_vb);
	//bottom_panel_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	bottom_panel_hb = memnew(HBoxContainer);
	bottom_panel_vb->add_child(bottom_panel_hb);

	log = memnew(EditorLog);

	add_bottom_panel_item(TTR("Output"), log);

	//left_split->set_dragger_visible(false);

	old_split_ofs = 0;

	center_split->connect("resized", this, "_vp_resized");

	/*PanelContainer *bottom_pc = memnew( PanelContainer );
	srt->add_child(bottom_pc);
	bottom_hb = memnew( HBoxContainer );
	bottom_pc->add_child(bottom_hb);*/

	//	center_vb->add_child( log->get_button() );
	//	log->get_button()->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	//progress_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	/*
	animation_menu = memnew( ToolButton );
	animation_menu->set_pos(Point2(500,0));
	animation_menu->set_size(Size2(20,20));
	animation_menu->set_toggle_mode(true);
	animation_menu->set_focus_mode(Control::FOCUS_NONE);
	menu_panel->add_child(animation_menu);
	animation_menu->set_icon(gui_base->get_icon("Animation","EditorIcons"));
	animation_menu->connect("pressed",this,"_animation_visibility_toggle");
*/

	call_dialog = memnew(CallDialog);
	call_dialog->hide();
	gui_base->add_child(call_dialog);

	orphan_resources = memnew(OrphanResourcesDialog);
	gui_base->add_child(orphan_resources);

	confirmation = memnew(ConfirmationDialog);
	gui_base->add_child(confirmation);
	confirmation->connect("confirmed", this, "_menu_confirm_current");

	accept = memnew(AcceptDialog);
	gui_base->add_child(accept);
	accept->connect("confirmed", this, "_menu_confirm_current");

	//	optimized_save = memnew( OptimizedSaveDialog(&editor_data) );
	//gui_base->add_child(optimized_save);
	//optimized_save->connect("confirmed",this,"_save_optimized");

	project_export = memnew(ProjectExport(&editor_data));
	gui_base->add_child(project_export);

	project_export_settings = memnew(ProjectExportDialog(this));
	gui_base->add_child(project_export_settings);

	//optimized_presets = memnew( OptimizedPresetsDialog(&editor_data) );
	//gui_base->add_child(optimized_presets);
	//optimized_presets->connect("confirmed",this,"_presets_optimized");

	//import_subscene = memnew( EditorSubScene );
	//gui_base->add_child(import_subscene);

	dependency_error = memnew(DependencyErrorDialog);
	gui_base->add_child(dependency_error);

	dependency_fixer = memnew(DependencyEditor);
	gui_base->add_child(dependency_fixer);

	settings_config_dialog = memnew(EditorSettingsDialog);
	gui_base->add_child(settings_config_dialog);

	project_settings = memnew(ProjectSettings(&editor_data));
	gui_base->add_child(project_settings);

	import_confirmation = memnew(ConfirmationDialog);
	import_confirmation->get_ok()->set_text(TTR("Re-Import"));
	import_confirmation->add_button(TTR("Update"), !OS::get_singleton()->get_swap_ok_cancel(), "update");
	import_confirmation->get_label()->set_align(Label::ALIGN_CENTER);
	import_confirmation->connect("confirmed", this, "_import_action", make_binds("re-import"));
	import_confirmation->connect("custom_action", this, "_import_action");
	gui_base->add_child(import_confirmation);

	open_recent_confirmation = memnew(ConfirmationDialog);
	add_child(open_recent_confirmation);
	open_recent_confirmation->connect("confirmed", this, "_open_recent_scene_confirm");

	run_settings_dialog = memnew(RunSettingsDialog);
	gui_base->add_child(run_settings_dialog);

	about = memnew(AcceptDialog);
	about->set_title(TTR("Thanks from the Godot community!"));
	about->get_ok()->set_text(TTR("Thanks!"));
	about->set_hide_on_ok(true);
	gui_base->add_child(about);
	VBoxContainer *vbc = memnew(VBoxContainer);
	HBoxContainer *hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->set_alignment(BoxContainer::ALIGN_CENTER);
	about->add_child(vbc);
	vbc->add_child(hbc);
	Label *about_text = memnew(Label);
	about_text->set_text(VERSION_FULL_NAME + String::utf8("\n\u00A9 2007-2017 Juan Linietsky, Ariel Manzur.\n\u00A9 2014-2017 ") + TTR("Godot Engine contributors") + "\n");
	TextureFrame *logo = memnew(TextureFrame);
	logo->set_texture(gui_base->get_icon("Logo", "EditorIcons"));
	hbc->add_child(logo);
	hbc->add_child(about_text);
	TabContainer *tc = memnew(TabContainer);
	tc->set_custom_minimum_size(Vector2(740, 300));
	vbc->add_child(tc);
	ScrollContainer *dev_base = memnew(ScrollContainer);
	dev_base->set_name(TTR("Developers"));
	tc->add_child(dev_base);
	HBoxContainer *dev_hbc = memnew(HBoxContainer);
	dev_hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dev_base->add_child(dev_hbc);
	for (int i = 0; i < 3; i++) {
		Label *dev_label = memnew(Label);
		dev_label->set_h_size_flags(Control::SIZE_EXPAND);
		dev_label->set_v_size_flags(Control::SIZE_FILL);
		dev_hbc->add_child(dev_label);
	}
	int dev_name_index = 0;
	int dev_name_column = 0;
	const int dev_index_max = AUTHORS_COUNT / 3 + (AUTHORS_COUNT % 3 == 0 ? 0 : 1);
	String dev_name = "";
	const char **dev_names_ptr = dev_names;
	while (*dev_names_ptr) {
		dev_name += String::utf8(*dev_names_ptr++);
		if (++dev_name_index == dev_index_max || !*dev_names_ptr) {
			dev_hbc->get_child(dev_name_column)->cast_to<Label>()->set_text(dev_name);
			dev_name_column++;
			dev_name = "";
			dev_name_index = 0;
		} else {
			dev_name += "\n";
		}
	}

	warning = memnew(AcceptDialog);
	gui_base->add_child(warning);

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

	reimport_dialog = memnew(EditorReImportDialog);
	gui_base->add_child(reimport_dialog);

	property_forward->connect("pressed", this, "_property_editor_forward");
	property_back->connect("pressed", this, "_property_editor_back");

	file_menu->get_popup()->connect("item_pressed", this, "_menu_option");
	object_menu->get_popup()->connect("item_pressed", this, "_menu_option");

	update_menu->get_popup()->connect("item_pressed", this, "_menu_option");
	settings_menu->get_popup()->connect("item_pressed", this, "_menu_option");

	file->connect("file_selected", this, "_dialog_action");
	file_templates->connect("file_selected", this, "_dialog_action");
	property_editor->connect("resource_selected", this, "_resource_selected");
	property_editor->connect("property_keyed", this, "_property_keyed");

	//plugin stuff

	file_server = memnew(EditorFileServer);

	editor_import_export->add_import_plugin(Ref<EditorTextureImportPlugin>(memnew(EditorTextureImportPlugin(this))));
	Ref<EditorSceneImportPlugin> _scene_import = memnew(EditorSceneImportPlugin(this));
	Ref<EditorSceneImporterCollada> _collada_import = memnew(EditorSceneImporterCollada);
	_scene_import->add_importer(_collada_import);
	//	Ref<EditorSceneImporterFBXConv> _fbxconv_import = memnew( EditorSceneImporterFBXConv);
	//	_scene_import->add_importer(_fbxconv_import);
	editor_import_export->add_import_plugin(_scene_import);
	// TODO: This plugin has no code, it should be either implemented or dropped (GH-3667)
	// editor_import_export->add_import_plugin( Ref<EditorSceneAnimationImportPlugin>( memnew(EditorSceneAnimationImportPlugin(this))));
	editor_import_export->add_import_plugin(Ref<EditorMeshImportPlugin>(memnew(EditorMeshImportPlugin(this))));
	editor_import_export->add_import_plugin(Ref<EditorFontImportPlugin>(memnew(EditorFontImportPlugin(this))));
	editor_import_export->add_import_plugin(Ref<EditorSampleImportPlugin>(memnew(EditorSampleImportPlugin(this))));
	editor_import_export->add_import_plugin(Ref<EditorTranslationImportPlugin>(memnew(EditorTranslationImportPlugin(this))));
	editor_import_export->add_import_plugin(Ref<EditorBitMaskImportPlugin>(memnew(EditorBitMaskImportPlugin(this))));

	editor_import_export->add_export_plugin(Ref<EditorTextureExportPlugin>(memnew(EditorTextureExportPlugin)));
	editor_import_export->add_export_plugin(Ref<EditorSampleExportPlugin>(memnew(EditorSampleExportPlugin)));
	editor_import_export->add_export_plugin(Ref<EditorSceneExportPlugin>(memnew(EditorSceneExportPlugin)));

	add_editor_plugin(memnew(AnimationPlayerEditorPlugin(this)));
	add_editor_plugin(memnew(CanvasItemEditorPlugin(this)));
	add_editor_plugin(memnew(SpatialEditorPlugin(this)));
	add_editor_plugin(memnew(ScriptEditorPlugin(this)));

	if (StreamPeerSSL::is_available()) {
		add_editor_plugin(memnew(AssetLibraryEditorPlugin(this)));
	} else {
		WARN_PRINT("Asset Library not available, as it requires SSL to work.");
	}
	//more visually meaningful to have this later
	raise_bottom_panel_item(AnimationPlayerEditor::singleton);

	add_editor_plugin(memnew(ShaderGraphEditorPlugin(this, true)));
	add_editor_plugin(memnew(ShaderGraphEditorPlugin(this, false)));
	add_editor_plugin(memnew(ShaderEditorPlugin(this, true)));
	add_editor_plugin(memnew(ShaderEditorPlugin(this, false)));
	add_editor_plugin(memnew(CameraEditorPlugin(this)));
	add_editor_plugin(memnew(SampleEditorPlugin(this)));
	add_editor_plugin(memnew(SampleLibraryEditorPlugin(this)));
	add_editor_plugin(memnew(ThemeEditorPlugin(this)));
	add_editor_plugin(memnew(MultiMeshEditorPlugin(this)));
	add_editor_plugin(memnew(MeshInstanceEditorPlugin(this)));
	add_editor_plugin(memnew(AnimationTreeEditorPlugin(this)));
	//add_editor_plugin( memnew( SamplePlayerEditorPlugin(this) ) ); - this is kind of useless at this point
	add_editor_plugin(memnew(MeshLibraryEditorPlugin(this)));
	//add_editor_plugin( memnew( StreamEditorPlugin(this) ) );
	add_editor_plugin(memnew(StyleBoxEditorPlugin(this)));
	add_editor_plugin(memnew(ParticlesEditorPlugin(this)));
	add_editor_plugin(memnew(ResourcePreloaderEditorPlugin(this)));
	add_editor_plugin(memnew(ItemListEditorPlugin(this)));
	//add_editor_plugin( memnew( RichTextEditorPlugin(this) ) );
	add_editor_plugin(memnew(CollisionPolygonEditorPlugin(this)));
	add_editor_plugin(memnew(CollisionPolygon2DEditorPlugin(this)));
	add_editor_plugin(memnew(TileSetEditorPlugin(this)));
	add_editor_plugin(memnew(TileMapEditorPlugin(this)));
	add_editor_plugin(memnew(SpriteFramesEditorPlugin(this)));
	add_editor_plugin(memnew(TextureRegionEditorPlugin(this)));
	add_editor_plugin(memnew(Particles2DEditorPlugin(this)));
	add_editor_plugin(memnew(Path2DEditorPlugin(this)));
	add_editor_plugin(memnew(PathEditorPlugin(this)));
	add_editor_plugin(memnew(BakedLightEditorPlugin(this)));
	add_editor_plugin(memnew(Polygon2DEditorPlugin(this)));
	add_editor_plugin(memnew(LightOccluder2DEditorPlugin(this)));
	add_editor_plugin(memnew(NavigationPolygonEditorPlugin(this)));
	add_editor_plugin(memnew(ColorRampEditorPlugin(this)));
	add_editor_plugin(memnew(CollisionShape2DEditorPlugin(this)));
	add_editor_plugin(memnew(TextureEditorPlugin(this)));
	add_editor_plugin(memnew(MaterialEditorPlugin(this)));
	add_editor_plugin(memnew(MeshEditorPlugin(this)));

	for (int i = 0; i < EditorPlugins::get_plugin_count(); i++)
		add_editor_plugin(EditorPlugins::create(i, this));

	resource_preview->add_preview_generator(Ref<EditorTexturePreviewPlugin>(memnew(EditorTexturePreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorPackedScenePreviewPlugin>(memnew(EditorPackedScenePreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorMaterialPreviewPlugin>(memnew(EditorMaterialPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorScriptPreviewPlugin>(memnew(EditorScriptPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorSamplePreviewPlugin>(memnew(EditorSamplePreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorMeshPreviewPlugin>(memnew(EditorMeshPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorBitmapPreviewPlugin>(memnew(EditorBitmapPreviewPlugin)));

	circle_step_msec = OS::get_singleton()->get_ticks_msec();
	circle_step_frame = OS::get_singleton()->get_frames_drawn();
	circle_step = 0;

	_rebuild_import_menu();

	editor_plugin_screen = NULL;
	editor_plugins_over = memnew(EditorPluginList);

	//	force_top_viewport(true);
	_edit_current();
	current = NULL;

	PhysicsServer::get_singleton()->set_active(false); // no physics by default if editor
	Physics2DServer::get_singleton()->set_active(false); // no physics by default if editor
	ScriptServer::set_scripting_enabled(false); // no scripting by default if editor

	Globals::get_singleton()->set("debug/indicators_enabled", true);
	Globals::get_singleton()->set("render/room_cull_enabled", false);

	reference_resource_mem = true;
	save_external_resources_mem = true;

	set_process(true);
	OS::get_singleton()->set_low_processor_usage_mode(true);

	if (0) { //not sure if i want this to happen after all

		//store project name in ssettings
		String project_name;
		//figure it out from path
		project_name = Globals::get_singleton()->get_resource_path().replace("\\", "/");
		print_line("path: " + project_name);
		if (project_name.length() && project_name[project_name.length() - 1] == '/')
			project_name = project_name.substr(0, project_name.length() - 1);

		project_name = project_name.replace("/", "::");

		if (project_name != "") {
			EditorSettings::get_singleton()->set("projects/" + project_name, Globals::get_singleton()->get_resource_path());
			EditorSettings::get_singleton()->raise_order("projects/" + project_name);
			EditorSettings::get_singleton()->save();
		}
	}

	//edited_scene=NULL;
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

	//	Panel *errors = memnew( Panel );
	load_errors = memnew(RichTextLabel);
	//	load_errors->set_readonly(true);
	load_error_dialog = memnew(AcceptDialog);
	load_error_dialog->add_child(load_errors);
	load_error_dialog->set_title(TTR("Load Errors"));
	load_error_dialog->set_child_rect(load_errors);
	gui_base->add_child(load_error_dialog);

	//EditorImport::add_importer( Ref<EditorImporterCollada>( memnew(EditorImporterCollada )));

	EditorFileSystem::get_singleton()->connect("sources_changed", this, "_sources_changed");
	EditorFileSystem::get_singleton()->connect("filesystem_changed", this, "_fs_changed");

	{
		List<StringName> tl;
		StringName ei = "EditorIcons";
		theme_base->get_theme()->get_icon_list(ei, &tl);
		for (List<StringName>::Element *E = tl.front(); E; E = E->next()) {

			if (!ObjectTypeDB::type_exists(E->get()))
				continue;
			icon_type_cache[E->get()] = theme_base->get_theme()->get_icon(E->get(), ei);
		}
	}

	export_godot3_dialog = memnew(FileDialog);
	export_godot3_dialog->set_access(FileDialog::ACCESS_FILESYSTEM);
	export_godot3_dialog->set_mode(FileDialog::MODE_OPEN_DIR);
	gui_base->add_child(export_godot3_dialog);
	export_godot3_dialog->connect("dir_selected", this, "_export_godot3_path");

	Node::set_human_readable_collision_renaming(true);

	pick_main_scene = memnew(ConfirmationDialog);
	gui_base->add_child(pick_main_scene);
	pick_main_scene->get_ok()->set_text("Select");
	pick_main_scene->connect("confirmed", this, "_menu_option", varray(SETTINGS_PICK_MAIN_SCENE));

	//	Ref<ImageTexture> it = gui_base->get_icon("logo","Icons");
	//	OS::get_singleton()->set_icon( it->get_data() );

	for (int i = 0; i < _init_callbacks.size(); i++)
		_init_callbacks[i]();

	editor_data.add_edited_scene(-1);
	editor_data.set_edited_scene(0);
	_update_scene_tabs();

	{

		_initializing_addons = true;
		Vector<String> addons;
		if (Globals::get_singleton()->has("editor_plugins/enabled")) {
			addons = Globals::get_singleton()->get("editor_plugins/enabled");
		}

		for (int i = 0; i < addons.size(); i++) {
			set_addon_plugin_enabled(addons[i], true);
		}
		_initializing_addons = false;
	}

	_load_docks();

	FileAccess::set_file_close_fail_notify_callback(_file_access_close_error_notify);

	ED_SHORTCUT("editor/editor_2d", TTR("Open 2D Editor"), KEY_F2);
	ED_SHORTCUT("editor/editor_3d", TTR("Open 3D Editor"), KEY_F3);
	ED_SHORTCUT("editor/editor_script", TTR("Open Script Editor"), KEY_F4);
	ED_SHORTCUT("editor/editor_help", TTR("Search Help"), KEY_F1);
	ED_SHORTCUT("editor/editor_assetlib", TTR("Open Asset Library"));
	ED_SHORTCUT("editor/editor_next", TTR("Open the next Editor"));
	ED_SHORTCUT("editor/editor_prev", TTR("Open the previous Editor"));
}

EditorNode::~EditorNode() {

	memdelete(EditorHelp::get_doc_data());
	memdelete(editor_selection);
	memdelete(editor_plugins_over);
	memdelete(file_server);
	EditorSettings::destroy();
}

/*
 * EDITOR PLUGIN LIST
 */

void EditorPluginList::make_visible(bool p_visible) {
	if (!plugins_list.empty()) {
		for (int i = 0; i < plugins_list.size(); i++) {
			plugins_list[i]->make_visible(p_visible);
		}
	}
}

void EditorPluginList::edit(Object *p_object) {
	if (!plugins_list.empty()) {
		for (int i = 0; i < plugins_list.size(); i++) {
			plugins_list[i]->edit(p_object);
		}
	}
}

bool EditorPluginList::forward_input_event(const InputEvent &p_event) {
	bool discard = false;
	if (!plugins_list.empty()) {
		for (int i = 0; i < plugins_list.size(); i++) {
			if (plugins_list[i]->forward_input_event(p_event)) {
				discard = true;
			}
		}
	}
	return discard;
}

bool EditorPluginList::forward_spatial_input_event(Camera *p_camera, const InputEvent &p_event) {
	bool discard = false;
	if (!plugins_list.empty()) {
		for (int i = 0; i < plugins_list.size(); i++) {
			if (plugins_list[i]->forward_spatial_input_event(p_camera, p_event)) {
				discard = true;
			}
		}
	}
	return discard;
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
