/*************************************************************************/
/*  editor_node.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "version.h"
#include "editor_node.h"
#include "print_string.h"
#include "editor_icons.h"
#include "editor_fonts.h"

#include "editor_help.h"
#include "core/io/resource_saver.h"
#include "core/io/resource_loader.h"
#include "servers/physics_2d_server.h"
#include "scene/resources/packed_scene.h"
#include "editor_settings.h"
#include "io_plugins/editor_import_collada.h"
#include "io_plugins/editor_scene_importer_fbxconv.h"
#include "globals.h"
#include <stdio.h>
#include "object_type_db.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "os/file_access.h"
#include "message_queue.h"
#include "path_remap.h"
#include "translation.h"
#include "pvrtc_compress.h"
#include "editor_file_system.h"
#include "register_exporters.h"
#include "bind/core_bind.h"
#include "io/zip_io.h"


// plugins
#include "plugins/sprite_frames_editor_plugin.h"
#include "plugins/canvas_item_editor_plugin.h"
#include "plugins/spatial_editor_plugin.h"
#include "plugins/sample_editor_plugin.h"
#include "plugins/sample_library_editor_plugin.h"
#include "plugins/sample_player_editor_plugin.h"
#include "plugins/camera_editor_plugin.h"
#include "plugins/style_box_editor_plugin.h"
#include "plugins/resource_preloader_editor_plugin.h"
#include "plugins/item_list_editor_plugin.h"
#include "plugins/stream_editor_plugin.h"
#include "plugins/multimesh_editor_plugin.h"
#include "plugins/mesh_editor_plugin.h"
#include "plugins/theme_editor_plugin.h"

#include "plugins/tile_map_editor_plugin.h"
#include "plugins/cube_grid_theme_editor_plugin.h"
#include "plugins/shader_editor_plugin.h"
#include "plugins/shader_graph_editor_plugin.h"
#include "plugins/path_editor_plugin.h"
#include "plugins/rich_text_editor_plugin.h"
#include "plugins/collision_polygon_editor_plugin.h"
#include "plugins/collision_polygon_2d_editor_plugin.h"
#include "plugins/script_editor_plugin.h"
#include "plugins/path_2d_editor_plugin.h"
#include "plugins/particles_editor_plugin.h"
#include "plugins/particles_2d_editor_plugin.h"
#include "plugins/animation_tree_editor_plugin.h"
#include "plugins/tile_set_editor_plugin.h"
#include "plugins/animation_player_editor_plugin.h"
#include "plugins/baked_light_editor_plugin.h"
#include "plugins/polygon_2d_editor_plugin.h"
// end
#include "tools/editor/io_plugins/editor_texture_import_plugin.h"
#include "tools/editor/io_plugins/editor_scene_import_plugin.h"
#include "tools/editor/io_plugins/editor_font_import_plugin.h"
#include "tools/editor/io_plugins/editor_sample_import_plugin.h"
#include "tools/editor/io_plugins/editor_translation_import_plugin.h"
#include "tools/editor/io_plugins/editor_mesh_import_plugin.h"



EditorNode *EditorNode::singleton=NULL;

void EditorNode::_update_title() {

	String appname = Globals::get_singleton()->get("application/name");
	String title = appname.empty()?String(VERSION_FULL_NAME):String(_MKSTR(VERSION_NAME) + String(" - ") + appname);
	String edited = edited_scene?edited_scene->get_filename():String();
	if (!edited.empty())
		title+=" - " + String(edited.get_file());
	if (unsaved_cache)
		title+=" (*)";

	OS::get_singleton()->set_window_title(title);

}

void EditorNode::_unhandled_input(const InputEvent& p_event) {

	if (p_event.type==InputEvent::KEY && p_event.key.pressed && !p_event.key.echo) {

		switch(p_event.key.scancode) {

			case KEY_F1:
				if (!p_event.key.mod.shift && !p_event.key.mod.command)
					_editor_select(3);
			break;
			case KEY_F2: _editor_select(0); break;
			case KEY_F3: _editor_select(1); break;
			case KEY_F4: _editor_select(2); break;
			case KEY_F5: _menu_option_confirm((p_event.key.mod.control&&p_event.key.mod.shift)?RUN_PLAY_CUSTOM_SCENE:RUN_PLAY,true); break;
			case KEY_F6: _menu_option_confirm(RUN_PLAY_SCENE,true); break;
			case KEY_F7: _menu_option_confirm(RUN_PAUSE,true); break;
			case KEY_F8: _menu_option_confirm(RUN_STOP,true); break;
		}

	}
}



void EditorNode::_notification(int p_what) {

	if (p_what==NOTIFICATION_EXIT_TREE) {
		editor_data.save_editor_external_data();

		log->deinit(); // do not get messages anymore
	}
	if (p_what==NOTIFICATION_PROCESS) {
	
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
			opening_prev=false;

		if (unsaved_cache != (saved_version!=editor_data.get_undo_redo().get_version())) {

			unsaved_cache = (saved_version!=editor_data.get_undo_redo().get_version());
			_update_title();
		}

		//get_root_node()->set_rect(viewport->get_global_rect());

		//update the circle
		uint64_t frame = OS::get_singleton()->get_frames_drawn();
		uint32_t tick = OS::get_singleton()->get_ticks_msec();

		if (frame!=circle_step_frame && (tick-circle_step_msec)>(1000/8)) {

			circle_step++;
			if (circle_step>=8)
				circle_step=0;

			circle_step_msec=tick;
			circle_step_frame=frame+1;

			update_menu->set_icon(gui_base->get_icon("Progress"+itos(circle_step+1),"EditorIcons"));

		}

		scene_root->set_size_override(true,Size2(Globals::get_singleton()->get("display/width"),Globals::get_singleton()->get("display/height")));

		editor_selection->update();

		{
			uint32_t p32 = AudioServer::get_singleton()->read_output_peak()>>8;

			float peak = p32==0? -80 : Math::linear2db(p32 / 65535.0);

			if (peak<-80)
				peak=-80;
			float vu = audio_vu->get_val();

			if (peak > vu) {
				audio_vu->set_val(peak);
			} else {
				float new_vu = vu - get_process_delta_time()*70.0;
				if (new_vu<-80)
					new_vu=-80;
				if (new_vu !=-80 && vu !=-80)
					audio_vu->set_val(new_vu);
			}

		}
	
	}
	if (p_what==NOTIFICATION_ENTER_TREE) {

		//MessageQueue::get_singleton()->push_call(this,"_get_scene_metadata");
		get_tree()->set_editor_hint(true);				
		get_tree()->get_root()->set_as_audio_listener(false);
		get_tree()->get_root()->set_as_audio_listener_2d(false);
		get_tree()->set_auto_accept_quit(false);
				//VisualServer::get_singleton()->viewport_set_hide_canvas(editor->get_scene_root()->get_viewport(),false);

		//import_monitor->scan_changes();
	}
	if (p_what==NOTIFICATION_READY) {

		VisualServer::get_singleton()->viewport_set_hide_scenario(get_scene_root()->get_viewport(),true);
		VisualServer::get_singleton()->viewport_set_hide_canvas(get_scene_root()->get_viewport(),true);
		_editor_select(1);

		if (defer_load_scene!="") {

			load_scene(defer_load_scene);
			defer_load_scene="";
		}

		if (defer_translatable!="") {

			Error ok = save_translatable_strings(defer_translatable);
			if (ok!=OK)
				OS::get_singleton()->set_exit_code(255);
			defer_translatable="";
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

		if (export_defer.platform!="") {

			project_export_settings->export_platform(export_defer.platform,export_defer.path,export_defer.debug,export_defer.password,true);
			export_defer.platform="";
		}

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

		if (bool(EDITOR_DEF("resources/auto_reload_modified_images",true))) {

			_menu_option_confirm(DEPENDENCY_LOAD_CHANGED_IMAGES,true);
		}

		EditorFileSystem::get_singleton()->scan_sources();

	}

	if (p_what == MainLoop::NOTIFICATION_WM_QUIT_REQUEST) {

		_menu_option_confirm(FILE_QUIT, false);
	};

}

void EditorNode::_fs_changed() {

	for(Set<FileDialog*>::Element *E=file_dialogs.front();E;E=E->next()) {

		E->get()->invalidate();
	}
}

void EditorNode::_sources_changed(bool p_exist) {

	if (p_exist) {

		sources_button->set_icon(gui_base->get_icon("DependencyChanged","EditorIcons"));
		sources_button->set_disabled(false);

	} else {

		sources_button->set_icon(gui_base->get_icon("DependencyOk","EditorIcons"));
		sources_button->set_disabled(true);

	}
}

void EditorNode::_vp_resized() {


}

void EditorNode::_node_renamed() {

	if (property_editor)
		property_editor->update_tree();
}


Error EditorNode::load_resource(const String& p_scene) {

	RES res = ResourceLoader::load(p_scene);
	ERR_FAIL_COND_V(!res.is_valid(),ERR_CANT_OPEN);

	edit_resource(res);

	return OK;
}


void EditorNode::edit_resource(const Ref<Resource>& p_resource) {

	_resource_selected(p_resource,"");
}

void EditorNode::edit_node(Node *p_node) {

	push_item(p_node);
}

void EditorNode::open_resource(const String& p_type) {


	file->set_mode(FileDialog::MODE_OPEN_FILE);

	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type(p_type,&extensions);

	file->clear_filters();
	for(int i=0;i<extensions.size();i++) {

		file->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
	}

	//file->set_current_path(current_path);

	file->popup_centered_ratio();
	current_option=RESOURCE_LOAD;
}

void EditorNode::save_resource(const Ref<Resource>& p_resource) {


	ERR_FAIL_COND(p_resource.is_null() || p_resource->get_path()=="" || p_resource->get_path().find("::")!=-1);
	resources_dock->save_resource(p_resource->get_path(),p_resource);
}

void EditorNode::save_resource_as(const Ref<Resource>& p_resource) {

	resources_dock->save_resource_as(p_resource);

}



void EditorNode::_menu_option(int p_option) {
	
	_menu_option_confirm(p_option,false);
}

void EditorNode::_menu_confirm_current() {
	
	_menu_option_confirm(current_option,true);
}


void EditorNode::_dialog_display_file_error(String p_file,Error p_error) {


	if (p_error) {
		
		current_option=-1;
		//accept->"()->hide();
		accept->get_ok()->set_text("I see..");
		
		switch(p_error) {
		
			case ERR_FILE_CANT_WRITE: {

				accept->set_text("Can't open file for writing: "+p_file.extension());
			} break;
			case ERR_FILE_UNRECOGNIZED: {
			
				accept->set_text("File format requested unknown: "+p_file.extension());
			} break;
			default: {
			
				accept->set_text("Error Saving.");
			}break;
		}
					
		accept->popup_centered(Size2(300,70));;
	}

}

void EditorNode::_get_scene_metadata() {

	Node *scene = edited_scene;

	if (!scene)
		return;


	if (scene->has_meta("__editor_plugin_states__")) {

		Dictionary md = scene->get_meta("__editor_plugin_states__");
		editor_data.set_editor_states(md);

	}

	if (scene->has_meta("__editor_run_settings__")) {

		Dictionary md = scene->get_meta("__editor_run_settings__");
		if (md.has("run_mode"))
			run_settings_dialog->set_run_mode(md["run_mode"]);
		if (md.has("custom_args"))
			run_settings_dialog->set_custom_arguments(md["custom_args"]);
	}


}

void EditorNode::_set_scene_metadata() {

	Node *scene = edited_scene;

	if (!scene)
		return;

	{ /* Editor States */
		Dictionary md = editor_data.get_editor_states();

		if (!md.empty()) {
			scene->set_meta("__editor_plugin_states__",md);
		}
	}

	{ /* Run Settings */


		Dictionary md;
		md["run_mode"]=run_settings_dialog->get_run_mode();
		md["custom_args"]=run_settings_dialog->get_custom_arguments();
		scene->set_meta("__editor_run_settings__",md);
	}




}

static Error _fix_object_paths(Object* obj, Node* root, String save_path) {

	Globals* g = Globals::get_singleton();

	String import_dir = root->get_meta("__editor_import_file__");
	import_dir = import_dir.get_base_dir();
	import_dir = DirAccess::normalize_path(import_dir);
	if (import_dir[import_dir.length()-1] != '/') {
		import_dir = import_dir + "/";
	};

	String resource_dir = DirAccess::normalize_path(g->get_resource_path());
	if (resource_dir[resource_dir.length()-1] != '/') {
		resource_dir = resource_dir + "/";
	};


	List<PropertyInfo> list;
	obj->get_property_list(&list, false);

	List<PropertyInfo>::Element *E = list.front();

	while (E) {

		Variant v = obj->get(E->get().name);
		if (v.get_type() == Variant::OBJECT) {

			Ref<Resource> res = (RefPtr)v;
			if (res.is_null()) {
				E = E->next();
				continue;
			}

			if (res->get_path() != "") {

				String res_path = res->get_path();
				res_path = Globals::get_singleton()->globalize_path(res_path);
				res_path = DirAccess::normalize_path(res_path);

				if (res_path.find(resource_dir) != 0) {

					// path of resource is not inside engine's resource path

					String new_path;

					if (res_path.find(import_dir) == 0) {

						// path of resource is relative to path of import file
						new_path = save_path + "/" + res_path.substr(import_dir.length(), res_path.length() - import_dir.length());

					} else {

						// path of resource is not relative to import file
						new_path = save_path + "/" + res_path.get_file();
					};

					res->set_path(g->localize_path(new_path));
					DirAccess* d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
					d->make_dir_recursive(new_path.get_base_dir());
					printf("copying from %ls to %ls\n", res_path.c_str(), new_path.c_str());
					Error err = d->copy(res_path, new_path);
					memdelete(d);
					ERR_FAIL_COND_V(err != OK, err);
				}

			} else {

				_fix_object_paths(res.operator->(), root, save_path);
			};
		};


		E = E->next();
	};

	return OK;
};

static Error _fix_imported_scene_paths(Node* node, Node* root, String save_path) {

	if (node == root || node->get_owner() == root) {
		Error e = _fix_object_paths(node, root, save_path);
		ERR_FAIL_COND_V(e != OK, e);
	};

	for (int i=0; i<node->get_child_count(); i++) {

		Error e = _fix_imported_scene_paths(node->get_child(i), root, save_path);
		ERR_FAIL_COND_V(e != OK, e);
	};

	return OK;
};


bool EditorNode::_find_and_save_edited_subresources(Object *obj,Set<RES>& processed,int32_t flags) {

	bool ret_changed=false;
	List<PropertyInfo> pi;
	obj->get_property_list(&pi);
	for (List<PropertyInfo>::Element *E=pi.front();E;E=E->next()) {

		if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
			continue;

		switch(E->get().type) {
			case Variant::OBJECT: {

				RES res = obj->get(E->get().name);

				if (res.is_null() || processed.has(res))
					break;

				processed.insert(res);

				bool changed = res->is_edited();
				res->set_edited(false);

				bool subchanged = _find_and_save_edited_subresources(res.ptr(),processed,flags);

				if (res->get_path().is_resource_file()) {
					if (changed || subchanged) {
						//save
						print_line("Also saving modified external resource: "+res->get_path());
						Error err = ResourceSaver::save(res->get_path(),res,flags);

					}
				} else {

					ret_changed=true;
				}


			} break;
			case Variant::ARRAY: {

				/*Array varray=p_variant;
				int len=varray.size();
				for(int i=0;i<len;i++) {

					Variant v=varray.get(i);
					_find_resources(v);
				}*/

			} break;
			case Variant::DICTIONARY: {

				/*
				Dictionary d=p_variant;
				List<Variant> keys;
				d.get_key_list(&keys);
				for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

					Variant v = d[E->get()];
					_find_resources(v);
				} */
			} break;
			default: {}
		}

	}

	return ret_changed;

}

void EditorNode::_save_edited_subresources(Node* scene,Set<RES>& processed,int32_t flags) {

	_find_and_save_edited_subresources(scene,processed,flags);

	for(int i=0;i<scene->get_child_count();i++) {

		Node *n = scene->get_child(i);
		if (n->get_owner()!=edited_scene)
			continue;
		_save_edited_subresources(n,processed,flags);
	}

}

void EditorNode::_save_scene(String p_file) {

	Node *scene = edited_scene;

	if (!scene) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("I see..");
		accept->set_text("This operation can't be done without a tree root.");
		accept->popup_centered(Size2(300,70));;
		return;
	}

	editor_data.apply_changes_in_editors();

	if (editor_plugin_screen) {
		scene->set_meta("__editor_plugin_screen__",editor_plugin_screen->get_name());
	}


	_set_scene_metadata();
	Ref<PackedScene> sdata = memnew( PackedScene );
	Error err = sdata->pack(scene);


	if (err!=OK) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("I see..");
		accept->set_text("Couldn't save scene. Likely dependencies (instances) couldn't be satisfied.");
		accept->popup_centered(Size2(300,70));;
		return;
	}

	sdata->set_import_metadata(scene_import_metadata);
	int flg=0;
	if (EditorSettings::get_singleton()->get("on_save/compress_binary_resources"))
		flg|=ResourceSaver::FLAG_COMPRESS;
	if (EditorSettings::get_singleton()->get("on_save/save_paths_as_relative"))
		flg|=ResourceSaver::FLAG_RELATIVE_PATHS;
	flg|=ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;


	err = ResourceSaver::save(p_file,sdata,flg);
	Set<RES> processed;
	_save_edited_subresources(scene,processed,flg);
	editor_data.save_editor_external_data();
	if (err==OK) {
		scene->set_filename( Globals::get_singleton()->localize_path(p_file) );
		//EditorFileSystem::get_singleton()->update_file(p_file,sdata->get_type());
		saved_version=editor_data.get_undo_redo().get_version();
		_update_title();
	} else {

		_dialog_display_file_error(p_file,err);
	}


};




void EditorNode::_import_action(const String& p_action) {
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
			accept->popup_centered(Size2(300,70));;
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
			accept->popup_centered(Size2(300,70));;
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
		accept->popup_centered(Size2(300,70));;
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
	

	switch(current_option) {
	
		case RESOURCE_LOAD: {

			RES res = ResourceLoader::load(p_file);
			if (res.is_null()) {

				current_option=-1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text("ok :(");
				accept->set_text("Failed to load resource.");
				return;
			};


			push_item(res.operator->() );
		} break;
		case FILE_OPEN_SCENE: {


			load_scene(p_file);
		} break;

		case FILE_SAVE_OPTIMIZED: {



		} break;
		case FILE_RUN_SCRIPT: {

			print_line("RUN: "+p_file);
			Ref<Script> scr = ResourceLoader::load(p_file,"Script",true);
			if (scr.is_null()) {
				add_io_error("Script Failed to Load:\n"+p_file);
				return;
			}
			if (!scr->is_tool()) {

				add_io_error("Script is not tool, will not be able to run:\n"+p_file);
				return;
			}

			Ref<EditorScript> es = memnew( EditorScript );
			es->set_script(scr.get_ref_ptr());
			es->set_editor(this);
			es->_run();

			get_undo_redo()->clear_history();
		} break;
		case FILE_DUMP_STRINGS: {

			save_translatable_strings(p_file);

		} break;
		case FILE_SAVE_SUBSCENE: {

			List<Node*> selection = editor_selection->get_selected_node_list();

			if (selection.size()!=1) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation requieres a single selected node.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			Node *base = selection.front()->get();

			Map<Node*,Node*> reown;
			reown[get_edited_scene()]=base;
			Node *copy = base->duplicate_and_reown(reown);
			if (copy) {

				Ref<PackedScene> sdata = memnew( PackedScene );
				Error err = sdata->pack(copy);
				memdelete(copy);

				if (err!=OK) {


					current_option=-1;
					//accept->get_cancel()->hide();
					accept->get_ok()->set_text("I see..");
					accept->set_text("Couldn't save subscene. Likely dependencies (instances) couldn't be satisfied.");
					accept->popup_centered(Size2(300,70));;
					return;
				}

				int flg=0;
				if (EditorSettings::get_singleton()->get("on_save/compress_binary_resources"))
					flg|=ResourceSaver::FLAG_COMPRESS;
				if (EditorSettings::get_singleton()->get("on_save/save_paths_as_relative"))
					flg|=ResourceSaver::FLAG_RELATIVE_PATHS;


				err = ResourceSaver::save(p_file,sdata,flg);
				if (err!=OK) {

					current_option=-1;
					//confirmation->get_cancel()->hide();
					accept->get_ok()->set_text("I see..");
					accept->set_text("Error saving scene.");
					accept->popup_centered(Size2(300,70));;
					break;
				}
		//EditorFileSystem::get_singleton()->update_file(p_file,sdata->get_type());

            } else {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("Error duplicating scene to save it.");
				accept->popup_centered(Size2(300,70));;
				break;

			}


		} break;


		case FILE_SAVE_SCENE:
		case FILE_SAVE_AS_SCENE: {

			if (file->get_mode()==FileDialog::MODE_SAVE_FILE) {

				_save_scene(p_file);
			}

		} break;

		case FILE_SAVE_AND_RUN: {
			if (file->get_mode()==FileDialog::MODE_SAVE_FILE) {

				_save_scene(p_file);
				_run(false);
			}
		} break;

		case FILE_EXPORT_MESH_LIBRARY: {

			Ref<MeshLibrary> ml;
			if (file_export_lib_merge->is_pressed() && FileAccess::exists(p_file)) {
				ml=ResourceLoader::load(p_file,"MeshLibrary");

				if (ml.is_null()) {
					current_option=-1;
					//accept->get_cancel()->hide();
					accept->get_ok()->set_text("I see..");
					accept->set_text("Can't load MeshLibrary for merging!.");
					accept->popup_centered(Size2(300,70));;
					return;
				}

			}

			if (ml.is_null()) {
				ml = Ref<MeshLibrary>( memnew( MeshLibrary ));
			}

			MeshLibraryEditor::update_library_file(edited_scene,ml,true);

			Error err = ResourceSaver::save(p_file,ml);
			if (err) {

				accept->get_ok()->set_text("I see..");
				accept->set_text("Error saving MeshLibrary!.");
				accept->popup_centered(Size2(300,70));;
				return;
			}


		} break;
		case FILE_EXPORT_TILESET: {

			Ref<TileSet> ml;
			if (file_export_lib_merge->is_pressed() && FileAccess::exists(p_file)) {
				ml=ResourceLoader::load(p_file,"TileSet");

				if (ml.is_null()) {
					current_option=-1;
					//accept->get_cancel()->hide();
					accept->get_ok()->set_text("I see..");
					accept->set_text("Can't load TileSet for merging!.");
					accept->popup_centered(Size2(300,70));;
					return;
				}

			}

			if (ml.is_null()) {
				ml = Ref<TileSet>( memnew( TileSet ));
			}

			TileSetEditor::update_library_file(edited_scene,ml,true);

			Error err = ResourceSaver::save(p_file,ml);
			if (err) {

				accept->get_ok()->set_text("I see..");
				accept->set_text("Error saving TileSet!.");
				accept->popup_centered(Size2(300,70));;
				return;
			}
		} break;

		case SETTINGS_LOAD_EXPORT_TEMPLATES: {

			FileAccess *fa=NULL;
			zlib_filefunc_def io = zipio_create_io_from_file(&fa);

			unzFile pkg = unzOpen2(p_file.utf8().get_data(), &io);
			if (!pkg) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("Can't open export templates zip.");
				accept->popup_centered(Size2(300,70));;
				return;

			}
			int ret = unzGoToFirstFile(pkg);

			int fc=0; //coun them

			while(ret==UNZ_OK) {
				fc++;
				ret = unzGoToNextFile(pkg);

			}

			ret = unzGoToFirstFile(pkg);

			EditorProgress p("ltask","Loading Export Templates",fc);
			print_line("BEGIN IMPORT");

			fc=0;

			while(ret==UNZ_OK) {

				//get filename
				unz_file_info info;
				char fname[16384];
				ret = unzGetCurrentFileInfo(pkg,&info,fname,16384,NULL,0,NULL,0);


				String file=fname;

				Vector<uint8_t> data;
				data.resize(info.uncompressed_size);

				//read
				ret = unzOpenCurrentFile(pkg);
				ret = unzReadCurrentFile(pkg,data.ptr(),data.size());
				unzCloseCurrentFile(pkg);

				print_line(fname);
				//for(int i=0;i<512;i++) {
				//	print_line(itos(data[i]));
				//}

				file=file.get_file();

				p.step("Importing: "+file,fc);
				print_line("IMPORT "+file);

				FileAccess *f = FileAccess::open(EditorSettings::get_singleton()->get_settings_path()+"/templates/"+file,FileAccess::WRITE);

				ERR_CONTINUE(!f);
				f->store_buffer(data.ptr(),data.size());

				memdelete(f);

				ret = unzGoToNextFile(pkg);
				fc++;
			}

			unzClose(pkg);

		} break;

		default: { //save scene?
		
			
			if (file->get_mode()==FileDialog::MODE_SAVE_FILE) {

				_save_scene(p_file);
			}
			
		} break;
	}
}



void EditorNode::push_item(Object *p_object,const String& p_property) {


	if (!p_object) {
		property_editor->edit(NULL);
		scene_tree_dock->set_selected(NULL);
		return;
	}

	uint32_t id = p_object->get_instance_ID();
	if (id!=editor_history.get_current()) {

		if (p_property=="")
			editor_history.add_object(id);
		else
			editor_history.add_object(id,p_property);
	}

	_edit_current();

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

	Node *scene = edited_scene;
	set_edited_scene(p_node);

	if (scene) {
		String path = scene->get_filename();
		p_node->set_filename(path);
		memdelete(scene);
	}



}




void EditorNode::_hide_top_editors() {

	if (editor_plugin_over)
		editor_plugin_over->make_visible(false);
	editor_plugin_over=NULL;
}

void EditorNode::_edit_current() {
	
	uint32_t current = editor_history.get_current();
	Object *current_obj = current>0 ? ObjectDB::get_instance(current) : NULL;

	this->current=current_obj;
	editor_path->update_path();


	if (!current_obj) {
		
		scene_tree_dock->set_selected(NULL);
		property_editor->edit( NULL );
		object_menu->set_disabled(true);
		return;
	}

	object_menu->set_disabled(true);

	if (current_obj->is_type("Resource")) {


		Resource *current_res = current_obj->cast_to<Resource>();
		ERR_FAIL_COND(!current_res);
		scene_tree_dock->set_selected(NULL);
		property_editor->edit( current_res );
		object_menu->set_disabled(false);

		resources_dock->add_resource(Ref<Resource>(current_res));
		top_pallete->set_current_tab(1);
	}


	if (current_obj->is_type("Node")) {

		Node * current_node = current_obj->cast_to<Node>();
		ERR_FAIL_COND(!current_node);
		ERR_FAIL_COND(!current_node->is_inside_tree());




		property_editor->edit( current_node );
		scene_tree_dock->set_selected(current_node);
		object_menu->get_popup()->clear();

		top_pallete->set_current_tab(0);

	}

	/* Take care of PLUGIN EDITOR */


	EditorPlugin *main_plugin = editor_data.get_editor(current_obj);

	if (main_plugin) {

		if (main_plugin!=editor_plugin_screen) {

			// update screen main_plugin
			if (editor_plugin_screen)
				editor_plugin_screen->make_visible(false);
			editor_plugin_screen=main_plugin;
			editor_plugin_screen->edit(current_obj);
			editor_plugin_screen->make_visible(true);


			for(int i=0;i<editor_table.size();i++) {
				if (editor_table[i]==main_plugin) {
					main_editor_tabs->set_current_tab(i);
					break;
				}
			}

		} else {

			editor_plugin_screen->edit(current_obj);
		}

	}

	EditorPlugin *sub_plugin = editor_data.get_subeditor(current_obj);

	if (sub_plugin) {


		if (editor_plugin_over)
			editor_plugin_over->make_visible(false);
		editor_plugin_over=sub_plugin;
		editor_plugin_over->edit(current_obj);
		editor_plugin_over->make_visible(true);
	} else if (editor_plugin_over) {

		editor_plugin_over->make_visible(false);
		editor_plugin_over=NULL;

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

	PopupMenu *p=object_menu->get_popup();

	p->clear();
	p->add_item("Copy Params",OBJECT_COPY_PARAMS);
	p->add_item("Set Params",OBJECT_PASTE_PARAMS);
	p->add_separator();
	p->add_item("Make Resources Unique",OBJECT_UNIQUE_RESOURCES);
	p->add_separator();
	p->add_icon_item(gui_base->get_icon("Help","EditorIcons"),"Class Reference",OBJECT_REQUEST_HELP);
	List<MethodInfo> methods;
	current_obj->get_method_list(&methods);


	if (!methods.empty()) {

		bool found=false;
		List<MethodInfo>::Element *I=methods.front();
		int i=0;
		while(I) {

			if (I->get().flags&METHOD_FLAG_EDITOR) {
				if (!found) {
					p->add_separator();
					found=true;
				}
				p->add_item(I->get().name.capitalize(),OBJECT_METHOD_BASE+i);
			}
			i++;
			I=I->next();
		}
	}

	//p->add_separator();
	//p->add_item("All Methods",OBJECT_CALL_METHOD);


	_update_keying();
}

void EditorNode::_resource_selected(const RES& p_res,const String& p_property) {
	

	if (p_res.is_null())
		return;

	RES r=p_res;
	push_item(r.operator->(),p_property);

}


void EditorNode::_run(bool p_current,const String& p_custom) {

	if (editor_run.get_status()==EditorRun::STATUS_PLAY) {

		play_button->set_pressed(!_playing_edited);
		play_scene_button->set_pressed(_playing_edited);
		return;
	}

	play_button->set_pressed(false);
	pause_button->set_pressed(false);
	play_scene_button->set_pressed(false);

	String current_filename;
	String run_filename;
	String args;



	if (p_current || (edited_scene && p_custom==edited_scene->get_filename())) {


		Node *scene = edited_scene;

		if (!scene) {

			current_option=-1;
			//accept->get_cancel()->hide();
			accept->get_ok()->set_text("I see..");
			accept->set_text("No scene to run exists.");
			accept->popup_centered(Size2(300,70));;
			return;
		}

		if (scene->get_filename()=="") {
			current_option=-1;
			//accept->get_cancel()->hide();
			/**/
			_menu_option_confirm(FILE_SAVE_BEFORE_RUN, false);
			return;

		}

		bool autosave = EDITOR_DEF("run/auto_save_before_running",true);

		if (autosave) {

			_menu_option(FILE_SAVE_SCENE);
		}

		if (run_settings_dialog->get_run_mode()==RunSettingsDialog::RUN_LOCAL_SCENE) {

			run_filename=scene->get_filename();
		} else {
			args=run_settings_dialog->get_custom_arguments();
			current_filename=scene->get_filename();
		}

	} else if (p_custom!="") {

		run_filename=p_custom;
	}

	if (run_filename=="") {

		//evidently, run the scene
		run_filename=GLOBAL_DEF("application/main_scene","");
		if (run_filename=="") {

			current_option=-1;
			//accept->get_cancel()->hide();
			accept->get_ok()->set_text("I see..");
			accept->set_text("No main scene has ever been defined.\nSelect one from \"Project Settings\" under the 'application' category.");
			accept->popup_centered(Size2(300,100));;
			return;
		}

	}


	if (bool(EDITOR_DEF("run/auto_save_before_running",true))) {

		if (unsaved_cache) {

			Node *scene = edited_scene;

			if (scene) { //only autosave if there is a scene obviously

				if (scene->get_filename()=="") {

					current_option=-1;
					//accept->get_cancel()->hide();
					accept->get_ok()->set_text("I see..");
					accept->set_text("Current scene was never saved, please save scene before running.");
					accept->popup_centered(Size2(300,70));;
					return;
				}

				_save_scene(scene->get_filename());
			}
		}

		editor_data.save_editor_external_data();
	}


	List<String> breakpoints;
	editor_data.get_editor_breakpoints(&breakpoints);

	Error error = editor_run.run(run_filename,args,breakpoints,current_filename);

	if (error!=OK) {

		current_option=-1;
		//confirmation->get_cancel()->hide();
		accept->get_ok()->set_text("I see..");
		accept->set_text("Could not start subprocess!");
		accept->popup_centered(Size2(300,70));;
		return;

	}

	emit_signal("play_pressed");
	if (p_current) {
		play_scene_button->set_pressed(true);
	} else {
		play_button->set_pressed(true);
	}

	_playing_edited=p_current;

}

void EditorNode::_cleanup_scene() {


	Node *scene = edited_scene;
	editor_selection->clear();
	editor_data.clear_editor_states();
	editor_history.clear();
	_hide_top_editors();
	animation_editor->cleanup();
	property_editor->edit(NULL);
	resources_dock->cleanup();
	scene_import_metadata.unref();
	set_edited_scene(NULL);
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

}

void EditorNode::_menu_option_confirm(int p_option,bool p_confirmed) {
	
	current_option=(MenuOptions)p_option;


	switch( p_option ) {
		case FILE_NEW_SCENE: {

			if (!p_confirmed) {
				confirmation->get_ok()->set_text("Yes");
				//confirmation->get_cancel()->show();
				confirmation->set_text("Start a New Scene? (Current will be lost)");
				confirmation->popup_centered(Size2(300,70));
				break;
			}


			_cleanup_scene();

			
		} break;
		case FILE_OPEN_SCENE: {
			
			
			//print_tree();
			file->set_mode(FileDialog::MODE_OPEN_FILE);
			//not for now?
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene",&extensions);
			file->clear_filters();
			for(int i=0;i<extensions.size();i++) {
				
				file->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
			}
			

			//file->set_current_path(current_path);
			Node *scene = edited_scene;
			if (scene) {
				file->set_current_path(scene->get_filename());
			};
			file->set_title("Open Scene");
			file->popup_centered_ratio();
			
		} break;
		case FILE_QUICK_OPEN_SCENE: {

			quick_open->popup("PackedScene");
			quick_open->set_title("Quick Open Scene..");

		} break;
		case FILE_QUICK_OPEN_SCRIPT: {


			quick_open->popup("Script");
			quick_open->set_title("Quick Open Script..");

		} break;
		case FILE_RUN_SCRIPT: {

			file_script->popup_centered_ratio();
		} break;
		case FILE_OPEN_PREV: {

			if (previous_scenes.empty())
				break;
			opening_prev=true;
			open_request(previous_scenes.back()->get());

		} break;
		case FILE_SAVE_SCENE: {


			Node *scene = edited_scene;
			if (scene && scene->get_filename()!="") {

				_save_scene(scene->get_filename());
				return;
			};
			// fallthrough to save_as
		};
		case FILE_SAVE_AS_SCENE: {
			
			Node *scene = edited_scene;
			
			if (!scene) {
				
				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a tree root.");
				accept->popup_centered(Size2(300,70));;
				break;				
			}
			
			file->set_mode(FileDialog::MODE_SAVE_FILE);
			bool relpaths = (scene->has_meta("__editor_relpaths__") && scene->get_meta("__editor_relpaths__").operator bool());


			List<String> extensions;
			Ref<PackedScene> sd = memnew( PackedScene );
			ResourceSaver::get_recognized_extensions(sd,&extensions);
			file->clear_filters();
			for(int i=0;i<extensions.size();i++) {

				file->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
			}
			
			//file->set_current_path(current_path);
			if (scene->get_filename()!="") {
				file->set_current_path(scene->get_filename());
				if (extensions.size()) {
					String ext=scene->get_filename().extension().to_lower();
					if (extensions.find(ext)==NULL) {
						file->set_current_path(scene->get_filename().replacen("."+ext,"."+extensions.front()->get()));
					}
				}
			} else {

				String existing;
				if (extensions.size()) {
					existing="new_scene."+extensions.front()->get().to_lower();
				}
				file->set_current_path(existing);

			}
			file->popup_centered_ratio();
			file->set_title("Save Scene As..");
			
		} break;

		case FILE_SAVE_BEFORE_RUN: {
			if (!p_confirmed) {
				accept->get_ok()->set_text("Yes");
				accept->set_text("This scene has never been saved. Save before running?");
				accept->popup_centered(Size2(300, 70));
				break;
			}

			_menu_option(FILE_SAVE_AS_SCENE);
			_menu_option_confirm(FILE_SAVE_AND_RUN, true);
		} break;

		case FILE_DUMP_STRINGS: {

			Node *scene = edited_scene;

			if (!scene) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a tree root.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			String cpath;
			if (scene->get_filename()!="") {
				cpath = scene->get_filename();

				String fn = cpath.substr(0,cpath.length() - cpath.extension().size());
				String ext=cpath.extension();
				cpath=fn+".pot";


			} else {
				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("Please save the scene first.");
				accept->popup_centered(Size2(300,70));;
				break;

			}

			bool relpaths = (scene->has_meta("__editor_relpaths__") && scene->get_meta("__editor_relpaths__").operator bool());

			file->set_mode(FileDialog::MODE_SAVE_FILE);

			file->set_current_path(cpath);
			file->set_title("Save Translatable Strings");
			file->popup_centered_ratio();


		} break;

		case FILE_SAVE_SUBSCENE: {

			Node *scene = edited_scene;

			if (!scene) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a scene.");
				accept->popup_centered(Size2(300,70));;
				break;
			}


			List<Node*> selection = editor_selection->get_selected_node_list();

			if (selection.size()!=1) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation requieres a single selected node.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			Node *tocopy = selection.front()->get();

			if (tocopy!=edited_scene && tocopy->get_filename()!="") {


				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done on instanced scenes.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			file->set_mode(FileDialog::MODE_SAVE_FILE);

			List<String> extensions;
			Ref<PackedScene> sd = memnew( PackedScene );
			ResourceSaver::get_recognized_extensions(sd,&extensions);
			file->clear_filters();
			for(int i=0;i<extensions.size();i++) {

				file->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
			}


			String existing;
			if (extensions.size()) {
				existing="new_scene."+extensions.front()->get().to_lower();
			}
			file->set_current_path(existing);


			file->popup_centered_ratio();
			file->set_title("Save Sub-Scene As..");
		} break;
		case FILE_SAVE_OPTIMIZED: {
			Node *scene = edited_scene;
#if 0
			if (!scene) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a tree root.");
				accept->popup_centered(Size2(300,70));;
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
				accept->popup_centered(Size2(300,70));;
				break;

			}
#endif
		} break;

		case FILE_EXPORT_PROJECT: {

			project_export_settings->popup_centered_ratio();
			/*
			String target = export_db->get_current_platform();
			Ref<EditorExporter> exporter = export_db->get_exporter(target);
			if (exporter.is_null()) {
				accept->set_text("No exporter for platform '"+target+"' yet.");
				accept->popup_centered(Size2(300,70));;
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

			if (!edited_scene) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a scene.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			List<String> extensions;
			Ref<MeshLibrary> ml( memnew( MeshLibrary) );
			ResourceSaver::get_recognized_extensions(ml,&extensions);
			file_export_lib->clear_filters();
			for(List<String>::Element *E=extensions.front();E;E=E->next()) {
				file_export_lib->add_filter("*."+E->get());
			}

			file_export_lib->popup_centered_ratio();
			file_export_lib->set_title("Export Mesh Library");

		} break;
		case FILE_EXPORT_TILESET: {

			List<String> extensions;
			Ref<TileSet> ml( memnew( TileSet) );
			ResourceSaver::get_recognized_extensions(ml,&extensions);
			file_export_lib->clear_filters();
			for(List<String>::Element *E=extensions.front();E;E=E->next()) {
				file_export_lib->add_filter("*."+E->get());
			}

			file_export_lib->popup_centered_ratio();
			file_export_lib->set_title("Export Tile Set");

		} break;

		case SETTINGS_EXPORT_PREFERENCES: {

			//project_export_settings->popup_centered_ratio();
		} break;
		case FILE_IMPORT_SUBSCENE: {

			//import_subscene->popup_centered_ratio();

			if (!edited_scene) {

				current_option=-1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a selected node.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			scene_tree_dock->import_subscene();

		} break;

		case FILE_QUIT: {
			
			if (!p_confirmed) {
				confirmation->get_ok()->set_text("Quit");
				//confirmation->get_cancel()->show();
				confirmation->set_text("Exit the Editor?");
				confirmation->popup_centered(Size2(300,70));
				break;
			}

			_menu_option_confirm(RUN_STOP,true);
			get_tree()->quit();
				
		} break;
		case FILE_EXTERNAL_OPEN_SCENE: {

			if (unsaved_cache && !p_confirmed) {

				confirmation->get_ok()->set_text("Open");
				//confirmation->get_cancel()->show();
				confirmation->set_text("Current scene not saved. Open anyway?");
				confirmation->popup_centered(Size2(300,70));
				break;

			}

			bool oprev=opening_prev;
			Error err = load_scene(external_file);
			if (err == OK && oprev) {
				previous_scenes.pop_back();
				opening_prev=false;
			}

		} break;

		case EDIT_UNDO: {

			if (OS::get_singleton()->get_mouse_button_state())
				break; // can't undo while mouse buttons are pressed

			String action  = editor_data.get_undo_redo().get_current_action_name();
			if (action!="")
				log->add_message("UNDO: "+action);

			editor_data.get_undo_redo().undo();
		} break;
		case EDIT_REDO: {

			if (OS::get_singleton()->get_mouse_button_state())
				break; // can't redo while mouse buttons are pressed

			editor_data.get_undo_redo().redo();
			String action  = editor_data.get_undo_redo().get_current_action_name();
			if (action!="")
				log->add_message("REDO: "+action);

		} break;
#if 0
		case NODE_EXTERNAL_INSTANCE: {


			if (!edited_scene) {

				current_option=-1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a selected node.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			Node *parent = scene_tree_editor->get_selected();

			if (!parent) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a selected node.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			Node*instanced_scene=SceneLoader::load(external_file,true);

			if (!instanced_scene) {

				current_option=-1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text("Ugh");
				accept->set_text(String("Error loading scene from ")+external_file);
				accept->popup_centered(Size2(300,70));;
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
		
		case OBJECT_REQUEST_HELP: {

			if (current) {
				_editor_select(3);
				emit_signal("request_help",current->get_type());
			}


		} break;
		case OBJECT_COPY_PARAMS: {
			
			editor_data.apply_changes_in_editors();;
			if (current)
				editor_data.copy_object_params(current);
		} break;
		case OBJECT_PASTE_PARAMS: {
			
			editor_data.apply_changes_in_editors();;
			if (current)
				editor_data.paste_object_params(current);
			editor_data.get_undo_redo().clear_history();
		} break;
		case OBJECT_UNIQUE_RESOURCES: {

			editor_data.apply_changes_in_editors();;
			if (current) {
				List<PropertyInfo> props;
				current->get_property_list(&props);
				Map<RES,RES> duplicates;
				for (List<PropertyInfo>::Element *E=props.front();E;E=E->next()) {

					if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
						continue;

					Variant v = current->get(E->get().name);
					if (v.is_ref()) {
						REF ref = v;
						if (ref.is_valid()) {

							RES res = ref;
							if (res.is_valid()) {

								if (!duplicates.has(res)) {
									duplicates[res]=res->duplicate();
								}
								res=duplicates[res];

								current->set(E->get().name,res);
							}

						}
					}

				}
			}

			editor_data.get_undo_redo().clear_history();
			if (editor_plugin_screen) { //reload editor plugin
				editor_plugin_over->edit(NULL);
				editor_plugin_over->edit(current);
			}

		} break;
		case OBJECT_CALL_METHOD: {
		
			editor_data.apply_changes_in_editors();;
			call_dialog->set_object(current);
			call_dialog->popup_centered_ratio();
		} break;
		case RUN_PLAY: {

			_run(false);

		} break;
		case RUN_PLAY_CUSTOM_SCENE: {

			quick_run->popup("PackedScene",true);
			quick_run->set_title("Quick Run Scene..");

		} break;
		case RUN_PAUSE: {

			emit_signal("pause_pressed");

		} break;
		case RUN_STOP: {

			if (editor_run.get_status()==EditorRun::STATUS_STOP)
				break;

			editor_run.stop();
			play_button->set_pressed(false);
			play_scene_button->set_pressed(false);
			pause_button->set_pressed(false);
			emit_signal("stop_pressed");

		} break;
		case RUN_PLAY_SCENE: {

			_run(true);

		} break;
		case RUN_SCENE_SETTINGS: {

			run_settings_dialog->popup_run_settings();
		} break;
		case RUN_SETTINGS: {

			project_settings->popup_project_settings();
		} break;
		case RUN_PROJECT_MANAGER: {

			if (!p_confirmed) {
				confirmation->get_ok()->set_text("Yes");
				confirmation->set_text("Open Project Manager? \n(Unsaved changes will be lost)");
				confirmation->popup_centered(Size2(300,70));
				break;
			}

			get_tree()->quit();
			String exec = OS::get_singleton()->get_executable_path();

			List<String> args;
			args.push_back ( "-path" );
			args.push_back (exec.get_base_dir() );

			OS::ProcessID pid=0;
			Error err = OS::get_singleton()->execute(exec,args,false,&pid);
			ERR_FAIL_COND(err);
		} break;
		case RUN_FILE_SERVER: {

			//file_server
			bool ischecked = fileserver_menu->get_popup()->is_item_checked( fileserver_menu->get_popup()->get_item_index(RUN_FILE_SERVER));

			if (ischecked) {
				file_server->stop();
				fileserver_menu->set_icon(gui_base->get_icon("FileServer","EditorIcons"));
				fileserver_menu->get_popup()->set_item_text( fileserver_menu->get_popup()->get_item_index(RUN_FILE_SERVER),"Enable File Server");
			} else {
				file_server->start();
				fileserver_menu->set_icon(gui_base->get_icon("FileServerActive","EditorIcons"));
				fileserver_menu->get_popup()->set_item_text( fileserver_menu->get_popup()->get_item_index(RUN_FILE_SERVER),"Disable File Server");
			}

			fileserver_menu->get_popup()->set_item_checked( fileserver_menu->get_popup()->get_item_index(RUN_FILE_SERVER),!ischecked);

		} break;
		case RUN_DEPLOY_DUMB_CLIENTS: {

			bool ischecked = fileserver_menu->get_popup()->is_item_checked( fileserver_menu->get_popup()->get_item_index(RUN_DEPLOY_DUMB_CLIENTS));
			fileserver_menu->get_popup()->set_item_checked( fileserver_menu->get_popup()->get_item_index(RUN_DEPLOY_DUMB_CLIENTS),!ischecked);
			run_native->set_deploy_dumb(!ischecked);

		} break;
		case SETTINGS_UPDATE_ALWAYS: {

			update_menu->get_popup()->set_item_checked(0,true);
			update_menu->get_popup()->set_item_checked(1,false);
			OS::get_singleton()->set_low_processor_usage_mode(false);
		} break;
		case SETTINGS_UPDATE_CHANGES: {

			update_menu->get_popup()->set_item_checked(0,false);
			update_menu->get_popup()->set_item_checked(1,true);
			OS::get_singleton()->set_low_processor_usage_mode(true);
		} break;
		case SETTINGS_PREFERENCES: {

			settings_config_dialog->popup_edit_settings();
		} break;
		case SETTINGS_IMPORT: {

			import_settings->popup_import_settings();
		} break;
		case SETTINGS_OPTIMIZED_PRESETS: {

			//optimized_presets->popup_centered_ratio();
		} break;
		case SETTINGS_SHOW_ANIMATION: {

			animation_panel_make_visible( ! animation_panel->is_visible() );

		} break;
		case SETTINGS_LOAD_EXPORT_TEMPLATES: {


			file_templates->popup_centered_ratio();

		} break;
		case SETTINGS_ABOUT: {

			about->popup_centered(Size2(500,130));
		} break;
		case SOURCES_REIMPORT: {

			reimport_dialog->popup_reimport();
		} break;
		case DEPENDENCY_LOAD_CHANGED_IMAGES: {


			List<Ref<Resource> > cached;
			ResourceCache::get_cached_resources(&cached);

			for(List<Ref<Resource> >::Element *E=cached.front();E;E=E->next()) {

				if (!E->get()->can_reload_from_file())
					continue;
				if (!FileAccess::exists(E->get()->get_path()))
					continue;
				uint64_t mt = FileAccess::get_modified_time(E->get()->get_path());
				if (mt!=E->get()->get_last_modified_time()) {
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
					accept->popup_centered(Size2(300,70));;
					break;

				}


				import_reload_fn = scene->get_filename();
				_save_scene(import_reload_fn);
				_cleanup_scene();


			}

*/


		} break;
		default: {
		
			if (p_option>=OBJECT_METHOD_BASE) {

				ERR_FAIL_COND(!current);

				int idx=p_option-OBJECT_METHOD_BASE;

				List<MethodInfo> methods;
				current->get_method_list(&methods);

				ERR_FAIL_INDEX( idx, methods.size() );
				String name=methods[idx].name;

				if (current)
					current->call(name);
			} else if (p_option>=IMPORT_PLUGIN_BASE) {

				Ref<EditorImportPlugin> p = editor_import_export->get_import_plugin(p_option-IMPORT_PLUGIN_BASE);
				if (p.is_valid()) {
					p->import_dialog();
				}

			}
		}
	}		
}


Control* EditorNode::get_viewport() {
	
	return viewport;
}



void EditorNode::_editor_select(int p_which) {

	static bool selecting=false;
	if (selecting)
		return;

	selecting=true;


	ERR_FAIL_INDEX(p_which,editor_table.size());

	main_editor_tabs->set_current_tab(p_which);

	selecting=false;


	EditorPlugin *new_editor = editor_table[p_which];
	ERR_FAIL_COND(!new_editor);

	if (editor_plugin_screen==new_editor)
		return;

	if (editor_plugin_screen) {
		editor_plugin_screen->make_visible(false);
	}

	editor_plugin_screen=new_editor;
	editor_plugin_screen->make_visible(true);
	editor_plugin_screen->selected_notify();
}

void EditorNode::add_editor_plugin(EditorPlugin *p_editor) {


	if (p_editor->has_main_screen()) {
	
		singleton->main_editor_tabs->add_tab(p_editor->get_name());
		singleton->editor_table.push_back(p_editor);
	}
	singleton->editor_data.add_editor_plugin( p_editor );
	singleton->add_child(p_editor);
}


void EditorNode::remove_editor_plugin(EditorPlugin *p_editor) {

	if (p_editor->has_main_screen()) {

		for(int i=0;i<singleton->main_editor_tabs->get_tab_count();i++) {

			if (p_editor->get_name()==singleton->main_editor_tabs->get_tab_title(i)) {

				singleton->main_editor_tabs->remove_tab(i);
				break;
			}
		}

		singleton->main_editor_tabs->add_tab(p_editor->get_name());
		singleton->editor_table.erase(p_editor);
	}
	singleton->remove_child(p_editor);
	singleton->editor_data.remove_editor_plugin( p_editor );

}


void EditorNode::set_edited_scene(Node *p_scene) {

	if (edited_scene) {
		if (edited_scene->get_parent()==scene_root)
			scene_root->remove_child(edited_scene);
		animation_editor->set_root(NULL);
	}	
	edited_scene=p_scene;
	if (edited_scene && edited_scene->cast_to<Popup>())
		edited_scene->cast_to<Popup>()->show(); //show popups
	scene_tree_dock->set_edited_scene(edited_scene);
	if (get_tree())
		get_tree()->set_edited_scene_root(edited_scene);

	if (edited_scene) {
		if (p_scene->get_parent()!=scene_root)
			scene_root->add_child(p_scene);
		animation_editor->set_root(p_scene);
	}


}

void EditorNode::_fetch_translatable_strings(const Object *p_object,Set<StringName>& strings) {


	List<String> tstrings;
	p_object->get_translatable_strings(&tstrings);
	for(List<String>::Element *E=tstrings.front();E;E=E->next())
		strings.insert(E->get());



	const Node * node = p_object->cast_to<Node>();

	if (!node)
		return;

	Ref<Script> script = node->get_script();
	if (script.is_valid())
		_fetch_translatable_strings(script.ptr(),strings);

	for(int i=0;i<node->get_child_count();i++) {

		Node *c=node->get_child(i);
		if (c->get_owner()!=get_edited_scene())
			continue;

		_fetch_translatable_strings(c,strings);
	}

}


Error EditorNode::save_translatable_strings(const String& p_to_file) {

	if (!is_inside_tree()) {
		defer_translatable=p_to_file;
		return OK;
	}

	ERR_FAIL_COND_V(!get_edited_scene(),ERR_INVALID_DATA);

	Set<StringName> strings;
	_fetch_translatable_strings(get_edited_scene(),strings);

	Error err;
	FileAccess *f = FileAccess::open(p_to_file,FileAccess::WRITE,&err);
	ERR_FAIL_COND_V(err,err);

	OS::Date date = OS::get_singleton()->get_date();
	OS::Time time = OS::get_singleton()->get_time();
	f->store_line("# Translation Strings Dump.");
	f->store_line("# Created By.");
	f->store_line("# \t"VERSION_FULL_NAME" (c) 2008-2014 Juan Linietsky, Ariel Manzur.");
	f->store_line("# From Scene: ");
	f->store_line("# \t"+get_edited_scene()->get_filename());
	f->store_line("");
	f->store_line("msgid \"\"");
	f->store_line("msgstr \"\"");
	f->store_line("\"Report-Msgid-Bugs-To: <define>\\n\"");
	f->store_line("\"POT-Creation-Date: "+itos(date.year)+"-"+itos(date.month)+"-"+itos(date.day)+" "+itos(time.hour)+":"+itos(time.min)+"0000\\n\"");
//	f->store_line("\"PO-Revision-Date: 2006-08-30 13:56-0700\\n\"");
//	f->store_line("\"Last-Translator: Rubén C. Díaz Alonso <outime@gmail.com>\\n\"");
	f->store_line("\"Language-Team: <define>\\n\"");
	f->store_line("\"MIME-Version: 1.0\\n\"");
	f->store_line("\"Content-Type: text/plain; charset=UTF-8\\n\"");
	f->store_line("\"Content-Transfer-Encoding: 8bit\\n\"");
	f->store_line("");

	for(Set<StringName>::Element *E=strings.front();E;E=E->next()) {

		String s = E->get();
		if (s=="" || s.strip_edges()=="")
			continue;
		Vector<String> substr = s.split("\n");
		ERR_CONTINUE(substr.size()==0);

		f->store_line("");

		if (substr.size()==1) {

			f->store_line("msgid \""+substr[0].c_escape()+"\"");
		} else {

			f->store_line("msgid \"\"");
			for(int i=0;i<substr.size();i++) {

				String s = substr[i];
				if (i!=substr.size()-1)
					s+="\n";
				f->store_line("\""+s.c_escape()+"\"");
			}
		}

		f->store_line("msgstr \"\"");

	}


	f->close();
	memdelete(f);

	return OK;

}

Error EditorNode::save_optimized_copy(const String& p_scene,const String& p_preset) {

#if 0

	if (!is_inside_scene()) {
		defer_optimize=p_scene;
		defer_optimize_preset=p_preset;
		return OK;
	}


	if (!get_edited_scene()) {

		get_scene()->quit();
		ERR_EXPLAIN("No scene to optimize (loading failed?");
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
		accept->popup_centered(Size2(300,70));;
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

	ERR_EXPLAIN("Preset '"+p_preset+"' references unexisting saver: "+type);
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
		accept->popup_centered(Size2(300,70));;
		return ERR_INVALID_DATA;

	}
	err = ResourceSaver::save(path,sdata,flags); //todo, saverSceneSaver::save(path,get_edited_scene(),flags,saver);

	if (err) {

		//accept->"()->hide();
		accept->get_ok()->set_text("I see..");
		accept->set_text("Error saving optimized scene: "+path);
		accept->popup_centered(Size2(300,70));;

		ERR_FAIL_COND_V(err,err);

	}

	project_settings->add_remapped_path(src_scene,path,platform);
#endif
	return OK;
}

Error EditorNode::load_scene(const String& p_scene) {

	if (!is_inside_tree()) {
		defer_load_scene = p_scene;
		return OK;
	}



	load_errors->clear();
	String lpath = Globals::get_singleton()->localize_path(p_scene);

	if (!lpath.begins_with("res://")) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("Ugh");
		accept->set_text("Error loading scene, it must be inside the project path. Use 'Import' to open the scene, then save it inside the project path.");
		accept->popup_centered(Size2(300,120));
		opening_prev=false;
		return ERR_FILE_NOT_FOUND;
	}

	_cleanup_scene(); // i'm sorry but this MUST happen to avoid modified resources to not be reloaded.

	Ref<PackedScene> sdata = ResourceLoader::load(lpath);
	if (!sdata.is_valid()) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("Ugh");
		accept->set_text("Error loading scene.");
		accept->popup_centered(Size2(300,70));;
		opening_prev=false;
		return ERR_FILE_NOT_FOUND;
	}

	Node*new_scene=sdata->instance(true);

	if (!new_scene) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("Ugh");
		accept->set_text("Error loading scene.");
		accept->popup_centered(Size2(300,70));;
		opening_prev=false;
		return ERR_FILE_NOT_FOUND;
	}

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
	set_edited_scene(new_scene);
	_get_scene_metadata();
	scene_tree_dock->set_selected(new_scene, true);
	property_editor->edit(new_scene);
	scene_import_metadata = sdata->get_import_metadata();

	editor_data.get_undo_redo().clear_history();
	saved_version=editor_data.get_undo_redo().get_version();
	_update_title();

	_add_to_recent_scenes(lpath);

	if (new_scene->has_meta("__editor_plugin_screen__")) {

		String editor = new_scene->get_meta("__editor_plugin_screen__");

		for(int i=0;i<editor_table.size();i++) {

			if (editor_table[i]->get_name()==editor) {
				_editor_select(i);
				break;
			}
		}
	}

	prev_scene->set_disabled(previous_scenes.size()==0);
	opening_prev=false;

	top_pallete->set_current_tab(0); //always go to scene

	push_item(new_scene);

	return OK;
}



void EditorNode::open_request(const String& p_path) {

	external_file=p_path;
	_menu_option_confirm(FILE_EXTERNAL_OPEN_SCENE,false);
}


Node* EditorNode::request_instance_scene(const String &p_path) {

	return scene_tree_dock->instance(p_path);

}

ScenesDock *EditorNode::get_scenes_dock() {

	return scenes_dock;
}

void EditorNode::_instance_request(const String& p_path){


	request_instance_scene(p_path);
}

void EditorNode::_property_keyed(const String& p_keyed,const Variant& p_value) {

	animation_editor->insert_value_key(p_keyed,p_value);
}

void EditorNode::_transform_keyed(Object *sp,const String& p_sub,const Transform& p_key) {

	Spatial *s=sp->cast_to<Spatial>();
	if (!s)
		return;
	animation_editor->insert_transform_key(s,p_sub,p_key);
}

void EditorNode::_update_keying() {

	//print_line("KR: "+itos(p_enabled));

	bool valid=false;

	if (animation_editor->has_keying()) {

		if (editor_history.get_path_size()>=1) {

			Object *obj = ObjectDB::get_instance(editor_history.get_path_object(0));
			if (obj && obj->cast_to<Node>()) {

				valid=true;
			}
		}

	}

	property_editor->set_keying(valid);

}


void EditorNode::_close_messages() {

//	left_split->set_dragger_visible(false);
	old_split_ofs = left_split->get_split_offset();
	left_split->set_split_offset(0);
//	scene_root_parent->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_END,0);
}

void EditorNode::_show_messages() {

//	left_split->set_dragger_visible(true);
	left_split->set_split_offset(old_split_ofs);
//	scene_root_parent->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_END,log->get_margin(MARGIN_TOP));

}

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
	_update_keying();

}

void EditorNode::_add_to_recent_scenes(const String& p_scene) {

	String base="_"+Globals::get_singleton()->get_resource_path().replace("\\","::").replace("/","::");
	Vector<String> rc = EDITOR_DEF(base+"/_recent_scenes",Array());
	String name = p_scene;
	name=name.replace("res://","");
	if (rc.find(name)!=-1)
		rc.erase(name);
	rc.insert(0,name);
	if (rc.size()>10)
		rc.resize(10);

	EditorSettings::get_singleton()->set(base+"/_recent_scenes",rc);
	EditorSettings::get_singleton()->save();
	_update_recent_scenes();

}

void EditorNode::_open_recent_scene(int p_idx) {

	String base="_"+Globals::get_singleton()->get_resource_path().replace("\\","::").replace("/","::");
	Vector<String> rc = EDITOR_DEF(base+"/_recent_scenes",Array());

	ERR_FAIL_INDEX(p_idx,rc.size());

	String path = "res://"+rc[p_idx];

	if (unsaved_cache) {
		_recent_scene=rc[p_idx];
		open_recent_confirmation->set_text("Discard current scene and open:\n'"+rc[p_idx]+"'");
		open_recent_confirmation->get_label()->set_align(Label::ALIGN_CENTER);
		open_recent_confirmation->popup_centered(Size2(400,100));
		return;
	}

	load_scene(rc[p_idx]);


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
		accept->popup_centered(Size2(300,70));;
		return;

	}

	project_settings->add_remapped_path(Globals::get_singleton()->localize_path(get_edited_scene()->get_filename()),Globals::get_singleton()->localize_path(path),platform);
#endif
}

void EditorNode::_open_recent_scene_confirm() {

	load_scene(_recent_scene);

}

void EditorNode::_update_recent_scenes() {

	String base="_"+Globals::get_singleton()->get_resource_path().replace("\\","::").replace("/","::");
	Vector<String> rc = EDITOR_DEF(base+"/_recent_scenes",Array());
	recent_scenes->clear();
	for(int i=0;i<rc.size();i++) {

		recent_scenes->add_item(rc[i],i);
	}

}

void EditorNode::hide_animation_player_editors() {

	emit_signal("hide_animation_player_editors");
}

void EditorNode::_quick_opened(const String& p_resource) {

	print_line("quick_opened");
	if (quick_open->get_base_type()=="PackedScene") {

		open_request(p_resource);
	} else {
		load_resource(p_resource);
	}

}

void EditorNode::_quick_run(const String& p_resource) {

	_run(false,p_resource);
}


void EditorNode::notify_child_process_exited() {

	play_button->set_pressed(false);
	play_scene_button->set_pressed(false);
	pause_button->set_pressed(false);
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




void EditorNode::add_io_error(const String& p_error) {

	_load_error_notify(singleton,p_error);
}

void EditorNode::_load_error_notify(void* p_ud,const String& p_text) {

	EditorNode*en=(EditorNode*)p_ud;
	en->load_errors->set_text(en->load_errors->get_text()+p_text+"\n");
	en->load_error_dialog->popup_centered_ratio(0.5);

}


bool EditorNode::_find_scene_in_use(Node* p_node,const String& p_path) const {

	if (p_node->get_filename()==p_path) {
		return true;
	}

	for(int i=0;i<p_node->get_child_count();i++) {

		if (_find_scene_in_use(p_node->get_child(i),p_path)) {
			return true;
		}
	}

	return false;
}


bool EditorNode::is_scene_in_use(const String& p_path) {

	Node *es = get_edited_scene();
	if (es)
		return _find_scene_in_use(es,p_path);
	return false;

}

void EditorNode::register_editor_types() {

	ObjectTypeDB::register_type<EditorPlugin>();
	ObjectTypeDB::register_type<EditorImportPlugin>();
	ObjectTypeDB::register_type<EditorScenePostImport>();
	ObjectTypeDB::register_type<EditorScript>();


	//ObjectTypeDB::register_type<EditorImporter>();
//	ObjectTypeDB::register_type<EditorPostImport>();
}

void EditorNode::unregister_editor_types() {

	_init_callbacks.clear();
}


void EditorNode::stop_child_process() {

	_menu_option_confirm(RUN_STOP,false);
}





void EditorNode::progress_add_task(const String& p_task,const String& p_label, int p_steps) {

	singleton->progress_dialog->add_task(p_task,p_label,p_steps);
}

void EditorNode::progress_task_step(const String& p_task, const String& p_state, int p_step) {

	singleton->progress_dialog->task_step(p_task,p_state,p_step);

}

void EditorNode::progress_end_task(const String& p_task) {

	singleton->progress_dialog->end_task(p_task);

}


void EditorNode::progress_add_task_bg(const String& p_task,const String& p_label, int p_steps) {

	singleton->progress_hb->add_task(p_task,p_label,p_steps);
}

void EditorNode::progress_task_step_bg(const String& p_task, int p_step) {

	singleton->progress_hb->task_step(p_task,p_step);

}

void EditorNode::progress_end_task_bg(const String& p_task) {

	singleton->progress_hb->end_task(p_task);

}


void EditorNode::_bind_methods() {
	

	ObjectTypeDB::bind_method("_menu_option",&EditorNode::_menu_option);
	ObjectTypeDB::bind_method("_menu_confirm_current",&EditorNode::_menu_confirm_current);
	ObjectTypeDB::bind_method("_dialog_action",&EditorNode::_dialog_action);
	ObjectTypeDB::bind_method("_resource_selected",&EditorNode::_resource_selected,DEFVAL(""));
	ObjectTypeDB::bind_method("_property_editor_forward",&EditorNode::_property_editor_forward);
	ObjectTypeDB::bind_method("_property_editor_back",&EditorNode::_property_editor_back);
	ObjectTypeDB::bind_method("_editor_select",&EditorNode::_editor_select);
	ObjectTypeDB::bind_method("_node_renamed",&EditorNode::_node_renamed);
	ObjectTypeDB::bind_method("edit_node",&EditorNode::edit_node);
	ObjectTypeDB::bind_method("_imported",&EditorNode::_imported);
	ObjectTypeDB::bind_method("_unhandled_input",&EditorNode::_unhandled_input);

	ObjectTypeDB::bind_method("_get_scene_metadata",&EditorNode::_get_scene_metadata);
	ObjectTypeDB::bind_method("set_edited_scene",&EditorNode::set_edited_scene);
	ObjectTypeDB::bind_method("open_request",&EditorNode::open_request);
	ObjectTypeDB::bind_method("_instance_request",&EditorNode::_instance_request);
	ObjectTypeDB::bind_method("_update_keying",&EditorNode::_update_keying);
	ObjectTypeDB::bind_method("_property_keyed",&EditorNode::_property_keyed);
	ObjectTypeDB::bind_method("_transform_keyed",&EditorNode::_transform_keyed);
	ObjectTypeDB::bind_method("_close_messages",&EditorNode::_close_messages);
	ObjectTypeDB::bind_method("_show_messages",&EditorNode::_show_messages);
	ObjectTypeDB::bind_method("_vp_resized",&EditorNode::_vp_resized);
	ObjectTypeDB::bind_method("_quick_opened",&EditorNode::_quick_opened);
	ObjectTypeDB::bind_method("_quick_run",&EditorNode::_quick_run);

	ObjectTypeDB::bind_method("_import_action",&EditorNode::_import_action);
	//ObjectTypeDB::bind_method("_import",&EditorNode::_import);
//	ObjectTypeDB::bind_method("_import_conflicts_solved",&EditorNode::_import_conflicts_solved);
	ObjectTypeDB::bind_method("_open_recent_scene",&EditorNode::_open_recent_scene);
	ObjectTypeDB::bind_method("_open_recent_scene_confirm",&EditorNode::_open_recent_scene_confirm);

	ObjectTypeDB::bind_method("_save_optimized",&EditorNode::_save_optimized);
	ObjectTypeDB::bind_method(_MD("animation_panel_make_visible","enable"),&EditorNode::animation_panel_make_visible);

	ObjectTypeDB::bind_method("stop_child_process",&EditorNode::stop_child_process);

	ObjectTypeDB::bind_method("_sources_changed",&EditorNode::_sources_changed);
	ObjectTypeDB::bind_method("_fs_changed",&EditorNode::_fs_changed);


	ADD_SIGNAL( MethodInfo("play_pressed") );
	ADD_SIGNAL( MethodInfo("pause_pressed") );
	ADD_SIGNAL( MethodInfo("stop_pressed") );
	ADD_SIGNAL( MethodInfo("hide_animation_player_editors") );
	ADD_SIGNAL( MethodInfo("request_help") );
	ADD_SIGNAL( MethodInfo("script_add_function_request",PropertyInfo(Variant::OBJECT,"obj"),PropertyInfo(Variant::STRING,"function"),PropertyInfo(Variant::STRING_ARRAY,"args")) );
	ADD_SIGNAL( MethodInfo("resource_saved",PropertyInfo(Variant::OBJECT,"obj")) );


	
}

Ref<Texture> EditorNode::_file_dialog_get_icon(const String& p_path) {


	EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_path(p_path.get_base_dir());
	if (efsd) {

		String file = p_path.get_file();
		for(int i=0;i<efsd->get_file_count();i++) {
			if (efsd->get_file(i)==file) {

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

void EditorNode::_file_dialog_unregister(FileDialog *p_dialog){

	singleton->file_dialogs.erase(p_dialog);
}

Vector<EditorNodeInitCallback> EditorNode::_init_callbacks;

Error EditorNode::export_platform(const String& p_platform, const String& p_path, bool p_debug,const String& p_password,bool p_quit_after) {

	export_defer.platform=p_platform;
	export_defer.path=p_path;
	export_defer.debug=p_debug;
	export_defer.password=p_password;

	return OK;
}



EditorNode::EditorNode() {

	EditorHelp::generate_doc(); //before any editor classes are crated

	singleton=this;

	FileAccess::set_backup_save(true);

	PathRemap::get_singleton()->clear_remaps();; //editor uses no remaps
	TranslationServer::get_singleton()->set_enabled(false);
	// load settings
	if (!EditorSettings::get_singleton())
		EditorSettings::create();

	ResourceLoader::set_abort_on_missing_resources(false);
	ResourceLoader::set_error_notify_func(this,_load_error_notify);

	ResourceLoader::set_timestamp_on_load(true);
	ResourceSaver::set_timestamp_on_save(true);

	_pvrtc_register_compressors();

	editor_selection = memnew( EditorSelection );

	EditorFileSystem *efs = memnew( EditorFileSystem );
	add_child(efs);

	//used for previews
	FileDialog::get_icon_func=_file_dialog_get_icon;
	FileDialog::register_func=_file_dialog_register;
	FileDialog::unregister_func=_file_dialog_unregister;

	editor_import_export = memnew( EditorImportExport );

	register_exporters();

	editor_import_export->load_config();

	GLOBAL_DEF("editor/main_run_args","$exec -path $path -scene $scene $main_scene");

	ObjectTypeDB::set_type_enabled("CollisionShape",true);
	ObjectTypeDB::set_type_enabled("CollisionShape2D",true);
	ObjectTypeDB::set_type_enabled("CollisionPolygon2D",true);
	//ObjectTypeDB::set_type_enabled("BodyVolumeConvexPolygon",true);

	gui_base = memnew( Panel );
	add_child(gui_base);
	gui_base->set_area_as_parent_rect();


	theme = Ref<Theme>( memnew( Theme ) );
	gui_base->set_theme( theme );
	editor_register_icons(theme);
	editor_register_fonts(theme);

	String global_font = EditorSettings::get_singleton()->get("global/font");
	if (global_font!="") {
		Ref<Font> fnt = ResourceLoader::load(global_font);
		if (fnt.is_valid()) {
			theme->set_default_theme_font(fnt);
		}
	}

	Ref<StyleBoxTexture> focus_sbt=memnew( StyleBoxTexture );
	focus_sbt->set_texture(theme->get_icon("EditorFocus","EditorIcons"));
	for(int i=0;i<4;i++) {
		focus_sbt->set_margin_size(Margin(i),16);
		focus_sbt->set_default_margin(Margin(i),16);
	}
	focus_sbt->set_draw_center(false);
	theme->set_stylebox("EditorFocus","EditorStyles",focus_sbt);


	progress_dialog = memnew( ProgressDialog );
	gui_base->add_child(progress_dialog);

	// take up all screen
	gui_base->set_anchor( MARGIN_RIGHT, Control::ANCHOR_END );
	gui_base->set_anchor( MARGIN_BOTTOM, Control::ANCHOR_END );
	gui_base->set_end( Point2(0,0) );
	
	main_vbox = memnew( VBoxContainer );
	gui_base->add_child(main_vbox);
	main_vbox->set_area_as_parent_rect(8);

	menu_hb = memnew( HBoxContainer );
	main_vbox->add_child(menu_hb);

	main_split = memnew( HSplitContainer );
	main_vbox->add_child(main_split);
	main_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);


	left_split = memnew( VSplitContainer );
	main_split->add_child(left_split);
	left_split->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	left_split->set_collapsed(false);

	top_split = memnew( VSplitContainer );
	left_split->add_child(top_split);
	top_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	top_split->set_collapsed(true);

	VBoxContainer *srt = memnew( VBoxContainer );
	srt->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	top_split->add_child(srt);
	srt->add_constant_override("separation",0);


	main_editor_tabs  = memnew( Tabs );
	main_editor_tabs->connect("tab_changed",this,"_editor_select");
	HBoxContainer *srth = memnew( HBoxContainer );
	srt->add_child( srth );
	Control *tec = memnew( Control );
	tec->set_custom_minimum_size(Size2(100,0));
	tec->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	srth->add_child(tec);
	srth->add_child(main_editor_tabs);
	tec = memnew( Control );
	tec->set_custom_minimum_size(Size2(100,0));
	srth->add_child(tec);
	tec->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	scene_root_parent = memnew( Panel );

	Ref<StyleBox> sp = scene_root_parent->get_stylebox("panel","TabContainer");
	scene_root_parent->add_style_override("panel",sp);
	/*scene_root_parent->set_anchor( MARGIN_RIGHT, Control::ANCHOR_END );
	scene_root_parent->set_anchor( MARGIN_BOTTOM, Control::ANCHOR_END );
	scene_root_parent->set_begin( Point2( 0, 0) );
	scene_root_parent->set_end( Point2( 0,80 ) );*/
	srt->add_child(scene_root_parent);
	scene_root_parent->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	scene_root_base = memnew( Control );
	scene_root_base->set_area_as_parent_rect();
	for(int i=0;i<4;i++) {
		scene_root_base->set_margin(Margin(i),sp->get_margin(Margin(i)));
	}
	scene_root_parent->add_child(scene_root_base);


	scene_root = memnew( Viewport );
	//scene_root_base->add_child(scene_root);
	scene_root->set_meta("_editor_disable_input",true);
	VisualServer::get_singleton()->viewport_set_hide_scenario(scene_root->get_viewport(),true);
	scene_root->set_as_audio_listener_2d(true);
	scene_root->set_size_override(true,Size2(Globals::get_singleton()->get("display/width"),Globals::get_singleton()->get("display/height")));

//	scene_root->set_world_2d( Ref<World2D>( memnew( World2D )) );


	viewport = memnew( Control );
	viewport->set_area_as_parent_rect(4);
	for(int i=0;i<4;i++) {
		viewport->set_margin(Margin(i),sp->get_margin(Margin(i)));
	}
	scene_root_parent->add_child(viewport);


	PanelContainer *pc = memnew( PanelContainer );
	top_split->add_child(pc);
	animation_vb = memnew( VBoxContainer );
	animation_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	pc->add_child(animation_vb);
	animation_panel=pc;
	animation_panel->hide();

	HBoxContainer *animation_hb = memnew( HBoxContainer);
	animation_vb->add_child(animation_hb);

	Label *l= memnew( Label );
	l->set_text("Animation:");
	//l->set_h_size_flags(Control::SIZE_);
	animation_hb->add_child(l);

	animation_panel_hb = memnew( HBoxContainer );
	animation_hb->add_child(animation_panel_hb);
	animation_panel_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);


	/*pd_anim = memnew( PaneDrag );
	animation_hb->add_child(pd_anim);
	pd_anim->connect("dragged",this,"_dragged");
	pd_anim->set_default_cursor_shape(Control::CURSOR_MOVE);
	pd_anim->hide();*/

	anim_close = memnew( TextureButton );
	animation_hb->add_child(anim_close);
	anim_close->connect("pressed",this,"animation_panel_make_visible",make_binds(false));
	anim_close->set_normal_texture( anim_close->get_icon("Close","EditorIcons"));
	anim_close->set_hover_texture( anim_close->get_icon("CloseHover","EditorIcons"));
	anim_close->set_pressed_texture( anim_close->get_icon("Close","EditorIcons"));


	PanelContainer *top_region = memnew( PanelContainer );
	top_region->add_style_override("panel",gui_base->get_stylebox("hover","Button"));
	HBoxContainer *left_menu_hb = memnew( HBoxContainer );
	top_region->add_child(left_menu_hb);
	menu_hb->add_child(top_region);

	PopupMenu *p;



	file_menu = memnew( MenuButton );
	file_menu->set_text("Scene");
	//file_menu->set_icon(gui_base->get_icon("Save","EditorIcons"));
	left_menu_hb->add_child( file_menu );

	prev_scene = memnew( ToolButton );
	prev_scene->set_icon(gui_base->get_icon("PrevScene","EditorIcons"));
	prev_scene->set_tooltip("Go to previously opened scene.");
	prev_scene->set_disabled(true);
	//left_menu_hb->add_child( prev_scene );
	prev_scene->connect("pressed",this,"_menu_option",make_binds(FILE_OPEN_PREV));
	//gui_base->add_child(prev_scene);
	prev_scene->set_pos(Point2(3,24));


	Separator *vs=NULL;

	file_menu->set_tooltip("Operations with scene files.");
	p=file_menu->get_popup();
	p->add_item("New Scene",FILE_NEW_SCENE);
	p->add_item("Open Scene..",FILE_OPEN_SCENE,KEY_MASK_CMD+KEY_O);
	p->add_item("Save Scene",FILE_SAVE_SCENE,KEY_MASK_CMD+KEY_S);
	p->add_item("Save Scene As..",FILE_SAVE_AS_SCENE,KEY_MASK_SHIFT+KEY_MASK_CMD+KEY_S);
	p->add_separator();
	p->add_item("Goto Prev. Scene",FILE_OPEN_PREV,KEY_MASK_SHIFT+KEY_MASK_CMD+KEY_P);
	p->add_submenu_item("Open Recent","RecentScenes",FILE_OPEN_RECENT);
	p->add_separator();
	p->add_item("Quick Open Scene..",FILE_QUICK_OPEN_SCENE,KEY_MASK_SHIFT+KEY_MASK_CMD+KEY_O);
	p->add_item("Quick Open Script..",FILE_QUICK_OPEN_SCRIPT,KEY_MASK_ALT+KEY_MASK_CMD+KEY_O);
	p->add_separator();

	PopupMenu *pm_export = memnew(PopupMenu );
	pm_export->set_name("Export");
	p->add_child(pm_export);
	p->add_submenu_item("Convert To..","Export");
	pm_export->add_item("Subscene..",FILE_SAVE_SUBSCENE);
	pm_export->add_item("Translatable Strings..",FILE_DUMP_STRINGS);
	pm_export->add_separator();
	pm_export->add_item("MeshLibrary..",FILE_EXPORT_MESH_LIBRARY);
	pm_export->add_item("TileSet..",FILE_EXPORT_TILESET);
	pm_export->connect("item_pressed",this,"_menu_option");

	p->add_separator();
	p->add_item("Undo",EDIT_UNDO,KEY_MASK_CMD+KEY_Z);
	p->add_item("Redo",EDIT_REDO,KEY_MASK_CMD+KEY_MASK_SHIFT+KEY_Z);
	p->add_separator();
	p->add_item("Run Script",FILE_RUN_SCRIPT,KEY_MASK_SHIFT+KEY_MASK_CMD+KEY_R);
	p->add_separator();
	p->add_item("Project Settings",RUN_SETTINGS);
	p->add_separator();
	p->add_item("Quit to Project List",RUN_PROJECT_MANAGER,KEY_MASK_SHIFT+KEY_MASK_CMD+KEY_Q);
	p->add_item("Quit",FILE_QUIT,KEY_MASK_CMD+KEY_Q);

	recent_scenes = memnew( PopupMenu );
	recent_scenes->set_name("RecentScenes");
	p->add_child(recent_scenes);
	recent_scenes->connect("item_pressed",this,"_open_recent_scene");

	//menu_hb->add_spacer();
#if 0
	node_menu = memnew( MenuButton );
	node_menu->set_text("Node");
	node_menu->set_pos( Point2( 50,0) );;
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

	import_menu = memnew( MenuButton );
	import_menu->set_tooltip("Import assets to the project.");
	import_menu->set_text("Import");
	//import_menu->set_icon(gui_base->get_icon("Save","EditorIcons"));
	left_menu_hb->add_child( import_menu );

	p=import_menu->get_popup();
	p->add_item("Sub-Scene",FILE_IMPORT_SUBSCENE);
	p->add_separator();
	p->connect("item_pressed",this,"_menu_option");

	export_button = memnew( ToolButton );
	export_button->set_tooltip("Export the project to many platforms.");
	export_button->set_text("Export");
	export_button->connect("pressed",this,"_menu_option",varray(FILE_EXPORT_PROJECT));
	export_button->set_focus_mode(Control::FOCUS_NONE);
	left_menu_hb->add_child(export_button);

	menu_hb->add_spacer();

	//Separator *s1 = memnew( VSeparator );
	//menu_panel->add_child(s1);
	//s1->set_pos(Point2(210,4));
	//s1->set_size(Point2(10,15));


	CenterContainer *play_cc = memnew( CenterContainer );
	play_cc->set_ignore_mouse(true);
	gui_base->add_child( play_cc );
	play_cc->set_area_as_parent_rect();
	play_cc->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_BEGIN,10);
	play_cc->set_margin(MARGIN_TOP,5);

	top_region = memnew( PanelContainer );
	top_region->add_style_override("panel",gui_base->get_stylebox("hover","Button"));
	play_cc->add_child(top_region);

	HBoxContainer *play_hb = memnew( HBoxContainer );
	top_region->add_child(play_hb);

	play_button = memnew( ToolButton );
	play_hb->add_child(play_button);
	play_button->set_toggle_mode(true);
	play_button->set_icon(gui_base->get_icon("MainPlay","EditorIcons"));
	play_button->set_focus_mode(Control::FOCUS_NONE);
	play_button->connect("pressed", this,"_menu_option",make_binds(RUN_PLAY));
	play_button->set_tooltip("Start the scene (F5).");



	pause_button = memnew( ToolButton );
	//menu_panel->add_child(pause_button); - not needed for now?
	pause_button->set_toggle_mode(true);
	pause_button->set_icon(gui_base->get_icon("Pause","EditorIcons"));
	pause_button->set_focus_mode(Control::FOCUS_NONE);
	pause_button->connect("pressed", this,"_menu_option",make_binds(RUN_PAUSE));
	pause_button->set_tooltip("Pause the scene (F7).");

	stop_button = memnew( ToolButton );
	play_hb->add_child(stop_button);
	//stop_button->set_toggle_mode(true);
	stop_button->set_focus_mode(Control::FOCUS_NONE);
	stop_button->set_icon(gui_base->get_icon("MainStop","EditorIcons"));
	stop_button->connect("pressed", this,"_menu_option",make_binds(RUN_STOP));
	stop_button->set_tooltip("Stop the scene (F8).");

	run_native = memnew( EditorRunNative);
	play_hb->add_child(run_native);
	native_play_button = memnew( MenuButton );
	native_play_button->set_text("NTV");
	menu_hb->add_child(native_play_button);
	native_play_button->hide();
	native_play_button->get_popup()->connect("item_pressed",this,"_run_in_device");

	VSeparator *s1 = memnew( VSeparator );
//	play_hb->add_child(s1);

	play_scene_button = memnew( ToolButton );
	play_hb->add_child(play_scene_button);
	play_scene_button->set_toggle_mode(true);
	play_scene_button->set_focus_mode(Control::FOCUS_NONE);
	play_scene_button->set_icon(gui_base->get_icon("PlayScene","EditorIcons"));
	play_scene_button->connect("pressed", this,"_menu_option",make_binds(RUN_PLAY_SCENE));
	play_scene_button->set_tooltip("Play the edited scene (F6).");

	play_custom_scene_button = memnew( ToolButton );
	play_hb->add_child(play_custom_scene_button);
	play_custom_scene_button->set_toggle_mode(true);
	play_custom_scene_button->set_focus_mode(Control::FOCUS_NONE);
	play_custom_scene_button->set_icon(gui_base->get_icon("PlayCustom","EditorIcons"));
	play_custom_scene_button->connect("pressed", this,"_menu_option",make_binds(RUN_PLAY_CUSTOM_SCENE));
	play_custom_scene_button->set_tooltip("Play custom scene ("+keycode_get_string(KEY_MASK_CMD|KEY_MASK_SHIFT|KEY_F5)+").");

	fileserver_menu = memnew( MenuButton );
	play_hb->add_child(fileserver_menu);
	fileserver_menu->set_flat(true);
	fileserver_menu->set_focus_mode(Control::FOCUS_NONE);
	fileserver_menu->set_icon(gui_base->get_icon("FileServer","EditorIcons"));
	//fileserver_menu->connect("pressed", this,"_menu_option",make_binds(RUN_PLAY_CUSTOM_SCENE));
	fileserver_menu->set_tooltip("Serve the project filesystem to remote clients.");

	p=fileserver_menu->get_popup();
	p->add_check_item("Enable File Server",RUN_FILE_SERVER);
	p->set_item_tooltip(p->get_item_index(RUN_FILE_SERVER),"Enable/Disable the File Server.");
	p->add_separator();
	p->add_check_item("Deploy Dumb Clients",RUN_DEPLOY_DUMB_CLIENTS);
	//p->set_item_checked( p->get_item_index(RUN_DEPLOY_DUMB_CLIENTS),true );
	p->set_item_tooltip(p->get_item_index(RUN_DEPLOY_DUMB_CLIENTS),"Deploy dumb clients when the File Server is active.");
	p->connect("item_pressed",this,"_menu_option");

	run_settings_button = memnew( ToolButton );
	//menu_hb->add_child(run_settings_button);
	//run_settings_button->set_toggle_mode(true);
	run_settings_button->set_focus_mode(Control::FOCUS_NONE);
	run_settings_button->set_icon(gui_base->get_icon("Run","EditorIcons"));
	run_settings_button->connect("pressed", this,"_menu_option",make_binds(RUN_SCENE_SETTINGS));


	/*
	run_settings_button = memnew( ToolButton );
	menu_panel->add_child(run_settings_button);
	run_settings_button->set_pos(Point2(305,0));
	run_settings_button->set_focus_mode(Control::FOCUS_NONE);
	run_settings_button->set_icon(gui_base->get_icon("Run","EditorIcons"));
	run_settings_button->connect("pressed", this,"_menu_option",make_binds(RUN_SETTINGS));
*/


	top_region = memnew( PanelContainer );
	top_region->add_style_override("panel",gui_base->get_stylebox("hover","Button"));
	HBoxContainer *right_menu_hb = memnew( HBoxContainer );
	top_region->add_child(right_menu_hb);
	menu_hb->add_child(top_region);


	settings_menu = memnew( MenuButton );
	settings_menu->set_text("Settings");
	//settings_menu->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	right_menu_hb->add_child( settings_menu );
	p=settings_menu->get_popup();


	//p->add_item("Export Settings",SETTINGS_EXPORT_PREFERENCES);
	p->add_item("Editor Settings",SETTINGS_PREFERENCES);
	//p->add_item("Optimization Presets",SETTINGS_OPTIMIZED_PRESETS);
	p->add_separator();
	p->add_check_item("Show Animation",SETTINGS_SHOW_ANIMATION,KEY_MASK_CMD+KEY_N);
	p->add_separator();
	p->add_item("Install Export Templates",SETTINGS_LOAD_EXPORT_TEMPLATES);
	p->add_separator();
	p->add_item("About",SETTINGS_ABOUT);


	sources_button = memnew( ToolButton );
	right_menu_hb->add_child(sources_button);
	sources_button->set_icon(gui_base->get_icon("DependencyOk","EditorIcons"));
	sources_button->connect("pressed",this,"_menu_option",varray(SOURCES_REIMPORT));
	sources_button->set_tooltip("Alerts when an external resource has changed.");

	//sources_button->connect();

/*
	Separator *s2 = memnew( VSeparator );
	menu_panel->add_child(s2);
	s2->set_pos(Point2(338,4));
	s2->set_size(Point2(10,15));
*/



	editor_hsplit = memnew( HSplitContainer );
	main_split->add_child(editor_hsplit);
	editor_hsplit->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	editor_vsplit = memnew( VSplitContainer );
	editor_hsplit->add_child(editor_vsplit);

	top_pallete = memnew( TabContainer );
	scene_tree_dock = memnew( SceneTreeDock(this,scene_root,editor_selection,editor_data) );
	scene_tree_dock->set_name("Scene");
	top_pallete->add_child(scene_tree_dock);

	resources_dock = memnew( ResourcesDock(this) );
	resources_dock->set_name("Resources");
	top_pallete->add_child(resources_dock);
	top_pallete->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	Control *editor_spacer = memnew( Control );
	editor_spacer->set_custom_minimum_size(Size2(260,200));
	editor_spacer->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor_vsplit->add_child( editor_spacer );
	editor_spacer->add_child( top_pallete );
	top_pallete->set_area_as_parent_rect();


	prop_pallete = memnew( TabContainer );

	prop_pallete->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	editor_spacer = memnew( Control );
	editor_spacer->set_custom_minimum_size(Size2(260,200));
	editor_spacer->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor_vsplit->add_child( editor_spacer );
	editor_spacer->add_child( prop_pallete );
	prop_pallete->set_area_as_parent_rect();

	VBoxContainer *prop_editor_base = memnew( VBoxContainer );
	prop_editor_base->set_name("Inspector"); // Properties?
	prop_pallete->add_child(prop_editor_base);

	HBoxContainer *prop_editor_hb = memnew( HBoxContainer );
	prop_editor_base->add_child(prop_editor_hb);
	editor_path = memnew( EditorPath(&editor_history) );
	editor_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	prop_editor_hb->add_child(editor_path);

	property_editor = memnew( PropertyEditor );
	property_editor->set_autoclear(true);
	property_editor->set_show_categories(true);
	property_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	
	property_editor->hide_top_label();

	prop_editor_base->add_child( property_editor );
	property_editor->set_undo_redo(&editor_data.get_undo_redo());


	
	property_back = memnew( ToolButton );
	property_back->set_icon( gui_base->get_icon("Back","EditorIcons") );
	property_back->set_flat(true);
	property_back->set_tooltip("Go to the previous edited object in history.");

	prop_editor_hb->add_child( property_back );

	property_forward = memnew( ToolButton );
	property_forward->set_icon( gui_base->get_icon("Forward","EditorIcons") );
	property_forward->set_flat(true);
	property_forward->set_tooltip("Go to the next edited object in history.");

	prop_editor_hb->add_child( property_forward );

	object_menu = memnew( MenuButton );
	object_menu->set_icon(gui_base->get_icon("Tools","EditorIcons"));
	prop_editor_hb->add_child( object_menu );
	object_menu->set_tooltip("Object properties.");


	scenes_dock = memnew( ScenesDock(this) );
	scenes_dock->set_name("FileSystem");
	prop_pallete->add_child(scenes_dock);
	scenes_dock->connect("open",this,"open_request");
	scenes_dock->connect("instance",this,"_instance_request");



	log = memnew( EditorLog );
	left_split->add_child(log);
	log->connect("close_request",this,"_close_messages");
	log->connect("show_request",this,"_show_messages");
	//left_split->set_dragger_visible(false);
	old_split_ofs=0;


	animation_editor = memnew( AnimationKeyEditor(get_undo_redo(),&editor_history,editor_selection) );
	animation_editor->set_anchor_and_margin(MARGIN_RIGHT,Control::ANCHOR_END,0);
	animation_editor->set_margin(MARGIN_BOTTOM,200);
	animation_editor->connect("keying_changed",this,"_update_keying");

	animation_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);


	animation_vb->add_child(animation_editor);
	left_split->connect("resized",this,"_vp_resized");


	animation_editor->hide();

	PanelContainer *bottom_pc = memnew( PanelContainer );
	main_vbox->add_child(bottom_pc);
	bottom_hb = memnew( HBoxContainer );
	bottom_pc->add_child(bottom_hb);

	bottom_hb->add_child( log->get_button() );
	log->get_button()->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	progress_hb = memnew( BackgroundProgress );
	bottom_hb->add_child(progress_hb);
	//progress_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	audio_vu = memnew( TextureProgress );
	CenterContainer *vu_cc = memnew( CenterContainer );
	vu_cc->add_child(audio_vu);
	bottom_hb->add_child(vu_cc);
	audio_vu->set_under_texture(gui_base->get_icon("VuEmpty","EditorIcons"));
	audio_vu->set_progress_texture(gui_base->get_icon("VuFull","EditorIcons"));
	audio_vu->set_max(24);
	audio_vu->set_min(-80);
	audio_vu->set_step(0.01);
	audio_vu->set_val(0);

	update_menu = memnew( MenuButton );
	update_menu->set_tooltip("Spins when the editor window repaints!");
	bottom_hb->add_child(update_menu);
	update_menu->set_icon(gui_base->get_icon("Progress1","EditorIcons"));
	p=update_menu->get_popup();
	p->add_check_item("Update Always",SETTINGS_UPDATE_ALWAYS);
	p->add_check_item("Update Changes",SETTINGS_UPDATE_CHANGES);
	p->set_item_checked(1,true);

	/*
	animation_menu = memnew( ToolButton );
	animation_menu->set_pos(Point2(500,0));
	animation_menu->set_size(Size2(20,20));
	animation_menu->set_toggle_mode(true);
	animation_menu->set_focus_mode(Control::FOCUS_NONE);
	menu_panel->add_child(animation_menu);
	animation_menu->set_icon(gui_base->get_icon("Animation","EditorIcons"));
	animation_menu->connect("pressed",this,"_animation_visibility_toggle");;
*/






	
	call_dialog = memnew( CallDialog );
	call_dialog->hide();
	gui_base->add_child( call_dialog );










	confirmation = memnew( ConfirmationDialog  );
	gui_base->add_child(confirmation);
	confirmation->connect("confirmed", this,"_menu_confirm_current");

	accept = memnew( AcceptDialog  );
	gui_base->add_child(accept);
	accept->connect("confirmed", this,"_menu_confirm_current");





//	optimized_save = memnew( OptimizedSaveDialog(&editor_data) );
	//gui_base->add_child(optimized_save);
	//optimized_save->connect("confirmed",this,"_save_optimized");

	project_export = memnew( ProjectExport(&editor_data) );
	gui_base->add_child(project_export);

	project_export_settings = memnew( ProjectExportDialog(this) );
	gui_base->add_child(project_export_settings);

	//optimized_presets = memnew( OptimizedPresetsDialog(&editor_data) );
	//gui_base->add_child(optimized_presets);
	//optimized_presets->connect("confirmed",this,"_presets_optimized");



	//import_subscene = memnew( EditorSubScene );
	//gui_base->add_child(import_subscene);




	
	settings_config_dialog = memnew( EditorSettingsDialog );
	gui_base->add_child(settings_config_dialog);

	project_settings = memnew( ProjectSettings(&editor_data) );
	gui_base->add_child(project_settings);

	import_confirmation = memnew( ConfirmationDialog );
	import_confirmation->get_ok()->set_text("Re-Import");
	import_confirmation->add_button("Update",!OS::get_singleton()->get_swap_ok_cancel(),"update");
	import_confirmation->get_label()->set_align(Label::ALIGN_CENTER);
	import_confirmation->connect("confirmed",this,"_import_action",make_binds("re-import"));
	import_confirmation->connect("custom_action",this,"_import_action");
	gui_base->add_child(import_confirmation);

	open_recent_confirmation = memnew( ConfirmationDialog );
	add_child(open_recent_confirmation);
	open_recent_confirmation->connect("confirmed",this,"_open_recent_scene_confirm");


	import_settings= memnew(ImportSettingsDialog(this));
	gui_base->add_child(import_settings);
	run_settings_dialog = memnew( RunSettingsDialog );
	gui_base->add_child( run_settings_dialog );



	about = memnew( AcceptDialog );
	about->set_title("Thanks so Much!");
	//about->get_cancel()->hide();
	about->get_ok()->set_text("Thanks!");
	about->set_hide_on_ok(true);
	Label *about_text = memnew( Label );
	about_text->set_text(VERSION_FULL_NAME"\n(c) 2008-2014 Juan Linietsky, Ariel Manzur.\n");
	about_text->set_pos(Point2(gui_base->get_icon("Logo","EditorIcons")->get_size().width+30,20));
	gui_base->add_child(about);
	about->add_child(about_text);
	TextureFrame *logo = memnew( TextureFrame );
	about->add_child(logo);
	logo->set_pos(Point2(20,20));
	logo->set_texture(gui_base->get_icon("Logo","EditorIcons") );




	file_templates = memnew( FileDialog );
	file_templates->set_title("Import Templates from ZIP file");

	gui_base->add_child( file_templates );
	file_templates->set_mode(FileDialog::MODE_OPEN_FILE);
	file_templates->set_access(FileDialog::ACCESS_FILESYSTEM);
	file_templates->clear_filters();
	file_templates->add_filter("*.tpz ; Template Package");


	file = memnew( FileDialog );
	gui_base->add_child(file);
	file->set_current_dir("res://");

	file_export = memnew( FileDialog );
	file_export->set_access(FileDialog::ACCESS_FILESYSTEM);
	gui_base->add_child(file_export);
	file_export->set_title("Export Project");
	file_export->connect("file_selected", this,"_dialog_action");

	file_export_lib = memnew( FileDialog );
	file_export_lib->set_title("Export Library");
	file_export_lib->set_mode(FileDialog::MODE_SAVE_FILE);
	file_export_lib->connect("file_selected", this,"_dialog_action");
	file_export_lib_merge = memnew( CheckButton );
	file_export_lib_merge->set_text("Merge With Existing");
	file_export_lib_merge->set_pressed(true);
	file_export_lib->get_vbox()->add_child(file_export_lib_merge);
	gui_base->add_child(file_export_lib);




	file_export_check = memnew( CheckButton );
	file_export_check->set_text("Enable Debugging");
	file_export_check->set_pressed(true);
	file_export_check->connect("pressed",this,"_export_debug_toggled");
	file_export->get_vbox()->add_margin_child("Debug:",file_export_check);
	file_export_password = memnew( LineEdit );
	file_export_password->set_secret(true);
	file_export_password->set_editable(false);
	file_export->get_vbox()->add_margin_child("Password:",file_export_password);


	file_script = memnew( FileDialog );
	file_script->set_title("Open & Run a Script");
	file_script->set_access(FileDialog::ACCESS_FILESYSTEM);
	file_script->set_mode(FileDialog::MODE_OPEN_FILE);
	List<String> sexts;
	ResourceLoader::get_recognized_extensions_for_type("Script",&sexts);
	for (List<String>::Element*E=sexts.front();E;E=E->next()) {
		file_script->add_filter("*."+E->get());
	}
	gui_base->add_child(file_script);
	file_script->connect("file_selected",this,"_dialog_action");

	reimport_dialog = memnew( EditorReImportDialog );
	gui_base->add_child(reimport_dialog);



	property_forward->connect("pressed", this,"_property_editor_forward");
	property_back->connect("pressed", this,"_property_editor_back");
		


	file_menu->get_popup()->connect("item_pressed", this,"_menu_option");
	object_menu->get_popup()->connect("item_pressed", this,"_menu_option");

	update_menu->get_popup()->connect("item_pressed", this,"_menu_option");
	settings_menu->get_popup()->connect("item_pressed", this,"_menu_option");


	file->connect("file_selected", this,"_dialog_action");
	file_templates->connect("file_selected", this,"_dialog_action");
	property_editor->connect("resource_selected", this,"_resource_selected");
	property_editor->connect("property_keyed", this,"_property_keyed");
	animation_editor->connect("resource_selected", this,"_resource_selected");
	//plugin stuff

	file_server = memnew( EditorFileServer );


	editor_import_export->add_import_plugin( Ref<EditorTextureImportPlugin>( memnew(EditorTextureImportPlugin(this,EditorTextureImportPlugin::MODE_TEXTURE_2D) )));
	editor_import_export->add_import_plugin( Ref<EditorTextureImportPlugin>( memnew(EditorTextureImportPlugin(this,EditorTextureImportPlugin::MODE_TEXTURE_3D) )));
	editor_import_export->add_import_plugin( Ref<EditorTextureImportPlugin>( memnew(EditorTextureImportPlugin(this,EditorTextureImportPlugin::MODE_ATLAS) )));
	Ref<EditorSceneImportPlugin> _scene_import =  memnew(EditorSceneImportPlugin(this) );
	Ref<EditorSceneImporterCollada> _collada_import = memnew( EditorSceneImporterCollada);
	_scene_import->add_importer(_collada_import);
//	Ref<EditorSceneImporterFBXConv> _fbxconv_import = memnew( EditorSceneImporterFBXConv);
//	_scene_import->add_importer(_fbxconv_import);
	editor_import_export->add_import_plugin( _scene_import);
	editor_import_export->add_import_plugin( Ref<EditorSceneAnimationImportPlugin>( memnew(EditorSceneAnimationImportPlugin(this))));
	editor_import_export->add_import_plugin( Ref<EditorMeshImportPlugin>( memnew(EditorMeshImportPlugin(this))));
	editor_import_export->add_import_plugin( Ref<EditorFontImportPlugin>( memnew(EditorFontImportPlugin(this))));
	editor_import_export->add_import_plugin( Ref<EditorSampleImportPlugin>( memnew(EditorSampleImportPlugin(this))));
	editor_import_export->add_import_plugin( Ref<EditorTranslationImportPlugin>( memnew(EditorTranslationImportPlugin(this))));


	for(int i=0;i<editor_import_export->get_import_plugin_count();i++) {
		    import_menu->get_popup()->add_item(editor_import_export->get_import_plugin(i)->get_visible_name(),IMPORT_PLUGIN_BASE+i);
	}

	editor_import_export->add_export_plugin( Ref<EditorTextureExportPlugin>( memnew(EditorTextureExportPlugin)));

	add_editor_plugin( memnew( CanvasItemEditorPlugin(this) ) );
	add_editor_plugin( memnew( SpatialEditorPlugin(this) ) );
	add_editor_plugin( memnew( ScriptEditorPlugin(this) ) );
	add_editor_plugin( memnew( EditorHelpPlugin(this) ) );
	add_editor_plugin( memnew( AnimationPlayerEditorPlugin(this) ) );
	add_editor_plugin( memnew( ShaderGraphEditorPlugin(this,true) ) );
	add_editor_plugin( memnew( ShaderGraphEditorPlugin(this,false) ) );
	add_editor_plugin( memnew( ShaderEditorPlugin(this,true) ) );
	add_editor_plugin( memnew( ShaderEditorPlugin(this,false) ) );
	add_editor_plugin( memnew( CameraEditorPlugin(this) ) );
	add_editor_plugin( memnew( SampleEditorPlugin(this) ) );
	add_editor_plugin( memnew( SampleLibraryEditorPlugin(this) ) );
	add_editor_plugin( memnew( ThemeEditorPlugin(this) ) );
	add_editor_plugin( memnew( MultiMeshEditorPlugin(this) ) );
	add_editor_plugin( memnew( MeshInstanceEditorPlugin(this) ) );
	add_editor_plugin( memnew( AnimationTreeEditorPlugin(this) ) );
	add_editor_plugin( memnew( SamplePlayerEditorPlugin(this) ) );
	add_editor_plugin( memnew( MeshLibraryEditorPlugin(this) ) );
	add_editor_plugin( memnew( StreamEditorPlugin(this) ) );
	add_editor_plugin( memnew( StyleBoxEditorPlugin(this) ) );
	add_editor_plugin( memnew( ParticlesEditorPlugin(this) ) );
	add_editor_plugin( memnew( ResourcePreloaderEditorPlugin(this) ) );
	add_editor_plugin( memnew( ItemListEditorPlugin(this) ) );
	add_editor_plugin( memnew( RichTextEditorPlugin(this) ) );
	add_editor_plugin( memnew( CollisionPolygonEditorPlugin(this) ) );
	add_editor_plugin( memnew( CollisionPolygon2DEditorPlugin(this) ) );
	add_editor_plugin( memnew( TileSetEditorPlugin(this) ) );
	add_editor_plugin( memnew( TileMapEditorPlugin(this) ) );
	add_editor_plugin( memnew( SpriteFramesEditorPlugin(this) ) );
	add_editor_plugin( memnew( Particles2DEditorPlugin(this) ) );
	add_editor_plugin( memnew( Path2DEditorPlugin(this) ) );
	add_editor_plugin( memnew( PathEditorPlugin(this) ) );
	add_editor_plugin( memnew( BakedLightEditorPlugin(this) ) );
	add_editor_plugin( memnew( Polygon2DEditorPlugin(this) ) );

	for(int i=0;i<EditorPlugins::get_plugin_count();i++)
		add_editor_plugin( EditorPlugins::create(i,this) );

	circle_step_msec=OS::get_singleton()->get_ticks_msec();
	circle_step_frame=OS::get_singleton()->get_frames_drawn();;
	circle_step=0;


	import_menu->get_popup()->add_separator();
	import_menu->get_popup()->add_item("Re-Import..",SETTINGS_IMPORT);

	editor_plugin_screen=NULL;
	editor_plugin_over=NULL;

//	force_top_viewport(true);
	_edit_current();
	current=NULL;

	PhysicsServer::get_singleton()->set_active(false); // no physics by default if editor
	Physics2DServer::get_singleton()->set_active(false); // no physics by default if editor
	ScriptServer::set_scripting_enabled(false); // no scripting by default if editor

	Globals::get_singleton()->set("debug/indicators_enabled",true);
	Globals::get_singleton()->set("render/room_cull_enabled",false);
	theme->set_color("prop_category","Editor",Color::hex(0x403d41ff));
	theme->set_color("prop_section","Editor",Color::hex(0x383539ff));
	theme->set_color("prop_subsection","Editor",Color::hex(0x343135ff));
	theme->set_color("fg_selected","Editor",Color::html("ffbd8e8e"));
	theme->set_color("fg_error","Editor",Color::html("ffbd8e8e"));

	reference_resource_mem=true;
	save_external_resources_mem=true;

	set_process(true);
	OS::get_singleton()->set_low_processor_usage_mode(true);


	if (0) { //not sure if i want this to happen after all

		//store project name in ssettings
		String project_name;
		//figure it out from path
		project_name=Globals::get_singleton()->get_resource_path().replace("\\","/");
		print_line("path: "+project_name);
		if (project_name.length() && project_name[project_name.length()-1]=='/')
			project_name=project_name.substr(0,project_name.length()-1);

		project_name=project_name.replace("/","::");

		if (project_name!="") {
			EditorSettings::get_singleton()->set("projects/"+project_name,Globals::get_singleton()->get_resource_path());
			EditorSettings::get_singleton()->raise_order("projects/"+project_name);
			EditorSettings::get_singleton()->save();
		}
	}


	edited_scene=NULL;
	saved_version=0;
	unsaved_cache=true;
	_last_instanced_scene=NULL;


	quick_open = memnew( EditorQuickOpen );
	gui_base->add_child(quick_open);
	quick_open->connect("quick_open",this,"_quick_opened");

	quick_run = memnew( EditorQuickOpen );
	gui_base->add_child(quick_run);
	quick_run->connect("quick_open",this,"_quick_run");


	_update_recent_scenes();


	editor_data.restore_editor_global_states();
	convert_old=false;
	opening_prev=false;
	set_process_unhandled_input(true);
	_playing_edited=false;

	load_errors = memnew( TextEdit );
	load_errors->set_readonly(true);
	load_error_dialog = memnew( AcceptDialog );
	load_error_dialog->add_child(load_errors);
	load_error_dialog->set_title("Load Errors");
	load_error_dialog->set_child_rect(load_errors);
	add_child(load_error_dialog);

	//EditorImport::add_importer( Ref<EditorImporterCollada>( memnew(EditorImporterCollada )));

	EditorFileSystem::get_singleton()->connect("sources_changed",this,"_sources_changed");
	EditorFileSystem::get_singleton()->connect("filesystem_changed",this,"_fs_changed");


	{
		List<StringName> tl;
		StringName ei = "EditorIcons";
		gui_base->get_theme()->get_icon_list(ei,&tl);
		for(List<StringName>::Element *E=tl.front();E;E=E->next()) {

			if (!ObjectTypeDB::type_exists(E->get()))
				continue;
			icon_type_cache[E->get()]=gui_base->get_theme()->get_icon(E->get(),ei);
		}
	}


	EditorSettings::get_singleton()->enable_plugins();
	Node::set_human_readable_collision_renaming(true);


//	Ref<ImageTexture> it = gui_base->get_icon("logo","Icons");
//	OS::get_singleton()->set_icon( it->get_data() );

	for(int i=0;i<_init_callbacks.size();i++)
		_init_callbacks[i]();

}


EditorNode::~EditorNode() {	

	memdelete(editor_selection);
	memdelete(file_server);
	EditorSettings::destroy();
}


