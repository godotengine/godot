/*************************************************************************/
/*  scenes_dock.cpp                                                      */
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
#include "scenes_dock.h"
#include "os/dir_access.h"
#include "os/file_access.h"
#include "globals.h"

#include "io/resource_loader.h"
#include "os/os.h"
#include "editor_node.h"

bool ScenesDock::_create_tree(TreeItem *p_parent,EditorFileSystemDirectory *p_dir) {

	String search_term = tree_filter->get_search_term();
	ScenesDockFilter::FilterOption file_filter = tree_filter->get_file_filter();

	TreeItem *item = tree->create_item(p_parent);
	item->set_text(0,p_dir->get_name()+"/");
	item->set_icon(0,get_icon("Folder","EditorIcons"));
	item->set_custom_bg_color(0,get_color("prop_subsection","Editor"));

	bool has_items=false;

	for(int i=0;i<p_dir->get_subdir_count();i++) {

		if (_create_tree(item,p_dir->get_subdir(i)))
			has_items=true;
	}

	for (int i=0;i<p_dir->get_file_count();i++) {

		String file_name = p_dir->get_file(i);
		String file_path = p_dir->get_file_path(i);

		// ScenesDockFilter::FILTER_PATH
		String search_from = file_path.right(6); // trim "res://"
		if (file_filter == ScenesDockFilter::FILTER_NAME)
			 search_from = file_name;
		else if (file_filter == ScenesDockFilter::FILTER_FOLDER)
			search_from = file_path.right(6).get_base_dir();

		if (search_term!="" && search_from.findn(search_term)==-1)
			continue;

		bool isfave = favorites.has(file_path);
		if (button_favorite->is_pressed() && !isfave)
			continue;

		TreeItem *fitem = tree->create_item(item);
		fitem->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		fitem->set_editable(0,true);
		fitem->set_checked(0,isfave);
		fitem->set_text(0,file_name);

		Ref<Texture> icon = get_icon( (has_icon(p_dir->get_file_type(i),"EditorIcons")?p_dir->get_file_type(i):String("Object")),"EditorIcons");
		fitem->set_icon(0, icon );


		fitem->set_metadata(0,file_path);
		//if (p_dir->files[i]->icon.is_valid()) {
//			fitem->set_icon(0,p_dir->files[i]->icon);
//		}
		has_items=true;

	}

	if (!has_items) {

		memdelete(item);
		return false;

	}

	return true;
}


void ScenesDock::_update_tree() {

	tree->clear();
	updating_tree=true;
	_create_tree(NULL,EditorFileSystem::get_singleton()->get_filesystem());
	updating_tree=false;

}


void ScenesDock::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {


			EditorFileSystem::get_singleton()->connect("filesystem_changed",this,"_update_tree");

			button_reload->set_icon( get_icon("Reload","EditorIcons"));
			button_favorite->set_icon( get_icon("Favorites","EditorIcons"));
			button_instance->set_icon( get_icon("Add","EditorIcons"));
			button_open->set_icon( get_icon("Folder","EditorIcons"));

			String path = Globals::get_singleton()->get_resource_path()+"/favorites.cfg";
			FileAccess *f=FileAccess::open(path,FileAccess::READ);
			if (f) {

				String l = f->get_line();

				while(l!="") {
					favorites.insert(l);
					l = f->get_line();

				}

				f->close();
				memdelete(f);
			}



			_update_tree(); //maybe it finished already
		} break;
		case NOTIFICATION_EXIT_TREE: {

		} break;
		case NOTIFICATION_PROCESS: {

		} break;
	}

}

void ScenesDock::_favorite_toggled() {

	if (updating_tree)
		return;


	TreeItem *sel = tree->get_selected();
	if (!sel)
		return; //?

	bool faved = sel->is_checked(0);
	String path = sel->get_metadata(0);
	if (faved)
		favorites.insert(path);
	else
		favorites.erase(path);

	timer->start();

}

void ScenesDock::_favorites_toggled(bool p_toggled) {

	_update_tree();
}

String ScenesDock::get_selected_path() const {

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return "";
	String path = sel->get_metadata(0);
	return "res://"+path;
}

void ScenesDock::_instance_pressed() {

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return;
	String path = sel->get_metadata(0);
	emit_signal("instance",path);
}

void ScenesDock::_open_pressed(){

	TreeItem *sel = tree->get_selected();
	if (!sel) {
		return;
	}
	String path = sel->get_metadata(0);

	if (ResourceLoader::get_resource_type(path)=="PackedScene") {

		editor->open_request(path);
	} else {

		/*

		RES res = ResourceLoader::load(path);
		if (res.is_null()) {


			return;
		}*/

		editor->load_resource(path);
	}

//	emit_signal("open",path);

}

void ScenesDock::_save_favorites() {

	String path = Globals::get_singleton()->get_resource_path()+"/favorites.cfg";
	FileAccess *f=FileAccess::open(path,FileAccess::WRITE);
	ERR_FAIL_COND(!f);
	for(Set<String>::Element *E=favorites.front();E;E=E->next() ) {

		CharString utf8f = E->get().utf8();
		f->store_buffer((const uint8_t*)utf8f.get_data(),utf8f.length());
		f->store_8('\n');
	}

	f->close();
	memdelete(f);
}

void ScenesDock::_rescan() {

	EditorFileSystem::get_singleton()->scan();
}

void ScenesDock::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_update_tree"),&ScenesDock::_update_tree);
	ObjectTypeDB::bind_method(_MD("_rescan"),&ScenesDock::_rescan);
	ObjectTypeDB::bind_method(_MD("_favorites_toggled"),&ScenesDock::_favorites_toggled);
	ObjectTypeDB::bind_method(_MD("_favorite_toggled"),&ScenesDock::_favorite_toggled);
	ObjectTypeDB::bind_method(_MD("_instance_pressed"),&ScenesDock::_instance_pressed);
	ObjectTypeDB::bind_method(_MD("_open_pressed"),&ScenesDock::_open_pressed);
	ObjectTypeDB::bind_method(_MD("_save_favorites"),&ScenesDock::_save_favorites);

	ADD_SIGNAL(MethodInfo("instance"));
	ADD_SIGNAL(MethodInfo("open"));

}

ScenesDock::ScenesDock(EditorNode *p_editor) {

	editor=p_editor;

	HBoxContainer *toolbar_hbc = memnew( HBoxContainer );
	add_child(toolbar_hbc);

	button_reload = memnew( Button );
	button_reload->set_flat(true);
	button_reload->connect("pressed",this,"_rescan");	
	toolbar_hbc->add_child(button_reload);

	button_favorite = memnew( Button );
	button_favorite->set_flat(true);
	button_favorite->set_toggle_mode(true);
	button_favorite->connect("toggled",this,"_favorites_toggled");
	toolbar_hbc->add_child(button_favorite);

	toolbar_hbc->add_spacer();

	button_open = memnew( Button );
	button_open->set_flat(true);
	button_open->connect("pressed",this,"_open_pressed");
	toolbar_hbc->add_child(button_open);

	button_instance = memnew( Button );
	button_instance->set_flat(true);
	button_instance->connect("pressed",this,"_instance_pressed");
	toolbar_hbc->add_child(button_instance);

	tree = memnew( Tree );
	tree_filter=memnew( ScenesDockFilter() );
	tree_filter->connect("filter_changed", this, "_update_tree");
	add_child(tree_filter);
	add_child(tree);

	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->connect("item_edited",this,"_favorite_toggled");
	tree->connect("item_activated",this,"_open_pressed");

	timer = memnew( Timer );
	timer->set_one_shot(true);
	timer->set_wait_time(2);
	timer->connect("timeout",this,"_save_favorites");
	add_child(timer);


	updating_tree=false;

}

ScenesDock::~ScenesDock() {

}

void ScenesDockFilter::_setup_filters() {

	filter_option->clear();
	filter_option->add_item("Path");
	filter_option->add_item("Name");
	filter_option->add_item("Folder");
#if 0
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("",&extensions);

	file_filter->add_item("All Files (*)");
	filters.push_back("*");

	List<String> filter_texts;
	for(int i=0;i<extensions.size();i++) {
		filter_texts.push_back("*."+extensions[i]+" ; "+extensions[i].to_upper());
		filters.push_back(extensions[i]);
	}
	for(int i=0;i<filter_texts.size();i++) {

		String flt=filter_texts[i].get_slice(";",0).strip_edges();
		String desc=filter_texts[i].get_slice(";",1).strip_edges();
		if (desc.length())
			file_filter->add_item(desc+" ( "+flt+" )");
		else
			file_filter->add_item("( "+flt+" )");
	}
#endif
}

void ScenesDockFilter::_command(int p_command) {
	switch (p_command) {

		case CMD_CLEAR_FILTER: {
			if (search_box->get_text()!="") {
				search_box->clear();
				emit_signal("filter_changed");
			}
		}break;
	}
}

void ScenesDockFilter::_search_text_changed(const String &p_newtext) {
	emit_signal("filter_changed");
}

String ScenesDockFilter::get_search_term() {
	return search_box->get_text().strip_edges();
}

ScenesDockFilter::FilterOption ScenesDockFilter::get_file_filter() {
	return _current_filter;
}

void ScenesDockFilter::_file_filter_selected(int p_idx) {
	FilterOption selected = (FilterOption)(filter_option->get_selected());
	if (_current_filter != selected ) {
		_current_filter = selected;
		emit_signal("filter_changed");
	}
}

void ScenesDockFilter::_notification(int p_what) {
	switch(p_what) {
		case NOTIFICATION_ENTER_TREE: {
			clear_search_button->set_icon(get_icon("CloseHover","EditorIcons"));
		} break;
	}
}

void ScenesDockFilter::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_command"),&ScenesDockFilter::_command);
	ObjectTypeDB::bind_method(_MD("_search_text_changed"), &ScenesDockFilter::_search_text_changed);
	ObjectTypeDB::bind_method(_MD("_file_filter_selected"), &ScenesDockFilter::_file_filter_selected);

	ADD_SIGNAL( MethodInfo("filter_changed") );
}

ScenesDockFilter::ScenesDockFilter() {

	_current_filter = FILTER_PATH;

	filter_option = memnew( OptionButton );
	filter_option->set_custom_minimum_size(Size2(60,10));
	filter_option->set_clip_text(true);
	filter_option->connect("item_selected", this, "_file_filter_selected");
	add_child(filter_option);

	_setup_filters();

	search_box = memnew( LineEdit );
	search_box->connect("text_changed",this,"_search_text_changed");
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(search_box);

	clear_search_button = memnew( ToolButton );
	clear_search_button->connect("pressed",this,"_command",make_binds(CMD_CLEAR_FILTER));
	add_child(clear_search_button);

}

