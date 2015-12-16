/*************************************************************************/
/*  scenes_dock.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include "editor_settings.h"

bool ScenesDock::_create_tree(TreeItem *p_parent,EditorFileSystemDirectory *p_dir) {


	TreeItem *item = tree->create_item(p_parent);
	String dname=p_dir->get_name();
	if (dname=="")
		dname="res://";

	item->set_text(0,dname);
	item->set_icon(0,get_icon("Folder","EditorIcons"));
	item->set_selectable(0,true);
	String lpath = p_dir->get_path();
	if (lpath!="res://" && lpath.ends_with("/")) {
		lpath=lpath.substr(0,lpath.length()-1);
	}
	item->set_metadata(0,lpath);
	if (lpath==path) {
		item->select(0);
	}


	//item->set_custom_bg_color(0,get_color("prop_subsection","Editor"));

	bool has_items=false;

	for(int i=0;i<p_dir->get_subdir_count();i++) {

		if (_create_tree(item,p_dir->get_subdir(i)))
			has_items=true;
	}
#if 0
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
#endif
	/*if (!has_items) {

		memdelete(item);
		return false;

	}*/

	return true;
}


void ScenesDock::_update_tree() {

	tree->clear();
	updating_tree=true;
	TreeItem *root = tree->create_item();
	TreeItem *favorites = tree->create_item(root);
	favorites->set_icon(0, get_icon("Favorites","EditorIcons") );
	favorites->set_text(0,"Favorites:");
	favorites->set_selectable(0,false);
	Vector<String> faves = 	EditorSettings::get_singleton()->get_favorite_dirs();
	for(int i=0;i<faves.size();i++) {
		if (!faves[i].begins_with("res://"))
			continue;

		TreeItem *ti = tree->create_item(favorites);
		String fv = faves[i];
		if (fv=="res://")
			ti->set_text(0,"/");
		else
			ti->set_text(0,faves[i].get_file());
		ti->set_icon(0,get_icon("Folder","EditorIcons"));
		ti->set_selectable(0,true);
		ti->set_metadata(0,faves[i]);
	}

	_create_tree(root,EditorFileSystem::get_singleton()->get_filesystem());
	updating_tree=false;

}


void ScenesDock::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			if (initialized)
				return;
			initialized=true;

			EditorFileSystem::get_singleton()->connect("filesystem_changed",this,"_fs_changed");

			button_reload->set_icon( get_icon("Reload","EditorIcons"));
			button_favorite->set_icon( get_icon("Favorites","EditorIcons"));
			button_fav_up->set_icon( get_icon("MoveUp","EditorIcons"));
			button_fav_down->set_icon( get_icon("MoveDown","EditorIcons"));
			button_instance->set_icon( get_icon("Add","EditorIcons"));
			button_open->set_icon( get_icon("Folder","EditorIcons"));
			button_back->set_icon( get_icon("Filesystem","EditorIcons"));
			display_mode->set_icon( get_icon("FileList","EditorIcons"));
			display_mode->connect("pressed",this,"_change_file_display");
			file_options->set_icon( get_icon("Tools","EditorIcons"));
			files->connect("item_activated",this,"_select_file");
			button_hist_next->connect("pressed",this,"_fw_history");
			button_hist_prev->connect("pressed",this,"_bw_history");

			button_hist_next->set_icon( get_icon("Forward","EditorIcons"));
			button_hist_prev->set_icon( get_icon("Back","EditorIcons"));
			file_options->get_popup()->connect("item_pressed",this,"_file_option");


			button_back->connect("pressed",this,"_go_to_tree",varray(),CONNECT_DEFERRED);
			current_path->connect("text_entered",this,"_go_to_dir");
			_update_tree(); //maybe it finished already

			if (EditorFileSystem::get_singleton()->is_scanning()) {
				_set_scannig_mode();
			}

		} break;
		case NOTIFICATION_PROCESS: {
			if (EditorFileSystem::get_singleton()->is_scanning()) {
				scanning_progress->set_val(EditorFileSystem::get_singleton()->get_scanning_progress()*100);
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {

		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {

			display_mode->set_pressed(int(EditorSettings::get_singleton()->get("file_dialog/display_mode"))==EditorFileDialog::DISPLAY_LIST);

			_change_file_display();
		} break;
	}

}




void ScenesDock::_dir_selected() {

	TreeItem *ti = tree->get_selected();
	if (!ti)
		return;
	String dir = ti->get_metadata(0);
	bool found=false;
	Vector<String> favorites = EditorSettings::get_singleton()->get_favorite_dirs();
	for(int i=0;i<favorites.size();i++) {

		if (favorites[i]==dir) {
			found=true;
			break;
		}
	}


	button_favorite->set_pressed(found);
	if (ti->get_parent() && ti->get_parent()->get_parent()==tree->get_root() && !ti->get_parent()->get_prev()) {

		//a favorite!!!
		button_fav_up->set_disabled(!ti->get_prev());
		button_fav_down->set_disabled(!ti->get_next());
	} else {
		button_fav_up->set_disabled(true);
		button_fav_down->set_disabled(true);

	}

}

void ScenesDock::_fav_up_pressed() {

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return ;

	if (!sel->get_prev())
		return;

	String sw = sel->get_prev()->get_metadata(0);
	String txt = sel->get_metadata(0);

	Vector<String> favorited = EditorSettings::get_singleton()->get_favorite_dirs();

	int a_idx=favorited.find(sw);
	int b_idx=favorited.find(txt);

	if (a_idx==-1 || b_idx==-1)
		return;
	SWAP(favorited[a_idx],favorited[b_idx]);

	EditorSettings::get_singleton()->set_favorite_dirs(favorited);

	_update_tree();

	if (!tree->get_root() || !tree->get_root()->get_children() || !tree->get_root()->get_children()->get_children())
		return;
	sel = tree->get_root()->get_children()->get_children();
	while(sel) {

		String t = sel->get_metadata(0);
		if (t==txt) {
			sel->select(0);
			return;
		}
		sel=sel->get_next();
	}
}

void ScenesDock::_fav_down_pressed() {

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return ;

	if (!sel->get_next())
		return;

	String sw = sel->get_next()->get_metadata(0);
	String txt = sel->get_metadata(0);

	Vector<String> favorited = EditorSettings::get_singleton()->get_favorite_dirs();

	int a_idx=favorited.find(sw);
	int b_idx=favorited.find(txt);

	if (a_idx==-1 || b_idx==-1)
		return;
	SWAP(favorited[a_idx],favorited[b_idx]);

	EditorSettings::get_singleton()->set_favorite_dirs(favorited);

	_update_tree();

	if (!tree->get_root() || !tree->get_root()->get_children() || !tree->get_root()->get_children()->get_children())
		return;
	sel = tree->get_root()->get_children()->get_children();
	while(sel) {

		String t = sel->get_metadata(0);
		if (t==txt) {
			sel->select(0);
			return;
		}
		sel=sel->get_next();
	}
}

void ScenesDock::_favorites_pressed() {

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return ;
	String dir = sel->get_metadata(0);

	int idx = -1;
	Vector<String> favorites = EditorSettings::get_singleton()->get_favorite_dirs();
	for(int i=0;i<favorites.size();i++) {

		if (favorites[i]==dir) {
			idx=i;
			break;
		}
	}

	if (button_favorite->is_pressed() && idx==-1) {
		favorites.push_back(dir);
		EditorSettings::get_singleton()->set_favorite_dirs(favorites);
		_update_tree();
	} else if (!button_favorite->is_pressed() && idx!=-1) {
		favorites.remove(idx);
		EditorSettings::get_singleton()->set_favorite_dirs(favorites);
		_update_tree();
	}

}

String ScenesDock::get_selected_path() const {

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return "";
	String path = sel->get_metadata(0);
	return "res://"+path;
}

void ScenesDock::_instance_pressed() {

	if (tree_mode)
	{
		TreeItem *sel = tree->get_selected();
		if (!sel)
			return;
		String path = sel->get_metadata(0);
	}
	else
	{
		int idx = -1;
		for (int i = 0; i<files->get_item_count(); i++) {
			if (files->is_selected(i))
			{
				idx = i;
				break;
			}
		}

		if (idx<0)
			return;

		path = files->get_item_metadata(idx);
	}

	emit_signal("instance",path);
}

void ScenesDock::_thumbnail_done(const String& p_path,const Ref<Texture>& p_preview, const Variant& p_udata) {

	if (p_preview.is_valid() && path==p_path.get_base_dir()) {

		int idx=p_udata;
		if (idx>=files->get_item_count())
			return;
		String fpath = files->get_item_metadata(idx);
		if (fpath!=p_path)
			return;
		files->set_item_icon(idx,p_preview);

	}

}

void ScenesDock::_change_file_display() {

	if (display_mode->is_pressed()) {
		display_mode->set_icon( get_icon("FileThumbnail","EditorIcons"));

	} else {
		display_mode->set_icon( get_icon("FileList","EditorIcons"));
	}

	_update_files(true);
}

void ScenesDock::_update_files(bool p_keep_selection) {

	Set<String> cselection;

	if (p_keep_selection) {

		for(int i=0;i<files->get_item_count();i++) {

			if (files->is_selected(i))
				cselection.insert(files->get_item_text(i));
		}
	}

	files->clear();

	EditorFileSystemDirectory *efd = EditorFileSystem::get_singleton()->get_path(path);
	if (!efd)
		return;

	int thumbnail_size = EditorSettings::get_singleton()->get("file_dialog/thumbnail_size");
	Ref<Texture> folder_thumbnail;
	Ref<Texture> file_thumbnail;

	bool use_thumbnails=!display_mode->is_pressed();

	if (use_thumbnails) { //thumbnails

		files->set_max_columns(0);
		files->set_icon_mode(ItemList::ICON_MODE_TOP);
		files->set_fixed_column_width(thumbnail_size*3/2);
		files->set_max_text_lines(2);
		files->set_min_icon_size(Size2(thumbnail_size,thumbnail_size));

		if (!has_icon("ResizedFolder","EditorIcons")) {
			Ref<ImageTexture> folder = get_icon("FolderBig","EditorIcons");
			Image img = folder->get_data();
			img.resize(thumbnail_size,thumbnail_size);
			Ref<ImageTexture> resized_folder = Ref<ImageTexture>( memnew( ImageTexture));
			resized_folder->create_from_image(img,0);
			Theme::get_default()->set_icon("ResizedFolder","EditorIcons",resized_folder);
		}

		folder_thumbnail = get_icon("ResizedFolder","EditorIcons");

		if (!has_icon("ResizedFile","EditorIcons")) {
			Ref<ImageTexture> file = get_icon("FileBig","EditorIcons");
			Image img = file->get_data();
			img.resize(thumbnail_size,thumbnail_size);
			Ref<ImageTexture> resized_file = Ref<ImageTexture>( memnew( ImageTexture));
			resized_file->create_from_image(img,0);
			Theme::get_default()->set_icon("ResizedFile","EditorIcons",resized_file);
		}

		file_thumbnail = get_icon("ResizedFile","EditorIcons");

	} else {

		files->set_icon_mode(ItemList::ICON_MODE_LEFT);
		files->set_max_columns(1);
		files->set_max_text_lines(1);
		files->set_fixed_column_width(0);
		files->set_min_icon_size(Size2());

	}


	if (path!="res://") {

		if (use_thumbnails) {
			files->add_item("..",folder_thumbnail,true);
		} else {
			files->add_item("..",get_icon("folder","FileDialog"),true);
		}

		String bd = path.get_base_dir();
		if (bd!="res://" && !bd.ends_with("/"))
			bd+="/";

		files->set_item_metadata(files->get_item_count()-1,bd);
	}

	for(int i=0;i<efd->get_subdir_count();i++) {

		String dname=efd->get_subdir(i)->get_name();


		if (use_thumbnails) {
			files->add_item(dname,folder_thumbnail,true);
		} else {
			files->add_item(dname,get_icon("folder","FileDialog"),true);
		}

		files->set_item_metadata(files->get_item_count()-1,path.plus_file(dname)+"/");

		if (cselection.has(dname))
			files->select(files->get_item_count()-1,false);
	}

	for(int i=0;i<efd->get_file_count();i++) {

		String fname=efd->get_file(i);
		String fp = path.plus_file(fname);


		String type = efd->get_file_type(i);
		Ref<Texture> type_icon;

		if (has_icon(type,"EditorIcons")) {
			type_icon=get_icon(type,"EditorIcons");
		} else {
			type_icon=get_icon("Object","EditorIcons");
		}

		if (use_thumbnails) {
			files->add_item(fname,file_thumbnail,true);
			files->set_item_metadata(files->get_item_count()-1,fp);
			files->set_item_tag_icon(files->get_item_count()-1,type_icon);
			EditorResourcePreview::get_singleton()->queue_resource_preview(fp,this,"_thumbnail_done",files->get_item_count()-1);
		} else {
			files->add_item(fname,type_icon,true);
			files->set_item_metadata(files->get_item_count()-1,fp);

		}

		if (cselection.has(fname))
			files->select(files->get_item_count()-1,false);

	}


}

void ScenesDock::_select_file(int p_idx) {

	files->select(p_idx,true);
	_open_pressed();
}

void ScenesDock::_go_to_tree() {

	tree->show();
	files->hide();
	path_hb->hide();
	_update_tree();
	tree->grab_focus();
	tree->ensure_cursor_is_visible();
	button_favorite->show();
	button_fav_up->show();
	button_fav_down->show();
	button_open->hide();
	button_instance->hide();
	button_open->hide();
	file_options->hide();
	tree_mode=true;
}

void ScenesDock::_go_to_dir(const String& p_dir){

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (da->change_dir(p_dir)==OK) {
		path=da->get_current_dir();
		_update_files(false);
	}
	current_path->set_text(path);
	memdelete(da);


}
void ScenesDock::_fs_changed() {

	button_hist_prev->set_disabled(history_pos==0);
	button_hist_next->set_disabled(history_pos+1==history.size());
	scanning_vb->hide();

	if (tree_mode) {

		tree->show();
		button_favorite->show();
		button_fav_up->show();
		button_fav_down->show();
		_update_tree();
	} else {
		files->show();
		path_hb->show();
		button_instance->show();
		button_open->show();
		file_options->show();
		_update_files(true);
	}

	set_process(false);
}

void ScenesDock::_set_scannig_mode() {

	tree->hide();
	button_favorite->hide();
	button_fav_up->hide();
	button_fav_down->hide();
	button_instance->hide();
	button_open->hide();
	file_options->hide();
	button_hist_prev->set_disabled(true);
	button_hist_next->set_disabled(true);
	scanning_vb->show();
	path_hb->hide();
	files->hide();
	set_process(true);
	if (EditorFileSystem::get_singleton()->is_scanning()) {
		scanning_progress->set_val(EditorFileSystem::get_singleton()->get_scanning_progress()*100);
	} else {
		scanning_progress->set_val(0);
	}

}

void ScenesDock::_fw_history() {

	if (history_pos<history.size()-1)
		history_pos++;

	path=history[history_pos];

	if (tree_mode) {
		_update_tree();
		tree->grab_focus();
		tree->ensure_cursor_is_visible();
	} else {
		_update_files(false);
		current_path->set_text(path);
	}

	button_hist_prev->set_disabled(history_pos==0);
	button_hist_next->set_disabled(history_pos+1==history.size());

}

void ScenesDock::_bw_history() {

	if (history_pos>0)
		history_pos--;

	path=history[history_pos];

	if (tree_mode) {
		_update_tree();
		tree->grab_focus();
		tree->ensure_cursor_is_visible();
	} else {
		_update_files(false);
		current_path->set_text(path);
	}

	button_hist_prev->set_disabled(history_pos==0);
	button_hist_next->set_disabled(history_pos+1==history.size());

}

void ScenesDock::_push_to_history() {

	history.resize(history_pos+1);
	if (history[history_pos]!=path) {
		history.push_back(path);
		history_pos++;
	}

	button_hist_prev->set_disabled(history_pos==0);
	button_hist_next->set_disabled(history_pos+1==history.size());

}


void ScenesDock::_find_inside_move_files(EditorFileSystemDirectory *efsd,Vector<String>& files) {

	for(int i=0;i<efsd->get_subdir_count();i++) {
		_find_inside_move_files(efsd->get_subdir(i),files);
	}
	for(int i=0;i<efsd->get_file_count();i++) {
		files.push_back(efsd->get_file_path(i));
	}

}

void ScenesDock::_find_remaps(EditorFileSystemDirectory *efsd,Map<String,String> &renames,List<String>& to_remaps) {

	for(int i=0;i<efsd->get_subdir_count();i++) {
		_find_remaps(efsd->get_subdir(i),renames,to_remaps);
	}
	for(int i=0;i<efsd->get_file_count();i++) {
		Vector<String> deps=efsd->get_file_deps(i);
		for(int j=0;j<deps.size();j++) {
			if (renames.has(deps[j])) {
				to_remaps.push_back(efsd->get_file_path(i));
				break;
			}
		}
	}
}


void ScenesDock::_rename_operation(const String& p_to_path) {

	if (move_files[0]==p_to_path) {
		EditorNode::get_singleton()->show_warning("Same source and destination files, doing nothing.");
		return;
	}
	if (FileAccess::exists(p_to_path)) {
		EditorNode::get_singleton()->show_warning("Target file exists, can't overwrite. Delete first.");
		return;
	}

	Map<String,String> renames;
	renames[move_files[0]]=p_to_path;

	List<String> remap;

	_find_remaps(EditorFileSystem::get_singleton()->get_filesystem(),renames,remap);
	print_line("found files to remap: "+itos(remap.size()));

	//perform remaps
	for(List<String>::Element *E=remap.front();E;E=E->next()) {

		Error err = ResourceLoader::rename_dependencies(E->get(),renames);
		print_line("remapping: "+E->get());

		if (err!=OK) {
			EditorNode::get_singleton()->add_io_error("Can't rename deps for:\n"+E->get()+"\n");
		}
	}

	//finally, perform moves

	DirAccess *da=DirAccess::create(DirAccess::ACCESS_RESOURCES);

	Error err = da->rename(move_files[0],p_to_path);
	print_line("moving file "+move_files[0]+" to "+p_to_path);
	if (err!=OK) {
		EditorNode::get_singleton()->add_io_error("Error moving file:\n"+move_files[0]+"\n");
	}

	//rescan everything
	memdelete(da);
	print_line("call rescan!");
	_rescan();
}


void ScenesDock::_move_operation(const String& p_to_path) {

	if (p_to_path==path) {
		EditorNode::get_singleton()->show_warning("Same source and destination paths, doing nothing.");
		return;
	}

	//find files inside dirs to be moved

	Vector<String> inside_files;

	for(int i=0;i<move_dirs.size();i++) {
		if (p_to_path.begins_with(move_dirs[i])) {
			EditorNode::get_singleton()->show_warning("Can't move directories to within themselves");
			return;
		}

		EditorFileSystemDirectory *efsd=EditorFileSystem::get_singleton()->get_path(move_dirs[i]);
		if (!efsd)
			continue;
		_find_inside_move_files(efsd,inside_files);
	}

	//make list of remaps
	Map<String,String> renames;
	String repfrom=path=="res://"?path:String(path+"/");
	String repto=p_to_path=="res://"?p_to_path:String(p_to_path+"/");

	for(int i=0;i<move_files.size();i++) {
		renames[move_files[i]]=move_files[i].replace_first(repfrom,repto);
		print_line("move file "+move_files[i]+" -> "+renames[move_files[i]]);
	}
	for(int i=0;i<inside_files.size();i++) {
		renames[inside_files[i]]=inside_files[i].replace_first(repfrom,repto);
		print_line("inside file "+inside_files[i]+" -> "+renames[inside_files[i]]);
	}

	//make list of files that will be run the remapping
	List<String> remap;

	_find_remaps(EditorFileSystem::get_singleton()->get_filesystem(),renames,remap);
	print_line("found files to remap: "+itos(remap.size()));

	//perform remaps
	for(List<String>::Element *E=remap.front();E;E=E->next()) {

		Error err = ResourceLoader::rename_dependencies(E->get(),renames);
		print_line("remapping: "+E->get());

		if (err!=OK) {
			EditorNode::get_singleton()->add_io_error("Can't rename deps for:\n"+E->get()+"\n");
		}
	}

	//finally, perform moves

	DirAccess *da=DirAccess::create(DirAccess::ACCESS_RESOURCES);

	for(int i=0;i<move_files.size();i++) {

		String to = move_files[i].replace_first(repfrom,repto);
		Error err = da->rename(move_files[i],to);
		print_line("moving file "+move_files[i]+" to "+to);
		if (err!=OK) {
			EditorNode::get_singleton()->add_io_error("Error moving file:\n"+move_files[i]+"\n");
		}
	}

	for(int i=0;i<move_dirs.size();i++) {

		String to = p_to_path.plus_file(move_dirs[i].get_file());
		Error err = da->rename(move_dirs[i],to);
		print_line("moving dir "+move_dirs[i]+" to "+to);
		if (err!=OK) {
			EditorNode::get_singleton()->add_io_error("Error moving dir:\n"+move_dirs[i]+"\n");
		}
	}

	memdelete(da);
	//rescan everything
	print_line("call rescan!");
	_rescan();

}

void ScenesDock::_file_option(int p_option) {

	switch(p_option) {

		case FILE_DEPENDENCIES: {

			int idx = files->get_current();
			if (idx<0 || idx>=files->get_item_count())
				break;
			String path = files->get_item_metadata(idx);
			deps_editor->edit(path);
		} break;
		case FILE_OWNERS: {

			int idx = files->get_current();
			if (idx<0 || idx>=files->get_item_count())
				break;
			String path = files->get_item_metadata(idx);
			owners_editor->show(path);
		} break;
		case FILE_MOVE: {

			move_dirs.clear();;
			move_files.clear();

			for(int i=0;i<files->get_item_count();i++) {

				String path = files->get_item_metadata(i);
				if (!files->is_selected(i))
					continue;

				 if (files->get_item_text(i)=="..") {
					 EditorNode::get_singleton()->show_warning("Can't operate on '..'");
					 return;
				 }

				if (path.ends_with("/")) {
					move_dirs.push_back(path.substr(0,path.length()-1));
				} else {
					move_files.push_back(path);
				}
			}


			if (move_dirs.empty() && move_files.size()==1) {

				rename_dialog->clear_filters();
				rename_dialog->add_filter("*."+move_files[0].extension());
				rename_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
				rename_dialog->set_current_path(move_files[0]);
				rename_dialog->popup_centered_ratio();
				rename_dialog->set_title("Pick New Name and Location For: "+move_files[0].get_file());


			} else {
				//just move
				move_dialog->popup_centered_ratio();
			}


		} break;
		case FILE_REMOVE: {

			Vector<String> torem;

			for(int i=0;i<files->get_item_count();i++) {

				String path = files->get_item_metadata(i);
				if (path.ends_with("/") || !files->is_selected(i))
					continue;
				torem.push_back(path);
			}

			if (torem.empty()) {
				EditorNode::get_singleton()->show_warning("No files selected!");
				break;
			}

			remove_dialog->show(torem);
			//1) find if used
			//2) warn

		} break;
		case FILE_INFO: {

		} break;

	}
}

void ScenesDock::_open_pressed(){


	if (tree_mode) {

		TreeItem *sel = tree->get_selected();
		if (!sel) {
			return;
		}
		path = sel->get_metadata(0);
		/*if (path!="res://" && path.ends_with("/")) {
			path=path.substr(0,path.length()-1);
		}*/

		tree_mode=false;

		tree->hide();
		files->show();
		path_hb->show();
		button_favorite->hide();
		button_fav_up->hide();
		button_fav_down->hide();
		button_instance->show();
		button_open->show();
		file_options->show();

		_update_files(false);

		current_path->set_text(path);

		_push_to_history();


	} else {

		int idx=-1;
		for(int i=0;i<files->get_item_count();i++) {
			if (files->is_selected(i)) {
				idx=i;
				break;
			}
		}

		if (idx<0)
			return;



		String path = files->get_item_metadata(idx);

		if (path.ends_with("/")) {
			if (path!="res://") {
				path=path.substr(0,path.length()-1);
			}
			this->path=path;
			_update_files(false);
			current_path->set_text(path);
			_push_to_history();
		} else {

			if (ResourceLoader::get_resource_type(path)=="PackedScene") {

				editor->open_request(path);
			} else {

				editor->load_resource(path);
			}
		}
	}

//	emit_signal("open",path);

}


void ScenesDock::_rescan() {

	_set_scannig_mode();
	EditorFileSystem::get_singleton()->scan();

}

void ScenesDock::fix_dependencies(const String& p_for_file) {
	deps_editor->edit(p_for_file);
}

void ScenesDock::open(const String& p_path) {


	String npath;
	String nfile;

	if (p_path.ends_with("/")) {

		if (p_path!="res://")
			npath=p_path.substr(0,p_path.length()-1);
		else
			npath="res://";
	} else {
		nfile=p_path.get_file();
		npath=p_path.get_base_dir();
	}

	path=npath;

	if (tree_mode && nfile=="") {
		_update_tree();
		tree->grab_focus();
		tree->call_deferred("ensure_cursor_is_visible");
		_push_to_history();
		return;
	} else if (tree_mode){
		_update_tree();
		tree->grab_focus();
		tree->ensure_cursor_is_visible();
		_open_pressed();
		current_path->set_text(path);
	} else {
		_update_files(false);
		_push_to_history();
	}

	for(int i=0;i<files->get_item_count();i++) {

		String md = files->get_item_metadata(i);
		if (md==p_path) {
			files->select(i,true);
			files->ensure_current_is_visible();
			break;
		}
	}

}

void ScenesDock::set_use_thumbnails(bool p_use) {

	display_mode->set_pressed(!p_use);
}

void ScenesDock::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_update_tree"),&ScenesDock::_update_tree);
	ObjectTypeDB::bind_method(_MD("_rescan"),&ScenesDock::_rescan);
	ObjectTypeDB::bind_method(_MD("_favorites_pressed"),&ScenesDock::_favorites_pressed);
	ObjectTypeDB::bind_method(_MD("_instance_pressed"),&ScenesDock::_instance_pressed);
	ObjectTypeDB::bind_method(_MD("_open_pressed"),&ScenesDock::_open_pressed);

	ObjectTypeDB::bind_method(_MD("_thumbnail_done"),&ScenesDock::_thumbnail_done);
	ObjectTypeDB::bind_method(_MD("_select_file"), &ScenesDock::_select_file);
	ObjectTypeDB::bind_method(_MD("_go_to_tree"), &ScenesDock::_go_to_tree);
	ObjectTypeDB::bind_method(_MD("_go_to_dir"), &ScenesDock::_go_to_dir);
	ObjectTypeDB::bind_method(_MD("_change_file_display"), &ScenesDock::_change_file_display);
	ObjectTypeDB::bind_method(_MD("_fw_history"), &ScenesDock::_fw_history);
	ObjectTypeDB::bind_method(_MD("_bw_history"), &ScenesDock::_bw_history);
	ObjectTypeDB::bind_method(_MD("_fs_changed"), &ScenesDock::_fs_changed);
	ObjectTypeDB::bind_method(_MD("_dir_selected"), &ScenesDock::_dir_selected);
	ObjectTypeDB::bind_method(_MD("_fav_up_pressed"), &ScenesDock::_fav_up_pressed);
	ObjectTypeDB::bind_method(_MD("_fav_down_pressed"), &ScenesDock::_fav_down_pressed);
	ObjectTypeDB::bind_method(_MD("_file_option"), &ScenesDock::_file_option);
	ObjectTypeDB::bind_method(_MD("_move_operation"), &ScenesDock::_move_operation);
	ObjectTypeDB::bind_method(_MD("_rename_operation"), &ScenesDock::_rename_operation);

	ADD_SIGNAL(MethodInfo("instance"));
	ADD_SIGNAL(MethodInfo("open"));

}

ScenesDock::ScenesDock(EditorNode *p_editor) {

	editor=p_editor;

	HBoxContainer *toolbar_hbc = memnew( HBoxContainer );
	add_child(toolbar_hbc);

	button_hist_prev = memnew( ToolButton );
	toolbar_hbc->add_child(button_hist_prev);
	button_hist_prev->set_disabled(true);
	button_hist_prev->set_tooltip("Previous Directory");

	button_hist_next = memnew( ToolButton );
	toolbar_hbc->add_child(button_hist_next);
	button_hist_next->set_disabled(true);
	button_hist_prev->set_focus_mode(FOCUS_NONE);
	button_hist_next->set_focus_mode(FOCUS_NONE);
	button_hist_next->set_tooltip("Next Directory");

	button_reload = memnew( Button );
	button_reload->set_flat(true);
	button_reload->connect("pressed",this,"_rescan");	
	toolbar_hbc->add_child(button_reload);
	button_reload->set_focus_mode(FOCUS_NONE);
	button_reload->set_tooltip("Re-Scan Filesystem");

	toolbar_hbc->add_spacer();

	button_fav_up = memnew( ToolButton );
	button_fav_up->set_flat(true);
	toolbar_hbc->add_child(button_fav_up);
	button_fav_up->set_disabled(true);
	button_fav_up->connect("pressed",this,"_fav_up_pressed");
	button_fav_up->set_tooltip("Move Favorite Up");

	button_fav_down = memnew( ToolButton );
	button_fav_down->set_flat(true);
	toolbar_hbc->add_child(button_fav_down);
	button_fav_down->set_disabled(true);
	button_fav_down->connect("pressed",this,"_fav_down_pressed");
	button_fav_down->set_tooltip("Move Favorite Down");

	button_favorite = memnew( Button );
	button_favorite->set_flat(true);
	button_favorite->set_toggle_mode(true);
	button_favorite->connect("pressed",this,"_favorites_pressed");
	toolbar_hbc->add_child(button_favorite);
	button_favorite->set_tooltip("Toggle folder status as Favorite");

	button_favorite->set_focus_mode(FOCUS_NONE);
	button_fav_up->set_focus_mode(FOCUS_NONE);
	button_fav_down->set_focus_mode(FOCUS_NONE);


	button_open = memnew( Button );
	button_open->set_flat(true);
	button_open->connect("pressed",this,"_open_pressed");
	toolbar_hbc->add_child(button_open);
	button_open->hide();
	button_open->set_focus_mode(FOCUS_NONE);
	button_open->set_tooltip("Open the selected file.\nOpen as scene if a scene, or as resource otherwise.");


	button_instance = memnew( Button );
	button_instance->set_flat(true);
	button_instance->connect("pressed",this,"_instance_pressed");
	toolbar_hbc->add_child(button_instance);
	button_instance->hide();
	button_instance->set_focus_mode(FOCUS_NONE);
	button_instance->set_tooltip("Instance the selected scene(s) as child of the selected node.");


	file_options = memnew( MenuButton );
	toolbar_hbc->add_child(file_options);
	file_options->get_popup()->add_item("Rename or Move",FILE_MOVE);
	file_options->get_popup()->add_item("Delete",FILE_REMOVE);
	file_options->get_popup()->add_separator();
	file_options->get_popup()->add_item("Edit Dependencies",FILE_DEPENDENCIES);
	file_options->get_popup()->add_item("View Owners",FILE_OWNERS);
	//file_options->get_popup()->add_item("Info",FILE_INFO);
	file_options->hide();
	file_options->set_focus_mode(FOCUS_NONE);
	file_options->set_tooltip("Miscenaneous options related to resources on disk.");

	tree = memnew( Tree );

	tree->set_hide_root(true);
	add_child(tree);


	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->connect("item_edited",this,"_favorite_toggled");
	tree->connect("item_activated",this,"_open_pressed");
	tree->connect("cell_selected",this,"_dir_selected");

	files = memnew( ItemList );
	files->set_v_size_flags(SIZE_EXPAND_FILL);
	files->set_select_mode(ItemList::SELECT_MULTI);

	path_hb = memnew( HBoxContainer );
	button_back  = memnew( ToolButton );
	path_hb->add_child(button_back);
	current_path=memnew( LineEdit );
	current_path->set_h_size_flags(SIZE_EXPAND_FILL);
	path_hb->add_child(current_path);
	display_mode = memnew( ToolButton );
	path_hb->add_child(display_mode);
	display_mode->set_toggle_mode(true);
	add_child(path_hb);
	path_hb->hide();


	add_child(files);
	files->hide();

	scanning_vb = memnew( VBoxContainer );
	Label *slabel = memnew( Label );
	slabel->set_text("Scanning Files,\nPlease Wait..");
	slabel->set_align(Label::ALIGN_CENTER);
	scanning_vb->add_child(slabel);
	scanning_progress = memnew( ProgressBar );
	scanning_vb->add_child(scanning_progress);
	add_child(scanning_vb);
	scanning_vb->hide();



	deps_editor = memnew( DependencyEditor );
	add_child(deps_editor);

	owners_editor = memnew( DependencyEditorOwners);
	add_child(owners_editor);

	remove_dialog = memnew( DependencyRemoveDialog);
	add_child(remove_dialog);

	move_dialog = memnew( EditorDirDialog );
	add_child(move_dialog);	
	move_dialog->connect("dir_selected",this,"_move_operation");
	move_dialog->get_ok()->set_text("Move");
	
	rename_dialog = memnew( EditorFileDialog );
	rename_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
	rename_dialog->connect("file_selected",this,"_rename_operation");
	add_child(rename_dialog);

	updating_tree=false;
	initialized=false;

	history.push_back("res://");
	history_pos=0;
	tree_mode=true;


}

ScenesDock::~ScenesDock() {

}

