/*************************************************************************/
/*  scenes_dock.cpp                                                      */
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
#include "filesystem_dock.h"
#include "globals.h"
#include "os/dir_access.h"
#include "os/file_access.h"

#include "editor_node.h"
#include "io/resource_loader.h"
#include "os/os.h"

#include "editor_settings.h"
#include "scene/main/viewport.h"

bool FileSystemDock::_create_tree(TreeItem *p_parent, EditorFileSystemDirectory *p_dir) {

	TreeItem *item = tree->create_item(p_parent);
	String dname = p_dir->get_name();
	if (dname == "")
		dname = "res://";

	item->set_text(0, dname);
	item->set_icon(0, get_icon("Folder", "EditorIcons"));
	item->set_selectable(0, true);
	String lpath = p_dir->get_path();
	if (lpath != "res://" && lpath.ends_with("/")) {
		lpath = lpath.substr(0, lpath.length() - 1);
	}
	item->set_metadata(0, lpath);
	if (lpath == path) {
		item->select(0);
	}

	for (int i = 0; i < p_dir->get_subdir_count(); i++)
		_create_tree(item, p_dir->get_subdir(i));

	return true;
}

void FileSystemDock::_update_tree() {

	tree->clear();
	updating_tree = true;
	TreeItem *root = tree->create_item();
	TreeItem *favorites = tree->create_item(root);
	favorites->set_icon(0, get_icon("Favorites", "EditorIcons"));
	favorites->set_text(0, TTR("Favorites:"));
	favorites->set_selectable(0, false);
	Vector<String> faves = EditorSettings::get_singleton()->get_favorite_dirs();
	for (int i = 0; i < faves.size(); i++) {
		if (!faves[i].begins_with("res://"))
			continue;

		TreeItem *ti = tree->create_item(favorites);
		String fv = faves[i];
		if (fv == "res://")
			ti->set_text(0, "/");
		else
			ti->set_text(0, faves[i].get_file());
		ti->set_icon(0, get_icon("Folder", "EditorIcons"));
		ti->set_selectable(0, true);
		ti->set_metadata(0, faves[i]);
	}

	_create_tree(root, EditorFileSystem::get_singleton()->get_filesystem());
	updating_tree = false;
}

void FileSystemDock::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_RESIZED: {

			bool new_mode = get_size().height < get_viewport_rect().size.height / 2;

			if (new_mode != split_mode) {

				split_mode = new_mode;

				//print_line("SPLIT MODE? "+itos(split_mode));
				if (split_mode) {

					file_list_vb->hide();
					tree->set_custom_minimum_size(Size2(0, 0));
					tree->set_v_size_flags(SIZE_EXPAND_FILL);
					button_back->show();
				} else {

					tree->show();
					file_list_vb->show();
					tree->set_custom_minimum_size(Size2(0, 200) * EDSCALE);
					tree->set_v_size_flags(SIZE_FILL);
					button_back->hide();
					if (!EditorFileSystem::get_singleton()->is_scanning()) {
						_fs_changed();
					}
				}
			}

		} break;
		case NOTIFICATION_ENTER_TREE: {

			if (initialized)
				return;
			initialized = true;

			EditorFileSystem::get_singleton()->connect("filesystem_changed", this, "_fs_changed");
			EditorResourcePreview::get_singleton()->connect("preview_invalidated", this, "_preview_invalidated");

			button_reload->set_icon(get_icon("Reload", "EditorIcons"));
			button_favorite->set_icon(get_icon("Favorites", "EditorIcons"));
			//button_instance->set_icon( get_icon("Add","EditorIcons"));
			//button_open->set_icon( get_icon("Folder","EditorIcons"));
			button_back->set_icon(get_icon("Filesystem", "EditorIcons"));
			if (display_mode == DISPLAY_THUMBNAILS) {
				button_display_mode->set_icon(get_icon("FileList", "EditorIcons"));
			} else {
				button_display_mode->set_icon(get_icon("FileThumbnail", "EditorIcons"));
			}
			button_display_mode->connect("pressed", this, "_change_file_display");
			//file_options->set_icon( get_icon("Tools","EditorIcons"));
			files->connect("item_activated", this, "_select_file");
			button_hist_next->connect("pressed", this, "_fw_history");
			button_hist_prev->connect("pressed", this, "_bw_history");
			search_icon->set_texture(get_icon("Zoom", "EditorIcons"));

			button_hist_next->set_icon(get_icon("Forward", "EditorIcons"));
			button_hist_prev->set_icon(get_icon("Back", "EditorIcons"));
			file_options->connect("item_pressed", this, "_file_option");
			folder_options->connect("item_pressed", this, "_folder_option");

			button_back->connect("pressed", this, "_go_to_tree", varray(), CONNECT_DEFERRED);
			current_path->connect("text_entered", this, "_go_to_dir");
			_update_tree(); //maybe it finished already

			if (EditorFileSystem::get_singleton()->is_scanning()) {
				_set_scanning_mode();
			}

		} break;
		case NOTIFICATION_PROCESS: {
			if (EditorFileSystem::get_singleton()->is_scanning()) {
				scanning_progress->set_val(EditorFileSystem::get_singleton()->get_scanning_progress() * 100);
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {

		} break;
		case NOTIFICATION_DRAG_BEGIN: {

			Dictionary dd = get_viewport()->gui_get_drag_data();
			if (tree->is_visible() && dd.has("type")) {
				if ((String(dd["type"]) == "files") || (String(dd["type"]) == "files_and_dirs") || (String(dd["type"]) == "resource")) {
					tree->set_drop_mode_flags(Tree::DROP_MODE_ON_ITEM);
				}
				if ((String(dd["type"]) == "favorite")) {
					tree->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);
				}
			}

		} break;
		case NOTIFICATION_DRAG_END: {

			tree->set_drop_mode_flags(0);

		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {

			int new_mode = int(EditorSettings::get_singleton()->get("filesystem_dock/display_mode"));

			if (new_mode != display_mode) {
				set_display_mode(new_mode);
			} else {
				_update_files(true);
			}
		} break;
	}
}

void FileSystemDock::_dir_selected() {

	TreeItem *ti = tree->get_selected();
	if (!ti)
		return;
	String dir = ti->get_metadata(0);
	bool found = false;
	Vector<String> favorites = EditorSettings::get_singleton()->get_favorite_dirs();
	for (int i = 0; i < favorites.size(); i++) {

		if (favorites[i] == dir) {
			found = true;
			break;
		}
	}

	button_favorite->set_pressed(found);

	if (!split_mode) {
		_open_pressed(); //go directly to dir
	}
}

void FileSystemDock::_dir_rmb_pressed(const Vector2 &p_pos) {
	folder_options->clear();
	folder_options->set_size(Size2(1, 1));

	folder_options->add_item(TTR("Expand all"), FOLDER_EXPAND_ALL);
	folder_options->add_item(TTR("Collapse all"), FOLDER_COLLAPSE_ALL);

	TreeItem *item = tree->get_selected();
	if (item) {
		String fpath = item->get_metadata(tree->get_selected_column());
		folder_options->add_separator();
		folder_options->add_item(TTR("Copy Path"), FOLDER_COPY_PATH);
		if (fpath != "res://") {
			folder_options->add_item(TTR("Rename.."), FOLDER_RENAME);
			folder_options->add_item(TTR("Move To.."), FOLDER_MOVE);
			folder_options->add_item(TTR("Delete"), FOLDER_REMOVE);
		}
		folder_options->add_separator();
		folder_options->add_item(TTR("New Folder.."), FOLDER_NEW_FOLDER);
		folder_options->add_item(TTR("Show In File Manager"), FOLDER_SHOW_IN_EXPLORER);
	}
	folder_options->set_pos(tree->get_global_pos() + p_pos);
	folder_options->popup();
}

void FileSystemDock::_make_dir_confirm() {
	String dir_name = make_dir_dialog_text->get_text().strip_edges();

	if (dir_name.length() == 0) {
		EditorNode::get_singleton()->show_warning(TTR("No name provided"));
		return;
	} else if (dir_name.find("/") != -1 || dir_name.find("\\") != -1 || dir_name.find(":") != -1) {
		EditorNode::get_singleton()->show_warning(TTR("Provided name contains invalid characters"));
		return;
	}

	print_line("Making folder " + dir_name + " in " + path);
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	Error err = da->change_dir(path);
	if (err == OK) {
		err = da->make_dir(dir_name);
	}
	memdelete(da);

	if (err == OK) {
		print_line("call rescan!");
		_rescan();
	} else {
		EditorNode::get_singleton()->show_warning(TTR("Could not create folder."));
	}
}

void FileSystemDock::_rename_operation_confirm() {

	String new_name = rename_dialog_text->get_text().strip_edges();
	if (new_name.length() == 0) {
		EditorNode::get_singleton()->show_warning(TTR("No name provided."));
		return;
	} else if (new_name.find("/") != -1 || new_name.find("\\") != -1 || new_name.find(":") != -1) {
		EditorNode::get_singleton()->show_warning(TTR("Name contains invalid characters."));
		return;
	}

	String old_path = to_rename.path.ends_with("/") ? to_rename.path.substr(0, to_rename.path.length() - 1) : to_rename.path;
	String new_path = old_path.get_base_dir().plus_file(new_name);
	if (old_path == new_path) {
		return;
	}

	//Present a more user friendly warning for name conflict
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (da->file_exists(new_path) || da->dir_exists(new_path)) {
		EditorNode::get_singleton()->show_warning(TTR("A file or folder with this name already exists."));
		memdelete(da);
		return;
	}
	memdelete(da);

	Map<String, String> renames;
	_try_move_item(to_rename, new_path, renames);
	_update_dependencies_after_move(renames);

	//Rescan everything
	print_line("call rescan!");
	_rescan();
}

void FileSystemDock::_move_operation_confirm(const String &p_to_path) {

	Map<String, String> renames;
	for (int i = 0; i < to_move.size(); i++) {
		String old_path = to_move[i].path.ends_with("/") ? to_move[i].path.substr(0, to_move[i].path.length() - 1) : to_move[i].path;
		String new_path = p_to_path.plus_file(old_path.get_file());
		_try_move_item(to_move[i], new_path, renames);
	}

	_update_dependencies_after_move(renames);
	print_line("call rescan!");
	_rescan();
}

void FileSystemDock::_try_move_item(const FileOrFolder &p_item, const String &p_new_path, Map<String, String> &p_renames) {
	//Ensure folder paths end with "/"
	String old_path = (p_item.is_file || p_item.path.ends_with("/")) ? p_item.path : (p_item.path + "/");
	String new_path = (p_item.is_file || p_new_path.ends_with("/")) ? p_new_path : (p_new_path + "/");

	if (new_path == old_path) {
		return;
	} else if (old_path == "res://") {
		EditorNode::get_singleton()->add_io_error(TTR("Cannot move/rename resources root."));
		return;
	} else if (!p_item.is_file && new_path.begins_with(old_path)) {
		//This check doesn't erroneously catch renaming to a longer name as folder paths always end with "/"
		EditorNode::get_singleton()->add_io_error(TTR("Cannot move a folder into itself.\n") + old_path + "\n");
		return;
	}

	//Build a list of files which will have new paths as a result of this operation
	Vector<String> changed_paths;
	if (p_item.is_file) {
		changed_paths.push_back(old_path);
	} else {
		_get_all_files_in_dir(EditorFileSystem::get_singleton()->get_path(old_path), changed_paths);
	}

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	print_line("Moving " + old_path + " -> " + new_path);
	Error err = da->rename(old_path, new_path);
	if (err == OK) {
		//Only treat as a changed dependency if it was successfully moved
		for (int i = 0; i < changed_paths.size(); ++i) {
			p_renames[changed_paths[i]] = changed_paths[i].replace_first(old_path, new_path);
			print_line("  Remap: " + changed_paths[i] + " -> " + p_renames[changed_paths[i]]);
		}
	} else {
		EditorNode::get_singleton()->add_io_error(TTR("Error moving:\n") + old_path + "\n");
	}
	memdelete(da);
}

void FileSystemDock::_get_all_files_in_dir(EditorFileSystemDirectory *efsd, Vector<String> &files) {
	if (efsd == NULL)
		return;

	for (int i = 0; i < efsd->get_subdir_count(); i++) {
		_get_all_files_in_dir(efsd->get_subdir(i), files);
	}
	for (int i = 0; i < efsd->get_file_count(); i++) {
		files.push_back(efsd->get_file_path(i));
	}
}

void FileSystemDock::_update_dependencies_after_move(Map<String, String> &p_renames) {
	//The following code assumes that the following holds:
	// 1) EditorFileSystem contains the old paths/folder structure from before the rename/move.
	// 2) ResourceLoader can use the new paths without needing to call rescan.
	List<String> remaps;
	_find_remaps(EditorFileSystem::get_singleton()->get_filesystem(), p_renames, remaps);
	for (int i = 0; i < remaps.size(); ++i) {
		//Because we haven't called a rescan yet the found remap might still be an old path itself.
		String file = p_renames.has(remaps[i]) ? p_renames[remaps[i]] : remaps[i];
		print_line("Remapping dependencies for: " + file);
		Error err = ResourceLoader::rename_dependencies(file, p_renames);
		if (err != OK) {
			EditorNode::get_singleton()->add_io_error(TTR("Unable to update dependencies:\n") + remaps[i] + "\n");
		}
	}
}

void FileSystemDock::_favorites_pressed() {

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return;
	String dir = sel->get_metadata(0);

	int idx = -1;
	Vector<String> favorites = EditorSettings::get_singleton()->get_favorite_dirs();
	for (int i = 0; i < favorites.size(); i++) {

		if (favorites[i] == dir) {
			idx = i;
			break;
		}
	}

	if (button_favorite->is_pressed() && idx == -1) {
		favorites.push_back(dir);
		EditorSettings::get_singleton()->set_favorite_dirs(favorites);
		_update_tree();
	} else if (!button_favorite->is_pressed() && idx != -1) {
		favorites.remove(idx);
		EditorSettings::get_singleton()->set_favorite_dirs(favorites);
		_update_tree();
	}
}

String FileSystemDock::get_selected_path() const {

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return "";
	String path = sel->get_metadata(0);
	return "res://" + path;
}

void FileSystemDock::navigate_to_path(const String &p_path) {
	// If the path is a file, do not only go to the directory in the tree, also select the file in the file list.
	String file_name = "";
	DirAccess *dirAccess = DirAccess::open("res://");
	if (dirAccess->file_exists(p_path)) {
		path = p_path.get_base_dir();
		file_name = p_path.get_file();
	} else if (dirAccess->dir_exists(p_path)) {
		path = p_path;
	} else {
		ERR_EXPLAIN(vformat(TTR("Cannot navigate to '%s' as it has not been found in the file system!"), p_path));
		ERR_FAIL();
	}

	current_path->set_text(path);
	_push_to_history();

	_update_tree();
	_update_files(false);

	if (!file_name.empty()) {
		for (int i = 0; i < files->get_item_count(); i++) {
			if (files->get_item_text(i) == file_name) {
				files->select(i, true);
				files->ensure_current_is_visible();
				break;
			}
		}
	}
}

String FileSystemDock::get_current_path() const {

	return path;
}

void FileSystemDock::_thumbnail_done(const String &p_path, const Ref<Texture> &p_preview, const Variant &p_udata) {

	bool valid = false;

	if (!search_box->is_hidden()) {
		valid = true;
	} else {
		valid = (path == p_path.get_base_dir());
	}

	if (p_preview.is_valid() && valid) {

		Array uarr = p_udata;
		int idx = uarr[0];
		String file = uarr[1];
		if (idx >= files->get_item_count())
			return;
		if (files->get_item_text(idx) != file)
			return;
		String fpath = files->get_item_metadata(idx);
		if (fpath != p_path)
			return;
		files->set_item_icon(idx, p_preview);
	}
}

void FileSystemDock::_change_file_display() {

	if (button_display_mode->is_pressed()) {
		display_mode = DISPLAY_LIST;
		button_display_mode->set_icon(get_icon("FileThumbnail", "EditorIcons"));
	} else {
		display_mode = DISPLAY_THUMBNAILS;
		button_display_mode->set_icon(get_icon("FileList", "EditorIcons"));
	}

	EditorSettings::get_singleton()->set("filesystem_dock/display_mode", display_mode);

	_update_files(true);
}

void FileSystemDock::_search(EditorFileSystemDirectory *p_path, List<FileInfo> *matches, int p_max_items) {

	if (matches->size() > p_max_items)
		return;

	for (int i = 0; i < p_path->get_subdir_count(); i++) {
		_search(p_path->get_subdir(i), matches, p_max_items);
	}

	String match = search_box->get_text();

	for (int i = 0; i < p_path->get_file_count(); i++) {
		String file = p_path->get_file(i);

		if (file.find(match) != -1) {

			FileInfo fi;
			fi.name = file;
			fi.type = p_path->get_file_type(i);
			fi.path = p_path->get_file_path(i);
			if (p_path->get_file_meta(i)) {
				if (p_path->is_missing_sources(i)) {
					fi.import_status = 3;
				} else if (p_path->have_sources_changed(i)) {
					fi.import_status = 2;
				} else {
					fi.import_status = 1;
				}
			} else {
				fi.import_status = 0;
			}
			for (int j = 0; j < p_path->get_source_count(i); j++) {
				String s = EditorImportPlugin::expand_source_path(p_path->get_source_file(i, j));
				if (p_path->is_source_file_missing(i, j)) {
					s += " (Missing)";
				}
				fi.sources.push_back(s);
			}

			matches->push_back(fi);
			if (matches->size() > p_max_items)
				return;
		}
	}
}

void FileSystemDock::_update_files(bool p_keep_selection) {

	Set<String> cselection;

	if (p_keep_selection) {

		for (int i = 0; i < files->get_item_count(); i++) {

			if (files->is_selected(i))
				cselection.insert(files->get_item_text(i));
		}
	}

	files->clear();

	current_path->set_text(path);

	EditorFileSystemDirectory *efd = EditorFileSystem::get_singleton()->get_path(path);
	if (!efd)
		return;

	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem_dock/thumbnail_size");
	thumbnail_size *= EDSCALE;
	Ref<Texture> folder_thumbnail;
	Ref<Texture> file_thumbnail;

	bool use_thumbnails = (display_mode == DISPLAY_THUMBNAILS);
	bool use_folders = search_box->get_text().length() == 0 && split_mode;

	if (use_thumbnails) { //thumbnails

		files->set_max_columns(0);
		files->set_icon_mode(ItemList::ICON_MODE_TOP);
		files->set_fixed_column_width(thumbnail_size * 3 / 2);
		files->set_max_text_lines(2);
		files->set_fixed_icon_size(Size2(thumbnail_size, thumbnail_size));

		if (!has_icon("ResizedFolder", "EditorIcons")) {
			Ref<ImageTexture> folder = get_icon("FolderBig", "EditorIcons");
			Image img = folder->get_data();
			img.resize(thumbnail_size, thumbnail_size);
			Ref<ImageTexture> resized_folder = Ref<ImageTexture>(memnew(ImageTexture));
			resized_folder->create_from_image(img, 0);
			Theme::get_default()->set_icon("ResizedFolder", "EditorIcons", resized_folder);
		}

		folder_thumbnail = get_icon("ResizedFolder", "EditorIcons");

		if (!has_icon("ResizedFile", "EditorIcons")) {
			Ref<ImageTexture> file = get_icon("FileBig", "EditorIcons");
			Image img = file->get_data();
			img.resize(thumbnail_size, thumbnail_size);
			Ref<ImageTexture> resized_file = Ref<ImageTexture>(memnew(ImageTexture));
			resized_file->create_from_image(img, 0);
			Theme::get_default()->set_icon("ResizedFile", "EditorIcons", resized_file);
		}

		file_thumbnail = get_icon("ResizedFile", "EditorIcons");

	} else {

		files->set_icon_mode(ItemList::ICON_MODE_LEFT);
		files->set_max_columns(1);
		files->set_max_text_lines(1);
		files->set_fixed_column_width(0);
		files->set_fixed_icon_size(Size2());
	}

	if (use_folders) {

		if (path != "res://") {

			if (use_thumbnails) {
				files->add_item("..", folder_thumbnail, true);
			} else {
				files->add_item("..", get_icon("folder", "FileDialog"), true);
			}

			String bd = path.get_base_dir();
			if (bd != "res://" && !bd.ends_with("/"))
				bd += "/";

			files->set_item_metadata(files->get_item_count() - 1, bd);
		}

		for (int i = 0; i < efd->get_subdir_count(); i++) {

			String dname = efd->get_subdir(i)->get_name();

			if (use_thumbnails) {
				files->add_item(dname, folder_thumbnail, true);
			} else {
				files->add_item(dname, get_icon("folder", "FileDialog"), true);
			}

			files->set_item_metadata(files->get_item_count() - 1, path.plus_file(dname) + "/");

			if (cselection.has(dname))
				files->select(files->get_item_count() - 1, false);
		}
	}

	List<FileInfo> filelist;

	if (search_box->get_text().length()) {

		if (search_box->get_text().length() > 1) {
			_search(EditorFileSystem::get_singleton()->get_filesystem(), &filelist, 128);
		}

		filelist.sort();
	} else {

		for (int i = 0; i < efd->get_file_count(); i++) {

			FileInfo fi;
			fi.name = efd->get_file(i);
			fi.path = path.plus_file(fi.name);
			fi.type = efd->get_file_type(i);
			if (efd->get_file_meta(i)) {
				if (efd->is_missing_sources(i)) {
					fi.import_status = 3;
				} else if (efd->have_sources_changed(i)) {
					fi.import_status = 2;
				} else {
					fi.import_status = 1;
				}

				for (int j = 0; j < efd->get_source_count(i); j++) {
					String s = EditorImportPlugin::expand_source_path(efd->get_source_file(i, j));
					if (efd->is_source_file_missing(i, j)) {
						s += " (Missing)";
					}
					fi.sources.push_back(s);
				}
			} else {
				fi.import_status = 0;
			}

			filelist.push_back(fi);
		}
	}

	StringName ei = "EditorIcons"; //make it faster..
	StringName oi = "Object";

	for (List<FileInfo>::Element *E = filelist.front(); E; E = E->next()) {
		String fname = E->get().name;
		String fp = E->get().path;
		StringName type = E->get().type;

		Ref<Texture> type_icon;

		String tooltip = fname;

		if (E->get().import_status == 0) {

			if (has_icon(type, ei)) {
				type_icon = get_icon(type, ei);
			} else {
				type_icon = get_icon(oi, ei);
			}
		} else if (E->get().import_status == 1) {
			type_icon = get_icon("DependencyOk", "EditorIcons");
		} else if (E->get().import_status == 2) {
			type_icon = get_icon("DependencyChanged", "EditorIcons");
			tooltip + "\nStatus: Needs Re-Import";
		} else if (E->get().import_status == 3) {
			type_icon = get_icon("ImportFail", "EditorIcons");
			tooltip + "\nStatus: Missing Dependencies";
		}

		if (E->get().sources.size()) {
			for (int i = 0; i < E->get().sources.size(); i++) {
				tooltip += "\nSource: " + E->get().sources[i];
			}
		}

		if (use_thumbnails) {
			files->add_item(fname, file_thumbnail, true);
			files->set_item_metadata(files->get_item_count() - 1, fp);
			files->set_item_tag_icon(files->get_item_count() - 1, type_icon);
			Array udata;
			udata.resize(2);
			udata[0] = files->get_item_count() - 1;
			udata[1] = fname;
			EditorResourcePreview::get_singleton()->queue_resource_preview(fp, this, "_thumbnail_done", udata);
		} else {
			files->add_item(fname, type_icon, true);
			files->set_item_metadata(files->get_item_count() - 1, fp);
		}

		if (cselection.has(fname))
			files->select(files->get_item_count() - 1, false);

		files->set_item_tooltip(files->get_item_count() - 1, tooltip);
	}
}

void FileSystemDock::_select_file(int p_idx) {

	files->select(p_idx, true);
	_file_option(FILE_OPEN);
}

void FileSystemDock::_go_to_tree() {

	tree->show();
	file_list_vb->hide();
	_update_tree();
	tree->grab_focus();
	tree->ensure_cursor_is_visible();
	button_favorite->show();
	//button_open->hide();
	//file_options->hide();
}

void FileSystemDock::_go_to_dir(const String &p_dir) {

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (da->change_dir(p_dir) == OK) {
		path = da->get_current_dir();
		_update_files(false);
	}
	current_path->set_text(path);
	memdelete(da);
}

void FileSystemDock::_preview_invalidated(const String &p_path) {

	if (display_mode == DISPLAY_THUMBNAILS && p_path.get_base_dir() == path && search_box->get_text() == String() && file_list_vb->is_visible()) {

		for (int i = 0; i < files->get_item_count(); i++) {

			if (files->get_item_metadata(i) == p_path) {
				//re-request preview
				Array udata;
				udata.resize(2);
				udata[0] = i;
				udata[1] = files->get_item_text(i);
				EditorResourcePreview::get_singleton()->queue_resource_preview(p_path, this, "_thumbnail_done", udata);
				break;
			}
		}
	}
}

void FileSystemDock::_fs_changed() {

	button_hist_prev->set_disabled(history_pos == 0);
	button_hist_next->set_disabled(history_pos + 1 == history.size());
	scanning_vb->hide();
	split_box->show();

	if (!tree->is_hidden()) {
		button_favorite->show();
		_update_tree();
	}

	if (!file_list_vb->is_hidden()) {

		_update_files(true);
	}

	set_process(false);
}

void FileSystemDock::_set_scanning_mode() {

	split_box->hide();
	button_hist_prev->set_disabled(true);
	button_hist_next->set_disabled(true);
	scanning_vb->show();
	set_process(true);
	if (EditorFileSystem::get_singleton()->is_scanning()) {
		scanning_progress->set_val(EditorFileSystem::get_singleton()->get_scanning_progress() * 100);
	} else {
		scanning_progress->set_val(0);
	}
}

void FileSystemDock::_fw_history() {

	if (history_pos < history.size() - 1)
		history_pos++;

	path = history[history_pos];

	if (!tree->is_hidden()) {
		_update_tree();
		tree->grab_focus();
		tree->ensure_cursor_is_visible();
	}

	if (!file_list_vb->is_hidden()) {
		_update_files(false);
		current_path->set_text(path);
	}

	button_hist_prev->set_disabled(history_pos == 0);
	button_hist_next->set_disabled(history_pos + 1 == history.size());
}

void FileSystemDock::_bw_history() {

	if (history_pos > 0)
		history_pos--;

	path = history[history_pos];

	if (!tree->is_hidden()) {
		_update_tree();
		tree->grab_focus();
		tree->ensure_cursor_is_visible();
	}

	if (!file_list_vb->is_hidden()) {
		_update_files(false);
		current_path->set_text(path);
	}

	button_hist_prev->set_disabled(history_pos == 0);
	button_hist_next->set_disabled(history_pos + 1 == history.size());
}

void FileSystemDock::_push_to_history() {

	history.resize(history_pos + 1);
	if (history[history_pos] != path) {
		history.push_back(path);
		history_pos++;
	}

	button_hist_prev->set_disabled(history_pos == 0);
	button_hist_next->set_disabled(history_pos + 1 == history.size());
}

void FileSystemDock::_find_inside_move_files(EditorFileSystemDirectory *efsd, Vector<String> &files) {

	for (int i = 0; i < efsd->get_subdir_count(); i++) {
		_find_inside_move_files(efsd->get_subdir(i), files);
	}
	for (int i = 0; i < efsd->get_file_count(); i++) {
		files.push_back(efsd->get_file_path(i));
	}
}

void FileSystemDock::_find_remaps(EditorFileSystemDirectory *efsd, Map<String, String> &renames, List<String> &to_remaps) {

	for (int i = 0; i < efsd->get_subdir_count(); i++) {
		_find_remaps(efsd->get_subdir(i), renames, to_remaps);
	}
	for (int i = 0; i < efsd->get_file_count(); i++) {
		Vector<String> deps = efsd->get_file_deps(i);
		for (int j = 0; j < deps.size(); j++) {
			if (renames.has(deps[j])) {
				to_remaps.push_back(efsd->get_file_path(i));
				break;
			}
		}
	}
}

void FileSystemDock::_rename_operation(const String &p_to_path) {

	if (move_files[0] == p_to_path) {
		EditorNode::get_singleton()->show_warning(TTR("Same source and destination files, doing nothing."));
		return;
	}
	if (FileAccess::exists(p_to_path)) {
		EditorNode::get_singleton()->show_warning("Target file exists, can't overwrite. Delete first.");
		return;
	}

	Map<String, String> renames;
	renames[move_files[0]] = p_to_path;

	List<String> remap;

	_find_remaps(EditorFileSystem::get_singleton()->get_filesystem(), renames, remap);
	print_line("found files to remap: " + itos(remap.size()));

	//perform remaps
	for (List<String>::Element *E = remap.front(); E; E = E->next()) {

		Error err = ResourceLoader::rename_dependencies(E->get(), renames);
		print_line("remapping: " + E->get());

		if (err != OK) {
			EditorNode::get_singleton()->add_io_error("Can't rename deps for:\n" + E->get() + "\n");
		}
	}

	//finally, perform moves

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);

	Error err = da->rename(move_files[0], p_to_path);
	print_line("moving file " + move_files[0] + " to " + p_to_path);
	if (err != OK) {
		EditorNode::get_singleton()->add_io_error("Error moving file:\n" + move_files[0] + "\n");
	}

	//rescan everything
	memdelete(da);
	print_line("call rescan!");
	_rescan();
}

void FileSystemDock::_move_operation(const String &p_to_path) {

	if (p_to_path == path) {
		EditorNode::get_singleton()->show_warning(TTR("Same source and destination paths, doing nothing."));
		return;
	}

	//find files inside dirs to be moved

	Vector<String> inside_files;

	for (int i = 0; i < move_dirs.size(); i++) {
		if (p_to_path.begins_with(move_dirs[i])) {
			EditorNode::get_singleton()->show_warning(TTR("Can't move directories to within themselves."));
			return;
		}

		EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_path(move_dirs[i]);
		if (!efsd)
			continue;
		_find_inside_move_files(efsd, inside_files);
	}

	//make list of remaps
	Map<String, String> renames;
	String repfrom = path == "res://" ? path : String(path + "/");
	String repto = p_to_path == "res://" ? p_to_path : String(p_to_path + "/");

	for (int i = 0; i < move_files.size(); i++) {
		renames[move_files[i]] = move_files[i].replace_first(repfrom, repto);
		print_line("move file " + move_files[i] + " -> " + renames[move_files[i]]);
	}
	for (int i = 0; i < inside_files.size(); i++) {
		renames[inside_files[i]] = inside_files[i].replace_first(repfrom, repto);
		print_line("inside file " + inside_files[i] + " -> " + renames[inside_files[i]]);
	}

	//make list of files that will be run the remapping
	List<String> remap;

	_find_remaps(EditorFileSystem::get_singleton()->get_filesystem(), renames, remap);
	print_line("found files to remap: " + itos(remap.size()));

	//perform remaps
	for (List<String>::Element *E = remap.front(); E; E = E->next()) {

		Error err = ResourceLoader::rename_dependencies(E->get(), renames);
		print_line("remapping: " + E->get());

		if (err != OK) {
			EditorNode::get_singleton()->add_io_error("Can't rename deps for:\n" + E->get() + "\n");
		}
	}

	//finally, perform moves

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);

	for (int i = 0; i < move_files.size(); i++) {

		String to = move_files[i].replace_first(repfrom, repto);
		Error err = da->rename(move_files[i], to);
		print_line("moving file " + move_files[i] + " to " + to);
		if (err != OK) {
			EditorNode::get_singleton()->add_io_error("Error moving file:\n" + move_files[i] + "\n");
		}
	}

	for (int i = 0; i < move_dirs.size(); i++) {

		String mdir = move_dirs[i];
		if (mdir == "res://")
			continue;

		if (mdir.ends_with("/")) {
			mdir = mdir.substr(0, mdir.length() - 1);
		}

		String to = p_to_path.plus_file(mdir.get_file());
		Error err = da->rename(mdir, to);
		print_line("moving dir " + mdir + " to " + to);
		if (err != OK) {
			EditorNode::get_singleton()->add_io_error("Error moving dir:\n" + move_dirs[i] + "\n");
		}
	}

	memdelete(da);
	//rescan everything
	print_line("call rescan!");
	_rescan();
}

void FileSystemDock::_file_option(int p_option) {

	switch (p_option) {

		case FILE_SHOW_IN_EXPLORER:
		case FILE_OPEN: {
			int idx = -1;
			int selectcount = files->get_selected_items().size();
			for (int i = 0; i < files->get_item_count(); i++) {
				if (files->is_selected(i)) {
					String path = files->get_item_metadata(i);
					if (selectcount == 1) {
						if (p_option == FILE_SHOW_IN_EXPLORER) {
							String dir = Globals::get_singleton()->globalize_path(path);
							dir = dir.substr(0, dir.find_last("/"));
							OS::get_singleton()->shell_open(String("file://") + dir);
							return;
						}
						if (path.ends_with("/")) {
							if (path != "res://") {
								path = path.substr(0, path.length() - 1);
							}
							this->path = path;
							_update_files(false);
							current_path->set_text(path);
							_push_to_history();
						} else {

							if (ResourceLoader::get_resource_type(path) == "PackedScene") {

								editor->open_request(path);
							} else {

								editor->load_resource(path);
							}
						}
					} else if (selectcount > 1) {
						if (ResourceLoader::get_resource_type(path) == "PackedScene") {
							editor->open_request(path);
						}
					}
				}
			}
		} break;
		case FILE_INSTANCE: {

			Vector<String> paths;

			for (int i = 0; i < files->get_item_count(); i++) {
				if (!files->is_selected(i))
					continue;
				String path = files->get_item_metadata(i);
				if (EditorFileSystem::get_singleton()->get_file_type(path) == "PackedScene") {
					paths.push_back(path);
				}
			}

			if (!paths.empty()) {
				emit_signal("instance", paths);
			}
		} break;
		case FILE_DEPENDENCIES: {

			int idx = files->get_current();
			if (idx < 0 || idx >= files->get_item_count())
				break;
			String path = files->get_item_metadata(idx);
			deps_editor->edit(path);
		} break;
		case FILE_OWNERS: {

			int idx = files->get_current();
			if (idx < 0 || idx >= files->get_item_count())
				break;
			String path = files->get_item_metadata(idx);
			owners_editor->show(path);
		} break;
		case FILE_MOVE: {

			move_dirs.clear();
			move_files.clear();

			for (int i = 0; i < files->get_item_count(); i++) {

				String path = files->get_item_metadata(i);
				if (!files->is_selected(i))
					continue;

				if (files->get_item_text(i) == "..") {
					EditorNode::get_singleton()->show_warning(TTR("Can't operate on '..'"));
					return;
				}

				if (path.ends_with("/")) {
					move_dirs.push_back(path.substr(0, path.length() - 1));
				} else {
					move_files.push_back(path);
				}
			}

			if (move_dirs.empty() && move_files.size() == 1) {

				rename_dialog->clear_filters();
				rename_dialog->add_filter("*." + move_files[0].extension());
				rename_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
				rename_dialog->set_current_path(move_files[0]);
				rename_dialog->popup_centered_ratio();
				rename_dialog->set_title(TTR("Pick New Name and Location For:") + " " + move_files[0].get_file());

			} else {
				//just move
				move_dialog->popup_centered_ratio();
			}

		} break;
		case FILE_REMOVE: {

			Vector<String> torem;

			for (int i = 0; i < files->get_item_count(); i++) {

				String path = files->get_item_metadata(i);
				if (path.ends_with("/") || !files->is_selected(i))
					continue;
				torem.push_back(path);
			}

			if (torem.empty()) {
				EditorNode::get_singleton()->show_warning(TTR("No files selected!"));
				break;
			}

			remove_dialog->show(torem);
			//1) find if used
			//2) warn

		} break;
		case FILE_INFO: {

		} break;
		case FILE_REIMPORT: {

			Vector<String> reimport;
			for (int i = 0; i < files->get_item_count(); i++) {

				if (!files->is_selected(i))
					continue;

				String path = files->get_item_metadata(i);
				reimport.push_back(path);
			}

			ERR_FAIL_COND(reimport.size() == 0);

			Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(reimport[0]);
			ERR_FAIL_COND(!rimd.is_valid());
			String editor = rimd->get_editor();

			if (editor.begins_with("texture_")) { //compatibility fix for old texture format
				editor = "texture";
			}

			Ref<EditorImportPlugin> rimp = EditorImportExport::get_singleton()->get_import_plugin_by_name(editor);
			ERR_FAIL_COND(!rimp.is_valid());

			if (reimport.size() == 1) {
				rimp->import_dialog(reimport[0]);
			} else {
				rimp->reimport_multiple_files(reimport);
			}

		} break;
		case FILE_COPY_PATH:

			int idx = files->get_current();
			if (idx < 0 || idx >= files->get_item_count())
				break;
			String path = files->get_item_metadata(idx);
			OS::get_singleton()->set_clipboard(path);
	}
}

void FileSystemDock::_folder_option(int p_option) {

	TreeItem *selected = tree->get_selected();

	switch (p_option) {
		case FOLDER_EXPAND_ALL:
		case FOLDER_COLLAPSE_ALL: {
			bool is_collapsed = (p_option == FOLDER_COLLAPSE_ALL);
			Vector<TreeItem *> needs_check;
			needs_check.push_back(selected);

			while (needs_check.size()) {
				needs_check[0]->set_collapsed(is_collapsed);

				TreeItem *child = needs_check[0]->get_children();
				while (child) {
					needs_check.push_back(child);
					child = child->get_next();
				}

				needs_check.remove(0);
			}
		} break;
		case FOLDER_MOVE: {
			to_move.clear();
			String fpath = selected->get_metadata(tree->get_selected_column());
			if (fpath != "res://") {
				fpath = fpath.ends_with("/") ? fpath.substr(0, fpath.length() - 1) : fpath;
				to_move.push_back(FileOrFolder(fpath, false));
				move_dir_dialog->popup_centered_ratio();
			}
		} break;
		case FOLDER_RENAME: {
			to_rename.path = selected->get_metadata(tree->get_selected_column());
			to_rename.is_file = false;
			if (to_rename.path != "res://") {
				String name = to_rename.path.ends_with("/") ? to_rename.path.substr(0, to_rename.path.length() - 1).get_file() : to_rename.path.get_file();
				rename_dir_dialog->set_title(TTR("Renaming folder:") + " " + name);
				rename_dialog_text->set_text(name);
				rename_dialog_text->select(0, name.length());
				rename_dir_dialog->popup_centered_minsize(Size2(250, 80) * EDSCALE);
				rename_dialog_text->grab_focus();
			}
		} break;
		case FOLDER_REMOVE: {
			Vector<String> remove_folders;
			Vector<String> remove_files;
			String fpath = selected->get_metadata(tree->get_selected_column());
			if (fpath != "res://") {
				remove_folders.push_back(fpath);
				remove_dialog->show(remove_folders);
			}
		} break;
		case FOLDER_NEW_FOLDER: {
			make_dir_dialog_text->set_text("new folder");
			make_dir_dialog_text->select_all();
			make_dir_dialog->popup_centered_minsize(Size2(250, 80) * EDSCALE);
			make_dir_dialog_text->grab_focus();
		} break;
		case FOLDER_COPY_PATH: {
			String fpath = selected->get_metadata(tree->get_selected_column());
			OS::get_singleton()->set_clipboard(fpath);
		} break;
		case FOLDER_SHOW_IN_EXPLORER: {
			String fpath = selected->get_metadata(tree->get_selected_column());
			String dir = Globals::get_singleton()->globalize_path(fpath);
			OS::get_singleton()->shell_open(String("file://") + dir);
		} break;
	}
}

void FileSystemDock::_open_pressed() {

	TreeItem *sel = tree->get_selected();
	if (!sel) {
		return;
	}
	path = sel->get_metadata(0);
	/*if (path!="res://" && path.ends_with("/")) {
		path=path.substr(0,path.length()-1);
	}*/

	//tree_mode=false;

	if (split_mode) {
		tree->hide();
		file_list_vb->show();
		button_favorite->hide();
	}

	//file_options->show();

	_update_files(false);
	current_path->set_text(path);
	_push_to_history();

	//	emit_signal("open",path);
}

void FileSystemDock::_search_changed(const String &p_text) {

	if (!search_box->is_visible())
		return; //wtf

	_update_files(false);
}

void FileSystemDock::_rescan() {

	_set_scanning_mode();
	EditorFileSystem::get_singleton()->scan();
}

void FileSystemDock::fix_dependencies(const String &p_for_file) {
	deps_editor->edit(p_for_file);
}

void FileSystemDock::focus_on_filter() {

	if (!search_box->is_visible()) {
		// Tree mode, switch to files list with search box
		tree->hide();
		file_list_vb->show();
		button_favorite->hide();
	}

	search_box->grab_focus();
}

void FileSystemDock::set_display_mode(int p_mode) {

	if (p_mode == display_mode)
		return;

	button_display_mode->set_pressed(p_mode == DISPLAY_LIST);
	_change_file_display();
}

Variant FileSystemDock::get_drag_data_fw(const Point2 &p_point, Control *p_from) {

	if (p_from == tree) {

		TreeItem *selected = tree->get_selected();
		if (!selected)
			return Variant();

		String path = selected->get_metadata(0);
		if (path == String())
			return Variant();
		if (!path.ends_with("/"))
			path = path + "/";
		Vector<String> paths;
		paths.push_back(path);
		Dictionary d = EditorNode::get_singleton()->drag_files(paths, p_from);

		if (selected->get_parent() && tree->get_root()->get_children() == selected->get_parent()) {
			//a favorite.. treat as such
			d["type"] = "favorite";
		}

		return d;
	}

	if (p_from == files) {

		List<int> seldirs;
		List<int> selfiles;

		for (int i = 0; i < files->get_item_count(); i++) {
			if (files->is_selected(i)) {
				String path = files->get_item_metadata(i);
				if (path.ends_with("/"))
					seldirs.push_back(i);
				else
					selfiles.push_back(i);
			}
		}

		if (seldirs.empty() && selfiles.empty())
			return Variant();
		//if (seldirs.size() && selfiles.size())
		//	return Variant(); //can't really mix files and dirs (i think?) - yes you can, commenting

		/*if (selfiles.size()==1) {
			Ref<Resource> resource = ResourceLoader::load(files->get_item_metadata(selfiles.front()->get()));
			if (resource.is_valid()) {
				return EditorNode::get_singleton()->drag_resource(resource,p_from);
			}
		}*/

		if (selfiles.size() > 0 && seldirs.size() == 0) {
			Vector<String> fnames;
			for (List<int>::Element *E = selfiles.front(); E; E = E->next()) {
				fnames.push_back(files->get_item_metadata(E->get()));
			}
			return EditorNode::get_singleton()->drag_files(fnames, p_from);
		}

		if (selfiles.size() > 0 || seldirs.size() > 0) {
			Vector<String> fnames;
			for (List<int>::Element *E = selfiles.front(); E; E = E->next()) {
				fnames.push_back(files->get_item_metadata(E->get()));
			}
			for (List<int>::Element *E = seldirs.front(); E; E = E->next()) {
				fnames.push_back(files->get_item_metadata(E->get()));
			}
			return EditorNode::get_singleton()->drag_files_and_dirs(fnames, p_from);
		}
	}

	return Variant();
}

bool FileSystemDock::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	Dictionary drag_data = p_data;

	if (drag_data.has("type") && String(drag_data["type"]) == "favorite") {

		//moving favorite around
		TreeItem *ti = tree->get_item_at_pos(p_point);
		if (!ti)
			return false;

		int what = tree->get_drop_section_at_pos(p_point);

		if (ti == tree->get_root()->get_children()) {
			return (what == 1); //the parent, first fav
		}
		if (ti->get_parent() && tree->get_root()->get_children() == ti->get_parent()) {
			return true; // a favorite
		}

		if (ti == tree->get_root()->get_children()->get_next()) {
			return (what == -1); //the tree, last fav
		}

		return false;
	}

	if (drag_data.has("type") && String(drag_data["type"]) == "resource") {
		return true;
	}

	if (drag_data.has("type") && (String(drag_data["type"]) == "files" || String(drag_data["type"]) == "files_and_dirs")) {

		Vector<String> fnames = drag_data["files"];

		if (p_from == files) {

			int at_pos = files->get_item_at_pos(p_point);
			if (at_pos != -1) {

				String dir = files->get_item_metadata(at_pos);
				if (dir.ends_with("/"))
					return true;
			}
		}

		if (p_from == tree) {

			TreeItem *ti = tree->get_item_at_pos(p_point);
			if (!ti)
				return false;
			String path = ti->get_metadata(0);

			if (path == String())
				return false;

			return true;
		}
	}

	return false;
}

void FileSystemDock::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {

	if (!can_drop_data_fw(p_point, p_data, p_from))
		return;
	Dictionary drag_data = p_data;

	if (drag_data.has("type") && String(drag_data["type"]) == "favorite") {

		//moving favorite around
		TreeItem *ti = tree->get_item_at_pos(p_point);
		if (!ti)
			return;

		Vector<String> files = drag_data["files"];

		ERR_FAIL_COND(files.size() != 1);

		String swap = files[0];
		if (swap != "res://" && swap.ends_with("/")) {
			swap = swap.substr(0, swap.length() - 1);
		}

		int what = tree->get_drop_section_at_pos(p_point);

		TreeItem *swap_item = NULL;

		if (ti == tree->get_root()->get_children()) {
			swap_item = tree->get_root()->get_children()->get_children();

		} else if (ti->get_parent() && tree->get_root()->get_children() == ti->get_parent()) {
			if (what == -1) {
				swap_item = ti;
			} else {
				swap_item = ti->get_next();
			}
		}

		String swap_with;

		if (swap_item) {
			swap_with = swap_item->get_metadata(0);
			if (swap_with != "res://" && swap_with.ends_with("/")) {
				swap_with = swap_with.substr(0, swap_with.length() - 1);
			}
		}

		if (swap == swap_with)
			return;

		Vector<String> dirs = EditorSettings::get_singleton()->get_favorite_dirs();

		ERR_FAIL_COND(dirs.find(swap) == -1);
		ERR_FAIL_COND(swap_with != String() && dirs.find(swap_with) == -1);

		dirs.erase(swap);

		if (swap_with == String()) {
			dirs.push_back(swap);
		} else {
			int idx = dirs.find(swap_with);
			dirs.insert(idx, swap);
		}

		EditorSettings::get_singleton()->set_favorite_dirs(dirs);
		_update_tree();
		return;
	}

	if (drag_data.has("type") && String(drag_data["type"]) == "resource") {
		Ref<Resource> res = drag_data["resource"];

		if (!res.is_valid()) {
			return;
		}

		if (p_from == tree) {

			TreeItem *ti = tree->get_item_at_pos(p_point);
			if (!ti)
				return;
			String path = ti->get_metadata(0);

			if (path == String())
				return;

			EditorNode::get_singleton()->save_resource_as(res, path);
			return;
		}

		if (p_from == files) {
			String save_path = path;

			int at_pos = files->get_item_at_pos(p_point);
			if (at_pos != -1) {
				String to_dir = files->get_item_metadata(at_pos);
				if (to_dir.ends_with("/")) {
					save_path = to_dir;
					if (save_path != "res://")
						save_path = save_path.substr(0, save_path.length() - 1);
				}
			}

			EditorNode::get_singleton()->save_resource_as(res, save_path);
			return;
		}
	}

	if (drag_data.has("type") && (String(drag_data["type"]) == "files" || String(drag_data["type"]) == "files_and_dirs")) {

		if (p_from == files || p_from == tree) {

			String to_dir;

			if (p_from == files) {

				int at_pos = files->get_item_at_pos(p_point);
				ERR_FAIL_COND(at_pos == -1);
				to_dir = files->get_item_metadata(at_pos);
			} else {
				TreeItem *ti = tree->get_item_at_pos(p_point);
				if (!ti)
					return;
				to_dir = ti->get_metadata(0);
				ERR_FAIL_COND(to_dir == String());
			}

			if (to_dir != "res://" && to_dir.ends_with("/")) {
				to_dir = to_dir.substr(0, to_dir.length() - 1);
			}

			Vector<String> fnames = drag_data["files"];
			move_files.clear();
			move_dirs.clear();

			for (int i = 0; i < fnames.size(); i++) {
				if (fnames[i].ends_with("/"))
					move_dirs.push_back(fnames[i]);
				else
					move_files.push_back(fnames[i]);
			}

			_move_operation(to_dir);
		}
	}
}

void FileSystemDock::_files_list_rmb_select(int p_item, const Vector2 &p_pos) {

	Vector<String> filenames;

	bool all_scenes = true;
	bool all_can_reimport = true;
	Set<String> types;

	for (int i = 0; i < files->get_item_count(); i++) {

		if (!files->is_selected(i))
			continue;

		String path = files->get_item_metadata(i);

		if (files->get_item_text(i) == "..") {
			// no operate on ..
			return;
		}

		if (path.ends_with("/")) {
			//no operate on dirs
			return;
		}

		EditorFileSystemDirectory *efsd = NULL;
		int pos;

		efsd = EditorFileSystem::get_singleton()->find_file(path, &pos);

		if (efsd) {

			if (!efsd->get_file_meta(pos)) {
				all_can_reimport = false;

			} else {
				Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(path);
				if (rimd.is_valid()) {

					String editor = rimd->get_editor();
					if (editor.begins_with("texture_")) { //compatibility fix for old texture format
						editor = "texture";
					}
					types.insert(editor);

				} else {
					all_can_reimport = false;
				}
			}
		} else {
			all_can_reimport = false;
		}

		filenames.push_back(path);
		if (EditorFileSystem::get_singleton()->get_file_type(path) != "PackedScene")
			all_scenes = false;
	}

	if (filenames.size() == 0)
		return;

	file_options->clear();
	file_options->set_size(Size2(1, 1));

	file_options->add_item(TTR("Open"), FILE_OPEN);
	if (all_scenes) {
		file_options->add_item(TTR("Instance"), FILE_INSTANCE);
	}

	file_options->add_separator();

	if (filenames.size() == 1) {
		file_options->add_item(TTR("Edit Dependencies.."), FILE_DEPENDENCIES);
		file_options->add_item(TTR("View Owners.."), FILE_OWNERS);
		file_options->add_separator();
	}

	if (filenames.size() == 1) {
		file_options->add_item(TTR("Copy Path"), FILE_COPY_PATH);
		file_options->add_item(TTR("Rename or Move.."), FILE_MOVE);
	} else {
		file_options->add_item(TTR("Move To.."), FILE_MOVE);
	}

	file_options->add_item(TTR("Delete"), FILE_REMOVE);

	//file_options->add_item(TTR("Info"),FILE_INFO);

	file_options->add_separator();
	file_options->add_item(TTR("Show In File Manager"), FILE_SHOW_IN_EXPLORER);

	if (all_can_reimport && types.size() == 1) { //all can reimport and are of the same type

		bool valid = true;
		Ref<EditorImportPlugin> rimp = EditorImportExport::get_singleton()->get_import_plugin_by_name(types.front()->get());
		if (rimp.is_valid()) {

			if (filenames.size() > 1 && !rimp->can_reimport_multiple_files()) {
				valid = false;
			}
		} else {
			valid = false;
		}

		if (valid) {
			file_options->add_separator();
			file_options->add_item(TTR("Re-Import.."), FILE_REIMPORT);
		}
	}

	file_options->set_pos(files->get_global_pos() + p_pos);
	file_options->popup();
}

void FileSystemDock::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_update_tree"), &FileSystemDock::_update_tree);
	ObjectTypeDB::bind_method(_MD("_rescan"), &FileSystemDock::_rescan);
	ObjectTypeDB::bind_method(_MD("_favorites_pressed"), &FileSystemDock::_favorites_pressed);
	//	ObjectTypeDB::bind_method(_MD("_instance_pressed"),&ScenesDock::_instance_pressed);
	ObjectTypeDB::bind_method(_MD("_open_pressed"), &FileSystemDock::_open_pressed);

	ObjectTypeDB::bind_method(_MD("_thumbnail_done"), &FileSystemDock::_thumbnail_done);
	ObjectTypeDB::bind_method(_MD("_select_file"), &FileSystemDock::_select_file);
	ObjectTypeDB::bind_method(_MD("_go_to_tree"), &FileSystemDock::_go_to_tree);
	ObjectTypeDB::bind_method(_MD("_go_to_dir"), &FileSystemDock::_go_to_dir);
	ObjectTypeDB::bind_method(_MD("navigate_to_path"), &FileSystemDock::navigate_to_path);
	ObjectTypeDB::bind_method(_MD("_change_file_display"), &FileSystemDock::_change_file_display);
	ObjectTypeDB::bind_method(_MD("_fw_history"), &FileSystemDock::_fw_history);
	ObjectTypeDB::bind_method(_MD("_bw_history"), &FileSystemDock::_bw_history);
	ObjectTypeDB::bind_method(_MD("_fs_changed"), &FileSystemDock::_fs_changed);
	ObjectTypeDB::bind_method(_MD("_dir_selected"), &FileSystemDock::_dir_selected);
	ObjectTypeDB::bind_method(_MD("_file_option"), &FileSystemDock::_file_option);
	ObjectTypeDB::bind_method(_MD("_move_operation"), &FileSystemDock::_move_operation);
	ObjectTypeDB::bind_method(_MD("_rename_operation"), &FileSystemDock::_rename_operation);

	ObjectTypeDB::bind_method(_MD("_folder_option"), &FileSystemDock::_folder_option);
	ObjectTypeDB::bind_method(_MD("_dir_rmb_pressed"), &FileSystemDock::_dir_rmb_pressed);
	ObjectTypeDB::bind_method(_MD("_make_dir_confirm"), &FileSystemDock::_make_dir_confirm);
	ObjectTypeDB::bind_method(_MD("_rename_operation_confirm"), &FileSystemDock::_rename_operation_confirm);
	ObjectTypeDB::bind_method(_MD("_move_operation_confirm"), &FileSystemDock::_move_operation_confirm);

	ObjectTypeDB::bind_method(_MD("_search_changed"), &FileSystemDock::_search_changed);

	ObjectTypeDB::bind_method(_MD("get_drag_data_fw"), &FileSystemDock::get_drag_data_fw);
	ObjectTypeDB::bind_method(_MD("can_drop_data_fw"), &FileSystemDock::can_drop_data_fw);
	ObjectTypeDB::bind_method(_MD("drop_data_fw"), &FileSystemDock::drop_data_fw);
	ObjectTypeDB::bind_method(_MD("_files_list_rmb_select"), &FileSystemDock::_files_list_rmb_select);

	ObjectTypeDB::bind_method(_MD("_preview_invalidated"), &FileSystemDock::_preview_invalidated);

	ADD_SIGNAL(MethodInfo("instance", PropertyInfo(Variant::STRING_ARRAY, "files")));
	ADD_SIGNAL(MethodInfo("open"));
}

FileSystemDock::FileSystemDock(EditorNode *p_editor) {

	editor = p_editor;

	HBoxContainer *toolbar_hbc = memnew(HBoxContainer);
	add_child(toolbar_hbc);

	button_hist_prev = memnew(ToolButton);
	toolbar_hbc->add_child(button_hist_prev);
	button_hist_prev->set_disabled(true);
	button_hist_prev->set_tooltip(TTR("Previous Directory"));

	button_hist_next = memnew(ToolButton);
	toolbar_hbc->add_child(button_hist_next);
	button_hist_next->set_disabled(true);
	button_hist_prev->set_focus_mode(FOCUS_NONE);
	button_hist_next->set_focus_mode(FOCUS_NONE);
	button_hist_next->set_tooltip(TTR("Next Directory"));

	current_path = memnew(LineEdit);
	current_path->set_h_size_flags(SIZE_EXPAND_FILL);
	toolbar_hbc->add_child(current_path);

	button_reload = memnew(Button);
	button_reload->set_flat(true);
	button_reload->connect("pressed", this, "_rescan");
	toolbar_hbc->add_child(button_reload);
	button_reload->set_focus_mode(FOCUS_NONE);
	button_reload->set_tooltip(TTR("Re-Scan Filesystem"));
	button_reload->hide();

	//toolbar_hbc->add_spacer();

	button_favorite = memnew(Button);
	button_favorite->set_flat(true);
	button_favorite->set_toggle_mode(true);
	button_favorite->connect("pressed", this, "_favorites_pressed");
	toolbar_hbc->add_child(button_favorite);
	button_favorite->set_tooltip(TTR("Toggle folder status as Favorite"));

	button_favorite->set_focus_mode(FOCUS_NONE);

	//	Control *spacer = memnew( Control);

	/*
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
	button_instance->set_tooltip(TTR("Instance the selected scene(s) as child of the selected node."));

*/
	file_options = memnew(PopupMenu);
	add_child(file_options);

	folder_options = memnew(PopupMenu);
	add_child(folder_options);

	move_dir_dialog = memnew(EditorDirDialog);
	move_dir_dialog->get_ok()->set_text(TTR("Move"));
	add_child(move_dir_dialog);
	move_dir_dialog->connect("dir_selected", this, "_move_operation_confirm");

	rename_dir_dialog = memnew(ConfirmationDialog);
	VBoxContainer *rename_dialog_vb = memnew(VBoxContainer);
	rename_dir_dialog->add_child(rename_dialog_vb);

	rename_dialog_text = memnew(LineEdit);
	rename_dialog_vb->add_margin_child(TTR("Name:"), rename_dialog_text);
	rename_dir_dialog->get_ok()->set_text(TTR("Rename"));
	add_child(rename_dir_dialog);
	rename_dir_dialog->set_child_rect(rename_dialog_vb);
	rename_dir_dialog->register_text_enter(rename_dialog_text);
	rename_dir_dialog->connect("confirmed", this, "_rename_operation_confirm");

	make_dir_dialog = memnew(ConfirmationDialog);
	make_dir_dialog->set_title(TTR("Create Folder"));

	VBoxContainer *make_folder_dialog_vb = memnew(VBoxContainer);
	make_dir_dialog_text = memnew(LineEdit);
	make_folder_dialog_vb->add_margin_child(TTR("Name:"), make_dir_dialog_text);
	add_child(make_dir_dialog);

	make_dir_dialog->add_child(make_folder_dialog_vb);
	make_dir_dialog->set_child_rect(make_folder_dialog_vb);
	make_dir_dialog->register_text_enter(make_dir_dialog_text);
	make_dir_dialog->connect("confirmed", this, "_make_dir_confirm");

	split_box = memnew(VSplitContainer);
	add_child(split_box);
	split_box->set_v_size_flags(SIZE_EXPAND_FILL);

	tree = memnew(Tree);

	tree->set_hide_root(true);
	split_box->add_child(tree);
	tree->set_drag_forwarding(this);
	tree->set_allow_rmb_select(true);
	//tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->connect("item_edited", this, "_favorite_toggled");
	tree->connect("item_activated", this, "_open_pressed");
	tree->connect("cell_selected", this, "_dir_selected");
	tree->connect("item_rmb_selected", this, "_dir_rmb_pressed");

	files = memnew(ItemList);
	files->set_v_size_flags(SIZE_EXPAND_FILL);
	files->set_select_mode(ItemList::SELECT_MULTI);
	files->set_drag_forwarding(this);
	files->connect("item_rmb_selected", this, "_files_list_rmb_select");
	files->set_allow_rmb_select(true);

	file_list_vb = memnew(VBoxContainer);
	split_box->add_child(file_list_vb);
	file_list_vb->set_v_size_flags(SIZE_EXPAND_FILL);

	path_hb = memnew(HBoxContainer);
	file_list_vb->add_child(path_hb);

	button_back = memnew(ToolButton);
	path_hb->add_child(button_back);
	button_back->hide();

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	path_hb->add_child(search_box);
	search_box->connect("text_changed", this, "_search_changed");

	search_icon = memnew(TextureFrame);
	search_icon->set_stretch_mode(TextureFrame::STRETCH_KEEP_CENTERED);
	path_hb->add_child(search_icon);

	button_display_mode = memnew(ToolButton);
	path_hb->add_child(button_display_mode);
	button_display_mode->set_toggle_mode(true);

	file_list_vb->add_child(files);

	scanning_vb = memnew(VBoxContainer);
	Label *slabel = memnew(Label);
	slabel->set_text("Scanning Files,\nPlease Wait..");
	slabel->set_align(Label::ALIGN_CENTER);
	scanning_vb->add_child(slabel);
	scanning_progress = memnew(ProgressBar);
	scanning_vb->add_child(scanning_progress);
	add_child(scanning_vb);
	scanning_vb->hide();

	deps_editor = memnew(DependencyEditor);
	add_child(deps_editor);

	owners_editor = memnew(DependencyEditorOwners(editor));
	add_child(owners_editor);

	remove_dialog = memnew(DependencyRemoveDialog);
	add_child(remove_dialog);

	move_dialog = memnew(EditorDirDialog);
	add_child(move_dialog);
	move_dialog->connect("dir_selected", this, "_move_operation");
	move_dialog->get_ok()->set_text(TTR("Move"));

	rename_dialog = memnew(EditorFileDialog);
	rename_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
	rename_dialog->connect("file_selected", this, "_rename_operation");
	add_child(rename_dialog);

	updating_tree = false;
	initialized = false;

	history.push_back("res://");
	history_pos = 0;

	split_mode = false;
	display_mode = DISPLAY_THUMBNAILS;

	path = "res://";

	add_constant_override("separation", 3);
}

FileSystemDock::~FileSystemDock() {
}
