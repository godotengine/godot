/*************************************************************************/
/*  editor_asset_installer.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "editor_asset_installer.h"

#include "core/io/zip_io.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "editor_node.h"
#include "progress_dialog.h"

void EditorAssetInstaller::_update_subitems(TreeItem *p_item, bool p_check, bool p_first) {
	if (p_check) {
		if (p_item->get_custom_color(0) == Color()) {
			p_item->set_checked(0, true);
		}
	} else {
		p_item->set_checked(0, false);
	}

	if (p_item->get_children()) {
		_update_subitems(p_item->get_children(), p_check);
	}

	if (!p_first && p_item->get_next()) {
		_update_subitems(p_item->get_next(), p_check);
	}
}

void EditorAssetInstaller::_uncheck_parent(TreeItem *p_item) {
	if (!p_item) {
		return;
	}

	bool any_checked = false;
	TreeItem *item = p_item->get_children();
	while (item) {
		if (item->is_checked(0)) {
			any_checked = true;
			break;
		}
		item = item->get_next();
	}

	if (!any_checked) {
		p_item->set_checked(0, false);
		_uncheck_parent(p_item->get_parent());
	}
}

void EditorAssetInstaller::_item_edited() {
	if (updating) {
		return;
	}

	TreeItem *item = tree->get_edited();
	if (!item) {
		return;
	}

	String path = item->get_metadata(0);

	updating = true;
	if (path == String() || item == tree->get_root()) { //a dir or root
		_update_subitems(item, item->is_checked(0), true);
	}

	if (item->is_checked(0)) {
		while (item) {
			item->set_checked(0, true);
			item = item->get_parent();
		}
	} else {
		_uncheck_parent(item->get_parent());
	}
	updating = false;
}

void EditorAssetInstaller::open(const String &p_path, int p_depth) {
	package_path = p_path;
	Set<String> files_sorted;

	FileAccess *src_f = nullptr;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	unzFile pkg = unzOpen2(p_path.utf8().get_data(), &io);
	if (!pkg) {
		error->set_text(vformat(TTR("Error opening asset file for \"%s\" (not in ZIP format)."), asset_name));
		return;
	}

	int ret = unzGoToFirstFile(pkg);

	while (ret == UNZ_OK) {
		//get filename
		unz_file_info info;
		char fname[16384];
		unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);

		String name = String::utf8(fname);
		files_sorted.insert(name);

		ret = unzGoToNextFile(pkg);
	}

	Map<String, Ref<Texture>> extension_guess;
	{
		extension_guess["bmp"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["dds"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["exr"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["hdr"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["jpg"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["jpeg"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["png"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["svg"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["svgz"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["tga"] = tree->get_icon("ImageTexture", "EditorIcons");
		extension_guess["webp"] = tree->get_icon("ImageTexture", "EditorIcons");

		extension_guess["wav"] = tree->get_icon("AudioStreamSample", "EditorIcons");
		extension_guess["ogg"] = tree->get_icon("AudioStreamOGGVorbis", "EditorIcons");
		extension_guess["mp3"] = tree->get_icon("AudioStreamMP3", "EditorIcons");

		extension_guess["scn"] = tree->get_icon("PackedScene", "EditorIcons");
		extension_guess["tscn"] = tree->get_icon("PackedScene", "EditorIcons");
		extension_guess["escn"] = tree->get_icon("PackedScene", "EditorIcons");
		extension_guess["dae"] = tree->get_icon("PackedScene", "EditorIcons");
		extension_guess["gltf"] = tree->get_icon("PackedScene", "EditorIcons");
		extension_guess["glb"] = tree->get_icon("PackedScene", "EditorIcons");

		extension_guess["gdshader"] = tree->get_icon("Shader", "EditorIcons");
		extension_guess["shader"] = tree->get_icon("Shader", "EditorIcons");
		extension_guess["gd"] = tree->get_icon("GDScript", "EditorIcons");
		if (Engine::get_singleton()->has_singleton("GodotSharp")) {
			extension_guess["cs"] = tree->get_icon("CSharpScript", "EditorIcons");
		} else {
			// Mark C# support as unavailable.
			extension_guess["cs"] = tree->get_icon("ImportFail", "EditorIcons");
		}
		extension_guess["vs"] = tree->get_icon("VisualScript", "EditorIcons");

		extension_guess["res"] = tree->get_icon("Resource", "EditorIcons");
		extension_guess["tres"] = tree->get_icon("Resource", "EditorIcons");
		extension_guess["atlastex"] = tree->get_icon("AtlasTexture", "EditorIcons");
		// By default, OBJ files are imported as Mesh resources rather than PackedScenes.
		extension_guess["obj"] = tree->get_icon("Mesh", "EditorIcons");

		extension_guess["txt"] = tree->get_icon("TextFile", "EditorIcons");
		extension_guess["md"] = tree->get_icon("TextFile", "EditorIcons");
		extension_guess["rst"] = tree->get_icon("TextFile", "EditorIcons");
		extension_guess["json"] = tree->get_icon("TextFile", "EditorIcons");
		extension_guess["yml"] = tree->get_icon("TextFile", "EditorIcons");
		extension_guess["yaml"] = tree->get_icon("TextFile", "EditorIcons");
		extension_guess["toml"] = tree->get_icon("TextFile", "EditorIcons");
		extension_guess["cfg"] = tree->get_icon("TextFile", "EditorIcons");
		extension_guess["ini"] = tree->get_icon("TextFile", "EditorIcons");
	}

	Ref<Texture> generic_extension = get_icon("Object", "EditorIcons");

	unzClose(pkg);

	updating = true;
	tree->clear();
	TreeItem *root = tree->create_item();
	root->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	root->set_checked(0, true);
	root->set_icon(0, get_icon("folder", "FileDialog"));
	root->set_text(0, "res://");
	root->set_editable(0, true);
	Map<String, TreeItem *> dir_map;

	int num_file_conflicts = 0;

	for (Set<String>::Element *E = files_sorted.front(); E; E = E->next()) {
		String path = E->get();
		int depth = p_depth;
		bool skip = false;
		while (depth > 0) {
			int pp = path.find("/");
			if (pp == -1) {
				skip = true;
				break;
			}
			path = path.substr(pp + 1, path.length());
			depth--;
		}

		if (skip || path == String()) {
			continue;
		}

		bool isdir = false;

		if (path.ends_with("/")) {
			//a directory
			path = path.substr(0, path.length() - 1);
			isdir = true;
		}

		int pp = path.find_last("/");

		TreeItem *parent;
		if (pp == -1) {
			parent = root;
		} else {
			String ppath = path.substr(0, pp);
			ERR_CONTINUE(!dir_map.has(ppath));
			parent = dir_map[ppath];
		}

		TreeItem *ti = tree->create_item(parent);
		ti->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		ti->set_checked(0, true);
		ti->set_editable(0, true);
		if (isdir) {
			dir_map[path] = ti;
			ti->set_text(0, path.get_file() + "/");
			ti->set_icon(0, get_icon("folder", "FileDialog"));
			ti->set_metadata(0, String());
		} else {
			String file = path.get_file();
			String extension = file.get_extension().to_lower();
			if (extension_guess.has(extension)) {
				ti->set_icon(0, extension_guess[extension]);
			} else {
				ti->set_icon(0, generic_extension);
			}
			ti->set_text(0, file);

			String res_path = "res://" + path;
			if (FileAccess::exists(res_path)) {
				num_file_conflicts += 1;
				ti->set_custom_color(0, get_color("error_color", "Editor"));
				ti->set_tooltip(0, vformat(TTR("%s (already exists)"), res_path));
				ti->set_checked(0, false);
			} else {
				ti->set_tooltip(0, res_path);
			}

			ti->set_metadata(0, res_path);
		}

		status_map[E->get()] = ti;
	}

	if (num_file_conflicts >= 1) {
		asset_contents->set_text(vformat(TTR("Contents of asset \"%s\" - %d file(s) conflict with your project:"), asset_name, num_file_conflicts));
	} else {
		asset_contents->set_text(vformat(TTR("Contents of asset \"%s\" - No files conflict with your project:"), asset_name));
	}

	popup_centered_ratio();
	updating = false;
}

void EditorAssetInstaller::ok_pressed() {
	FileAccess *src_f = nullptr;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	unzFile pkg = unzOpen2(package_path.utf8().get_data(), &io);
	if (!pkg) {
		error->set_text(vformat(TTR("Error opening asset file for \"%s\" (not in ZIP format)."), asset_name));
		return;
	}

	int ret = unzGoToFirstFile(pkg);

	Vector<String> failed_files;

	ProgressDialog::get_singleton()->add_task("uncompress", TTR("Uncompressing Assets"), status_map.size());

	int idx = 0;
	while (ret == UNZ_OK) {
		//get filename
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);

		String name = String::utf8(fname);

		if (status_map.has(name) && status_map[name]->is_checked(0)) {
			String path = status_map[name]->get_metadata(0);
			if (path == String()) { // a dir

				String dirpath;
				TreeItem *t = status_map[name];
				while (t) {
					dirpath = t->get_text(0) + dirpath;
					t = t->get_parent();
				}

				if (dirpath.ends_with("/")) {
					dirpath = dirpath.substr(0, dirpath.length() - 1);
				}

				DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
				da->make_dir(dirpath);
				memdelete(da);

			} else {
				Vector<uint8_t> data;
				data.resize(info.uncompressed_size);

				//read
				unzOpenCurrentFile(pkg);
				unzReadCurrentFile(pkg, data.ptrw(), data.size());
				unzCloseCurrentFile(pkg);

				FileAccess *f = FileAccess::open(path, FileAccess::WRITE);
				if (f) {
					f->store_buffer(data.ptr(), data.size());
					memdelete(f);
				} else {
					failed_files.push_back(path);
				}

				ProgressDialog::get_singleton()->task_step("uncompress", path, idx);
			}
		}

		idx++;
		ret = unzGoToNextFile(pkg);
	}

	ProgressDialog::get_singleton()->end_task("uncompress");
	unzClose(pkg);

	if (failed_files.size()) {
		String msg = vformat(TTR("The following files failed extraction from asset \"%s\":"), asset_name) + "\n\n";
		for (int i = 0; i < failed_files.size(); i++) {
			if (i > 15) {
				msg += "\n" + vformat(TTR("(and %s more files)"), itos(failed_files.size() - i));
				break;
			}
			msg += failed_files[i];
		}
		if (EditorNode::get_singleton() != nullptr) {
			EditorNode::get_singleton()->show_warning(msg);
		}
	} else {
		if (EditorNode::get_singleton() != nullptr) {
			EditorNode::get_singleton()->show_warning(vformat(TTR("Asset \"%s\" installed successfully!"), asset_name), TTR("Success!"));
		}
	}
	EditorFileSystem::get_singleton()->scan_changes();
}

void EditorAssetInstaller::set_asset_name(const String &p_asset_name) {
	asset_name = p_asset_name;
}

String EditorAssetInstaller::get_asset_name() const {
	return asset_name;
}

void EditorAssetInstaller::_bind_methods() {
	ClassDB::bind_method("_item_edited", &EditorAssetInstaller::_item_edited);
}

EditorAssetInstaller::EditorAssetInstaller() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	asset_contents = memnew(Label);
	vb->add_child(asset_contents);

	tree = memnew(Tree);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->connect("item_edited", this, "_item_edited");
	vb->add_child(tree);

	error = memnew(AcceptDialog);
	add_child(error);
	get_ok()->set_text(TTR("Install"));
	set_title(TTR("Asset Installer"));

	updating = false;

	set_hide_on_ok(true);
}
