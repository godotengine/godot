/**************************************************************************/
/*  editor_asset_installer.cpp                                            */
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

#include "editor_asset_installer.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/zip_io.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_toaster.h"
#include "editor/progress_dialog.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"

void EditorAssetInstaller::_item_checked() {
	if (updating || !tree->get_edited()) {
		return;
	}

	updating = true;
	TreeItem *item = tree->get_edited();
	item->propagate_check(0);
	updating = false;
}

void EditorAssetInstaller::_check_propagated_to_item(Object *p_obj, int column) {
	TreeItem *affected_item = Object::cast_to<TreeItem>(p_obj);
	if (affected_item && affected_item->get_custom_color(0) != Color()) {
		affected_item->set_checked(0, false);
		affected_item->propagate_check(0, false);
	}
}

void EditorAssetInstaller::open_asset(const String &p_path, bool p_autoskip_toplevel) {
	package_path = p_path;
	asset_files.clear();

	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);

	unzFile pkg = unzOpen2(p_path.utf8().get_data(), &io);
	if (!pkg) {
		EditorToaster::get_singleton()->popup_str(vformat(TTR("Error opening asset file for \"%s\" (not in ZIP format)."), asset_name), EditorToaster::SEVERITY_ERROR);
		return;
	}

	int ret = unzGoToFirstFile(pkg);

	while (ret == UNZ_OK) {
		//get filename
		unz_file_info info;
		char fname[16384];
		unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);

		String source_name = String::utf8(fname);

		// Create intermediate directories if they aren't reported by unzip.
		// We are only interested in subfolders, so skip the root slash.
		int separator = source_name.find("/", 1);
		while (separator != -1) {
			String dir_name = source_name.substr(0, separator + 1);
			if (!dir_name.is_empty() && !asset_files.has(dir_name)) {
				asset_files.insert(dir_name);
			}

			separator = source_name.find("/", separator + 1);
		}

		if (!source_name.is_empty() && !asset_files.has(source_name)) {
			asset_files.insert(source_name);
		}

		ret = unzGoToNextFile(pkg);
	}

	unzClose(pkg);

	_check_has_toplevel();
	// Default to false, unless forced.
	skip_toplevel = p_autoskip_toplevel;
	skip_toplevel_check->set_pressed(!skip_toplevel_check->is_disabled() && skip_toplevel);

	_rebuild_tree();
}

void EditorAssetInstaller::_rebuild_tree() {
	updating = true;
	tree->clear();

	TreeItem *root = tree->create_item();
	root->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	root->set_checked(0, true);
	root->set_icon(0, get_theme_icon(SNAME("folder"), SNAME("FileDialog")));
	root->set_text(0, "res://");
	root->set_editable(0, true);

	HashMap<String, TreeItem *> directory_item_map;
	int num_file_conflicts = 0;

	bool first = true;
	for (const String &E : asset_files) {
		if (first) {
			first = false;

			if (!toplevel_prefix.is_empty() && skip_toplevel) {
				continue;
			}
		}

		String path = E; // We're going to mutate it.
		if (!toplevel_prefix.is_empty() && skip_toplevel) {
			path = path.trim_prefix(toplevel_prefix);
		}

		bool is_directory = false;
		if (path.ends_with("/")) {
			// Directory.
			path = path.trim_suffix("/");
			is_directory = true;
		}

		TreeItem *parent_item;

		int separator = path.rfind("/");
		if (separator == -1) {
			parent_item = root;
		} else {
			String parent_path = path.substr(0, separator);
			HashMap<String, TreeItem *>::Iterator I = directory_item_map.find(parent_path);
			ERR_CONTINUE(!I);
			parent_item = I->value;
		}

		TreeItem *ti;
		if (is_directory) {
			ti = _create_dir_item(parent_item, path, directory_item_map);
		} else {
			ti = _create_file_item(parent_item, path, &num_file_conflicts);
		}
		file_item_map[E] = ti;
	}

	asset_title_label->set_text(asset_name);
	if (num_file_conflicts >= 1) {
		asset_conflicts_label->set_text(vformat(TTRN("%d file conflicts with your project", "%d files conflict with your project", num_file_conflicts), num_file_conflicts));
		asset_conflicts_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
	} else {
		asset_conflicts_label->set_text(TTR("No files conflict with your project"));
		asset_conflicts_label->remove_theme_color_override("font_color");
	}

	popup_centered_ratio(0.5);
	updating = false;
}

TreeItem *EditorAssetInstaller::_create_dir_item(TreeItem *p_parent, const String &p_path, HashMap<String, TreeItem *> &p_item_map) {
	TreeItem *ti = tree->create_item(p_parent);
	ti->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	ti->set_checked(0, true);
	ti->set_editable(0, true);

	ti->set_text(0, p_path.get_file() + "/");
	ti->set_icon(0, get_theme_icon(SNAME("folder"), SNAME("FileDialog")));
	ti->set_metadata(0, String());

	p_item_map[p_path] = ti;
	return ti;
}

TreeItem *EditorAssetInstaller::_create_file_item(TreeItem *p_parent, const String &p_path, int *r_conflicts) {
	TreeItem *ti = tree->create_item(p_parent);
	ti->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	ti->set_checked(0, true);
	ti->set_editable(0, true);

	String file = p_path.get_file();
	String extension = file.get_extension().to_lower();
	if (extension_icon_map.has(extension)) {
		ti->set_icon(0, extension_icon_map[extension]);
	} else {
		ti->set_icon(0, generic_extension_icon);
	}
	ti->set_text(0, file);

	String res_path = "res://" + p_path;
	if (FileAccess::exists(res_path)) {
		*r_conflicts += 1;
		ti->set_custom_color(0, get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		ti->set_tooltip_text(0, vformat(TTR("%s (already exists)"), res_path));
		ti->set_checked(0, false);
		ti->propagate_check(0);
	} else {
		ti->set_tooltip_text(0, res_path);
	}

	ti->set_metadata(0, res_path);

	return ti;
}

void EditorAssetInstaller::_check_has_toplevel() {
	// Check if the file structure has a distinct top-level directory. This is typical
	// for archives generated by GitHub, etc, but not for manually created ZIPs.

	toplevel_prefix = "";
	skip_toplevel_check->set_pressed(false);
	skip_toplevel_check->set_disabled(true);
	skip_toplevel_check->set_tooltip_text(TTR("This asset doesn't have a root directory, so it can't be ignored."));

	if (asset_files.is_empty()) {
		return;
	}

	String first_asset;
	for (const String &E : asset_files) {
		if (first_asset.is_empty()) { // Checking the first file/directory.
			if (!E.ends_with("/")) {
				return; // No directories in this asset.
			}

			// We will match everything else against this directory.
			first_asset = E;
			continue;
		}

		if (!E.begins_with(first_asset)) {
			return; // Found a file or a directory that doesn't share the same base path.
		}
	}

	toplevel_prefix = first_asset;
	skip_toplevel_check->set_disabled(false);
	skip_toplevel_check->set_tooltip_text("");
}

void EditorAssetInstaller::_set_skip_toplevel(bool p_checked) {
	skip_toplevel = p_checked;
	_rebuild_tree();
}

void EditorAssetInstaller::ok_pressed() {
	_install_asset();
}

void EditorAssetInstaller::_install_asset() {
	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);

	unzFile pkg = unzOpen2(package_path.utf8().get_data(), &io);
	if (!pkg) {
		EditorToaster::get_singleton()->popup_str(vformat(TTR("Error opening asset file for \"%s\" (not in ZIP format)."), asset_name), EditorToaster::SEVERITY_ERROR);
		return;
	}

	Vector<String> failed_files;
	int ret = unzGoToFirstFile(pkg);

	ProgressDialog::get_singleton()->add_task("uncompress", TTR("Uncompressing Assets"), file_item_map.size());

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	for (int idx = 0; ret == UNZ_OK; ret = unzGoToNextFile(pkg), idx++) {
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
		if (ret != UNZ_OK) {
			break;
		}

		String source_name = String::utf8(fname);
		if (!file_item_map.has(source_name) || (!file_item_map[source_name]->is_checked(0) && !file_item_map[source_name]->is_indeterminate(0))) {
			continue;
		}

		String path = file_item_map[source_name]->get_metadata(0);
		if (path.is_empty()) { // Directory.
			// TODO: Metadata can be used to store the entire path of directories too,
			// so this tree iteration can be avoided.
			String dir_path;
			TreeItem *t = file_item_map[source_name];
			while (t) {
				dir_path = t->get_text(0) + dir_path;
				t = t->get_parent();
			}

			if (dir_path.ends_with("/")) {
				dir_path = dir_path.substr(0, dir_path.length() - 1);
			}

			da->make_dir_recursive(dir_path);
		} else {
			Vector<uint8_t> uncomp_data;
			uncomp_data.resize(info.uncompressed_size);

			unzOpenCurrentFile(pkg);
			unzReadCurrentFile(pkg, uncomp_data.ptrw(), uncomp_data.size());
			unzCloseCurrentFile(pkg);

			// Ensure that the target folder exists.
			da->make_dir_recursive(path.get_base_dir());

			Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
			if (f.is_valid()) {
				f->store_buffer(uncomp_data.ptr(), uncomp_data.size());
			} else {
				failed_files.push_back(path);
			}

			ProgressDialog::get_singleton()->task_step("uncompress", path, idx);
		}
	}

	ProgressDialog::get_singleton()->end_task("uncompress");
	unzClose(pkg);

	if (failed_files.size()) {
		String msg = vformat(TTR("The following files failed extraction from asset \"%s\":"), asset_name) + "\n\n";
		for (int i = 0; i < failed_files.size(); i++) {
			if (i > 10) {
				msg += "\n" + vformat(TTR("(and %s more files)"), itos(failed_files.size() - i));
				break;
			}
			msg += "\n" + failed_files[i];
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

void EditorAssetInstaller::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			generic_extension_icon = get_editor_theme_icon(SNAME("Object"));

			extension_icon_map.clear();
			{
				extension_icon_map["bmp"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["dds"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["exr"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["hdr"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["jpg"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["jpeg"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["png"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["svg"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["tga"] = get_editor_theme_icon(SNAME("ImageTexture"));
				extension_icon_map["webp"] = get_editor_theme_icon(SNAME("ImageTexture"));

				extension_icon_map["wav"] = get_editor_theme_icon(SNAME("AudioStreamWAV"));
				extension_icon_map["ogg"] = get_editor_theme_icon(SNAME("AudioStreamOggVorbis"));
				extension_icon_map["mp3"] = get_editor_theme_icon(SNAME("AudioStreamMP3"));

				extension_icon_map["scn"] = get_editor_theme_icon(SNAME("PackedScene"));
				extension_icon_map["tscn"] = get_editor_theme_icon(SNAME("PackedScene"));
				extension_icon_map["escn"] = get_editor_theme_icon(SNAME("PackedScene"));
				extension_icon_map["dae"] = get_editor_theme_icon(SNAME("PackedScene"));
				extension_icon_map["gltf"] = get_editor_theme_icon(SNAME("PackedScene"));
				extension_icon_map["glb"] = get_editor_theme_icon(SNAME("PackedScene"));

				extension_icon_map["gdshader"] = get_editor_theme_icon(SNAME("Shader"));
				extension_icon_map["gdshaderinc"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["gd"] = get_editor_theme_icon(SNAME("GDScript"));
				if (Engine::get_singleton()->has_singleton("GodotSharp")) {
					extension_icon_map["cs"] = get_editor_theme_icon(SNAME("CSharpScript"));
				} else {
					// Mark C# support as unavailable.
					extension_icon_map["cs"] = get_editor_theme_icon(SNAME("ImportFail"));
				}

				extension_icon_map["res"] = get_editor_theme_icon(SNAME("Resource"));
				extension_icon_map["tres"] = get_editor_theme_icon(SNAME("Resource"));
				extension_icon_map["atlastex"] = get_editor_theme_icon(SNAME("AtlasTexture"));
				// By default, OBJ files are imported as Mesh resources rather than PackedScenes.
				extension_icon_map["obj"] = get_editor_theme_icon(SNAME("Mesh"));

				extension_icon_map["txt"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["md"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["rst"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["json"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["yml"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["yaml"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["toml"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["cfg"] = get_editor_theme_icon(SNAME("TextFile"));
				extension_icon_map["ini"] = get_editor_theme_icon(SNAME("TextFile"));
			}
		} break;
	}
}

void EditorAssetInstaller::_bind_methods() {
}

EditorAssetInstaller::EditorAssetInstaller() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	HBoxContainer *asset_status = memnew(HBoxContainer);
	vb->add_child(asset_status);

	Label *asset_label = memnew(Label);
	asset_label->set_text(TTR("Contents of asset:"));
	asset_status->add_child(asset_label);

	asset_title_label = memnew(Label);
	asset_title_label->set_theme_type_variation("HeaderSmall");
	asset_status->add_child(asset_title_label);

	asset_status->add_spacer();

	asset_conflicts_label = memnew(Label);
	asset_conflicts_label->set_theme_type_variation("HeaderSmall");
	asset_status->add_child(asset_conflicts_label);

	skip_toplevel_check = memnew(CheckBox);
	skip_toplevel_check->set_text(TTR("Ignore the root directory when extracting files"));
	skip_toplevel_check->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
	skip_toplevel_check->connect("toggled", callable_mp(this, &EditorAssetInstaller::_set_skip_toplevel));
	vb->add_child(skip_toplevel_check);

	tree = memnew(Tree);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->connect("item_edited", callable_mp(this, &EditorAssetInstaller::_item_checked));
	tree->connect("check_propagated_to_item", callable_mp(this, &EditorAssetInstaller::_check_propagated_to_item));
	vb->add_child(tree);

	set_title(TTR("Select Asset Files to Install"));
	set_ok_button_text(TTR("Install"));
	set_hide_on_ok(true);
}
