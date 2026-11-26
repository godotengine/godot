/**************************************************************************/
/*  project_operations.cpp                                                */
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

#include "project_operations.h"

#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/zip_io.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/themes/editor_icons.h"

Error ProjectOperations::validate_origin_zip(const String &p_path, String &r_zip_root) {
	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);

	unzFile pkg = unzOpen2(p_path.utf8().get_data(), &io);
	if (!pkg) {
		unzClose(pkg);
		return ERR_CANT_OPEN;
	}

	int ret = unzGoToFirstFile(pkg);
	while (ret == UNZ_OK) {
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
		ERR_FAIL_COND_V_MSG(ret != UNZ_OK, ERR_INVALID_DATA, "Failed to get current file info.");

		String name = String::utf8(fname);

		// Skip the __MACOSX directory created by macOS's built-in file zipper.
		if (name.begins_with("__MACOSX")) {
			ret = unzGoToNextFile(pkg);
			continue;
		}

		if (name.get_file() == "project.godot") {
			r_zip_root = name.get_base_dir();
			if (!r_zip_root.ends_with("/")) {
				r_zip_root += "/";
			}
			break; // ret == UNZ_OK.
		}

		ret = unzGoToNextFile(pkg);
	}

	if (ret == UNZ_END_OF_LIST_OF_FILE) {
		unzClose(pkg);
		return ERR_FILE_NOT_FOUND;
	}

	unzClose(pkg);
	return OK;
}

Error ProjectOperations::validate_origin_folder(const String &p_path) {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	if (d->dir_exists(p_path) && d->file_exists(p_path.path_join("project.godot"))) {
		return OK;
	}
	return FAILED;
}

Error ProjectOperations::validate_destination(const String &p_path, bool &r_is_folder_empty, bool p_create_folder, bool *r_subdir_exists) {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	String base_path = p_create_folder ? p_path.get_base_dir() : p_path;

	if (!d->dir_exists(base_path)) {
		return ERR_FILE_NOT_FOUND;
	}

	if (p_create_folder) {
		if (!d->dir_exists(p_path)) {
			if (r_subdir_exists) {
				*r_subdir_exists = false;
			}
			return OK;
		}
	}

	// Check if the directory is empty. Not an error, but we want to warn the user.
	if (d->change_dir(p_path) == OK) {
		d->list_dir_begin();
		String n = d->get_next();
		while (!n.is_empty()) {
			if (n[0] != '.') {
				// Allow `.`, `..` (reserved current/parent folder names)
				// and hidden files/folders to be present.
				// For instance, this lets users initialize a Git repository
				// and still be able to create a project in the directory afterwards.
				r_is_folder_empty = false;
				break;
			}
			n = d->get_next();
		}
		d->list_dir_end();
	}

	return OK;
}

bool ProjectOperations::is_destination_inside_origin(const String &p_destination_path, const String &p_origin_path) {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (!d->exists(p_origin_path)) {
		return false;
	}

	String base_path = p_origin_path;
	String duplicate_target = p_destination_path;

	// Ensure the paths end with a slash
	if (!base_path.ends_with("/")) {
		base_path += "/";
	}

	if (!duplicate_target.ends_with("/")) {
		duplicate_target += "/";
	}

	bool is_subdirectory_or_equal;

	if (d->is_case_sensitive(base_path) || d->is_case_sensitive(duplicate_target)) {
		is_subdirectory_or_equal = duplicate_target.begins_with(base_path);
	} else {
		base_path = base_path.to_lower();
		String target_lower = duplicate_target.to_lower();
		is_subdirectory_or_equal = target_lower.begins_with(base_path);
	}

	return is_subdirectory_or_equal;
}

Error ProjectOperations::create_project(const String &p_path, const HashMap<String, Variant> &p_settings, const EditorVCSInterface::VCSMetadata &p_vcs_metadata) {
	PackedStringArray project_features = ProjectSettings::get_required_features();
	ProjectSettings::CustomMap initial_settings;

	merge_project_settings(p_settings, initial_settings);

	Variant renderer_type = initial_settings["rendering/renderer/rendering_method"];
	if (renderer_type == "forward_plus") {
		project_features.push_back("Forward Plus");
	} else if (renderer_type == "mobile") {
		project_features.push_back("Mobile");
	} else if (renderer_type == "gl_compatibility") {
		project_features.push_back("GL Compatibility");
		// Also change the default renderer for the mobile override.
		initial_settings["rendering/renderer/rendering_method.mobile"] = "gl_compatibility";
	} else {
		WARN_PRINT("Unknown renderer type. Please report this as a bug on GitHub.");
	}

	project_features.sort();
	initial_settings["application/config/features"] = project_features;
	initial_settings["application/config/icon"] = "res://icon.svg";
	ProjectSettings::CustomMap extra_settings = EditorNode::get_initial_settings();

	merge_project_settings(extra_settings, initial_settings);

	Error err = ProjectSettings::get_singleton()->save_custom(p_path.path_join("project.godot"), initial_settings, Vector<String>(), false);
	if (err != OK) {
		return ERR_FILE_CANT_WRITE;
	}

	// Store default project icon in SVG format.
	Ref<FileAccess> fa_icon = FileAccess::open(p_path.path_join("icon.svg"), FileAccess::WRITE, &err);
	if (err != OK) {
		return ERR_FILE_CANT_OPEN;
	}
	fa_icon->store_string(get_default_project_icon());

	EditorVCSInterface::create_vcs_metadata_files(p_vcs_metadata, p_path);

	// Ensures external editors and IDEs use UTF-8 encoding.
	const String editor_config_path = p_path.path_join(".editorconfig");
	Ref<FileAccess> f = FileAccess::open(editor_config_path, FileAccess::WRITE);
	if (f.is_null()) {
		// .editorconfig isn't so critical.
		ERR_PRINT("Couldn't create .editorconfig in project path.");
	} else {
		f->store_line("root = true");
		f->store_line("");
		f->store_line("[*]");
		f->store_line("charset = utf-8");
		f->close();
		FileAccess::set_hidden_attribute(editor_config_path, true);
	}
	return OK;
}

Error ProjectOperations::duplicate_project(const String &p_origin_path, const String &p_destination_path) {
	Ref<DirAccess> dir = DirAccess::open(p_origin_path);
	if (!dir.is_valid()) {
		return FAILED;
	}
	return dir->copy_dir(".", p_destination_path, -1, true);
}

Error ProjectOperations::install_zip_project(const String &p_origin_path, const String &p_destination_path, const String &p_root_path, Vector<String> &r_failed_files) {
	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);

	unzFile pkg = unzOpen2(p_origin_path.utf8().get_data(), &io);
	if (!pkg) {
		return ERR_CANT_OPEN;
	}

	int ret = unzGoToFirstFile(pkg);

	while (ret == UNZ_OK) {
		//get filename
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
		ERR_FAIL_COND_V_MSG(ret != UNZ_OK, ERR_INVALID_DATA, "Failed to get current file info.");

		String name = String::utf8(fname);

		// Skip the __MACOSX directory created by macOS's built-in file zipper.
		if (name.begins_with("__MACOSX")) {
			ret = unzGoToNextFile(pkg);
			continue;
		}

		String rel_path = name.trim_prefix(p_root_path);
		if (rel_path.is_empty()) { // Root.
		} else if (rel_path.ends_with("/")) { // Directory.
			Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			da->make_dir(p_destination_path.path_join(rel_path));
		} else { // File.
			Vector<uint8_t> uncomp_data;
			uncomp_data.resize(info.uncompressed_size);

			unzOpenCurrentFile(pkg);
			ret = unzReadCurrentFile(pkg, uncomp_data.ptrw(), uncomp_data.size());
			ERR_BREAK_MSG(ret < 0, vformat("An error occurred while attempting to read from file: %s. This file will not be used.", rel_path));
			unzCloseCurrentFile(pkg);

			Ref<FileAccess> f = FileAccess::open(p_destination_path.path_join(rel_path), FileAccess::WRITE);
			if (f.is_valid()) {
				f->store_buffer(uncomp_data.ptr(), uncomp_data.size());
			} else {
				r_failed_files.push_back(rel_path);
			}
		}

		ret = unzGoToNextFile(pkg);
	}

	unzClose(pkg);
	return OK;
}

static bool is_zip_file(Ref<DirAccess> p_d, const String &p_path) {
	return p_path.get_extension() == "zip" && p_d->file_exists(p_path);
}

String ProjectOperations::resolve_origin_path(const String &p_path, bool &r_is_zip) {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String new_path = p_path;

	if (is_zip_file(d, new_path)) {
		r_is_zip = true;
	} else if (is_zip_file(d, new_path.strip_edges())) {
		r_is_zip = true;
		new_path = new_path.strip_edges();
	} else {
		r_is_zip = false;
	}

	return new_path;
}

Error ProjectOperations::create_project_dir(const String &p_path) {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (!d->dir_exists(p_path) && d->make_dir(p_path) != OK) {
		return FAILED;
	}
	return OK;
}

#ifdef ANDROID_ENABLED
void ProjectOperations::create_nomedia_file(const String &p_path) {
	// Create a .nomedia file to hide assets from media apps on Android.
	// Android 11 has some issues with nomedia files, so it's disabled there. See GH-106479, GH-105399 for details.
	// NOTE: Nomedia file is also handled during the first filesystem scan. See editor_file_system.cpp -> EditorFileSystem::scan().
	String sdk_version = OS::get_singleton()->get_version().get_slicec('.', 0);
	if (sdk_version != "30") {
		const String nomedia_file_path = p_path.path_join(".nomedia");
		Ref<FileAccess> f2 = FileAccess::open(nomedia_file_path, FileAccess::WRITE);
		if (f2.is_null()) {
			// .nomedia isn't so critical.
			ERR_PRINT("Couldn't create .nomedia in project path.");
		} else {
			f2->close();
		}
	}
}
#endif

void ProjectOperations::merge_project_settings(const HashMap<String, Variant> &p_added_settings, ProjectSettings::CustomMap &r_base_settings) {
	for (const KeyValue<String, Variant> &setting : p_added_settings) {
		r_base_settings[setting.key] = setting.value;
	}
}

Error ProjectOperations::edit_existing_project_settings(const HashMap<String, Variant> &p_settings, const String &p_path, Ref<ConfigFile> p_project_godot) {
	for (const KeyValue<String, Variant> &setting : p_settings) {
		PackedStringArray section_and_key = setting.key.split("/", true, 1);

		p_project_godot->set_value(section_and_key[0], section_and_key[1], setting.value);
	}

	return p_project_godot->save(p_path);
}
