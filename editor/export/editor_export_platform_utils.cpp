/**************************************************************************/
/*  editor_export_platform_utils.cpp                                      */
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

#include "editor_export_platform_utils.h"

#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/extension/gdextension.h"
#include "core/io/file_access_encrypted.h"
#include "core/io/file_access_pack.h"
#include "core/math/random_pcg.h"
#include "core/version.h"
#include "editor/export/editor_export_platform.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/file_system/editor_paths.h"

/**
 * EditorExportPlatformUtils::AsyncPckFileDependenciesState
 */
void EditorExportPlatformUtils::AsyncPckFileDependenciesState::add_to_file_dependencies(const String &p_file) {
	String file = p_file.strip_edges().simplify_path();

	if (file_dependencies.has(file)) {
		return;
	}

	List<String> dependencies;
	ResourceLoader::get_dependencies(file, &dependencies);
	for (const String &dependency : dependencies) {
		String dependency_path = EditorExportPlatformUtils::get_path_from_dependency(dependency);
		if (!file_dependencies.has(file)) {
			HashSet<String> *dependency_list = &file_dependencies_lists.push_back({})->get();
			file_dependencies[file] = dependency_list;
		}
		file_dependencies[file]->insert(dependency_path);
		add_to_file_dependencies(dependency_path);
	}
}

void EditorExportPlatformUtils::AsyncPckFileDependenciesState::add_to_file_dependencies(const HashSet<String> &p_file_set) {
	for (const String &file : p_file_set) {
		if (file.ends_with("/")) {
			continue;
		}
		add_to_file_dependencies(file);
	}
}

HashMap<String, const HashSet<String> *> EditorExportPlatformUtils::AsyncPckFileDependenciesState::get_file_dependencies_of(const HashSet<String> &p_file_set) {
	HashMap<String, const HashSet<String> *> dependencies;
	for (const String &file : p_file_set) {
		_get_file_dependencies_of(file, dependencies);
	}
	return dependencies;
}

HashMap<String, const HashSet<String> *> EditorExportPlatformUtils::AsyncPckFileDependenciesState::get_file_dependencies_of(const String &p_file) {
	HashMap<String, const HashSet<String> *> dependencies;
	_get_file_dependencies_of(p_file, dependencies);
	return dependencies;
}

void EditorExportPlatformUtils::AsyncPckFileDependenciesState::_get_file_dependencies_of(const String &p_file, HashMap<String, const HashSet<String> *> &p_dependencies) {
	if (p_dependencies.has(p_file)) {
		return;
	}
	if (!file_dependencies.has(p_file)) {
		p_dependencies[p_file] = {};
		return;
	}
	ERR_FAIL_NULL(file_dependencies[p_file]);

	p_dependencies[p_file] = file_dependencies[p_file];
	for (const String &file_dependency : *file_dependencies[p_file]) {
		_get_file_dependencies_of(file_dependency, p_dependencies);
	}
}

/**
 * EditorExportPlatformUtils
 */
String EditorExportPlatformUtils::get_path_from_dependency(const String &p_dependency) {
	String path = p_dependency;
	if (path.contains("::")) {
		return path.get_slice("::", 2);
	}
	if (path.begins_with("uid://")) {
		return ResourceUID::get_singleton()->uid_to_path(path);
	}
	return path.simplify_path();
}

int EditorExportPlatformUtils::get_pad(int p_alignment, int p_n) {
	int rest = p_n % p_alignment;
	int pad = 0;
	if (rest > 0) {
		pad = p_alignment - rest;
	};

	return pad;
}

Variant EditorExportPlatformUtils::get_project_setting(const Ref<EditorExportPreset> &p_preset, const StringName &p_name) {
	if (p_preset.is_valid()) {
		return p_preset->get_project_setting(p_name);
	} else {
		return GLOBAL_GET(p_name);
	}
}

bool EditorExportPlatformUtils::encrypt_and_store_directory(Ref<FileAccess> p_fd, EditorExportPlatformData::PackData &p_pack_data, const Vector<uint8_t> &p_key, uint64_t p_seed, uint64_t p_file_base) {
	Ref<FileAccessEncrypted> fae;
	Ref<FileAccess> fhead = p_fd;

	// Amount of files.
	fhead->store_32(p_pack_data.file_ofs.size());

	if (!p_key.is_empty()) {
		uint64_t seed = p_seed;
		fae.instantiate();
		if (fae.is_null()) {
			return false;
		}

		Vector<uint8_t> iv;
		if (seed != 0) {
			for (int i = 0; i < p_pack_data.file_ofs.size(); i++) {
				for (int64_t j = 0; j < p_pack_data.file_ofs[i].path_utf8.length(); j++) {
					seed = ((seed << 5) + seed) ^ p_pack_data.file_ofs[i].path_utf8.get_data()[j];
				}
				for (int64_t j = 0; j < p_pack_data.file_ofs[i].md5.size(); j++) {
					seed = ((seed << 5) + seed) ^ p_pack_data.file_ofs[i].md5[j];
				}
				seed = ((seed << 5) + seed) ^ (p_pack_data.file_ofs[i].ofs - p_file_base);
				seed = ((seed << 5) + seed) ^ p_pack_data.file_ofs[i].size;
			}

			RandomPCG rng = RandomPCG(seed);
			iv.resize(16);
			for (int i = 0; i < 16; i++) {
				iv.write[i] = rng.rand() % 256;
			}
		}

		Error err = fae->open_and_parse(fhead, p_key, FileAccessEncrypted::MODE_WRITE_AES256, false, iv);
		if (err != OK) {
			return false;
		}

		fhead = fae;
	}
	for (int i = 0; i < p_pack_data.file_ofs.size(); i++) {
		uint32_t string_len = p_pack_data.file_ofs[i].path_utf8.length();
		uint32_t pad = EditorExportPlatformUtils::get_pad(4, string_len);

		fhead->store_32(string_len + pad);
		fhead->store_buffer((const uint8_t *)p_pack_data.file_ofs[i].path_utf8.get_data(), string_len);
		for (uint32_t j = 0; j < pad; j++) {
			fhead->store_8(0);
		}

		fhead->store_64(p_pack_data.file_ofs[i].ofs - p_file_base);
		fhead->store_64(p_pack_data.file_ofs[i].size); // Pay attention here, this is where file is.
		fhead->store_buffer(p_pack_data.file_ofs[i].md5.ptr(), 16); // Also save md5 for file.
		uint32_t flags = 0;
		if (p_pack_data.file_ofs[i].encrypted) {
			flags |= PACK_FILE_ENCRYPTED;
		}
		if (p_pack_data.file_ofs[i].removal) {
			flags |= PACK_FILE_REMOVAL;
		}
		if (p_pack_data.file_ofs[i].delta) {
			flags |= PACK_FILE_DELTA;
		}
		fhead->store_32(flags);
	}

	if (fae.is_valid()) {
		fhead.unref();
		fae.unref();
	}
	return true;
}

Error EditorExportPlatformUtils::encrypt_and_store_data(Ref<FileAccess> p_fd, const String &p_path, const Vector<uint8_t> &p_data, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key, uint64_t p_seed, bool &r_encrypt) {
	r_encrypt = false;
	for (const String &enc_in_filter : p_enc_in_filters) {
		if (p_path.matchn(enc_in_filter) || p_path.trim_prefix("res://").matchn(enc_in_filter)) {
			r_encrypt = true;
			break;
		}
	}

	for (const String &enc_ex_filter : p_enc_ex_filters) {
		if (p_path.matchn(enc_ex_filter) || p_path.trim_prefix("res://").matchn(enc_ex_filter)) {
			r_encrypt = false;
			break;
		}
	}

	Ref<FileAccessEncrypted> fae;
	Ref<FileAccess> ftmp = p_fd;
	if (r_encrypt) {
		Vector<uint8_t> iv;
		if (p_seed != 0) {
			uint64_t seed = p_seed;

			const uint8_t *ptr = p_data.ptr();
			int64_t len = p_data.size();
			for (int64_t i = 0; i < len; i++) {
				seed = ((seed << 5) + seed) ^ ptr[i];
			}

			RandomPCG rng = RandomPCG(seed);
			iv.resize(16);
			for (int i = 0; i < 16; i++) {
				iv.write[i] = rng.rand() % 256;
			}
		}

		fae.instantiate();
		ERR_FAIL_COND_V(fae.is_null(), ERR_FILE_CANT_OPEN);

		Error err = fae->open_and_parse(ftmp, p_key, FileAccessEncrypted::MODE_WRITE_AES256, false, iv);
		ERR_FAIL_COND_V(err != OK, ERR_FILE_CANT_OPEN);
		ftmp = fae;
	}

	// Store file content.
	ftmp->store_buffer(p_data.ptr(), p_data.size());

	if (fae.is_valid()) {
		ftmp.unref();
		fae.unref();
	}
	return OK;
}

Error EditorExportPlatformUtils::store_temp_file(const String &p_simplified_path, const PackedByteArray &p_data, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const PackedByteArray &p_key, uint64_t p_seed, bool p_delta, PackedByteArray &r_enc_data, EditorExportPlatformData::SavedData &r_sd) {
	Error err = OK;
	Ref<FileAccess> ftmp = FileAccess::create_temp(FileAccess::WRITE_READ, "export", "tmp", false, &err);
	if (err != OK) {
		return err;
	}
	r_sd.path_utf8 = p_simplified_path.trim_prefix("res://").utf8();
	r_sd.ofs = 0;
	r_sd.size = p_data.size();
	r_sd.delta = p_delta;
	err = EditorExportPlatformUtils::encrypt_and_store_data(ftmp, p_simplified_path, p_data, p_enc_in_filters, p_enc_ex_filters, p_key, p_seed, r_sd.encrypted);
	if (err != OK) {
		return err;
	}

	r_enc_data.resize(ftmp->get_length());
	ftmp->seek(0);
	ftmp->get_buffer(r_enc_data.ptrw(), r_enc_data.size());
	ftmp.unref();

	// Store MD5 of original file.
	{
		unsigned char hash[16];
		CryptoCore::md5(p_data.ptr(), p_data.size(), hash);
		r_sd.md5.resize(16);
		for (int i = 0; i < 16; i++) {
			r_sd.md5.write[i] = hash[i];
		}
	}
	return OK;
}

// Utility method used to create a directory.
Error EditorExportPlatformUtils::create_directory(const String &p_dir) {
	if (DirAccess::exists(p_dir)) {
		return OK;
	}
	Ref<DirAccess> filesystem_da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	ERR_FAIL_COND_V_MSG(filesystem_da.is_null(), ERR_CANT_CREATE, "Cannot create directory '" + p_dir + "'.");
	Error err = filesystem_da->make_dir_recursive(p_dir);
	ERR_FAIL_COND_V_MSG(err, ERR_CANT_CREATE, "Cannot create directory '" + p_dir + "'.");
	return OK;
}

// Writes p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error EditorExportPlatformUtils::store_file_at_path(const String &p_path, const PackedByteArray &p_data) {
	String dir = p_path.get_base_dir();
	Error err = EditorExportPlatformUtils::create_directory(dir);
	if (err != OK) {
		return err;
	}
	Ref<FileAccess> fa = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(fa.is_null(), ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");
	fa->store_buffer(p_data.ptr(), p_data.size());
	return OK;
}

// Writes string p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error EditorExportPlatformUtils::store_string_at_path(const String &p_path, const String &p_data) {
	String dir = p_path.get_base_dir();
	Error err = EditorExportPlatformUtils::create_directory(dir);
	if (err != OK) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_error("Unable to write data into " + p_path);
		}
		return err;
	}
	Ref<FileAccess> fa = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(fa.is_null(), ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");
	fa->store_string(p_data);
	return OK;
}

PackedByteArray EditorExportPlatformUtils::convert_string_encryption_key_to_bytes(const String &p_encryption_key) {
	PackedByteArray key;
	key.resize_initialized(32);
	ERR_FAIL_COND_V(p_encryption_key.length() != 64, key);

	for (int i = 0; i < 32; i++) {
		int v = 0;
		if (i * 2 < p_encryption_key.length()) {
			char32_t ct = p_encryption_key[i * 2];
			if (is_digit(ct)) {
				ct = ct - '0';
			} else if (ct >= 'a' && ct <= 'f') {
				ct = 10 + ct - 'a';
			}
			v |= ct << 4;
		}

		if (i * 2 + 1 < p_encryption_key.length()) {
			char32_t ct = p_encryption_key[i * 2 + 1];
			if (is_digit(ct)) {
				ct = ct - '0';
			} else if (ct >= 'a' && ct <= 'f') {
				ct = 10 + ct - 'a';
			}
			v |= ct;
		}
		key.write[i] = v;
	}

	return key;
}

void EditorExportPlatformUtils::export_find_resources(EditorFileSystemDirectory *p_dir, HashSet<String> &p_paths) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		EditorExportPlatformUtils::export_find_resources(p_dir->get_subdir(i), p_paths);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "TextFile") {
			continue;
		}
		p_paths.insert(p_dir->get_file_path(i));
	}
}

void EditorExportPlatformUtils::export_find_customized_resources(const Ref<EditorExportPreset> &p_preset, EditorFileSystemDirectory *p_dir, EditorExportPreset::FileExportMode p_mode, HashSet<String> &p_paths) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		EditorFileSystemDirectory *subdir = p_dir->get_subdir(i);
		EditorExportPlatformUtils::export_find_customized_resources(p_preset, subdir, p_preset->get_file_export_mode(subdir->get_path(), p_mode), p_paths);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "TextFile") {
			continue;
		}
		String path = p_dir->get_file_path(i);
		EditorExportPreset::FileExportMode file_mode = p_preset->get_file_export_mode(path, p_mode);
		if (file_mode != EditorExportPreset::MODE_FILE_REMOVE) {
			p_paths.insert(path);
		}
	}
}

void EditorExportPlatformUtils::export_find_dependencies(const String &p_path, HashSet<String> &p_paths) {
	if (p_paths.has(p_path)) {
		return;
	}

	p_paths.insert(p_path);

	EditorFileSystemDirectory *dir;
	int file_idx;
	dir = EditorFileSystem::get_singleton()->find_file(p_path, &file_idx);
	if (!dir) {
		return;
	}

	PackedStringArray deps = dir->get_file_deps(file_idx);

	for (const String &dep : deps) {
		EditorExportPlatformUtils::export_find_dependencies(dep, p_paths);
	}
}

void EditorExportPlatformUtils::export_find_preset_resources(const Ref<EditorExportPreset> &p_preset, HashSet<String> &p_paths) {
	EditorExportPreset::ExportFilter export_filter = p_preset->get_export_filter();
	switch (export_filter) {
		case EditorExportPreset::EXPORT_ALL_RESOURCES:
		case EditorExportPreset::EXCLUDE_SELECTED_RESOURCES: {
			EditorExportPlatformUtils::export_find_resources(EditorFileSystem::get_singleton()->get_filesystem(), p_paths);

			if (export_filter == EditorExportPreset::EXCLUDE_SELECTED_RESOURCES) {
				PackedStringArray excluded_resources = p_preset->get_files_to_export();
				for (const String &excluded_resource : excluded_resources) {
					p_paths.erase(excluded_resource);
				}
			}
		} break;

		case EditorExportPreset::EXPORT_CUSTOMIZED: {
			EditorExportPlatformUtils::export_find_customized_resources(p_preset, EditorFileSystem::get_singleton()->get_filesystem(), p_preset->get_file_export_mode("res://"), p_paths);
		} break;

		case EditorExportPreset::EXPORT_SELECTED_SCENES:
		case EditorExportPreset::EXPORT_SELECTED_RESOURCES: {
			bool scenes_only = export_filter == EditorExportPreset::EXPORT_SELECTED_SCENES;

			PackedStringArray files = p_preset->get_files_to_export();
			for (const String &file : files) {
				if (scenes_only && ResourceLoader::get_resource_type(file) != "PackedScene") {
					continue;
				}

				EditorExportPlatformUtils::export_find_dependencies(file, p_paths);
			}

			// Add autoload resources and their dependencies
			List<PropertyInfo> props;
			ProjectSettings::get_singleton()->get_property_list(&props);

			for (const PropertyInfo &pi : props) {
				if (!pi.name.begins_with("autoload/")) {
					continue;
				}

				String autoload_path = EditorExportPlatformUtils::get_project_setting(p_preset, pi.name);

				if (autoload_path.begins_with("*")) {
					autoload_path = autoload_path.substr(1);
				}

				EditorExportPlatformUtils::export_find_dependencies(autoload_path, p_paths);
			}
		} break;
	}

	//add native icons to non-resource include list
	EditorExportPlatformUtils::edit_filter_list(p_paths, String("*.icns"), false);
	EditorExportPlatformUtils::edit_filter_list(p_paths, String("*.ico"), false);

	EditorExportPlatformUtils::edit_filter_list(p_paths, p_preset->get_include_filter(), false);
	EditorExportPlatformUtils::edit_filter_list(p_paths, p_preset->get_exclude_filter(), true);

	// Ignore import files, since these are automatically added to the jar later with the resources
	EditorExportPlatformUtils::edit_filter_list(p_paths, String("*.import"), true);
}

void EditorExportPlatformUtils::edit_files_with_filter(Ref<DirAccess> &da, const Vector<String> &p_filters, HashSet<String> &r_list, bool exclude) {
	da->list_dir_begin();
	String cur_dir = da->get_current_dir().replace_char('\\', '/');
	if (!cur_dir.ends_with("/")) {
		cur_dir += "/";
	}
	String cur_dir_no_prefix = cur_dir.replace("res://", "");

	Vector<String> dirs;
	String f = da->get_next();
	while (!f.is_empty()) {
		if (da->current_is_dir()) {
			dirs.push_back(f);
		} else {
			String fullpath = cur_dir + f;
			// Test also against path without res:// so that filters like `file.txt` can work.
			String fullpath_no_prefix = cur_dir_no_prefix + f;
			for (const String &filter : p_filters) {
				if (fullpath.matchn(filter) || fullpath_no_prefix.matchn(filter)) {
					if (!exclude) {
						r_list.insert(fullpath);
					} else {
						r_list.erase(fullpath);
					}
				}
			}
		}
		f = da->get_next();
	}

	da->list_dir_end();

	for (const String &dir : dirs) {
		if (dir.begins_with(".")) {
			continue;
		}

		if (EditorFileSystem::_should_skip_directory(cur_dir + dir)) {
			continue;
		}

		da->change_dir(dir);
		EditorExportPlatformUtils::edit_files_with_filter(da, p_filters, r_list, exclude);
		da->change_dir("..");
	}
}

void EditorExportPlatformUtils::edit_filter_list(HashSet<String> &r_list, const String &p_filter, bool exclude) {
	if (p_filter.is_empty()) {
		return;
	}
	Vector<String> split = p_filter.split(",");
	Vector<String> filters;
	for (int i = 0; i < split.size(); i++) {
		String f = split[i].strip_edges();
		if (f.is_empty()) {
			continue;
		}
		filters.push_back(f);
	}

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	ERR_FAIL_COND(da.is_null());
	EditorExportPlatformUtils::edit_files_with_filter(da, filters, r_list, exclude);
}

Vector<uint8_t> EditorExportPlatformUtils::filter_extension_list_config_file(const String &p_config_path, const HashSet<String> &p_paths) {
	Ref<FileAccess> f = FileAccess::open(p_config_path, FileAccess::READ);
	if (f.is_null()) {
		ERR_FAIL_V_MSG(Vector<uint8_t>(), "Can't open file from path '" + String(p_config_path) + "'.");
	}
	Vector<uint8_t> data;
	while (!f->eof_reached()) {
		String l = f->get_line().strip_edges();
		if (p_paths.has(l)) {
			data.append_array(l.to_utf8_buffer());
			data.append('\n');
		}
	}
	return data;
}

Vector<String> EditorExportPlatformUtils::get_forced_export_files(const Ref<EditorExportPreset> &p_preset) {
	Vector<String> files;

	files.push_back(ProjectSettings::get_singleton()->get_global_class_list_path());

	String icon = ResourceUID::ensure_path(get_project_setting(p_preset, "application/config/icon"));
	String splash = ResourceUID::ensure_path(get_project_setting(p_preset, "application/boot_splash/image"));
	if (!icon.is_empty() && FileAccess::exists(icon)) {
		files.push_back(icon);
	}
	if (!splash.is_empty() && FileAccess::exists(splash) && icon != splash) {
		files.push_back(splash);
	}
	String resource_cache_file = ResourceUID::get_cache_file();
	if (FileAccess::exists(resource_cache_file)) {
		files.push_back(resource_cache_file);
	}

	String extension_list_config_file = GDExtension::get_extension_list_config_file();
	if (FileAccess::exists(extension_list_config_file)) {
		files.push_back(extension_list_config_file);
	}

	return files;
}

HashMap<String, PackedByteArray> EditorExportPlatformUtils::get_internal_export_files(const Ref<EditorExportPlatform> &p_editor_export_platform, const Ref<EditorExportPreset> &p_preset, bool p_debug) {
	HashMap<String, PackedByteArray> files;

	// Text server support data.
	if (TS->has_feature(TextServer::FEATURE_USE_SUPPORT_DATA)) {
		bool include_data = (bool)get_project_setting(p_preset, "internationalization/locale/include_text_server_data");
		if (!include_data) {
			Vector<String> translations = get_project_setting(p_preset, "internationalization/locale/translations");
			translations.push_back(get_project_setting(p_preset, "internationalization/locale/fallback"));
			for (const String &t : translations) {
				if (TS->is_locale_using_support_data(t)) {
					include_data = true;
					break;
				}
			}
		}
		if (include_data) {
			String ts_name = TS->get_support_data_filename();
			String ts_target = "res://" + ts_name;
			if (!ts_name.is_empty()) {
				bool export_ok = false;
				if (FileAccess::exists(ts_target)) { // Include user supplied data file.
					const PackedByteArray &ts_data = FileAccess::get_file_as_bytes(ts_target);
					if (!ts_data.is_empty()) {
						p_editor_export_platform->add_message(EditorExportPlatformData::EXPORT_MESSAGE_INFO, TTR("Export"), TTR("Using user provided text server data, text display in the exported project might be broken if export template was built with different ICU version!"));
						files[ts_target] = ts_data;
						export_ok = true;
					}
				} else {
					String current_version = GODOT_VERSION_FULL_CONFIG;
					String template_path = EditorPaths::get_singleton()->get_export_templates_dir().path_join(current_version);
					if (p_debug && p_preset->has("custom_template/debug") && p_preset->get("custom_template/debug") != "") {
						template_path = p_preset->get("custom_template/debug").operator String().get_base_dir();
					} else if (!p_debug && p_preset->has("custom_template/release") && p_preset->get("custom_template/release") != "") {
						template_path = p_preset->get("custom_template/release").operator String().get_base_dir();
					}
					String data_file_name = template_path.path_join(ts_name);
					if (FileAccess::exists(data_file_name)) {
						const PackedByteArray &ts_data = FileAccess::get_file_as_bytes(data_file_name);
						if (!ts_data.is_empty()) {
							print_line("Using text server data from export templates.");
							files[ts_target] = ts_data;
							export_ok = true;
						}
					} else {
						const PackedByteArray &ts_data = TS->get_support_data();
						if (!ts_data.is_empty()) {
							p_editor_export_platform->add_message(EditorExportPlatformData::EXPORT_MESSAGE_INFO, TTR("Export"), TTR("Using editor embedded text server data, text display in the exported project might be broken if export template was built with different ICU version!"));
							files[ts_target] = ts_data;
							export_ok = true;
						}
					}
				}
				if (!export_ok) {
					p_editor_export_platform->add_message(EditorExportPlatformData::EXPORT_MESSAGE_WARNING, TTR("Export"), TTR("Missing text server data, text display in the exported project might be broken!"));
				}
			}
		}
	}

	return files;
}
