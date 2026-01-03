/**************************************************************************/
/*  editor_file_system.cpp                                                */
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

#include "editor_file_system.h"

#include "core/config/project_settings.h"
#include "core/extension/gdextension_manager.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "core/object/worker_thread_pool.h"
#include "core/os/os.h"
#include "core/variant/variant_parser.h"
#include "editor/doc/editor_help.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_paths.h"
#include "editor/inspector/editor_resource_preview.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "editor/settings/project_settings_editor.h"
#include "scene/resources/packed_scene.h"

EditorFileSystem *EditorFileSystem::singleton = nullptr;
int EditorFileSystem::nb_files_total = 0;
EditorFileSystem::ScannedDirectory *EditorFileSystem::first_scan_root_dir = nullptr;

//the name is the version, to keep compatibility with different versions of Godot
#define CACHE_FILE_NAME "filesystem_cache10"

int EditorFileSystemDirectory::find_file_index(const String &p_file) const {
	for (int i = 0; i < files.size(); i++) {
		if (files[i]->file == p_file) {
			return i;
		}
	}
	return -1;
}

int EditorFileSystemDirectory::find_dir_index(const String &p_dir) const {
	for (int i = 0; i < subdirs.size(); i++) {
		if (subdirs[i]->name == p_dir) {
			return i;
		}
	}

	return -1;
}

void EditorFileSystemDirectory::force_update() {
	// We set modified_time to 0 to force `EditorFileSystem::_scan_fs_changes` to search changes in the directory
	modified_time = 0;
}

int EditorFileSystemDirectory::get_subdir_count() const {
	return subdirs.size();
}

EditorFileSystemDirectory *EditorFileSystemDirectory::get_subdir(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, subdirs.size(), nullptr);
	return subdirs[p_idx];
}

int EditorFileSystemDirectory::get_file_count() const {
	return files.size();
}

String EditorFileSystemDirectory::get_file(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, files.size(), "");

	return files[p_idx]->file;
}

String EditorFileSystemDirectory::get_path() const {
	int parents = 0;
	const EditorFileSystemDirectory *efd = this;
	// Determine the level of nesting.
	while (efd->parent) {
		parents++;
		efd = efd->parent;
	}

	if (parents == 0) {
		return "res://";
	}

	// Using PackedStringArray, because the path is built in reverse order.
	PackedStringArray path_bits;
	// Allocate an array based on nesting. It will store path bits.
	path_bits.resize(parents + 2); // Last String is empty, so paths end with /.
	String *path_write = path_bits.ptrw();
	path_write[0] = "res:/";

	efd = this;
	for (int i = parents; i > 0; i--) {
		path_write[i] = efd->name;
		efd = efd->parent;
	}
	return String("/").join(path_bits);
}

String EditorFileSystemDirectory::get_file_path(int p_idx) const {
	return get_path().path_join(get_file(p_idx));
}

Vector<String> EditorFileSystemDirectory::get_file_deps(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, files.size(), Vector<String>());
	Vector<String> deps;

	for (int i = 0; i < files[p_idx]->deps.size(); i++) {
		String dep = files[p_idx]->deps[i];
		int sep_idx = dep.find("::"); //may contain type information, unwanted
		if (sep_idx != -1) {
			dep = dep.substr(0, sep_idx);
		}
		ResourceUID::ID uid = ResourceUID::get_singleton()->text_to_id(dep);
		if (uid != ResourceUID::INVALID_ID) {
			//return proper dependency resource from uid
			if (ResourceUID::get_singleton()->has_id(uid)) {
				dep = ResourceUID::get_singleton()->get_id_path(uid);
			} else {
				continue;
			}
		}
		deps.push_back(dep);
	}
	return deps;
}

bool EditorFileSystemDirectory::get_file_import_is_valid(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, files.size(), false);
	return files[p_idx]->import_valid;
}

uint64_t EditorFileSystemDirectory::get_file_modified_time(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, files.size(), 0);
	return files[p_idx]->modified_time;
}

uint64_t EditorFileSystemDirectory::get_file_import_modified_time(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, files.size(), 0);
	return files[p_idx]->import_modified_time;
}

String EditorFileSystemDirectory::get_file_script_class_name(int p_idx) const {
	return files[p_idx]->class_info.name;
}

String EditorFileSystemDirectory::get_file_script_class_extends(int p_idx) const {
	return files[p_idx]->class_info.extends;
}

String EditorFileSystemDirectory::get_file_script_class_icon_path(int p_idx) const {
	return files[p_idx]->class_info.icon_path;
}

String EditorFileSystemDirectory::get_file_icon_path(int p_idx) const {
	return files[p_idx]->class_info.icon_path;
}

StringName EditorFileSystemDirectory::get_file_type(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, files.size(), "");
	return files[p_idx]->type;
}

StringName EditorFileSystemDirectory::get_file_resource_script_class(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, files.size(), "");
	return files[p_idx]->resource_script_class;
}

String EditorFileSystemDirectory::get_name() {
	return name;
}

EditorFileSystemDirectory *EditorFileSystemDirectory::get_parent() {
	return parent;
}

void EditorFileSystemDirectory::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_subdir_count"), &EditorFileSystemDirectory::get_subdir_count);
	ClassDB::bind_method(D_METHOD("get_subdir", "idx"), &EditorFileSystemDirectory::get_subdir);
	ClassDB::bind_method(D_METHOD("get_file_count"), &EditorFileSystemDirectory::get_file_count);
	ClassDB::bind_method(D_METHOD("get_file", "idx"), &EditorFileSystemDirectory::get_file);
	ClassDB::bind_method(D_METHOD("get_file_path", "idx"), &EditorFileSystemDirectory::get_file_path);
	ClassDB::bind_method(D_METHOD("get_file_type", "idx"), &EditorFileSystemDirectory::get_file_type);
	ClassDB::bind_method(D_METHOD("get_file_script_class_name", "idx"), &EditorFileSystemDirectory::get_file_script_class_name);
	ClassDB::bind_method(D_METHOD("get_file_script_class_extends", "idx"), &EditorFileSystemDirectory::get_file_script_class_extends);
	ClassDB::bind_method(D_METHOD("get_file_import_is_valid", "idx"), &EditorFileSystemDirectory::get_file_import_is_valid);
	ClassDB::bind_method(D_METHOD("get_name"), &EditorFileSystemDirectory::get_name);
	ClassDB::bind_method(D_METHOD("get_path"), &EditorFileSystemDirectory::get_path);
	ClassDB::bind_method(D_METHOD("get_parent"), &EditorFileSystemDirectory::get_parent);
	ClassDB::bind_method(D_METHOD("find_file_index", "name"), &EditorFileSystemDirectory::find_file_index);
	ClassDB::bind_method(D_METHOD("find_dir_index", "name"), &EditorFileSystemDirectory::find_dir_index);
}

EditorFileSystemDirectory::EditorFileSystemDirectory() {
	modified_time = 0;
	parent = nullptr;
}

EditorFileSystemDirectory::~EditorFileSystemDirectory() {
	for (FileInfo *fi : files) {
		memdelete(fi);
	}

	for (EditorFileSystemDirectory *dir : subdirs) {
		memdelete(dir);
	}
}

EditorFileSystem::ScannedDirectory::~ScannedDirectory() {
	for (ScannedDirectory *dir : subdirs) {
		memdelete(dir);
	}
}

void EditorFileSystem::_load_first_scan_root_dir() {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	first_scan_root_dir = memnew(ScannedDirectory);
	first_scan_root_dir->full_path = "res://";

	nb_files_total = _scan_new_dir(first_scan_root_dir, d);
}

void EditorFileSystem::scan_for_uid() {
	// Load file structure into memory.
	_load_first_scan_root_dir();

	// Load extensions for which an .import should exists.
	List<String> extensionsl;
	HashSet<String> import_extensions;
	ResourceFormatImporter::get_singleton()->get_recognized_extensions(&extensionsl);
	for (const String &E : extensionsl) {
		import_extensions.insert(E);
	}

	// Scan the file system to load uid.
	_scan_for_uid_directory(first_scan_root_dir, import_extensions);

	// It's done, resetting the callback method to prevent a second scan.
	ResourceUID::scan_for_uid_on_startup = nullptr;
}

void EditorFileSystem::_scan_for_uid_directory(const ScannedDirectory *p_scan_dir, const HashSet<String> &p_import_extensions) {
	for (ScannedDirectory *scan_sub_dir : p_scan_dir->subdirs) {
		_scan_for_uid_directory(scan_sub_dir, p_import_extensions);
	}

	for (const String &scan_file : p_scan_dir->files) {
		const String ext = scan_file.get_extension().to_lower();

		if (ext == "uid" || ext == "import") {
			continue;
		}

		const String path = p_scan_dir->full_path.path_join(scan_file);
		ResourceUID::ID uid = ResourceUID::INVALID_ID;
		if (_validate_file_extension(scan_file, p_import_extensions)) {
			if (FileAccess::exists(path + ".import")) {
				uid = ResourceFormatImporter::get_singleton()->get_resource_uid(path);
			}
		} else {
			uid = ResourceLoader::get_resource_uid(path);
		}

		if (uid != ResourceUID::INVALID_ID) {
			if (!ResourceUID::get_singleton()->has_id(uid)) {
				ResourceUID::get_singleton()->add_id(uid, path);
			}
		}
	}
}

void EditorFileSystem::_update_global_script_class_activation() {
	Vector<StringName> to_removes;
	for (KeyValue<StringName, ScriptClassAlternatives> &E : global_script_class_alternatives) {
		ScriptClassAlternatives &scas = E.value;
		if (scas.alternatives.is_empty()) {
			to_removes.push_back(E.key);
			continue;
		}
		if (!first_scan && scas.active) {
			continue;
		}
		if (!scas.active) {
			script_classes_updated = true;
			if (scas.active_uid != ResourceUID::INVALID_ID) {
				for (KeyValue<String, EditorFileInfo *> &F : scas.alternatives) {
					if (F.value->uid == scas.active_uid) { // Think of it as a file move.
						scas.active_path = F.key;
						scas.active = true;
						break;
					}
				}
			}
			if (!scas.active) {
				scas.active = true;
				if (scas.alternatives.has(scas.active_path)) { // Maybe the internal files have changed.
					scas.active_uid = scas.alternatives[scas.active_path]->uid;
				} else { // Seems to have been removed.
					scas.active_path = scas.alternatives.begin()->key;
					scas.active_uid = scas.alternatives.begin()->value->uid;
				}
			}
		}
		EditorFileInfo *fi = scas.alternatives[scas.active_path];
		ERR_CONTINUE(fi == nullptr);
		fi->status |= EditorFileInfo::IS_ACTIVE_GLOBAL_CLASS_ALTERNATIVE;
		_check_loader_or_saver_changed(fi->class_info);
		ScriptServer::add_global_class(fi->class_info.name, fi->class_info.extends, fi->class_info.lang, scas.active_path, fi->class_info.is_abstract, fi->class_info.is_tool, fi->uid);
		EditorNode::get_editor_data().script_class_set_icon_path(fi->class_info.name, fi->class_info.icon_path);
		EditorNode::get_editor_data().script_class_set_name(scas.active_path, fi->class_info.name);
		{
			MutexLock update_script_lock(update_script_mutex);
			update_script_paths_documentation.insert(scas.active_path);
		}
	}
	for (StringName &E : to_removes) {
		script_classes_updated = true;
		global_script_class_alternatives.erase(E);
		EditorNode::get_editor_data().script_class_clear_icon_path(E);
	}
	_reload_loader_or_saver();
}

void EditorFileSystem::_first_scan_filesystem() {
	EditorProgress ep = EditorProgress("first_scan_filesystem", TTR("Project initialization"), 5);
	HashSet<String> existing_class_names;
	HashSet<String> extensions;

	if (!first_scan_root_dir) {
		ep.step(TTR("Scanning file structure..."), 0, true);
		_load_first_scan_root_dir();
	}

	// Preloading GDExtensions file extensions to prevent looping on all the resource loaders
	// for each files in _first_scan_process_scripts.
	List<String> gdextension_extensions;
	ResourceLoader::get_recognized_extensions_for_type("GDExtension", &gdextension_extensions);

	// This loads the global class names from the scripts and ensures that even if the
	// global_script_class_cache.cfg was missing or invalid, the global class names are valid in ScriptServer.
	// At the same time, to prevent looping multiple times in all files, it looks for extensions.
	ep.step(TTR("Loading global class names..."), 1, true);
	_first_scan_process_scripts(first_scan_root_dir, gdextension_extensions, existing_class_names, extensions);
	_update_global_script_class_activation();

	// Removing invalid global class to prevent having invalid paths in ScriptServer.
	bool save_scripts = _remove_invalid_global_class_names(existing_class_names);

	// If a global class is found or removed, we sync global_script_class_cache.cfg with the ScriptServer
	if (!existing_class_names.is_empty() || save_scripts) {
		EditorNode::get_editor_data().script_class_save_global_classes();
	}

	// Processing extensions to add new extensions or remove invalid ones.
	// Important to do it in the first scan so custom types, new class names, custom importers, etc...
	// from extensions are ready to go before plugins, autoloads and resources validation/importation.
	// At this point, a restart of the editor should not be needed so we don't use the return value.
	ep.step(TTR("Verifying GDExtensions..."), 2, true);
	GDExtensionManager::get_singleton()->ensure_extensions_loaded(extensions);

	// Now that all the global class names should be loaded, create autoloads and plugins.
	// This is done after loading the global class names because autoloads and plugins can use
	// global class names.
	ep.step(TTR("Creating autoload scripts..."), 3, true);
	ProjectSettingsEditor::get_singleton()->init_autoloads();

	ep.step(TTR("Initializing plugins..."), 4, true);
	EditorNode::get_singleton()->init_plugins();

	ep.step(TTR("Starting file scan..."), 5, true);
	update_extensions();
}

void EditorFileSystem::_first_scan_process_scripts(const ScannedDirectory *p_scan_dir, List<String> &p_gdextension_extensions, HashSet<String> &p_existing_class_names, HashSet<String> &p_extensions) {
	for (ScannedDirectory *scan_sub_dir : p_scan_dir->subdirs) {
		_first_scan_process_scripts(scan_sub_dir, p_gdextension_extensions, p_existing_class_names, p_extensions);
	}

	for (const String &scan_file : p_scan_dir->files) {
		// Optimization to skip the ResourceLoader::get_resource_type for files
		// that are not scripts. Some loader get_resource_type methods read the file
		// which can be very slow on large projects.
		const String ext = scan_file.get_extension().to_lower();
		bool is_script = false;
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			if (ScriptServer::get_language(i)->get_extension() == ext) {
				is_script = true;
				break;
			}
		}

		bool is_gdextension = p_gdextension_extensions.find(ext) != nullptr; // Check for GDExtensions.

		if (is_script || is_gdextension) {
			const String path = p_scan_dir->full_path.path_join(scan_file);
			const String type = ResourceLoader::get_resource_type(path);

			if (is_script && ClassDB::is_parent_class(type, Script::get_class_static())) {
				const ScriptClassInfo &info = _get_global_script_class(type, path);

				EditorFileInfo *fi = memnew(EditorFileInfo);
				fi->status |= EditorFileInfo::IS_SCRIPT | EditorFileInfo::IS_ORPHAN;
				fi->file = scan_file;
				fi->type = type;
				_script_class_info_update(fi, path, &info);
				fi->uid = ResourceLoader::get_resource_uid(path);
				script_file_info[path] = fi;

				if (info.name.is_empty() || info.lang.is_empty()) {
					continue;
				}
				p_existing_class_names.insert(info.name);

				if (ScriptServer::is_global_class(info.name)) {
					ScriptClassAlternatives &scas = global_script_class_alternatives[info.name];
					if (scas.active_path.is_empty()) { // Query only once.
						scas.active_path = ScriptServer::get_global_class_path(info.name);
						scas.active_uid = ScriptServer::get_global_class_uid(info.name);
					}
					if (!scas.active && scas.active_uid == fi->uid && scas.active_path == path) {
						scas.active = true; // A perfect match.
					}
				}
			} else if (is_gdextension && type == GDExtension::get_class_static()) {
				p_extensions.insert(path);
			}
		}
	}
}
// Restore the file info cache to the last saved state from the cache file.
bool EditorFileSystem::_load_filesystem_from_cache() {
	const String fscache = EditorPaths::get_singleton()->get_project_settings_dir().path_join(CACHE_FILE_NAME);
	Ref<FileAccess> f = FileAccess::open(fscache, FileAccess::READ);
	if (f.is_null()) {
		return false;
	}

	// Read the disk cache.
	Vector<String> dirs;
	String cpath;
	EditorFileSystemDirectory *cur_dir = nullptr;
	nb_files_total = 0;
	bool is_first_line = true;
	while (!f->eof_reached()) {
		String l = f->get_line().strip_edges();
		if (is_first_line) {
			filesystem_settings_version_for_import = l;
			revalidate_import_files = filesystem_settings_version_for_import != ResourceFormatImporter::get_singleton()->get_import_settings_hash();
			is_first_line = false;
			continue;
		}
		if (l.is_empty()) {
			continue;
		}

		// Directory entries.
		if (l.begins_with("::")) {
			Vector<String> split = l.split("::");
			ERR_CONTINUE(split.size() != 3);
			cpath = split[1];
			if (cpath == "res://") {
				cur_dir = filesystem;
				cur_dir->modified_time = split[2].to_int();
				continue;
			}

			EditorFileSystemDirectory *parent = cur_dir;
			Vector<String> prev_dirs = dirs;
			dirs = cpath.substr(6, cpath.length() - 7).split("/");

			int common_ancestor_count = 0;
			int amount = prev_dirs.size();
			bool fetch = false;
			while (common_ancestor_count != amount) {
				if (fetch) {
					parent = parent->parent;
					amount--;
				} else {
					fetch = prev_dirs[common_ancestor_count] != dirs[common_ancestor_count];
					if (fetch) {
						continue;
					}
					fetch = common_ancestor_count == dirs.size() - 1;
					common_ancestor_count++;
				}
			}

			while (common_ancestor_count < dirs.size() - 1) {
				int idx = parent->find_dir_index(dirs[common_ancestor_count]);
				if (idx == -1) {
					EditorFileSystemDirectory *lost = memnew(EditorFileSystemDirectory);
					lost->name = dirs[common_ancestor_count];
					lost->parent = parent;
					lost->parent->subdirs.push_back(lost);
				} else {
					parent = parent->get_subdir(idx);
				}
				common_ancestor_count++;
			}

			cur_dir = memnew(EditorFileSystemDirectory);
			cur_dir->name = dirs[dirs.size() - 1];
			cur_dir->parent = parent;
			cur_dir->parent->subdirs.push_back(cur_dir);
			cur_dir->modified_time = split[2].to_int();
			continue;
		}

		// The last section (deps) may contain the same splitter, so limit the maxsplit to 8 to get the complete deps.
		Vector<String> split = l.split("::", true, 8);
		ERR_CONTINUE(split.size() < 9);

		const String path = cpath.path_join(split[0]);
		const bool is_reused = script_file_info.has(path);

		EditorFileInfo *fi = is_reused ? script_file_info[path] : memnew(EditorFileInfo);
		cur_dir->files.push_back(fi);
		fi->parent = cur_dir;
		fi->status &= ~(EditorFileInfo::FILE_ADD | EditorFileInfo::IS_ORPHAN);

		fi->file = split[0];
		_category_validate(fi, path);

		if (is_reused) {
			const StringName old_type = split[1].get_slicec('/', 0);
			if (old_type != fi->type) {
				if (!old_type.is_empty()) {
					fi->status |= EditorFileInfo::TYPE_REMOVE;
				}
				if (!fi->type.is_empty()) {
					fi->status |= EditorFileInfo::TYPE_ADD;
				}
			}
			_create_actions_from_uid_change(fi, path, split[2].to_int());
		} else {
			fi->type = split[1].get_slicec('/', 0);
			if (fi->type.is_empty() || fi->type == "OtherFile" || fi->type == "TextFile") {
				fi->status &= ~EditorFileInfo::AS_RESOURCE;
			} else if (fi->type == PackedScene::get_class_static()) {
				fi->status |= EditorFileInfo::IS_PACKEDSCENE;
			} else if (ClassDB::is_parent_class(fi->type, Script::get_class_static())) {
				fi->status |= EditorFileInfo::IS_SCRIPT;
			}
			fi->uid = split[2].to_int();
		}

		fi->resource_script_class = split[1].get_slicec('/', 1);
		fi->modified_time = split[3].to_int();
		fi->import_modified_time = split[4].to_int();
		fi->import_valid = split[5].to_int() != 0;
		fi->import_group_file = split[6].strip_edges();
		{
			const Vector<String> &slices = split[7].split("<>");
			ERR_CONTINUE(slices.size() < 7);
			if (!is_reused) {
				fi->class_info.name = slices[0];
				fi->class_info.extends = slices[1];
				fi->class_info.icon_path = slices[2];
				fi->class_info.is_abstract = slices[3].to_int();
				fi->class_info.is_tool = slices[4].to_int();
			}
			fi->import_md5 = slices[5];
			fi->import_dest_paths = slices[6].split("<*>", false); // Make sure the path is not empty.
		}
		fi->deps = split[8].strip_edges().split("<>", false);

		nb_files_total++;
	}

	return true;
}

void EditorFileSystem::_scan_filesystem() {
	// On the first scan, the first_scan_root_dir is created in _first_scan_filesystem.
	ERR_FAIL_COND(!scanning || (first_scan && !first_scan_root_dir));

	bool update = !first_scan || _load_filesystem_from_cache();

	EditorProgressBG scan_progress("efs", "ScanFS", 1000);
	ScanProgress sp;
	sp.hi = nb_files_total;
	sp.progress = &scan_progress;

	// On the first scan, the first_scan_root_dir is created in _first_scan_filesystem.
	if (first_scan) {
		ResourceUID::scan_for_uid_on_startup = nullptr;
	}

	if (update) {
		_scan_fs_changes(filesystem, sp);
	} else {
		_process_file_system(first_scan_root_dir, filesystem, sp, nullptr);
	}

	_update_scan_uid_actions();
	if (first_scan) {
		memdelete(first_scan_root_dir);
		first_scan_root_dir = nullptr;
	}
}

void EditorFileSystem::_save_filesystem_cache() {
	group_file_cache.clear();

	String fscache = EditorPaths::get_singleton()->get_project_settings_dir().path_join(CACHE_FILE_NAME);

	Ref<FileAccess> f = FileAccess::open(fscache, FileAccess::WRITE);
	ERR_FAIL_COND_MSG(f.is_null(), "Cannot create file '" + fscache + "'. Check user write permissions.");

	f->store_line(filesystem_settings_version_for_import);
	_save_filesystem_cache(filesystem, f);
}

void EditorFileSystem::_thread_func(void *_userdata) {
	EditorFileSystem *sd = (EditorFileSystem *)_userdata;
	sd->_scan_filesystem();
	sd->scanning_done.set();
}

bool EditorFileSystem::_is_test_for_reimport_needed(EditorFileInfo *p_file, uint64_t p_modified_time, uint64_t p_import_modified_time) {
	if (p_modified_time != p_file->modified_time) {
		return true;
	}
	if (p_import_modified_time != p_file->import_modified_time) {
		return true;
	}
	if (!reimport_on_missing_imported_files) {
		return false;
	}
	for (const String &path : p_file->import_dest_paths) {
		if (!FileAccess::exists(path)) {
			return true;
		}
	}
	return false;
}

bool EditorFileSystem::_test_for_reimport(const String &p_path, EditorFileInfo *p_file) {
	if (p_file->import_md5.is_empty()) {
		// Marked as reimportation needed.
		return true;
	}
	String new_md5 = FileAccess::get_md5(p_path + ".import");
	if (p_file->import_md5 != new_md5) {
		return true;
	}

	Error err;
	Ref<FileAccess> f = FileAccess::open(p_path + ".import", FileAccess::READ, &err);

	if (f.is_null()) { // No import file, reimport.
		return true;
	}

	VariantParser::StreamFile stream;
	stream.f = f;

	String assign;
	Variant value;
	VariantParser::Tag next_tag;

	int lines = 0;
	String error_text;

	Vector<String> to_check;

	String importer_name;
	String source_file = "";
	String source_md5 = "";
	Vector<String> dest_files;
	String dest_md5 = "";
	int version = 0;
	bool found_uid = false;
	Variant meta;

	while (true) {
		assign = Variant();
		next_tag.fields.clear();
		next_tag.name = String();

		err = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, nullptr, true);
		if (err == ERR_FILE_EOF) {
			break;
		} else if (err != OK) {
			ERR_PRINT("ResourceFormatImporter::load - '" + p_path + ".import:" + itos(lines) + "' error '" + error_text + "'.");
			// Parse error, skip and let user attempt manual reimport to avoid reimport loop.
			return false;
		}

		if (!assign.is_empty()) {
			if (assign == "valid" && value.operator bool() == false) {
				// Invalid import (failed previous import), skip and let user attempt manual reimport to avoid reimport loop.
				return false;
			}
			if (assign.begins_with("path")) {
				to_check.push_back(value);
			} else if (assign == "files") {
				Array fa = value;
				for (const Variant &check_path : fa) {
					to_check.push_back(check_path);
				}
			} else if (assign == "importer_version") {
				version = value;
			} else if (assign == "importer") {
				importer_name = value;
			} else if (assign == "uid") {
				found_uid = true;
			} else if (assign == "source_file") {
				source_file = value;
			} else if (assign == "dest_files") {
				dest_files = value;
			} else if (assign == "metadata") {
				meta = value;
			}

		} else if (next_tag.name != "remap" && next_tag.name != "deps") {
			break;
		}
	}

	if (importer_name == "keep" || importer_name == "skip") {
		return false; // Keep mode, do not reimport.
	}

	if (!found_uid) {
		return true; // UID not found, old format, reimport.
	}

	// Imported files are gone, reimport.
	for (const String &E : to_check) {
		if (!FileAccess::exists(E)) {
			return true;
		}
	}

	Ref<ResourceImporter> importer = ResourceFormatImporter::get_singleton()->get_importer_by_name(importer_name);

	if (importer.is_null()) {
		return true; // The importer has possibly changed, try to reimport.
	}

	if (importer->get_format_version() > version) {
		return true; // Version changed, reimport.
	}

	if (!importer->are_import_settings_valid(p_path, meta)) {
		// Reimport settings are out of sync with project settings, reimport.
		return true;
	}

	// Read the md5's from a separate file (so the import parameters aren't dependent on the file version).
	String base_path = ResourceFormatImporter::get_singleton()->get_import_base_path(p_path);
	Ref<FileAccess> md5s = FileAccess::open(base_path + ".md5", FileAccess::READ, &err);
	if (md5s.is_null()) { // No md5's stored for this resource.
		return true;
	}

	VariantParser::StreamFile md5_stream;
	md5_stream.f = md5s;

	while (true) {
		assign = Variant();
		next_tag.fields.clear();
		next_tag.name = String();

		err = VariantParser::parse_tag_assign_eof(&md5_stream, lines, error_text, next_tag, assign, value, nullptr, true);

		if (err == ERR_FILE_EOF) {
			break;
		} else if (err != OK) {
			ERR_PRINT("ResourceFormatImporter::load - '" + p_path + ".import.md5:" + itos(lines) + "' error '" + error_text + "'.");
			return false; // Parse error.
		}
		if (!assign.is_empty()) {
			if (assign == "source_md5") {
				source_md5 = value;
			} else if (assign == "dest_md5") {
				dest_md5 = value;
			}
		}
	}

	// Check source md5 matching.
	if (!source_file.is_empty() && source_file != p_path) {
		return true; // File was moved, reimport.
	}

	if (source_md5.is_empty()) {
		return true; // Lacks md5, so just reimport.
	}

	String md5 = FileAccess::get_md5(p_path);
	if (md5 != source_md5) {
		return true;
	}

	if (!dest_files.is_empty() && !dest_md5.is_empty()) {
		md5 = FileAccess::get_multiple_md5(dest_files);
		if (md5 != dest_md5) {
			return true;
		}
	}

	return false; // Nothing changed.
}

Vector<String> EditorFileSystem::_get_import_dest_paths(const String &p_path) {
	Error err;
	Ref<FileAccess> f = FileAccess::open(p_path + ".import", FileAccess::READ, &err);

	if (f.is_null()) { // No import file, reimport.
		return Vector<String>();
	}

	VariantParser::StreamFile stream;
	stream.f = f;

	String assign;
	Variant value;
	VariantParser::Tag next_tag;

	int lines = 0;
	String error_text;

	Vector<String> dest_paths;
	String importer_name;

	while (true) {
		assign = Variant();
		next_tag.fields.clear();
		next_tag.name = String();

		err = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, nullptr, true);
		if (err == ERR_FILE_EOF) {
			break;
		} else if (err != OK) {
			ERR_PRINT("ResourceFormatImporter::load - '" + p_path + ".import:" + itos(lines) + "' error '" + error_text + "'.");
			// Parse error, skip and let user attempt manual reimport to avoid reimport loop.
			return Vector<String>();
		}

		if (!assign.is_empty()) {
			if (assign == "valid" && value.operator bool() == false) {
				// Invalid import (failed previous import), skip and let user attempt manual reimport to avoid reimport loop.
				return Vector<String>();
			}
			if (assign == "dest_files") {
				Array fa = value;
				for (const Variant &dest_path : fa) {
					dest_paths.push_back(dest_path);
				}
			} else if (assign == "importer") {
				importer_name = value;
			}
		} else if (next_tag.name != "remap" && next_tag.name != "deps") {
			break;
		}
	}

	if (importer_name == "keep" || importer_name == "skip") {
		return Vector<String>();
	}

	return dest_paths;
}

bool EditorFileSystem::_import_support_abort_scan(const Vector<String> &reimports) {
	if (import_support_queries.is_empty()) {
		return false;
	}
	HashMap<String, int> import_support_test;
	Vector<bool> import_support_tested;
	import_support_tested.resize(import_support_queries.size());
	for (int i = 0; i < import_support_queries.size(); i++) {
		import_support_tested.write[i] = false;
		if (import_support_queries[i]->is_active()) {
			Vector<String> extensions = import_support_queries[i]->get_file_extensions();
			for (int j = 0; j < extensions.size(); j++) {
				import_support_test.insert(extensions[j], i);
			}
		}
	}

	if (import_support_test.is_empty()) {
		return false; //well nothing to do
	}

	for (int i = 0; i < reimports.size(); i++) {
		const String file = reimports[i].get_file();
		for (KeyValue<String, int> &E : import_support_test) {
			if (file.right(E.key.length() + 1).nocasecmp_to("." + E.key) == 0) {
				import_support_tested.write[E.value] = true;
			}
		}
	}

	for (int i = 0; i < import_support_tested.size(); i++) {
		if (import_support_tested[i]) {
			if (import_support_queries.write[i]->query()) {
				return true;
			}
		}
	}

	return false;
}

void EditorFileSystem::_reset_uid_points() {
	scan_uid_actions.clear();
	ItemUIDAction ia;
	uid_newly_add_end = scan_uid_actions.push_back(ia);
	uid_move_end = scan_uid_actions.push_back(ia);
}

void EditorFileSystem::_reset_points() {
	scan_actions.clear();
	ItemAction ia;
	normal = scan_actions.push_back(ia);
	remove_point = scan_actions.push_back(ia);
}

void EditorFileSystem::_create_action(EditorFileSystemDirectory *p_dir, EditorFileInfo *p_fi, const String &p_path, const ItemAction::Action p_action, const ItemAction::Step p_step, const ResourceUID::ID p_old_uid) {
	ItemAction ia;
	ia.action = p_action;
	ia.path = p_path;
	ia.file = p_fi;
	ia.dir = p_dir;
	switch (p_step) {
		case ItemAction::STEP_NORMAL: {
			scan_actions.insert_before(normal, ia);
		} break;
		case ItemAction::STEP_CLEAR_STATUS: {
			scan_actions.insert_after(normal, ia);
		} break;
		case ItemAction::STEP_FILE_REMOVE: {
			scan_actions.insert_before(remove_point, ia);
		} break;
		case ItemAction::STEP_DIR_REMOVE: {
			scan_actions.insert_after(remove_point, ia);
		} break;
		case ItemAction::STEP_MAX: {
		} break;
	}
}

void EditorFileSystem::_create_uid_action(EditorFileInfo *p_fi, const String &p_path, const ItemUIDAction::UIDAction p_action, const ItemUIDAction::UIDStep p_step, const ResourceUID::ID p_old_uid) {
	ItemUIDAction ia;
	ia.action = p_action;
	ia.old_uid = p_old_uid;
	ia.path = p_path;
	ia.file = p_fi;
	switch (p_step) {
		case ItemUIDAction::STEP_UID_VALIDATE: {
			scan_uid_actions.push_front(ia);
		} break;
		case ItemUIDAction::STEP_UID_NEWLY_ADD: {
			scan_uid_actions.insert_before(uid_newly_add_end, ia);
		} break;
		case ItemUIDAction::STEP_UID_REMOVE: {
			scan_uid_actions.insert_after(uid_newly_add_end, ia);
		} break;
		case ItemUIDAction::STEP_UID_PENDING_ADD: {
			scan_uid_actions.insert_before(uid_move_end, ia);
		} break;
		case ItemUIDAction::STEP_UID_REGENERATE: {
			scan_uid_actions.insert_after(uid_move_end, ia);
		} break;
		case ItemUIDAction::STEP_UID_MAX: {
		} break;
	}
}

void EditorFileSystem::_create_actions_from_uid_change(EditorFileInfo *p_fi, const String &p_path, const ResourceUID::ID p_old_uid) {
	ERR_FAIL_NULL(p_fi);

	if (!(p_fi->status & EditorFileInfo::AS_RESOURCE) || p_fi->status & EditorFileInfo::FILE_REMOVE) {
		if (p_old_uid == ResourceUID::INVALID_ID) {
			return;
		}
		// Files that are no longer as resource or no longer tracked.
		_create_uid_action(p_fi, p_path, ItemUIDAction::ACTION_UID_REMOVE, ItemUIDAction::STEP_UID_REMOVE, p_old_uid);
		return;
	} else if (p_fi->status & EditorFileInfo::FILE_ADD) {
		if (p_fi->uid == ResourceUID::INVALID_ID) {
			// Newly added files. It is also possible that the skip/keep type import file was removed.
			_create_uid_action(p_fi, p_path, ItemUIDAction::ACTION_UID_ADD, ItemUIDAction::STEP_UID_NEWLY_ADD, p_old_uid);
		} else {
			// The file was moved or duplicated.
			_create_uid_action(p_fi, p_path, ItemUIDAction::ACTION_UID_PENDING_ADD, ItemUIDAction::STEP_UID_PENDING_ADD, p_old_uid);
		}
		return;
	}

	if (p_old_uid != ResourceUID::INVALID_ID) {
		if (p_fi->uid == p_old_uid) {
			if (first_scan) {
				// The UID cache may be invalid. Edge case.
				_create_uid_action(p_fi, p_path, ItemUIDAction::ACTION_UID_PENDING_ADD, ItemUIDAction::STEP_UID_VALIDATE, p_old_uid);
			}
			return;
		}
		// The file was overwritten. Remove the old uid.
		_create_uid_action(p_fi, p_path, ItemUIDAction::ACTION_UID_REMOVE, ItemUIDAction::STEP_UID_REMOVE, p_old_uid);
	}

	if (p_fi->uid == ResourceUID::INVALID_ID) {
		// The internal files are removed. Re-add(create) the uid.
		_create_uid_action(p_fi, p_path, ItemUIDAction::ACTION_UID_ADD, ItemUIDAction::STEP_UID_PENDING_ADD, p_old_uid);
	} else {
		// The file may be overwritten.
		_create_uid_action(p_fi, p_path, ItemUIDAction::ACTION_UID_PENDING_ADD, ItemUIDAction::STEP_UID_PENDING_ADD, p_old_uid);
	}
}

bool EditorFileSystem::_update_scan_uid_actions() {
	bool fs_changed = false;

	HashMap<ResourceUID::ID, String> retracks;
	HashMap<ResourceUID::ID, String> untracks;
	HashMap<ResourceUID::ID, Vector<String>> duplicates;
	HashMap<String, ResourceUID::ID> tracks;

	EditorProgress *ep = nullptr;
	if (scan_uid_actions.size() > (EditorFileSystem::ItemUIDAction::STEP_UID_MAX / 2 + 1)) {
		ep = memnew(EditorProgress("_update_scan_uid_actions", TTR("Scanning uid actions..."), scan_uid_actions.size()));
	}
	int step_count = 0;
	for (const ItemUIDAction &ia : scan_uid_actions) {
		switch (ia.action) {
			case ItemUIDAction::ACTION_UID_NONE: {
				continue;
			} break;
			case ItemUIDAction::ACTION_UID_ADD: {
				print_verbose(vformat("[%.6f] [ADD UID] old uid: %s, uid: %s, path: %s.",
						OS::get_singleton()->get_ticks_usec() / 1000000.0f,
						ResourceUID::get_singleton()->id_to_text(ia.old_uid),
						ResourceUID::get_singleton()->id_to_text(ia.file->uid),
						ia.path));
				if (ia.old_uid == ResourceUID::INVALID_ID || ResourceUID::get_singleton()->has_id(ia.old_uid)) {
					ia.file->uid = ResourceUID::get_singleton()->create_id_for_path(ia.path);
				} else {
					ia.file->uid = ia.old_uid;
				}
				ResourceUID::get_singleton()->add_id(ia.file->uid, ia.path);
				tracks[ia.path] = ia.file->uid;
				fs_changed = true;
				// Update internal file.
				if (ia.file->status & EditorFileInfo::IS_IMPORTABLE) {
					const String internal_path = ia.path + ".import";
					Ref<ConfigFile> cfg;
					cfg.instantiate();
					Error err = cfg->load(internal_path);
					if (err == OK) {
						cfg->set_value("remap", "uid", ResourceUID::get_singleton()->id_to_text(ia.file->uid));
						err = cfg->save(internal_path);
						ia.file->import_modified_time = FileAccess::get_modified_time(internal_path);
					}
				} else if (ia.file->status & EditorFileInfo::HAS_CUSTOM_UID_SUPPORT) {
					ResourceSaver::set_uid(ia.path, ia.file->uid);
					ia.file->modified_time = FileAccess::get_modified_time(ia.path);
				} else {
					const String internal_path = ia.path + ".uid";
					Ref<FileAccess> f = FileAccess::open(internal_path, FileAccess::WRITE);
					if (f.is_valid()) {
						f->store_line(ResourceUID::get_singleton()->id_to_text(ia.file->uid));
					}
				}
				print_verbose(vformat("[ADD UID] %s, %s", ia.path, ResourceUID::get_singleton()->id_to_text(ia.file->uid)));
			} break;
			case ItemUIDAction::ACTION_UID_REMOVE: {
				print_verbose(vformat("[%.6f] [REMOVE UID] old uid: %s, uid: %s, path: %s.",
						OS::get_singleton()->get_ticks_usec() / 1000000.0f,
						ResourceUID::get_singleton()->id_to_text(ia.old_uid),
						ResourceUID::get_singleton()->id_to_text(ia.file->uid),
						ia.path));
				untracks[ia.old_uid] = ia.path;
				fs_changed = true;
				if (!first_scan || ResourceUID::get_singleton()->has_id(ia.old_uid)) {
					ResourceUID::get_singleton()->remove_id(ia.old_uid);
				}
			} break;
			case ItemUIDAction::ACTION_UID_PENDING_ADD: {
				print_verbose(vformat("[%.6f] [TEST ADD UID] old uid: %s, uid: %s, path: %s.",
						OS::get_singleton()->get_ticks_usec() / 1000000.0f,
						ResourceUID::get_singleton()->id_to_text(ia.old_uid),
						ResourceUID::get_singleton()->id_to_text(ia.file->uid),
						ia.path));
				if (ResourceUID::get_singleton()->has_id(ia.file->uid)) {
					const String cache_uid_path = ResourceUID::get_singleton()->get_id_path(ia.file->uid);
					if (cache_uid_path != ia.path) {
						duplicates[ia.file->uid].push_back(ia.path);
						WARN_PRINT(vformat("Duplicate UID detected for Resource at \"%s\".\nOld Resource path: \"%s\". The new file UID was changed automatically.", ia.path, cache_uid_path));
						if (ia.old_uid != ResourceUID::INVALID_ID && ia.old_uid != ia.file->uid && ResourceUID::get_singleton()->has_id(ia.old_uid)) {
							// Files may be overwritten.
							_create_uid_action(ia.file, ia.path, ItemUIDAction::ACTION_UID_ADD, ItemUIDAction::STEP_UID_PENDING_ADD, ia.old_uid);
						} else {
							// Files may be duplicate.
							_create_uid_action(ia.file, ia.path, ItemUIDAction::ACTION_UID_ADD, ItemUIDAction::STEP_UID_REGENERATE);
						}
						step_count--;
					}
				} else {
					retracks[ia.file->uid] = ia.path;
					ResourceUID::get_singleton()->add_id(ia.file->uid, ia.path);
				}
			} break;
			default: {
			} break;
		}

		if (ep) {
			ep->step(ia.path, step_count++, false);
		}
	}
	memdelete_notnull(ep);

	if (!retracks.is_empty() || !untracks.is_empty() || !tracks.is_empty()) {
		struct MoveItems {
			String origin;
			String target;
		};
		HashMap<ResourceUID::ID, MoveItems> moves;

		print_verbose("======= Scan UID Analysis Start ======");

		for (KeyValue<ResourceUID::ID, String> &E : untracks) {
			HashMap<ResourceUID::ID, String>::Iterator I = retracks.find(E.key);
			if (!I) {
				print_verbose(vformat("[Untracked] %s, at %s.", ResourceUID::get_singleton()->id_to_text(E.key), E.value));
				continue;
			}
			MoveItems mi;
			mi.origin = E.value;
			mi.target = I->value;
			moves.insert(E.key, mi);

			retracks.erase(E.key);
		}

		for (KeyValue<ResourceUID::ID, MoveItems> &E : moves) {
			print_verbose(vformat("[Moved] %s, from %s to %s.", ResourceUID::get_singleton()->id_to_text(E.key), E.value.origin, E.value.target));
			untracks.erase(E.key);
		}

		for (KeyValue<ResourceUID::ID, Vector<String>> &E : duplicates) {
			for (const String &F : E.value) {
				HashMap<String, ResourceUID::ID>::Iterator I = tracks.find(F);
				ERR_CONTINUE(I == nullptr);
				print_verbose(vformat("[Duplicated] %s, from %s to %s, %s.", ResourceUID::get_singleton()->id_to_text(E.key), ResourceUID::get_singleton()->get_id_path(E.key), F, ResourceUID::get_singleton()->id_to_text(I->value)));
				tracks.erase(F);
			}
		}

		for (KeyValue<String, ResourceUID::ID> &E : tracks) {
			print_verbose(vformat("[Tracked] %s, at %s.", ResourceUID::get_singleton()->id_to_text(E.value), E.key));
		}

		for (KeyValue<ResourceUID::ID, String> &E : retracks) {
			print_verbose(vformat("[Retracked] %s, at %s.", ResourceUID::get_singleton()->id_to_text(E.key), E.value));
		}

		print_verbose("======= Scan UID Analysis End ======");
	}

	if (!first_scan) {
		_update_global_script_class_activation();
	}

	_reset_uid_points();
	return fs_changed;
}

bool EditorFileSystem::_update_scan_actions() {
	update_actions_queued = false;
	sources_changed.clear();

	// We need to update the script global class names before the reimports to be sure that
	// all the importer classes that depends on class names will work.
	_update_script_classes();

	bool fs_changed = false;

	Vector<String> reimports;
	Vector<String> overwrites;
	Vector<String> reloads;

	EditorProgress *ep = nullptr;
	if (scan_actions.size() > (EditorFileSystem::ItemAction::STEP_MAX / 2 + 1)) {
		ep = memnew(EditorProgress("_update_scan_actions", TTR("Scanning actions..."), scan_actions.size()));
	}

	int step_count = 0;
	for (const ItemAction &ia : scan_actions) {
		switch (ia.action) {
			case ItemAction::ACTION_NONE: {
				continue;
			} break;
			case ItemAction::ACTION_DIR_ADD: {
				// ERR_CONTINUE(!ia.dir);
				fs_changed = true;
			} break;
			case ItemAction::ACTION_DIR_REMOVE: {
				ERR_CONTINUE(!ia.dir);
				memdelete(ia.dir);
				fs_changed = true;
			} break;
			case ItemAction::ACTION_FILE_REMOVE: {
				ERR_CONTINUE(!ia.file);
				if (ia.file->status & EditorFileInfo::TYPE_REMOVE) {
					overwrites.push_back(ia.path);
				}
				ia.file->status &= ~EditorFileInfo::TEMPORARY;
				memdelete(ia.file);
				fs_changed = true;
			} break;
			case ItemAction::ACTION_FILE_ADD: {
				ERR_CONTINUE(!ia.file);
				ia.file->status &= ~EditorFileInfo::TEMPORARY;
				fs_changed = true;
			} break;
			case ItemAction::ACTION_FILE_UPDATE: {
				ERR_CONTINUE(!ia.file);
				if (!(ia.file->status & EditorFileInfo::IS_IMPORTABLE)) {
					if (ia.file->status & EditorFileInfo::TYPE_REMOVE) {
						overwrites.push_back(ia.path);
					} else if (!(ia.file->status & EditorFileInfo::TYPE_CHANGED)) {
						reloads.push_back(ia.path);
					}
				}
				ia.file->status &= ~EditorFileInfo::TEMPORARY;
				fs_changed = true;
			} break;
			case ItemAction::ACTION_FILE_REIMPORT: {
				if (ia.file) {
					ia.file->status &= ~EditorFileInfo::TEMPORARY;
				}
				reimports.push_back(ia.path);
			} break;
			default: {
			} break;
		}

		if (ep) {
			ep->step(ia.path, step_count++, false);
		}
	}

	memdelete_notnull(ep);

	if (_scan_extensions()) {
		//needs editor restart
		//extensions also may provide filetypes to be imported, so they must run before importing
		if (EditorNode::immediate_confirmation_dialog(TTR("Some extensions need the editor to restart to take effect."), first_scan ? TTR("Restart") : TTR("Save & Restart"), TTR("Continue"))) {
			if (!first_scan) {
				EditorNode::get_singleton()->save_all_scenes();
			}
			EditorNode::get_singleton()->restart_editor();
			//do not import
			return true;
		}
	}

	if (!reimports.is_empty()) {
		if (_import_support_abort_scan(reimports)) {
			return true;
		}

		reimport_files(reimports);
	} else {
		// Reimport files will update the uid cache file so if nothing was reimported, update it manually.
		ResourceUID::get_singleton()->update_cache();
	}

	if (first_scan) {
		// Only on first scan this is valid and updated, then settings changed.
		revalidate_import_files = false;
		filesystem_settings_version_for_import = ResourceFormatImporter::get_singleton()->get_import_settings_hash();
	}

	if (fs_changed) {
		_save_filesystem_cache();
	}

	// Moving the processing of pending updates before the resources_reload event to be sure all global class names
	// are updated. Script.cpp listens on resources_reload and reloads updated scripts.
	_process_update_pending();

	for (const String &path : overwrites) {
		Ref<Resource> res = ResourceCache::get_ref(path);
		if (res.is_null()) {
			continue;
		}
		res->set_path("");
	}

	if (reloads.size()) {
		emit_signal(SNAME("resources_reload"), reloads);
	}
	_reset_points();

	return fs_changed;
}

void EditorFileSystem::scan() {
	if (false /*&& bool(Globals::get_singleton()->get("debug/disable_scan"))*/) {
		return;
	}

	if (scanning || scanning_changes || thread.is_started()) {
		return;
	}

	// The first scan must be on the main thread because, after the first scan and update
	// of global class names, we load the plugins and autoloads. These need to
	// be added on the main thread because they are nodes, and we need to wait for them
	// to be loaded to continue the scan and reimportations.
	if (first_scan) {
		_first_scan_filesystem();
#ifdef ANDROID_ENABLED
		// Create a .nomedia file to hide assets from media apps on Android.
		// Android 11 has some issues with nomedia files, so it's disabled there. See GH-106479 and GH-105399 for details.
		// NOTE: Nomedia file is also handled in project manager. See project_dialog.cpp ->  ProjectDialog::ok_pressed().
		String sdk_version = OS::get_singleton()->get_version().get_slicec('.', 0);
		if (sdk_version != "30") {
			const String nomedia_file_path = ProjectSettings::get_singleton()->get_resource_path().path_join(".nomedia");
			if (!FileAccess::exists(nomedia_file_path)) {
				Ref<FileAccess> f = FileAccess::open(nomedia_file_path, FileAccess::WRITE);
				if (f.is_null()) {
					// .nomedia isn't so critical.
					ERR_PRINT("Couldn't create .nomedia in project path.");
				} else {
					f->close();
				}
			}
		}
#endif
	}

	sources_changed.clear();
	scanning_done.clear();
	scanning = true;
	scan_total = 0;

	if (!use_threads) {
		_scan_filesystem();
		scanning_done.set();
		dirty_directories.clear();
		scan_changes_pending = false;
		_update_scan_actions();
		// Update all icons so they are loaded for the FileSystemDock.
		_update_files_icon_path();
		extensions_changed = false;
		scanning = false;
		// Set first_scan to false before the signals so the function doing_first_scan can return false
		// in editor_node to start the export if needed.
		first_scan = false;
		ResourceImporter::load_on_startup = nullptr;
		emit_signal(SNAME("filesystem_changed"));
		emit_signal(SNAME("sources_changed"), sources_changed.size() > 0);
	} else {
		ERR_FAIL_COND(thread.is_started());
		set_process(true);
		Thread::Settings s;
		s.priority = Thread::PRIORITY_LOW;
		thread.start(_thread_func, this, s);
	}
}

void EditorFileSystem::ScanProgress::increment() {
	current++;
	float ratio = current / MAX(hi, 1.0f);
	if (progress) {
		progress->step(ratio * 1000.0f);
	}
	EditorFileSystem::singleton->scan_total = ratio;
}

int EditorFileSystem::_scan_new_dir(ScannedDirectory *p_dir, Ref<DirAccess> &da) {
	List<String> dirs;
	List<String> files;

	String cd = da->get_current_dir();

	da->list_dir_begin();
	while (true) {
		String f = da->get_next();
		if (f.is_empty()) {
			break;
		}

		if (da->current_is_hidden()) {
			continue;
		}

		if (da->current_is_dir()) {
			if (f.begins_with(".")) { // Ignore special and . / ..
				continue;
			}

			if (_should_skip_directory(cd.path_join(f))) {
				continue;
			}

			dirs.push_back(f);

		} else {
			files.push_back(f);
		}
	}

	da->list_dir_end();

	dirs.sort_custom<FileNoCaseComparator>();
	files.sort_custom<FileNoCaseComparator>();

	int nb_files_total_scan = 0;

	for (const String &dir : dirs) {
		if (da->change_dir(dir) == OK) {
			String d = da->get_current_dir();

			if (d == cd || !d.begins_with(cd)) {
				da->change_dir(cd); // Avoid recursion.
			} else {
				ScannedDirectory *sd = memnew(ScannedDirectory);
				sd->name = dir;
				sd->full_path = p_dir->full_path.path_join(sd->name);

				nb_files_total_scan += _scan_new_dir(sd, da);

				p_dir->subdirs.push_back(sd);

				da->change_dir("..");
			}
		} else {
			ERR_PRINT("Cannot go into subdir '" + dir + "'.");
		}
	}

	p_dir->files = files;
	nb_files_total_scan += files.size();
	p_dir->count = nb_files_total_scan;

	return nb_files_total_scan;
}

// Only called when a file info is added or when the file extension changes.
void EditorFileSystem::_category_validate(EditorFileInfo *p_file, const String &p_path) {
	if (ResourceLoader::has_custom_uid_support(p_path)) {
		p_file->status |= EditorFileInfo::HAS_CUSTOM_UID_SUPPORT;
	} else {
		p_file->status &= ~EditorFileInfo::HAS_CUSTOM_UID_SUPPORT;
	}

	p_file->status &= ~EditorFileInfo::CATEGORY_CHANGED; // Clear the category bits.

	if (_validate_file_extension(p_file->file, import_extensions)) {
		p_file->status |= EditorFileInfo::IS_IMPORTABLE;
	}
	if (_validate_file_extension(p_file->file, other_file_extensions)) {
		p_file->status |= EditorFileInfo::IS_OTHER;
	}
	if (_validate_file_extension(p_file->file, textfile_extensions)) {
		p_file->status |= EditorFileInfo::IS_TEXT;
	}
}

// Analyze whether the type has changed and mark the types and changes of concern.
void EditorFileSystem::_type_analysis(EditorFileInfo *p_file, const StringName &p_new_type) {
	const StringName old_type = p_file->type;
	p_file->type = p_new_type;
	if (p_file->type.is_empty()) {
		if (p_file->status & EditorFileInfo::IS_OTHER) {
			p_file->type = "OtherFile";
		} else if (p_file->status & EditorFileInfo::IS_TEXT) {
			p_file->type = "TextFile";
		}
		p_file->status &= ~EditorFileInfo::AS_RESOURCE;
	} else {
		p_file->status |= EditorFileInfo::AS_RESOURCE;
	}

	if (old_type != p_file->type) {
		if (!old_type.is_empty()) {
			p_file->status |= EditorFileInfo::TYPE_REMOVE;
		}
		if (!p_file->type.is_empty()) {
			p_file->status |= EditorFileInfo::TYPE_ADD;
		}
		if (p_file->type == PackedScene::get_class_static()) {
			p_file->status |= EditorFileInfo::IS_PACKEDSCENE;
		} else if (ClassDB::is_parent_class(p_file->type, Script::get_class_static())) {
			p_file->status |= EditorFileInfo::IS_SCRIPT;
		}
	}
}

void EditorFileSystem::_import_validate(EditorFileInfo *p_file, const String &p_path) {
	ERR_FAIL_NULL(p_file);
	const ResourceUID::ID old_uid = p_file->uid;
	const bool is_file_updated = !(p_file->status & EditorFileInfo::FILE_ADD);
	if (!FileAccess::exists(p_path + ".import")) {
		p_file->uid = ResourceUID::INVALID_ID;
		_create_actions_from_uid_change(p_file, p_path, old_uid);
		if (is_file_updated) {
			_create_action(nullptr, p_file, p_path, ItemAction::ACTION_FILE_UPDATE);
		}
		_create_action(nullptr, p_file, p_path, ItemAction::ACTION_FILE_REIMPORT); // Import directly.
		return;
	}

	const uint64_t mt = FileAccess::get_modified_time(p_path);
	const uint64_t import_mt = FileAccess::get_modified_time(p_path + ".import");
	StringName new_type;
	ResourceFormatImporter::get_singleton()->get_resource_import_info(p_path, new_type, p_file->uid, p_file->import_group_file);
	_type_analysis(p_file, new_type);

	const bool is_uid_unchanged = old_uid == p_file->uid;
	if (!is_uid_unchanged || first_scan) {
		_create_actions_from_uid_change(p_file, p_path, old_uid);
		if (is_file_updated) {
			_create_action(nullptr, p_file, p_path, ItemAction::ACTION_FILE_UPDATE);
		}
	}

	if (is_uid_unchanged && !_is_test_for_reimport_needed(p_file, mt, import_mt) &&
			(!first_scan || !revalidate_import_files || ResourceFormatImporter::get_singleton()->are_import_settings_valid(p_path))) {
		return;
	}
	bool need_reimport = _test_for_reimport(p_path, p_file);
	if (need_reimport) {
		if (is_file_updated && is_uid_unchanged && !first_scan) {
			_create_action(nullptr, p_file, p_path, ItemAction::ACTION_FILE_UPDATE);
		}
		_create_action(nullptr, p_file, p_path, ItemAction::ACTION_FILE_REIMPORT); // Must reimport.
		Vector<String> dependencies = _get_dependencies(p_path);
		for (const String &dep : dependencies) {
			const String &dependency_path = dep.contains("::") ? dep.get_slice("::", 0) : dep;
			if (_validate_file_extension(dependency_path, import_extensions)) {
				// Import these.
				_create_action(nullptr, nullptr, dependency_path, ItemAction::ACTION_FILE_REIMPORT);
			}
		}
		return;
	}
	// Must not reimport, all was good.
	// Update modified times, md5 and destination paths, to avoid reimport.
	p_file->modified_time = mt;
	p_file->import_modified_time = import_mt;
	p_file->import_dest_paths = _get_import_dest_paths(p_path);
}

void EditorFileSystem::_check_loader_or_saver_changed(const ScriptClassInfo &p_class_info) {
	if (p_class_info.extends == ResourceFormatLoader::get_class_static()) {
		loader_changed = true;
		need_update_extensions = true;
	} else if (p_class_info.extends == ResourceFormatSaver::get_class_static()) {
		saver_changed = true;
	}
}

void EditorFileSystem::_reload_loader_or_saver() {
	if (saver_changed) {
		saver_changed = false;
		ResourceSaver::remove_custom_savers();
		ResourceSaver::add_custom_savers();
	}
	if (loader_changed) {
		loader_changed = false;
		ResourceLoader::remove_custom_loaders();
		ResourceLoader::add_custom_loaders();
	}
}

void EditorFileSystem::_global_script_class_info_remove(EditorFileInfo *p_file, const String &p_path) {
	ERR_FAIL_NULL(p_file);
	ScriptClassInfo &sci = p_file->class_info;
	HashMap<StringName, ScriptClassAlternatives>::Iterator E = global_script_class_alternatives.find(sci.name);
	ERR_FAIL_NULL_MSG(E, vformat("The file %s of the untracked global class %s was detected and removed.", p_path, sci.name));
	ScriptClassAlternatives &scas = E->value;
	scas.alternatives.erase(p_path);
	p_file->status &= ~EditorFileInfo::IS_GLOBAL_CLASS_ALTERNATIVE;
	if (p_file->status & EditorFileInfo::IS_ACTIVE_GLOBAL_CLASS_ALTERNATIVE) {
		p_file->status &= ~EditorFileInfo::IS_ACTIVE_GLOBAL_CLASS_ALTERNATIVE;
		_check_loader_or_saver_changed(sci);
		scas.active = false;
		ScriptServer::remove_global_class(sci.name);
		EditorHelp::remove_doc(sci.name);
		EditorNode::get_editor_data().script_class_clear_name(p_path);
	}
}

void EditorFileSystem::_global_script_class_info_add(EditorFileInfo *p_file, const String &p_path) {
	ERR_FAIL_NULL(p_file);
	ScriptClassAlternatives &scas = global_script_class_alternatives[p_file->class_info.name];
	scas.alternatives[p_path] = p_file;
}

void EditorFileSystem::_script_class_info_update(EditorFileInfo *p_file, const String &p_path, const ScriptClassInfo *p_sci) {
	ERR_FAIL_NULL(p_file);
	ERR_FAIL_NULL(p_sci);
	if (p_file->status & EditorFileInfo::IS_GLOBAL_CLASS_ALTERNATIVE) {
		_global_script_class_info_remove(p_file, p_path);
	}
	ScriptClassInfo &sci = p_file->class_info;
	sci = *p_sci;
	if (sci.name.is_empty() || sci.lang.is_empty()) {
		return; // No need to add global class info.
	}
	p_file->status |= EditorFileInfo::IS_GLOBAL_CLASS_ALTERNATIVE;
	_global_script_class_info_add(p_file, p_path);
}

void EditorFileSystem::_file_info_add(EditorFileSystemDirectory *p_dir, const String &p_dir_path, const String &p_file, bool p_insert) {
	const String path = p_dir_path.path_join(p_file);
	const bool is_reused = first_scan && script_file_info.has(path);

	EditorFileInfo *fi = is_reused ? script_file_info[path] : memnew(EditorFileInfo);
	fi->parent = p_dir;
	if (p_insert) {
		p_insert = false;
		for (int idx = 0; idx < p_dir->files.size(); idx++) {
			if (fi->file.filenocasecmp_to(p_dir->files[idx]->file) < 0) {
				p_dir->files.insert(idx, fi);
				p_insert = true;
				break;
			}
		}
		if (!p_insert) {
			p_dir->files.push_back(fi);
		}
	} else {
		p_dir->files.push_back(fi);
	}
	fi->status &= ~EditorFileInfo::IS_ORPHAN;
	// fi->status |= EditorFileInfo::FILE_ADD;
	_create_action(p_dir, fi, path, ItemAction::ACTION_FILE_ADD);

	fi->file = p_file;
	_category_validate(fi, path);

	if (fi->status & EditorFileSystemDirectory::FileInfo::IS_IMPORTABLE) {
		_import_validate(fi, path);
		return;
	}

	if (is_reused) {
		if (!fi->type.is_empty()) {
			fi->status |= EditorFileInfo::TYPE_ADD;
		}
	} else {
		_type_analysis(fi, ResourceLoader::get_resource_type(path));
		fi->uid = ResourceLoader::get_resource_uid(path);
		if (fi->status & EditorFileInfo::IS_PACKEDSCENE) {
			_queue_update_scene_groups(path);
		} else if (fi->status & EditorFileInfo::IS_SCRIPT) {
			const ScriptClassInfo &sci = _get_global_script_class(fi->type, path);
			_script_class_info_update(fi, path, &sci);
		}
	}

	fi->modified_time = FileAccess::get_modified_time(path);
	fi->import_md5 = "";
	fi->deps = _get_dependencies(path);
	fi->resource_script_class = ResourceLoader::get_resource_script_class(path);
	fi->import_group_file = ResourceLoader::get_import_group_file(path);
	fi->import_modified_time = 0;
	fi->import_valid = !(fi->status & EditorFileInfo::AS_RESOURCE) || ResourceLoader::is_import_valid(path);

	_create_actions_from_uid_change(fi, path, fi->uid);

	// Update preview
	EditorResourcePreview::get_singleton()->check_for_invalidation(path);
}

void EditorFileSystem::_file_info_remove(EditorFileInfo *p_file, const String &p_path, const int p_idx) {
	if (p_file->status & EditorFileInfo::FILE_REMOVE) {
		return;
	}
	p_file->status |= EditorFileInfo::FILE_REMOVE | EditorFileInfo::TYPE_REMOVE;
	if (p_idx != -1) {
		// Immediately remove it from the tree, but do not immediately release it.
		p_file->parent->files.remove_at(p_idx);
		_create_action(nullptr, p_file, p_path, ItemAction::ACTION_FILE_REMOVE, ItemAction::STEP_FILE_REMOVE);
	}

	_delete_internal_files(p_path);
	_create_actions_from_uid_change(p_file, p_path, p_file->uid);

	if (p_file->status & EditorFileInfo::IS_PACKEDSCENE) {
		_queue_update_scene_groups(p_path);
	} else if (p_file->status & EditorFileInfo::IS_GLOBAL_CLASS_ALTERNATIVE) {
		_global_script_class_info_remove(p_file, p_path);
	}
}

void EditorFileSystem::_file_info_update(EditorFileInfo *p_file, const String &p_path) {
	// There is only one state for a file in the same frame.
	// Files that have already been marked will not be marked again.
	if (p_file->status & EditorFileInfo::FILE_CHANGED) {
		return;
	}

	const uint32_t old_status = p_file->status;
	if (extensions_changed && !first_scan) {
		_category_validate(p_file, p_path);
	}

	if (p_file->status & EditorFileInfo::IS_IMPORTABLE) {
		// TODO: Clean up the previous state.
		// if (old_status & EditorFileInfo::IS_PACKEDSCENE) {
		// 	_queue_update_scene_groups(p_path);
		// }
		// if (old_status & EditorFileInfo::IS_GLOBAL_CLASS_ALTERNATIVE) {
		// 	_global_script_class_info_remove(p_file, p_path);
		// }

		_import_validate(p_file, p_path);
		return;
	} else if (old_status & EditorFileInfo::IS_IMPORTABLE) {
		// TODO: Clear code.
	}

	// Since the timestamp obtained is accurate to 1 second, so in projects using Git for version control,
	// all files may have the same timestamp. Swapping directory structures may not change file timestamps.
	// It is still necessary to compare the modified time, type, and uid.
	const uint64_t mt = FileAccess::get_modified_time(p_path);
	if (!first_scan || !(p_file->status & EditorFileInfo::IS_SCRIPT)) {
		_type_analysis(p_file, ResourceLoader::get_resource_type(p_path));
	}
	const ResourceUID::ID old_uid = p_file->uid;
	if ((p_file->status & EditorFileInfo::AS_RESOURCE)) {
		p_file->uid = ResourceLoader::get_resource_uid(p_path);
	}

	if (mt == p_file->modified_time && !(p_file->status & EditorFileInfo::TYPE_CHANGED)) {
		if (!(p_file->status & EditorFileInfo::AS_RESOURCE)) {
			return; // No uid, no internal files.
		}

		if (old_uid == p_file->uid) {
			// The UID is not changed, but prevents invalidation of the UID cache on editor startup.
			if (first_scan && !(p_file->status & EditorFileInfo::IS_SCRIPT)) {
				_create_actions_from_uid_change(p_file, p_path, old_uid);
				_create_action(nullptr, p_file, p_path, ItemAction::ACTION_FILE_UPDATE);
			}
			return;
		}
	}

	p_file->modified_time = mt;
	p_file->import_modified_time = 0;
	if (!first_scan || !(p_file->status & EditorFileInfo::IS_SCRIPT)) {
		_create_actions_from_uid_change(p_file, p_path, old_uid);
	}
	_create_action(nullptr, p_file, p_path, ItemAction::ACTION_FILE_UPDATE, ItemAction::STEP_CLEAR_STATUS);

	p_file->resource_script_class = ResourceLoader::get_resource_script_class(p_path);

	if ((p_file->status | old_status) & EditorFileInfo::IS_PACKEDSCENE) {
		_queue_update_scene_groups(p_path);
	}
	if ((p_file->status | old_status) & EditorFileInfo::IS_SCRIPT) {
		const ScriptClassInfo &sci = _get_global_script_class(p_file->type, p_path);
		_script_class_info_update(p_file, p_path, &sci);
	}

	p_file->import_group_file = ResourceLoader::get_import_group_file(p_path);
	p_file->deps = _get_dependencies(p_path);
	p_file->import_valid = !(p_file->status & EditorFileInfo::AS_RESOURCE) || ResourceLoader::is_import_valid(p_path);

	if (ClassDB::is_parent_class(p_file->type, Resource::get_class_static())) {
		// files_to_update_icon_path.push_back(p_file);
	}
	// Update preview
	EditorResourcePreview::get_singleton()->check_for_invalidation(p_path);
}

int EditorFileSystem::_dir_info_remove(EditorFileSystemDirectory *p_dir, const String &p_path, const int p_idx) {
	if (p_idx != -1) {
		p_dir->parent->subdirs.remove_at(p_idx);
		_create_action(p_dir, nullptr, p_path, ItemAction::ACTION_DIR_REMOVE, ItemAction::STEP_DIR_REMOVE, ResourceUID::INVALID_ID);
	}

	int count = p_dir->files.size();

	for (int i = 0; i < p_dir->files.size(); i++) {
		EditorFileInfo *fi = p_dir->files[i];
		_file_info_remove(fi, p_path.path_join(fi->file), -1);
	}

	for (int i = 0; i < p_dir->subdirs.size(); i++) {
		EditorFileSystemDirectory *sub_dir = p_dir->subdirs[i];
		count += _dir_info_remove(sub_dir, p_path.path_join(sub_dir->name), -1);
	}

	return count;
}

void EditorFileSystem::_process_file_system(const ScannedDirectory *p_scan_dir, EditorFileSystemDirectory *p_dir, ScanProgress &p_progress, HashSet<String> *r_processed_files) {
	p_dir->modified_time = FileAccess::get_modified_time(p_scan_dir->full_path);

	for (ScannedDirectory *scan_sub_dir : p_scan_dir->subdirs) {
		EditorFileSystemDirectory *sub_dir = memnew(EditorFileSystemDirectory);
		sub_dir->parent = p_dir;
		sub_dir->name = scan_sub_dir->name;
		p_dir->subdirs.push_back(sub_dir);
		_process_file_system(scan_sub_dir, sub_dir, p_progress, r_processed_files);
	}

	for (const String &scan_file : p_scan_dir->files) {
		if (!_validate_file_extension(scan_file, valid_extensions)) {
			p_progress.increment();
			continue; // Invalid.
		}
		_file_info_add(p_dir, p_scan_dir->full_path, scan_file, false);
		p_progress.increment();
	}
}

void EditorFileSystem::_scan_fs_changes(EditorFileSystemDirectory *p_dir, ScanProgress &p_progress, bool p_recursive) {
	p_recursive |= p_dir->recursive;
	p_dir->dirty = false;
	p_dir->recursive = false;

	bool updated_dir = false;
	const String cd = p_dir->get_path();
	int diff_nb_files = 0;

	const uint64_t current_mtime = FileAccess::get_modified_time(cd);

	if (scanning || force_detect || extensions_changed || current_mtime != p_dir->modified_time) {
		updated_dir = true;
		p_dir->modified_time = current_mtime;
		// Ooooops, dir changed, see what's going on.

		// First mark everything as not verified.

		for (int i = 0; i < p_dir->files.size(); i++) {
			p_dir->files[i]->verified = false;
		}

		for (int i = 0; i < p_dir->subdirs.size(); i++) {
			p_dir->subdirs[i]->verified = false;
		}

		diff_nb_files -= p_dir->files.size();

		// Then scan files and directories and check what's different.

		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);

		Error ret = da->change_dir(cd);
		ERR_FAIL_COND_MSG(ret != OK, "Cannot change to '" + cd + "' folder.");

		da->list_dir_begin();
		while (true) {
			String f = da->get_next();
			if (f.is_empty()) {
				break;
			}

			if (da->current_is_hidden()) {
				continue;
			}

			if (da->current_is_dir()) {
				if (f.begins_with(".")) { // Ignore special and . / ..
					continue;
				}

				int idx = p_dir->find_dir_index(f);
				if (idx == -1) {
					const String dir_path = cd.path_join(f);
					if (_should_skip_directory(dir_path)) {
						continue;
					}

					EditorFileSystemDirectory *efd = memnew(EditorFileSystemDirectory);
					efd->parent = p_dir;
					efd->name = f;

					for (idx = 0; idx < p_dir->subdirs.size(); idx++) {
						if (efd->name.filenocasecmp_to(p_dir->subdirs[idx]->name) < 0) {
							break;
						}
					}
					if (idx == p_dir->subdirs.size()) {
						p_dir->subdirs.push_back(efd);
					} else {
						p_dir->subdirs.insert(idx, efd);
					}

					if (first_scan) {
						Vector<String> dirs = dir_path.substr(6, dir_path.length() - 6).split("/");
						ScannedDirectory *dir = first_scan_root_dir;
						for (String &D : dirs) {
							for (ScannedDirectory *SD : dir->subdirs) {
								if (SD->name == D) {
									dir = SD;
									break;
								}
							}
						}
						p_progress.hi += dir->count;
						diff_nb_files += dir->count;
						_process_file_system(dir, efd, p_progress, nullptr);
					} else {
						ScannedDirectory sd;
						sd.name = f;
						sd.full_path = dir_path;
						Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
						d->change_dir(dir_path);
						int nb_files_dir = _scan_new_dir(&sd, d);
						p_progress.hi += nb_files_dir;
						diff_nb_files += nb_files_dir;
						_process_file_system(&sd, efd, p_progress, nullptr);
					}

					_create_action(p_dir, nullptr, dir_path, ItemAction::ACTION_DIR_ADD);
				} else {
					p_dir->subdirs[idx]->verified = true;
				}
			} else {
				if (!_validate_file_extension(f, valid_extensions)) {
					continue; // Invalid.
				}

				int idx = p_dir->find_file_index(f);
				if (idx == -1) {
					// Never seen this file, add actition to add it.
					_file_info_add(p_dir, cd, f, true);
					diff_nb_files++;
				} else {
					p_dir->files[idx]->verified = true;
				}
			}
		}

		da->list_dir_end();
	}

	for (int i = p_dir->files.size() - 1; i >= 0; i--) {
		EditorFileInfo *fi = p_dir->files[i];
		const String path = cd.path_join(fi->file);

		if (updated_dir && !fi->verified) {
			// This file was removed, add action to remove it.
			_file_info_remove(fi, path, i);
			diff_nb_files--;
			continue;
		}
		if (fi->status & EditorFileInfo::FILE_ADD) {
			continue; // Newly added.
		}
		_file_info_update(fi, path);

		p_progress.increment();
	}

	for (int i = p_dir->subdirs.size() - 1; i >= 0; i--) {
		EditorFileSystemDirectory *sub_dir = p_dir->subdirs[i];
		const String sub_dir_path = cd.path_join(sub_dir->name);
		if ((updated_dir && !sub_dir->verified) || _should_skip_directory(sub_dir_path)) {
			// Add all the files of the folder to be sure _update_scan_actions process the removed files
			// for global class names.
			diff_nb_files -= _dir_info_remove(sub_dir, sub_dir_path, i);
			continue;
		}
		if (p_recursive || sub_dir->dirty) {
			_scan_fs_changes(sub_dir, p_progress, p_recursive);
		}
	}

	nb_files_total = MAX(nb_files_total + diff_nb_files, 0);
}

void EditorFileSystem::scan_fs_changes(ScanProgress &p_progress) {
	List<EditorFileSystemDirectory *>::Element *E = dirty_directories.front();
	ERR_FAIL_NULL(E);
	while (E) {
		EditorFileSystemDirectory *efsd = E->get();
		E = E->next();
		dirty_directories.pop_front();
		if (!efsd->dirty) {
			continue;
		}
		_scan_fs_changes(efsd, p_progress, efsd->recursive);
	}
	_update_scan_uid_actions();
}

void EditorFileSystem::_pending_scan_fs_changes(EditorFileSystemDirectory *p_dir, bool p_recursive) {
	if (full_scan_pending) {
		return;
	}
	set_process(true);
	if (p_dir == filesystem && p_recursive) {
		full_scan_pending = true;
		return;
	}
	p_dir->dirty = true;
	p_dir->recursive |= p_recursive;
	dirty_directories.push_back(p_dir);
	scan_changes_pending = true;
}

void EditorFileSystem::pending_scan_fs_changes(const String &p_dir, bool p_recursive) {
	if (full_scan_pending) {
		return;
	}
	EditorFileSystemDirectory *efd = get_filesystem_path(p_dir);
	ERR_FAIL_NULL(efd);
	_pending_scan_fs_changes(efd, p_recursive);
}

void EditorFileSystem::_delete_internal_files(const String &p_file) {
	if (FileAccess::exists(p_file)) {
		return; // It is just ignored because it is not supported, so there is no need to delete its internal files.
	}
	if (FileAccess::exists(p_file + ".import")) {
		List<String> paths;
		ResourceFormatImporter::get_singleton()->get_internal_resource_path_list(p_file, &paths);
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		for (const String &E : paths) {
			da->remove(E);
			da->remove(E.get_basename() + ".md5");
		}
		da->remove(p_file + ".import");
	}
	if (FileAccess::exists(p_file + ".uid")) {
		DirAccess::remove_absolute(p_file + ".uid");
	}
}

void EditorFileSystem::_thread_func_sources(void *_userdata) {
	EditorFileSystem *efs = (EditorFileSystem *)_userdata;
	if (efs->filesystem) {
		EditorProgressBG pr("sources", TTR("ScanSources"), 1000);
		ScanProgress sp;
		sp.progress = &pr;
		sp.hi = efs->nb_files_total;
		efs->scan_fs_changes(sp);
	}
	efs->scanning_changes_done.set();
}

bool EditorFileSystem::_remove_invalid_global_class_names(const HashSet<String> &p_existing_class_names) {
	LocalVector<StringName> global_classes;
	bool must_save = false;
	ScriptServer::get_global_class_list(global_classes);
	for (const StringName &class_name : global_classes) {
		if (!p_existing_class_names.has(class_name)) {
			ScriptServer::remove_global_class(class_name);
			EditorHelp::remove_doc(class_name);
			must_save = true;
		}
	}
	return must_save;
}

String EditorFileSystem::_get_file_by_class_name(EditorFileSystemDirectory *p_dir, const String &p_class_name, EditorFileInfo *&r_file_info) {
	for (EditorFileInfo *fi : p_dir->files) {
		if (fi->class_info.name == p_class_name) {
			r_file_info = fi;
			return p_dir->get_path().path_join(fi->file);
		}
	}

	for (EditorFileSystemDirectory *sub_dir : p_dir->subdirs) {
		String file = _get_file_by_class_name(sub_dir, p_class_name, r_file_info);
		if (!file.is_empty()) {
			return file;
		}
	}
	r_file_info = nullptr;
	return "";
}

void EditorFileSystem::_scan_dirs_changes(bool p_full_scan) {
	ERR_FAIL_NULL(filesystem);

	if (first_scan || // Prevent a premature changes scan from inhibiting the first full scan
			scanning || scanning_changes || thread.is_started() || updating_scan_actions) {
		if (p_full_scan) {
			full_scan_pending = true;
		}
		if (updating_scan_actions) {
			set_process(true);
		}
		return;
	}

	if (p_full_scan) {
		dirty_directories.clear();
		filesystem->dirty = true;
		filesystem->recursive = true;
		dirty_directories.push_back(filesystem);
	}

	sources_changed.clear();
	scanning_changes = true;
	scanning_changes_done.clear();

	if (!use_threads) {
		if (filesystem) {
			EditorProgressBG pr("sources", TTR("ScanSources"), 1000);
			ScanProgress sp;
			sp.progress = &pr;
			sp.hi = nb_files_total;
			scan_total = 0;
			scan_fs_changes(sp);
			if (_update_scan_actions()) {
				emit_signal(SNAME("filesystem_changed"));
			}
		}
		extensions_changed = false;
		scanning_changes = false;
		scanning_changes_done.set();
		emit_signal(SNAME("sources_changed"), sources_changed.size() > 0);
	} else {
		ERR_FAIL_COND(thread_sources.is_started());
		set_process(true);
		scan_total = 0;
		Thread::Settings s;
		s.priority = Thread::PRIORITY_LOW;
		thread_sources.start(_thread_func_sources, this, s);
	}
}

void EditorFileSystem::scan_changes() {
	_scan_dirs_changes();
}

void EditorFileSystem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_EXIT_TREE: {
			Thread &active_thread = thread.is_started() ? thread : thread_sources;
			if (use_threads && active_thread.is_started()) {
				while (!scanning_done.is_set()) {
					OS::get_singleton()->delay_usec(1000);
				}
				active_thread.wait_to_finish();
				WARN_PRINT("Scan thread aborted...");
				set_process(false);
			}

			dirty_directories.clear();
			scan_changes_pending = false;
		} break;

		case NOTIFICATION_PROCESS: {
			if (use_threads) {
				/** This hack exists because of the EditorProgress nature
				 *  of processing events recursively. This needs to be rewritten
				 *  at some point entirely, but in the meantime the following
				 *  hack prevents deadlock on import.
				 */

				static bool prevent_recursive_process_hack = false;
				if (prevent_recursive_process_hack) {
					break;
				}

				prevent_recursive_process_hack = true;

				if (scanning_changes && scanning_changes_done.is_set()) {
					set_process(false);

					if (thread_sources.is_started()) {
						thread_sources.wait_to_finish();
					}
					bool changed = _update_scan_actions();
					// Set first_scan to false before the signals so the function doing_first_scan can return false
					// in editor_node to start the export if needed.
					first_scan = false;
					extensions_changed = false;
					scanning_changes = false;
					ResourceImporter::load_on_startup = nullptr;
					if (changed) {
						emit_signal(SNAME("filesystem_changed"));
					}
					emit_signal(SNAME("sources_changed"), sources_changed.size() > 0);
				} else if (scanning && scanning_done.is_set()) {
					set_process(false);

					dirty_directories.clear();
					scan_changes_pending = false;
					thread.wait_to_finish();
					_update_scan_actions();
					// Update all icons so they are loaded for the FileSystemDock.
					_update_files_icon_path();
					extensions_changed = false;
					scanning = false;
					// Set first_scan to false before the signals so the function doing_first_scan can return false
					// in editor_node to start the export if needed.
					first_scan = false;
					ResourceImporter::load_on_startup = nullptr;
					emit_signal(SNAME("filesystem_changed"));
					emit_signal(SNAME("sources_changed"), sources_changed.size() > 0);
				}

				prevent_recursive_process_hack = false;
			}
			if (is_scanning()) {
				break;
			}
			if (!first_scan && need_update_extensions) {
				need_update_extensions = false;
				_reload_loader_or_saver();
				update_extensions();
				break;
			}
			// Another scan needs to be triggered during the scan.
			if (scan_changes_pending || full_scan_pending) {
				set_process(false);
				scan_changes_pending = false;
				bool full_scan = full_scan_pending;
				full_scan_pending = false;
				_scan_dirs_changes(full_scan);
				break;
			}
			// The calls to _update_scan_actions() triggered by update_files() are merged into one.
			// This is usually triggered when saving files.
			if (update_actions_queued && !updating_scan_actions) {
				updating_scan_actions = true;
				set_process(false);
				_update_scan_uid_actions();
				if (!first_scan && need_update_extensions) {
					need_update_extensions = false;
					update_extensions();
				}
				if (_update_scan_actions()) {
					emit_signal(SNAME("filesystem_changed"));
				}
				updating_scan_actions = false;
			}
		} break;
	}
}

bool EditorFileSystem::is_scanning() const {
	return scanning || scanning_changes || first_scan;
}

float EditorFileSystem::get_scanning_progress() const {
	return scan_total;
}

EditorFileSystemDirectory *EditorFileSystem::get_filesystem() {
	return filesystem;
}

void EditorFileSystem::_save_filesystem_cache(EditorFileSystemDirectory *p_dir, Ref<FileAccess> p_file) {
	if (!p_dir) {
		return; //none
	}
	p_file->store_line("::" + p_dir->get_path() + "::" + String::num_int64(p_dir->modified_time));

	for (int i = 0; i < p_dir->files.size(); i++) {
		const EditorFileInfo *file_info = p_dir->files[i];
		if (!file_info->import_group_file.is_empty()) {
			group_file_cache.insert(file_info->import_group_file);
		}

		String type = file_info->type;
		if (file_info->resource_script_class) {
			type += "/" + String(file_info->resource_script_class);
		}

		PackedStringArray cache_string;
		cache_string.append(file_info->file);
		cache_string.append(type);
		cache_string.append(itos(file_info->uid));
		cache_string.append(itos(file_info->modified_time));
		cache_string.append(itos(file_info->import_modified_time));
		cache_string.append(itos(file_info->import_valid));
		cache_string.append(file_info->import_group_file);
		cache_string.append(String("<>").join({ file_info->class_info.name, file_info->class_info.extends, file_info->class_info.icon_path, itos(file_info->class_info.is_abstract), itos(file_info->class_info.is_tool), file_info->import_md5, String("<*>").join(file_info->import_dest_paths) }));
		cache_string.append(String("<>").join(file_info->deps));

		p_file->store_line(String("::").join(cache_string));
	}

	for (int i = 0; i < p_dir->subdirs.size(); i++) {
		_save_filesystem_cache(p_dir->subdirs[i], p_file);
	}
}

bool EditorFileSystem::_find_file(const String &p_file, EditorFileSystemDirectory **r_d, int &r_file_pos) const {
	//todo make faster

	if (!filesystem || !scanning_done.is_set()) {
		return false;
	}

	String f = ProjectSettings::get_singleton()->localize_path(p_file);

	// Note: Only checks if base directory is case sensitive.
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	bool fs_case_sensitive = dir->is_case_sensitive("res://");

	if (!f.begins_with("res://")) {
		return false;
	}
	f = f.substr(6);
	f = f.replace_char('\\', '/');

	Vector<String> path = f.split("/");

	if (path.is_empty()) {
		return false;
	}
	String file = path[path.size() - 1];
	path.resize(path.size() - 1);

	EditorFileSystemDirectory *fs = filesystem;

	for (int i = 0; i < path.size(); i++) {
		if (path[i].begins_with(".")) {
			return false;
		}

		int idx = -1;
		for (int j = 0; j < fs->get_subdir_count(); j++) {
			if (fs_case_sensitive) {
				if (fs->get_subdir(j)->get_name() == path[i]) {
					idx = j;
					break;
				}
			} else {
				if (fs->get_subdir(j)->get_name().to_lower() == path[i].to_lower()) {
					idx = j;
					break;
				}
			}
		}

		if (idx == -1) {
			// Only create a missing directory in memory when it exists on disk.
			if (!dir->dir_exists(fs->get_path().path_join(path[i]))) {
				return false;
			}
			EditorFileSystemDirectory *efsd = memnew(EditorFileSystemDirectory);

			efsd->name = path[i];
			efsd->parent = fs;

			int idx2 = 0;
			for (int j = 0; j < fs->get_subdir_count(); j++) {
				if (efsd->name.filenocasecmp_to(fs->get_subdir(j)->get_name()) < 0) {
					break;
				}
				idx2++;
			}

			if (idx2 == fs->get_subdir_count()) {
				fs->subdirs.push_back(efsd);
			} else {
				fs->subdirs.insert(idx2, efsd);
			}
			fs = efsd;
		} else {
			fs = fs->get_subdir(idx);
		}
	}

	int cpos = -1;
	for (int i = 0; i < fs->files.size(); i++) {
		if (fs_case_sensitive) {
			if (fs->files[i]->file == file) {
				cpos = i;
				break;
			}
		} else {
			if (fs->files[i]->file.to_lower() == file.to_lower()) {
				cpos = i;
				break;
			}
		}
	}

	r_file_pos = cpos;
	*r_d = fs;

	return cpos != -1;
}

String EditorFileSystem::get_file_type(const String &p_file) const {
	EditorFileSystemDirectory *fs = nullptr;
	int cpos = -1;

	if (!_find_file(p_file, &fs, cpos)) {
		return "";
	}

	return fs->files[cpos]->type;
}

EditorFileSystemDirectory *EditorFileSystem::find_file(const String &p_file, int *r_index) const {
	if (!filesystem || !scanning_done.is_set()) {
		return nullptr;
	}

	EditorFileSystemDirectory *fs = nullptr;
	int cpos = -1;
	if (!_find_file(p_file, &fs, cpos)) {
		return nullptr;
	}

	if (r_index) {
		*r_index = cpos;
	}

	return fs;
}

ResourceUID::ID EditorFileSystem::get_file_uid(const String &p_path) const {
	int file_idx;
	EditorFileSystemDirectory *directory = find_file(p_path, &file_idx);

	if (!directory) {
		return ResourceUID::INVALID_ID;
	}
	return directory->files[file_idx]->uid;
}

EditorFileSystemDirectory *EditorFileSystem::get_filesystem_path(const String &p_path) {
	if (!filesystem || !scanning_done.is_set()) {
		return nullptr;
	}

	String f = ProjectSettings::get_singleton()->localize_path(p_path);

	if (!f.begins_with("res://")) {
		return nullptr;
	}

	f = f.substr(6);
	f = f.replace_char('\\', '/');
	if (f.is_empty()) {
		return filesystem;
	}

	if (f.ends_with("/")) {
		f = f.substr(0, f.length() - 1);
	}

	Vector<String> path = f.split("/");

	if (path.is_empty()) {
		return nullptr;
	}

	EditorFileSystemDirectory *fs = filesystem;

	for (int i = 0; i < path.size(); i++) {
		int idx = -1;
		for (int j = 0; j < fs->get_subdir_count(); j++) {
			if (fs->get_subdir(j)->get_name() == path[i]) {
				idx = j;
				break;
			}
		}

		if (idx == -1) {
			return nullptr;
		} else {
			fs = fs->get_subdir(idx);
		}
	}

	return fs;
}

Vector<String> EditorFileSystem::_get_dependencies(const String &p_path) {
	// Avoid error spam on first opening of a not yet imported project by treating the following situation
	// as a benign one, not letting the file open error happen: the resource is of an importable type but
	// it has not been imported yet.
	if (ResourceFormatImporter::get_singleton()->recognize_path(p_path)) {
		const String &internal_path = ResourceFormatImporter::get_singleton()->get_internal_resource_path(p_path);
		if (!internal_path.is_empty() && !FileAccess::exists(internal_path)) { // If path is empty (error), keep the code flow to the error.
			return Vector<String>();
		}
	}

	List<String> deps;
	ResourceLoader::get_dependencies(p_path, &deps);

	Vector<String> ret;
	for (const String &E : deps) {
		ret.push_back(E);
	}

	return ret;
}

EditorFileSystem::ScriptClassInfo EditorFileSystem::_get_global_script_class(const String &p_type, const String &p_path) const {
	ScriptClassInfo info;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptLanguage *lang = ScriptServer::get_language(i);
		if (lang->handles_global_class_type(p_type)) {
			info.lang = lang->get_name();
			info.name = lang->get_global_class_name(p_path, &info.extends, &info.icon_path, &info.is_abstract, &info.is_tool);
			break;
		}
	}
	return info;
}

void EditorFileSystem::_update_file_icon_path(EditorFileInfo *file_info) {
	String icon_path;
	if (file_info->resource_script_class != StringName()) {
		icon_path = EditorNode::get_editor_data().script_class_get_icon_path(file_info->resource_script_class);
	} else if (file_info->class_info.icon_path.is_empty() && !file_info->deps.is_empty()) {
		const String &script_dep = file_info->deps[0]; // Assuming the first dependency is a script.
		const String &script_path = script_dep.contains("::") ? script_dep.get_slice("::", 2) : script_dep;
		if (!script_path.is_empty()) {
			String *cached = file_icon_cache.getptr(script_path);
			if (cached) {
				icon_path = *cached;
			} else {
				if (ClassDB::is_parent_class(ResourceLoader::get_resource_type(script_path), Script::get_class_static())) {
					int script_file;
					EditorFileSystemDirectory *efsd = find_file(script_path, &script_file);
					if (efsd) {
						icon_path = efsd->files[script_file]->class_info.icon_path;
					}
				}
				file_icon_cache.insert(script_path, icon_path);
			}
		}
	}

	if (icon_path.is_empty() && !file_info->type.is_empty()) {
		Ref<Texture2D> icon = EditorNode::get_singleton()->get_class_icon(file_info->type);
		if (icon.is_valid()) {
			icon_path = icon->get_path();
		}
	}

	file_info->class_info.icon_path = icon_path;
}

void EditorFileSystem::_update_files_icon_path(EditorFileSystemDirectory *edp) {
	if (!edp) {
		edp = filesystem;
		file_icon_cache.clear();
	}
	for (EditorFileSystemDirectory *sub_dir : edp->subdirs) {
		_update_files_icon_path(sub_dir);
	}
	for (EditorFileInfo *fi : edp->files) {
		_update_file_icon_path(fi);
	}
}

void EditorFileSystem::_update_script_classes() {
	if (!script_classes_updated) {
		// Ensure the global class file is always present; it's essential for exports to work.
		if (!FileAccess::exists(ProjectSettings::get_singleton()->get_global_class_list_path())) {
			EditorNode::get_editor_data().script_class_save_global_classes();
		}
		return;
	}
	script_classes_updated = false;
	EditorNode::get_editor_data().script_class_save_global_classes();
	emit_signal("script_classes_updated");

	// TODO: The scripts have been modified, so a rescan may be necessary.

	// Rescan custom loaders and savers.
	// Doing the following here because the `filesystem_changed` signal fires multiple times and isn't always followed by script classes update.
	// So I thought it's better to do this when script classes really get updated
}

void EditorFileSystem::_update_script_documentation() {
	if (update_script_paths_documentation.is_empty()) {
		return;
	}

	MutexLock update_script_lock(update_script_mutex);

	EditorProgress *ep = nullptr;
	if (update_script_paths_documentation.size() > 1) {
		if (MessageQueue::get_singleton()->is_flushing()) {
			// Use background progress when message queue is flushing.
			ep = memnew(EditorProgress("update_script_paths_documentation", TTR("Updating scripts documentation"), update_script_paths_documentation.size(), false, true));
		} else {
			ep = memnew(EditorProgress("update_script_paths_documentation", TTR("Updating scripts documentation"), update_script_paths_documentation.size()));
		}
	}

	int step_count = 0;
	for (const String &path : update_script_paths_documentation) {
		int index = -1;
		EditorFileSystemDirectory *efd = find_file(path, &index);

		if (!efd || index < 0) {
			// The file was removed
			EditorHelp::remove_script_doc_by_path(path);
			continue;
		}

		if (efd->files[index]->status & EditorFileInfo::IS_PACKEDSCENE) {
			Ref<PackedScene> packed_scene = ResourceLoader::load(path);
			if (packed_scene.is_null()) {
				continue;
			}
			Ref<SceneState> state = packed_scene->get_state();
			if (state.is_null()) {
				continue;
			}
			Vector<Ref<Resource>> sub_resources = state->get_sub_resources();
			for (Ref<Resource> &sub_resource : sub_resources) {
				Ref<Script> scr = sub_resource;
				if (scr.is_null()) {
					continue;
				}
				for (const DocData::ClassDoc &cd : scr->get_documentation()) {
					EditorHelp::add_doc(cd);
					if (!first_scan) {
						// Update the documentation in the Script Editor if it is open.
						ScriptEditor::get_singleton()->update_doc(cd.name);
					}
				}
			}
			continue;
		}

		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptLanguage *lang = ScriptServer::get_language(i);
			if (lang->supports_documentation() && efd->files[index]->type == lang->get_type()) {
				bool should_reload_script = _should_reload_script(path);
				Ref<Script> scr = ResourceLoader::load(path);
				if (scr.is_null()) {
					continue;
				}
				if (should_reload_script) {
					// Reloading the script from disk. Otherwise, the ResourceLoader::load will
					// return the last loaded version of the script (without the modifications).
					scr->reload_from_file();
				}
				for (const DocData::ClassDoc &cd : scr->get_documentation()) {
					EditorHelp::add_doc(cd);
					if (!first_scan) {
						// Update the documentation in the Script Editor if it is open.
						ScriptEditor::get_singleton()->update_doc(cd.name);
					}
				}
			}
		}

		if (ep) {
			ep->step(efd->files[index]->file, step_count++, false);
		}
	}

	memdelete_notnull(ep);

	update_script_paths_documentation.clear();
}

bool EditorFileSystem::_should_reload_script(const String &p_path) {
	if (first_scan) {
		return false;
	}

	Ref<Script> scr = ResourceCache::get_ref(p_path);
	if (scr.is_null()) {
		// Not a script or not already loaded.
		return false;
	}

	// Scripts are reloaded via the script editor if they are currently opened.
	if (ScriptEditor::get_singleton()->get_open_scripts().has(scr)) {
		return false;
	}

	return true;
}

void EditorFileSystem::_process_update_pending() {
	_update_script_classes();
	// Parse documentation second, as it requires the class names to be loaded
	// because _update_script_documentation loads the scripts completely.
	if (!EditorNode::is_cmdline_mode()) {
		_update_script_documentation();
		_update_pending_scene_groups();
	}
}

void EditorFileSystem::_update_scene_groups() {
	if (update_scene_paths.is_empty()) {
		return;
	}

	EditorProgress *ep = nullptr;
	if (update_scene_paths.size() > 20) {
		ep = memnew(EditorProgress("update_scene_groups", TTR("Updating Scene Groups"), update_scene_paths.size()));
	}
	int step_count = 0;

	{
		MutexLock update_scene_lock(update_scene_mutex);
		for (const String &path : update_scene_paths) {
			ProjectSettings::get_singleton()->remove_scene_groups_cache(path);

			int index = -1;
			EditorFileSystemDirectory *efd = find_file(path, &index);

			if (!efd || index < 0) {
				// The file was removed.
				continue;
			}

			const HashSet<StringName> scene_groups = PackedScene::get_scene_groups(path);
			if (!scene_groups.is_empty()) {
				ProjectSettings::get_singleton()->add_scene_groups_cache(path, scene_groups);
			}

			if (ep) {
				ep->step(efd->files[index]->file, step_count++, false);
			}
		}

		memdelete_notnull(ep);
		update_scene_paths.clear();
	}

	ProjectSettings::get_singleton()->save_scene_groups_cache();
}

void EditorFileSystem::_update_pending_scene_groups() {
	if (!FileAccess::exists(ProjectSettings::get_singleton()->get_scene_groups_cache_path())) {
		_get_all_scenes(get_filesystem(), update_scene_paths);
	}
	_update_scene_groups();
}

void EditorFileSystem::_queue_update_scene_groups(const String &p_path) {
	{
		MutexLock update_scene_lock(update_scene_mutex);
		update_scene_paths.insert(p_path);
	}
	{
		MutexLock update_script_lock(update_script_mutex);
		update_script_paths_documentation.insert(p_path);
	}
}

void EditorFileSystem::_get_all_scenes(EditorFileSystemDirectory *p_dir, HashSet<String> &r_list) {
	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->files[i]->status & EditorFileInfo::IS_PACKEDSCENE) {
			r_list.insert(p_dir->get_file_path(i));
		}
	}

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_get_all_scenes(p_dir->get_subdir(i), r_list);
	}
}

void EditorFileSystem::update_file(const String &p_file) {
	ERR_FAIL_COND(p_file.is_empty());
	update_files({ p_file });
}

void EditorFileSystem::update_files(const Vector<String> &p_files) {
	bool updated = false;
	for (const String &file : p_files) {
		ERR_CONTINUE(file.is_empty());
		EditorFileSystemDirectory *fs = nullptr;
		int cpos = -1;

		if (!_find_file(file, &fs, cpos)) {
			if (!fs) {
				continue;
			}
		}

		if (!FileAccess::exists(file)) {
			// Was removed.
			if (cpos == -1) { // Might've never been part of the editor file system (*.* files deleted in Open dialog).
				_delete_internal_files(file);
			} else {
				_file_info_remove(fs->files[cpos], file, cpos);
				updated = true;
			}
		} else {
			if (cpos == -1) {
				// The file did not exist, it was added.
				_file_info_add(fs, file.get_base_dir(), file.get_file(), true);
			} else {
				_file_info_update(fs->files[cpos], file);
			}
			updated = true;
		}
	}
	if (!updated || is_scanning() || update_actions_queued) {
		return;
	}
	set_process(true);
	update_actions_queued = true;
}

HashSet<String> EditorFileSystem::get_valid_extensions() const {
	return valid_extensions;
}

void EditorFileSystem::_register_global_class_script(const String &p_search_path, const String &p_target_path, const EditorFileInfo *p_fi) {
	ScriptServer::remove_global_class_by_path(p_search_path); // First remove, just in case it changed.

	if (p_fi->class_info.name.is_empty()) {
		return;
	}

	String lang = p_fi->class_info.lang;
	if (lang.is_empty()) {
		for (int j = 0; j < ScriptServer::get_language_count(); j++) {
			if (ScriptServer::get_language(j)->handles_global_class_type(p_fi->type)) {
				lang = ScriptServer::get_language(j)->get_name();
				break;
			}
		}
	}
	ERR_FAIL_COND(lang.is_empty()); // No lang found that can handle this global class.

	ScriptServer::add_global_class(p_fi->class_info.name, p_fi->class_info.extends, lang, p_target_path, p_fi->class_info.is_abstract, p_fi->class_info.is_tool, p_fi->uid);
	EditorNode::get_editor_data().script_class_set_icon_path(p_fi->class_info.name, p_fi->class_info.icon_path);
	EditorNode::get_editor_data().script_class_set_name(p_target_path, p_fi->class_info.name);
}

void EditorFileSystem::register_global_class_script(const String &p_search_path, const String &p_target_path) {
	int index_file;
	EditorFileSystemDirectory *efsd = find_file(p_search_path, &index_file);
	if (efsd) {
		const EditorFileInfo *fi = efsd->files[index_file];
		EditorFileSystem::get_singleton()->_register_global_class_script(p_search_path, p_target_path, fi);
	} else {
		ScriptServer::remove_global_class_by_path(p_search_path);
	}
}

Error EditorFileSystem::_reimport_group(const String &p_group_file, const Vector<String> &p_files) {
	String importer_name;

	HashMap<String, HashMap<StringName, Variant>> source_file_options;
	HashMap<String, ResourceUID::ID> uids;
	HashMap<String, String> base_paths;
	for (int i = 0; i < p_files.size(); i++) {
		Ref<ConfigFile> config;
		config.instantiate();
		Error err = config->load(p_files[i] + ".import");
		ERR_CONTINUE(err != OK);
		ERR_CONTINUE(!config->has_section_key("remap", "importer"));
		String file_importer_name = config->get_value("remap", "importer");
		ERR_CONTINUE(file_importer_name.is_empty());

		if (!importer_name.is_empty() && importer_name != file_importer_name) {
			EditorNode::get_singleton()->show_warning(vformat(TTR("There are multiple importers for different types pointing to file %s, import aborted"), p_group_file));
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		}

		ResourceUID::ID uid = ResourceUID::INVALID_ID;

		if (config->has_section_key("remap", "uid")) {
			String uidt = config->get_value("remap", "uid");
			uid = ResourceUID::get_singleton()->text_to_id(uidt);
		}

		uids[p_files[i]] = uid;

		source_file_options[p_files[i]] = HashMap<StringName, Variant>();
		importer_name = file_importer_name;

		if (importer_name == "keep" || importer_name == "skip") {
			continue; //do nothing
		}

		Ref<ResourceImporter> importer = ResourceFormatImporter::get_singleton()->get_importer_by_name(importer_name);
		ERR_FAIL_COND_V(importer.is_null(), ERR_FILE_CORRUPT);
		List<ResourceImporter::ImportOption> options;
		importer->get_import_options(p_files[i], &options);
		//set default values
		for (const ResourceImporter::ImportOption &E : options) {
			source_file_options[p_files[i]][E.option.name] = E.default_value;
		}

		if (config->has_section("params")) {
			Vector<String> sk = config->get_section_keys("params");
			for (const String &param : sk) {
				Variant value = config->get_value("params", param);
				//override with whatever is in file
				source_file_options[p_files[i]][param] = value;
			}
		}

		base_paths[p_files[i]] = ResourceFormatImporter::get_singleton()->get_import_base_path(p_files[i]);
	}

	if (importer_name == "keep" || importer_name == "skip") {
		return OK; // (do nothing)
	}

	ERR_FAIL_COND_V(importer_name.is_empty(), ERR_UNCONFIGURED);

	Ref<ResourceImporter> importer = ResourceFormatImporter::get_singleton()->get_importer_by_name(importer_name);

	Error err = importer->import_group_file(p_group_file, source_file_options, base_paths);

	//all went well, overwrite config files with proper remaps and md5s
	for (const KeyValue<String, HashMap<StringName, Variant>> &E : source_file_options) {
		const String &file = E.key;
		String base_path = ResourceFormatImporter::get_singleton()->get_import_base_path(file);
		Vector<String> dest_paths;
		ResourceUID::ID uid = uids[file];
		{
			Ref<FileAccess> f = FileAccess::open(file + ".import", FileAccess::WRITE);
			ERR_FAIL_COND_V_MSG(f.is_null(), ERR_FILE_CANT_OPEN, "Cannot open import file '" + file + ".import'.");

			//write manually, as order matters ([remap] has to go first for performance).
			f->store_line("[remap]");
			f->store_line("");
			f->store_line("importer=\"" + importer->get_importer_name() + "\"");
			int version = importer->get_format_version();
			if (version > 0) {
				f->store_line("importer_version=" + itos(version));
			}
			if (!importer->get_resource_type().is_empty()) {
				f->store_line("type=\"" + importer->get_resource_type() + "\"");
			}

			if (uid == ResourceUID::INVALID_ID) {
				uid = ResourceUID::get_singleton()->create_id_for_path(file);
			}

			f->store_line("uid=\"" + ResourceUID::get_singleton()->id_to_text(uid) + "\""); // Store in readable format.

			if (err == OK) {
				String path = base_path + "." + importer->get_save_extension();
				f->store_line("path=\"" + path + "\"");
				dest_paths.push_back(path);
			}

			f->store_line("group_file=" + Variant(p_group_file).get_construct_string());

			if (err == OK) {
				f->store_line("valid=true");
			} else {
				f->store_line("valid=false");
			}
			f->store_line("[deps]\n");

			f->store_line("");

			f->store_line("source_file=" + Variant(file).get_construct_string());
			if (dest_paths.size()) {
				Array dp;
				for (int i = 0; i < dest_paths.size(); i++) {
					dp.push_back(dest_paths[i]);
				}
				f->store_line("dest_files=" + Variant(dp).get_construct_string() + "\n");
			}
			f->store_line("[params]");
			f->store_line("");

			//store options in provided order, to avoid file changing. Order is also important because first match is accepted first.

			List<ResourceImporter::ImportOption> options;
			importer->get_import_options(file, &options);
			//set default values
			for (const ResourceImporter::ImportOption &F : options) {
				String base = F.option.name;
				Variant v = F.default_value;
				if (source_file_options[file].has(base)) {
					v = source_file_options[file][base];
				}
				String value;
				VariantWriter::write_to_string(v, value);
				f->store_line(base + "=" + value);
			}
		}

		// Store the md5's of the various files. These are stored separately so that the .import files can be version controlled.
		{
			Ref<FileAccess> md5s = FileAccess::open(base_path + ".md5", FileAccess::WRITE);
			ERR_FAIL_COND_V_MSG(md5s.is_null(), ERR_FILE_CANT_OPEN, "Cannot open MD5 file '" + base_path + ".md5'.");

			md5s->store_line("source_md5=\"" + FileAccess::get_md5(file) + "\"");
			if (dest_paths.size()) {
				md5s->store_line("dest_md5=\"" + FileAccess::get_multiple_md5(dest_paths) + "\"\n");
			}
		}

		EditorFileSystemDirectory *fs = nullptr;
		int cpos = -1;
		bool found = _find_file(file, &fs, cpos);
		ERR_FAIL_COND_V_MSG(!found, ERR_UNCONFIGURED, vformat("Can't find file '%s' during group reimport.", file));

		EditorFileInfo *fi = fs->files[cpos];
		//update modified times, to avoid reimport
		fi->modified_time = FileAccess::get_modified_time(file);
		fi->import_modified_time = FileAccess::get_modified_time(file + ".import");
		fi->import_md5 = FileAccess::get_md5(file + ".import");
		fi->import_dest_paths = dest_paths;
		fi->deps = _get_dependencies(file);
		fi->uid = uid;
		_type_analysis(fi, importer->get_resource_type());
		fi->import_valid = err == OK;

		if (ResourceUID::get_singleton()->has_id(uid)) {
			ResourceUID::get_singleton()->set_id(uid, file);
		} else {
			ResourceUID::get_singleton()->add_id(uid, file);
		}

		//if file is currently up, maybe the source it was loaded from changed, so import math must be updated for it
		//to reload properly
		Ref<Resource> r = ResourceCache::get_ref(file);

		if (r.is_valid()) {
			if (!r->get_import_path().is_empty()) {
				String dst_path = ResourceFormatImporter::get_singleton()->get_internal_resource_path(file);
				r->set_import_path(dst_path);
				r->set_import_last_modified_time(0);
			}
		}

		EditorResourcePreview::get_singleton()->check_for_invalidation(file);
	}

	return err;
}

Error EditorFileSystem::_reimport_file(const String &p_file, const HashMap<StringName, Variant> &p_custom_options, const String &p_custom_importer, Variant *p_generator_parameters, bool p_update_file_system) {
	print_verbose(vformat("EditorFileSystem: Importing file: %s", p_file));
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	EditorFileSystemDirectory *fs = nullptr;
	int cpos = -1;
	EditorFileInfo *fi = nullptr;
	if (p_update_file_system) {
		bool found = _find_file(p_file, &fs, cpos);
		ERR_FAIL_COND_V_MSG(!found, ERR_FILE_NOT_FOUND, vformat("Can't find file '%s' during file reimport.", p_file));
		fi = fs->files[cpos];
	}

	//try to obtain existing params

	HashMap<StringName, Variant> params = p_custom_options;
	String importer_name; //empty by default though

	if (!p_custom_importer.is_empty()) {
		importer_name = p_custom_importer;
	}

	ResourceUID::ID uid = p_update_file_system ? fi->uid : ResourceUID::INVALID_ID;
	Variant generator_parameters;
	String group_file;
	if (p_generator_parameters) {
		generator_parameters = *p_generator_parameters;
	}

	if (FileAccess::exists(p_file + ".import")) {
		//use existing
		Ref<ConfigFile> cf;
		cf.instantiate();
		Error err = cf->load(p_file + ".import");
		if (err == OK) {
			if (cf->has_section("params")) {
				Vector<String> sk = cf->get_section_keys("params");
				for (const String &E : sk) {
					if (!params.has(E)) {
						params[E] = cf->get_value("params", E);
					}
				}
			}

			if (cf->has_section("remap")) {
				if (p_custom_importer.is_empty()) {
					importer_name = cf->get_value("remap", "importer");
				}

				if (!p_update_file_system && cf->has_section_key("remap", "uid")) {
					String uidt = cf->get_value("remap", "uid");
					uid = ResourceUID::get_singleton()->text_to_id(uidt);
				}

				if (cf->has_section_key("remap", "group_file")) {
					group_file = cf->get_value("remap", "group_file");
				}

				if (!p_generator_parameters) {
					if (cf->has_section_key("remap", "generator_parameters")) {
						generator_parameters = cf->get_value("remap", "generator_parameters");
					}
				}
			}
		}
	}

	if (importer_name == "keep" || importer_name == "skip") {
		//keep files, do nothing.
		if (p_update_file_system) {
			fi->modified_time = FileAccess::get_modified_time(p_file);
			fi->import_modified_time = FileAccess::get_modified_time(p_file + ".import");
			fi->import_md5 = FileAccess::get_md5(p_file + ".import");
			fi->import_dest_paths = Vector<String>();
			fi->deps.clear();
			fi->type = "";
			fi->import_valid = false;
			EditorResourcePreview::get_singleton()->check_for_invalidation(p_file);
		}
		return OK;
	}
	Ref<ResourceImporter> importer;
	bool load_default = false;
	//find the importer
	if (!importer_name.is_empty()) {
		importer = ResourceFormatImporter::get_singleton()->get_importer_by_name(importer_name);
	}

	if (importer.is_null()) {
		//not found by name, find by extension
		importer = ResourceFormatImporter::get_singleton()->get_importer_by_file(p_file);
		load_default = true;
		if (importer.is_null()) {
			ERR_FAIL_V_MSG(ERR_FILE_CANT_OPEN, "BUG: File queued for import, but can't be imported, importer for type '" + importer_name + "' not found.");
		}
	}

	if (FileAccess::exists(p_file + ".import")) {
		// We only want to handle compat for existing files, not new ones.
		importer->handle_compatibility_options(params);
	}

	//mix with default params, in case a parameter is missing

	List<ResourceImporter::ImportOption> opts;
	importer->get_import_options(p_file, &opts);
	for (const ResourceImporter::ImportOption &E : opts) {
		if (!params.has(E.option.name)) { //this one is not present
			params[E.option.name] = E.default_value;
		}
	}

	if (load_default && ProjectSettings::get_singleton()->has_setting("importer_defaults/" + importer->get_importer_name())) {
		//use defaults if exist
		Dictionary d = GLOBAL_GET("importer_defaults/" + importer->get_importer_name());

		for (const KeyValue<Variant, Variant> &kv : d) {
			params[kv.key] = kv.value;
		}
	}

	if (uid == ResourceUID::INVALID_ID) {
		uid = ResourceUID::get_singleton()->create_id_for_path(p_file);
	}

	//finally, perform import!!
	String base_path = ResourceFormatImporter::get_singleton()->get_import_base_path(p_file);

	List<String> import_variants;
	List<String> gen_files;
	Variant meta;
	Error err = importer->import(uid, p_file, base_path, params, &import_variants, &gen_files, &meta);

	// As import is complete, save the .import file.

	Vector<String> dest_paths;
	{
		Ref<FileAccess> f = FileAccess::open(p_file + ".import", FileAccess::WRITE);
		ERR_FAIL_COND_V_MSG(f.is_null(), ERR_FILE_CANT_OPEN, "Cannot open file from path '" + p_file + ".import'.");

		// Write manually, as order matters ([remap] has to go first for performance).
		f->store_line("[remap]");
		f->store_line("");
		f->store_line("importer=\"" + importer->get_importer_name() + "\"");
		int version = importer->get_format_version();
		if (version > 0) {
			f->store_line("importer_version=" + itos(version));
		}
		if (!importer->get_resource_type().is_empty()) {
			f->store_line("type=\"" + importer->get_resource_type() + "\"");
		}

		f->store_line("uid=\"" + ResourceUID::get_singleton()->id_to_text(uid) + "\""); // Store in readable format.
		if (!group_file.is_empty()) {
			f->store_line("group_file=\"" + group_file + "\"");
		}

		if (err == OK) {
			if (importer->get_save_extension().is_empty()) {
				//no path
			} else if (import_variants.size()) {
				//import with variants
				for (const String &E : import_variants) {
					String path = base_path.c_escape() + "." + E + "." + importer->get_save_extension();

					f->store_line("path." + E + "=\"" + path + "\"");
					dest_paths.push_back(path);
				}
			} else {
				String path = base_path + "." + importer->get_save_extension();
				f->store_line("path=\"" + path + "\"");
				dest_paths.push_back(path);
			}

		} else {
			f->store_line("valid=false");
		}

		if (meta != Variant()) {
			f->store_line("metadata=" + meta.get_construct_string());
		}

		if (generator_parameters != Variant()) {
			f->store_line("generator_parameters=" + generator_parameters.get_construct_string());
		}

		f->store_line("");

		f->store_line("[deps]\n");

		if (gen_files.size()) {
			Array genf;
			for (const String &E : gen_files) {
				genf.push_back(E);
				if (dest_paths.has(E)) {
					continue; // Files in formats such as obj will generate duplicate paths.
				}
				dest_paths.push_back(E);
			}

			String value;
			VariantWriter::write_to_string(genf, value);
			f->store_line("files=" + value);
			f->store_line("");
		}

		f->store_line("source_file=" + Variant(p_file).get_construct_string());

		if (dest_paths.size()) {
			Array dp;
			for (int i = 0; i < dest_paths.size(); i++) {
				dp.push_back(dest_paths[i]);
			}
			f->store_line("dest_files=" + Variant(dp).get_construct_string());
		}
		f->store_line("");

		f->store_line("[params]");
		f->store_line("");

		// Store options in provided order, to avoid file changing. Order is also important because first match is accepted first.

		for (const ResourceImporter::ImportOption &E : opts) {
			String base = E.option.name;
			String value;
			VariantWriter::write_to_string(params[base], value);
			f->store_line(base + "=" + value);
		}
	}

	// Store the md5's of the various files. These are stored separately so that the .import files can be version controlled.
	{
		Ref<FileAccess> md5s = FileAccess::open(base_path + ".md5", FileAccess::WRITE);
		ERR_FAIL_COND_V_MSG(md5s.is_null(), ERR_FILE_CANT_OPEN, "Cannot open MD5 file '" + base_path + ".md5'.");

		md5s->store_line("source_md5=\"" + FileAccess::get_md5(p_file) + "\"");
		if (dest_paths.size()) {
			md5s->store_line("dest_md5=\"" + FileAccess::get_multiple_md5(dest_paths) + "\"\n");
		}
	}

	if (p_update_file_system) {
		// Update modified times, to avoid reimport.
		fi->modified_time = FileAccess::get_modified_time(p_file);
		fi->import_modified_time = FileAccess::get_modified_time(p_file + ".import");
		fi->import_md5 = FileAccess::get_md5(p_file + ".import");
		fi->import_dest_paths = dest_paths;
		fi->deps = _get_dependencies(p_file);
		_type_analysis(fi, importer->get_resource_type());
		fi->uid = uid;
		fi->import_valid = (fi->status & EditorFileInfo::AS_RESOURCE) || ResourceLoader::is_import_valid(p_file);
	}

	for (const String &path : gen_files) {
		Ref<Resource> cached = ResourceCache::get_ref(path);
		if (cached.is_valid()) {
			cached->reload_from_file();
		}
	}

	if (ResourceUID::get_singleton()->has_id(uid)) {
		ResourceUID::get_singleton()->set_id(uid, p_file);
	} else {
		ResourceUID::get_singleton()->add_id(uid, p_file);
	}

	// If file is currently up, maybe the source it was loaded from changed, so import math must be updated for it
	// to reload properly.
	Ref<Resource> r = ResourceCache::get_ref(p_file);
	if (r.is_valid()) {
		if (!r->get_import_path().is_empty()) {
			String dst_path = ResourceFormatImporter::get_singleton()->get_internal_resource_path(p_file);
			r->set_import_path(dst_path);
			r->set_import_last_modified_time(0);
		}
	}

	EditorResourcePreview::get_singleton()->check_for_invalidation(p_file);

	print_verbose(vformat("EditorFileSystem: \"%s\" import took %d ms.", p_file, OS::get_singleton()->get_ticks_msec() - start_time));

	ERR_FAIL_COND_V_MSG(err != OK, ERR_FILE_UNRECOGNIZED, "Error importing '" + p_file + "'.");
	return OK;
}

void EditorFileSystem::_find_group_files(EditorFileSystemDirectory *efd, HashMap<String, Vector<String>> &group_files, HashSet<String> &groups_to_reimport) {
	int fc = efd->files.size();
	const EditorFileInfo *const *files = efd->files.ptr();
	for (int i = 0; i < fc; i++) {
		if (groups_to_reimport.has(files[i]->import_group_file)) {
			if (!group_files.has(files[i]->import_group_file)) {
				group_files[files[i]->import_group_file] = Vector<String>();
			}
			group_files[files[i]->import_group_file].push_back(efd->get_file_path(i));
		}
	}

	for (int i = 0; i < efd->get_subdir_count(); i++) {
		_find_group_files(efd->get_subdir(i), group_files, groups_to_reimport);
	}
}

void EditorFileSystem::reimport_file_with_custom_parameters(const String &p_file, const String &p_importer, const HashMap<StringName, Variant> &p_custom_params) {
	Vector<String> reloads;
	reloads.append(p_file);

	// Emit the resource_reimporting signal for the single file before the actual importation.
	emit_signal(SNAME("resources_reimporting"), reloads);

	_reimport_file(p_file, p_custom_params, p_importer);

	// Emit the resource_reimported signal for the single file we just reimported.
	emit_signal(SNAME("resources_reimported"), reloads);
}

Error EditorFileSystem::_copy_file(const String &p_from, const String &p_to) {
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (FileAccess::exists(p_from + ".import")) {
		Error err = da->copy(p_from, p_to);
		if (err != OK) {
			return err;
		}

		// Save the new .import file
		Ref<ConfigFile> cfg;
		cfg.instantiate();
		cfg->load(p_from + ".import");
		String importer_name = cfg->get_value("remap", "importer");

		if (importer_name == "keep" || importer_name == "skip") {
			err = da->copy(p_from + ".import", p_to + ".import");
			return err;
		}

		// Roll a new uid for this copied .import file to avoid conflict.
		ResourceUID::ID res_uid = ResourceUID::get_singleton()->create_id_for_path(p_to);
		cfg->set_value("remap", "uid", ResourceUID::get_singleton()->id_to_text(res_uid));
		err = cfg->save(p_to + ".import");
		if (err != OK) {
			return err;
		}

		// Make sure it's immediately added to the map so we can remap dependencies if we want to after this.
		ResourceUID::get_singleton()->add_id(res_uid, p_to);
	} else if (ResourceLoader::get_resource_uid(p_from) == ResourceUID::INVALID_ID) {
		// Files which do not use an uid can just be copied.
		Error err = da->copy(p_from, p_to);
		if (err != OK) {
			return err;
		}
	} else {
		// Load the resource and save it again in the new location (this generates a new UID).
		Error err = OK;
		Ref<Resource> res = ResourceCache::get_ref(p_from);
		if (res.is_null()) {
			res = ResourceLoader::load(p_from, "", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
		} else {
			bool edited = false;
			List<Ref<Resource>> cached;
			ResourceCache::get_cached_resources(&cached);
			for (Ref<Resource> &resource : cached) {
				if (!resource->is_edited()) {
					continue;
				}
				if (!resource->get_path().begins_with(p_from)) {
					continue;
				}
				// The resource or one of its built-in resources is edited.
				edited = true;
				resource->set_edited(false);
			}

			if (edited) {
				// Save cached resources to prevent changes from being lost and to prevent discrepancies.
				EditorNode::get_singleton()->save_resource(res);
			}
		}
		if (err == OK && res.is_valid()) {
			err = ResourceSaver::save(res, p_to, ResourceSaver::FLAG_COMPRESS);
			if (err != OK) {
				return err;
			}
		} else if (err != OK) {
			// When loading files like text files the error is OK but the resource is still null.
			// We can ignore such files.
			return err;
		}
	}
	return OK;
}

bool EditorFileSystem::_copy_directory(const String &p_from, const String &p_to, HashMap<String, String> *p_files) {
	Ref<DirAccess> old_dir = DirAccess::open(p_from);
	ERR_FAIL_COND_V(old_dir.is_null(), false);

	Error err = make_dir_recursive(p_to);
	if (err != OK && err != ERR_ALREADY_EXISTS) {
		return false;
	}

	bool success = true;
	old_dir->set_include_navigational(false);
	old_dir->list_dir_begin();

	for (String F = old_dir->_get_next(); !F.is_empty(); F = old_dir->_get_next()) {
		if (old_dir->current_is_dir()) {
			success = _copy_directory(p_from.path_join(F), p_to.path_join(F), p_files) && success;
		} else if (F.get_extension() != "import" && F.get_extension() != "uid") {
			(*p_files)[p_from.path_join(F)] = p_to.path_join(F);
		}
	}
	return success;
}

void EditorFileSystem::_queue_refresh_filesystem() {
	if (refresh_queued) {
		return;
	}
	refresh_queued = true;
	get_tree()->connect(SNAME("process_frame"), callable_mp(this, &EditorFileSystem::_refresh_filesystem), CONNECT_ONE_SHOT);
}

void EditorFileSystem::_refresh_filesystem() {
	for (const ObjectID &id : folders_to_sort) {
		EditorFileSystemDirectory *dir = ObjectDB::get_instance<EditorFileSystemDirectory>(id);
		if (dir) {
			dir->subdirs.sort_custom<DirectoryComparator>();
		}
	}
	folders_to_sort.clear();

	if (dirty_directories.is_empty()) {
		// Avoid calling _update_scan_actions() repeatedly.
		_update_scan_actions();
	}
	emit_signal(SNAME("filesystem_changed"));

	refresh_queued = false;
}

void EditorFileSystem::_reimport_thread(uint32_t p_index, ImportThreadData *p_import_data) {
	ResourceLoader::set_is_import_thread(true);
	int file_idx = p_import_data->reimport_from + int(p_index);
	_reimport_file(p_import_data->reimport_files[file_idx].path);
	ResourceLoader::set_is_import_thread(false);

	p_import_data->imported_sem->post();
}

void EditorFileSystem::reimport_files(const Vector<String> &p_files) {
	ERR_FAIL_COND_MSG(importing, "Attempted to call reimport_files() recursively, this is not allowed.");
	importing = true;

	Vector<String> reloads;

	EditorProgress *ep = memnew(EditorProgress("reimport", TTR("(Re)Importing Assets"), p_files.size()));

	// The method reimport_files runs on the main thread, and if VSync is enabled
	// or Update Continuously is disabled, Main::Iteration takes longer each frame.
	// Each EditorProgress::step can trigger a redraw, and when there are many files to import,
	// this could lead to a slow import process, especially when the editor is unfocused.
	// Temporarily disabling VSync and low_processor_usage_mode while reimporting fixes this.
	const bool old_low_processor_usage_mode = OS::get_singleton()->is_in_low_processor_usage_mode();
	const DisplayServer::VSyncMode old_vsync_mode = DisplayServer::get_singleton()->window_get_vsync_mode(DisplayServer::MAIN_WINDOW_ID);
	OS::get_singleton()->set_low_processor_usage_mode(false);
	DisplayServer::get_singleton()->window_set_vsync_mode(DisplayServer::VSyncMode::VSYNC_DISABLED);

	Vector<ImportFile> reimport_files;

	HashSet<String> groups_to_reimport;

	for (int i = 0; i < p_files.size(); i++) {
		ep->step(TTR("Preparing files to reimport..."), i, false);

		String file = p_files[i];

		ResourceUID::ID uid = ResourceUID::get_singleton()->text_to_id(file);
		if (uid != ResourceUID::INVALID_ID && ResourceUID::get_singleton()->has_id(uid)) {
			file = ResourceUID::get_singleton()->get_id_path(uid);
		}

		String group_file = ResourceFormatImporter::get_singleton()->get_import_group_file(file);

		if (group_file_cache.has(file)) {
			// Maybe the file itself is a group!
			groups_to_reimport.insert(file);
			// Groups do not belong to groups.
			group_file = String();
		} else if (groups_to_reimport.has(file)) {
			// Groups do not belong to groups.
			group_file = String();
		} else if (!group_file.is_empty()) {
			// It's a group file, add group to import and skip this file.
			groups_to_reimport.insert(group_file);
		} else {
			// It's a regular file.
			ImportFile ifile;
			ifile.path = file;
			ResourceFormatImporter::get_singleton()->get_import_order_threads_and_importer(file, ifile.order, ifile.threaded, ifile.importer);
			reloads.push_back(file);
			reimport_files.push_back(ifile);
		}

		// Group may have changed, so also update group reference.
		EditorFileSystemDirectory *fs = nullptr;
		int cpos = -1;
		if (_find_file(file, &fs, cpos)) {
			fs->files.write[cpos]->import_group_file = group_file;
		}
	}

	reimport_files.sort();

	ep->step(TTR("Executing pre-reimport operations..."), 0, true);

	// Emit the resource_reimporting signal for the single file before the actual importation.
	emit_signal(SNAME("resources_reimporting"), reloads);

#ifdef THREADS_ENABLED
	bool use_multiple_threads = GLOBAL_GET("editor/import/use_multiple_threads");
#else
	bool use_multiple_threads = false;
#endif

	int from = 0;
	Semaphore imported_sem;
	for (int i = 0; i < reimport_files.size(); i++) {
		if (groups_to_reimport.has(reimport_files[i].path)) {
			from = i + 1;
			continue;
		}

		if (use_multiple_threads && reimport_files[i].threaded) {
			if (i + 1 == reimport_files.size() || reimport_files[i + 1].importer != reimport_files[from].importer || groups_to_reimport.has(reimport_files[i + 1].path)) {
				if (from - i == 0) {
					// Single file, do not use threads.
					ep->step(reimport_files[i].path.get_file(), i, false);
					_reimport_file(reimport_files[i].path);
				} else {
					Ref<ResourceImporter> importer = ResourceFormatImporter::get_singleton()->get_importer_by_name(reimport_files[from].importer);
					if (importer.is_null()) {
						ERR_PRINT(vformat("Invalid importer for \"%s\".", reimport_files[from].importer));
						from = i + 1;
						continue;
					}

					importer->import_threaded_begin();

					ImportThreadData tdata;
					tdata.reimport_from = from;
					tdata.reimport_files = reimport_files.ptr();
					tdata.imported_sem = &imported_sem;

					int item_count = i - from + 1;
					WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_template_group_task(this, &EditorFileSystem::_reimport_thread, &tdata, item_count, -1, false, vformat(TTR("Import resources of type: %s"), reimport_files[from].importer));

					int imported_count = 0;
					while (true) {
						while (true) {
							ep->step(reimport_files[imported_count].path.get_file(), from + imported_count, false);
							if (imported_sem.try_wait()) {
								imported_count++;
								break;
							}
						}
						if (imported_count == item_count) {
							break;
						}
					}

					WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);
					DEV_ASSERT(!imported_sem.try_wait());

					importer->import_threaded_end();
				}

				from = i + 1;
			}

		} else {
			ep->step(reimport_files[i].path.get_file(), i, false);
			_reimport_file(reimport_files[i].path);

			// We need to increment the counter, maybe the next file is multithreaded
			// and doesn't have the same importer.
			from = i + 1;
		}
	}

	// Reimport groups.

	from = reimport_files.size();

	if (groups_to_reimport.size()) {
		HashMap<String, Vector<String>> group_files;
		_find_group_files(filesystem, group_files, groups_to_reimport);
		for (const KeyValue<String, Vector<String>> &E : group_files) {
			ep->step(E.key.get_file(), from++, false);
			Error err = _reimport_group(E.key, E.value);
			reloads.push_back(E.key);
			reloads.append_array(E.value);
			if (err == OK) {
				_reimport_file(E.key);
			}
		}
	}
	ep->step(TTR("Finalizing Asset Import..."), p_files.size());

	ResourceUID::get_singleton()->update_cache(); // After reimporting, update the cache.
	_save_filesystem_cache();

	memdelete_notnull(ep);

	_process_update_pending();

	// Revert to previous values to restore editor settings for VSync and Update Continuously.
	OS::get_singleton()->set_low_processor_usage_mode(old_low_processor_usage_mode);
	DisplayServer::get_singleton()->window_set_vsync_mode(old_vsync_mode);

	importing = false;

	ep = memnew(EditorProgress("reimport", TTR("(Re)Importing Assets"), p_files.size()));
	ep->step(TTR("Executing post-reimport operations..."), 0, true);
	if (!is_scanning()) {
		emit_signal(SNAME("filesystem_changed"));
	}
	emit_signal(SNAME("resources_reimported"), reloads);
	memdelete_notnull(ep);
}

Error EditorFileSystem::reimport_append(const String &p_file, const HashMap<StringName, Variant> &p_custom_options, const String &p_custom_importer, Variant p_generator_parameters) {
	Vector<String> reloads;
	reloads.append(p_file);

	// Emit the resource_reimporting signal for the single file before the actual importation.
	emit_signal(SNAME("resources_reimporting"), reloads);

	Error ret = _reimport_file(p_file, p_custom_options, p_custom_importer, &p_generator_parameters);

	// Emit the resource_reimported signal for the single file we just reimported.
	emit_signal(SNAME("resources_reimported"), reloads);
	return ret;
}

Error EditorFileSystem::_resource_import(const String &p_path) {
	Vector<String> files;
	files.push_back(p_path);

	singleton->update_file(p_path);
	singleton->reimport_files(files);

	return OK;
}

Ref<Resource> EditorFileSystem::_load_resource_on_startup(ResourceFormatImporter *p_importer, const String &p_path, Error *r_error, bool p_use_sub_threads, float *r_progress, ResourceFormatLoader::CacheMode p_cache_mode) {
	ERR_FAIL_NULL_V(p_importer, Ref<Resource>());

	if (!FileAccess::exists(p_path)) {
		ERR_FAIL_V_MSG(Ref<Resource>(), vformat("Failed loading resource: %s. The file doesn't seem to exist.", p_path));
	}

	// Fail silently. Hopefully the resource is not yet imported.
	Ref<Resource> res = p_importer->load_internal(p_path, r_error, p_use_sub_threads, r_progress, p_cache_mode, true);
	if (res.is_valid()) {
		return res;
	}

	// Retry after importing the resource.
	if (singleton->_reimport_file(p_path, HashMap<StringName, Variant>(), "", nullptr, false) != OK) {
		return Ref<Resource>();
	}
	return p_importer->load_internal(p_path, r_error, p_use_sub_threads, r_progress, p_cache_mode, false);
}

bool EditorFileSystem::_should_skip_directory(const String &p_path) {
	String project_data_path = ProjectSettings::get_singleton()->get_project_data_path();
	if (p_path == project_data_path || p_path.begins_with(project_data_path + "/")) {
		return true;
	}

	if (FileAccess::exists(p_path.path_join("project.godot"))) {
		// Skip if another project inside this.
		if (EditorFileSystem::get_singleton() == nullptr || EditorFileSystem::get_singleton()->first_scan) {
			WARN_PRINT_ONCE(vformat("Detected another project.godot at %s. The folder will be ignored.", p_path));
		}
		return true;
	}

	if (FileAccess::exists(p_path.path_join(".gdignore"))) {
		// Skip if a `.gdignore` file is inside this.
		return true;
	}

	return false;
}

bool EditorFileSystem::is_group_file(const String &p_path) const {
	return group_file_cache.has(p_path);
}

void EditorFileSystem::_move_group_files(EditorFileSystemDirectory *efd, const String &p_group_file, const String &p_new_location) {
	int fc = efd->files.size();
	EditorFileInfo *const *files = efd->files.ptrw();
	for (int i = 0; i < fc; i++) {
		if (files[i]->import_group_file == p_group_file) {
			files[i]->import_group_file = p_new_location;

			Ref<ConfigFile> config;
			config.instantiate();
			String path = efd->get_file_path(i) + ".import";
			Error err = config->load(path);
			if (err != OK) {
				continue;
			}
			if (config->has_section_key("remap", "group_file")) {
				config->set_value("remap", "group_file", p_new_location);
			}

			Vector<String> sk = config->get_section_keys("params");
			for (const String &param : sk) {
				//not very clean, but should work
				String value = config->get_value("params", param);
				if (value == p_group_file) {
					config->set_value("params", param, p_new_location);
				}
			}

			config->save(path);
		}
	}

	for (int i = 0; i < efd->get_subdir_count(); i++) {
		_move_group_files(efd->get_subdir(i), p_group_file, p_new_location);
	}
}

void EditorFileSystem::move_group_file(const String &p_path, const String &p_new_path) {
	if (get_filesystem()) {
		_move_group_files(get_filesystem(), p_path, p_new_path);
		if (group_file_cache.has(p_path)) {
			group_file_cache.erase(p_path);
			group_file_cache.insert(p_new_path);
		}
	}
}

Error EditorFileSystem::make_dir_recursive(const String &p_path, const String &p_base_path) {
	Error err;
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (!p_base_path.is_empty()) {
		err = da->change_dir(p_base_path);
		ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot open base directory '" + p_base_path + "'.");
	}

	if (da->dir_exists(p_path)) {
		return ERR_ALREADY_EXISTS;
	}

	err = da->make_dir_recursive(p_path);
	if (err != OK) {
		return err;
	}

	const String path = da->get_current_dir();
	EditorFileSystemDirectory *parent = get_filesystem_path(path);
	ERR_FAIL_NULL_V(parent, ERR_FILE_NOT_FOUND);
	folders_to_sort.insert(parent->get_instance_id());

	const PackedStringArray folders = p_path.trim_prefix(path).split("/", false);
	bool exists = true;
	for (const String &folder : folders) {
		if (exists) {
			const int current = parent->find_dir_index(folder);
			if (current > -1) {
				parent = parent->get_subdir(current);
				continue;
			}
			exists = false;
			_pending_scan_fs_changes(parent);
		}

		EditorFileSystemDirectory *efd = memnew(EditorFileSystemDirectory);
		efd->parent = parent;
		efd->name = folder;
		parent->subdirs.push_back(efd);
		parent = efd;
	}

	_queue_refresh_filesystem();
	return OK;
}

Error EditorFileSystem::copy_file(const String &p_from, const String &p_to) {
	Error err = _copy_file(p_from, p_to);
	if (err != OK) {
		return err;
	}

	EditorFileSystemDirectory *parent = get_filesystem_path(p_to.get_base_dir());
	ERR_FAIL_NULL_V(parent, ERR_FILE_NOT_FOUND);

	_pending_scan_fs_changes(parent, false);

	_queue_refresh_filesystem();
	return OK;
}

Error EditorFileSystem::copy_directory(const String &p_from, const String &p_to) {
	// Recursively copy directories and build a map of files to copy.
	HashMap<String, String> files;
	bool success = _copy_directory(p_from, p_to, &files);

	// Copy the files themselves
	if (success) {
		EditorProgress *ep = nullptr;
		if (files.size() > 10) {
			ep = memnew(EditorProgress("copy_directory", TTR("Copying files..."), files.size()));
		}
		int i = 0;
		for (const KeyValue<String, String> &tuple : files) {
			if (_copy_file(tuple.key, tuple.value) != OK) {
				success = false;
			}
			if (ep) {
				ep->step(tuple.key.get_file(), i++, false);
			}
		}
		memdelete_notnull(ep);
	}

	// Now remap any internal dependencies (within the folder) to use the new files.
	if (success) {
		EditorProgress *ep = nullptr;
		if (files.size() > 10) {
			ep = memnew(EditorProgress("copy_directory", TTR("Remapping dependencies..."), files.size()));
		}
		int i = 0;
		for (const KeyValue<String, String> &tuple : files) {
			if (ResourceLoader::rename_dependencies(tuple.value, files) != OK) {
				success = false;
			}
			update_file(tuple.value);
			if (ep) {
				ep->step(tuple.key.get_file(), i++, false);
			}
		}
		memdelete_notnull(ep);
	}

	EditorFileSystemDirectory *efd = get_filesystem_path(p_to);
	ERR_FAIL_NULL_V(efd, FAILED);
	ERR_FAIL_NULL_V(efd->get_parent(), FAILED);

	folders_to_sort.insert(efd->get_parent()->get_instance_id());

	_pending_scan_fs_changes(efd);

	_queue_refresh_filesystem();
	return success ? OK : FAILED;
}

ResourceUID::ID EditorFileSystem::_resource_saver_get_resource_id_for_path(const String &p_path, bool p_generate) {
	if (!p_path.is_resource_file() || p_path.begins_with(ProjectSettings::get_singleton()->get_project_data_path())) {
		// Saved externally (configuration file) or internal file, do not assign an ID.
		return ResourceUID::INVALID_ID;
	}

	EditorFileSystemDirectory *fs = nullptr;
	int cpos = -1;

	if (!singleton->_find_file(p_path, &fs, cpos)) {
		// Fallback to ResourceLoader if filesystem cache fails (can happen during scanning etc.).
		ResourceUID::ID fallback = ResourceLoader::get_resource_uid(p_path);
		if (fallback != ResourceUID::INVALID_ID) {
			return fallback;
		}

		if (p_generate) {
			return ResourceUID::get_singleton()->create_id_for_path(p_path); // Just create a new one, we will be notified of save anyway and fetch the right UID at that time, to keep things simple.
		} else {
			return ResourceUID::INVALID_ID;
		}
	} else if (fs->files[cpos]->uid != ResourceUID::INVALID_ID) {
		return fs->files[cpos]->uid;
	} else if (p_generate) {
		return ResourceUID::get_singleton()->create_id_for_path(p_path); // Just create a new one, we will be notified of save anyway and fetch the right UID at that time, to keep things simple.
	} else {
		return ResourceUID::INVALID_ID;
	}
}

static void _scan_extensions_dir(EditorFileSystemDirectory *d, HashSet<String> &extensions) {
	int fc = d->get_file_count();
	for (int i = 0; i < fc; i++) {
		if (d->get_file_type(i) == GDExtension::get_class_static()) {
			extensions.insert(d->get_file_path(i));
		}
	}
	int dc = d->get_subdir_count();
	for (int i = 0; i < dc; i++) {
		_scan_extensions_dir(d->get_subdir(i), extensions);
	}
}
bool EditorFileSystem::_scan_extensions() {
	EditorFileSystemDirectory *d = get_filesystem();
	HashSet<String> extensions;

	_scan_extensions_dir(d, extensions);

	return GDExtensionManager::get_singleton()->ensure_extensions_loaded(extensions);
}

void EditorFileSystem::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_filesystem"), &EditorFileSystem::get_filesystem);
	ClassDB::bind_method(D_METHOD("is_scanning"), &EditorFileSystem::is_scanning);
	ClassDB::bind_method(D_METHOD("get_scanning_progress"), &EditorFileSystem::get_scanning_progress);
	ClassDB::bind_method(D_METHOD("scan"), &EditorFileSystem::scan);
	ClassDB::bind_method(D_METHOD("scan_sources"), &EditorFileSystem::scan_changes);
	ClassDB::bind_method(D_METHOD("update_file", "path"), &EditorFileSystem::update_file);
	ClassDB::bind_method(D_METHOD("get_filesystem_path", "path"), &EditorFileSystem::get_filesystem_path);
	ClassDB::bind_method(D_METHOD("get_file_type", "path"), &EditorFileSystem::get_file_type);
	ClassDB::bind_method(D_METHOD("reimport_files", "files"), &EditorFileSystem::reimport_files);

	ADD_SIGNAL(MethodInfo("filesystem_changed"));
	ADD_SIGNAL(MethodInfo("script_classes_updated"));
	ADD_SIGNAL(MethodInfo("sources_changed", PropertyInfo(Variant::BOOL, "exist")));
	ADD_SIGNAL(MethodInfo("resources_reimporting", PropertyInfo(Variant::PACKED_STRING_ARRAY, "resources")));
	ADD_SIGNAL(MethodInfo("resources_reimported", PropertyInfo(Variant::PACKED_STRING_ARRAY, "resources")));
	ADD_SIGNAL(MethodInfo("resources_reload", PropertyInfo(Variant::PACKED_STRING_ARRAY, "resources")));
}

void EditorFileSystem::update_extensions() {
	valid_extensions.clear();
	import_extensions.clear();
	textfile_extensions.clear();
	other_file_extensions.clear();

	List<String> extensionsl;
	ResourceLoader::get_recognized_extensions_for_type("", &extensionsl);
	for (const String &E : extensionsl) {
		valid_extensions.insert(E);
	}

	const Vector<String> textfile_ext = ((String)(EDITOR_GET("docks/filesystem/textfile_extensions"))).split(",", false);
	for (const String &E : textfile_ext) {
		if (valid_extensions.has(E)) {
			continue;
		}
		valid_extensions.insert(E);
		textfile_extensions.insert(E);
	}
	const Vector<String> other_file_ext = ((String)(EDITOR_GET("docks/filesystem/other_file_extensions"))).split(",", false);
	for (const String &E : other_file_ext) {
		if (valid_extensions.has(E)) {
			continue;
		}
		valid_extensions.insert(E);
		other_file_extensions.insert(E);
	}

	extensionsl.clear();
	ResourceFormatImporter::get_singleton()->get_recognized_extensions(&extensionsl);
	for (const String &E : extensionsl) {
		import_extensions.insert(E);
	}

	extensions_changed = true;
	if (!first_scan) {
		_scan_dirs_changes();
	}
}

bool EditorFileSystem::_validate_file_extension(const String &p_file, const HashSet<String> &p_extensions) {
	const String file = p_file.get_file();
	for (const String &E : p_extensions) {
		if (file.right(E.length() + 1).nocasecmp_to("." + E) == 0) {
			return true;
		}
	}

	return false;
}

void EditorFileSystem::add_import_format_support_query(Ref<EditorFileSystemImportFormatSupportQuery> p_query) {
	ERR_FAIL_COND(import_support_queries.has(p_query));
	import_support_queries.push_back(p_query);
}
void EditorFileSystem::remove_import_format_support_query(Ref<EditorFileSystemImportFormatSupportQuery> p_query) {
	import_support_queries.erase(p_query);
}

EditorFileSystem::EditorFileSystem() {
#ifdef THREADS_ENABLED
	use_threads = true;
#endif

	ResourceLoader::import = _resource_import;
	reimport_on_missing_imported_files = GLOBAL_GET("editor/import/reimport_missing_imported_files");
	singleton = this;
	filesystem = memnew(EditorFileSystemDirectory); //like, empty
	filesystem->parent = nullptr;

	const int detect_mode = EDITOR_GET("filesystem/on_scan/detect_mode");
	force_detect = detect_mode == 1;
	if (!force_detect) {
		// This should probably also work on Unix and use the string it returns for FAT32 or exFAT
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		force_detect = DirAccess::exists("res://.git") || da->get_filesystem_type() == "FAT32" || da->get_filesystem_type() == "EXFAT";
	}

	scan_total = 0;
	ResourceSaver::set_get_resource_id_for_path(_resource_saver_get_resource_id_for_path);

	// Set the callback method that the ResourceFormatImporter will use
	// if resources are loaded during the first scan.
	ResourceImporter::load_on_startup = _load_resource_on_startup;

	_reset_uid_points();
	_reset_points();
}

EditorFileSystem::~EditorFileSystem() {
	if (first_scan) {
		// The scan thread is aborted during first scan.
		for (KeyValue<String, EditorFileInfo *> &E : script_file_info) {
			if (E.value->status & EditorFileInfo::IS_ORPHAN) {
				memdelete(E.value);
			}
		}
		if (first_scan_root_dir) {
			memdelete(first_scan_root_dir);
		}
	}
	if (filesystem) {
		memdelete(filesystem);
	}
	filesystem = nullptr;
	ResourceSaver::set_get_resource_id_for_path(nullptr);
}
