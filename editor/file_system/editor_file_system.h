/**************************************************************************/
/*  editor_file_system.h                                                  */
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

#pragma once

#include "core/io/dir_access.h"
#include "core/io/resource_importer.h"
#include "core/io/resource_loader.h"
#include "core/os/thread.h"
#include "core/os/thread_safe.h"
#include "core/templates/hash_set.h"
#include "core/templates/safe_refcount.h"
#include "scene/main/node.h"

class FileAccess;

struct EditorProgressBG;
class EditorFileSystemDirectory : public Object {
	GDCLASS(EditorFileSystemDirectory, Object);

	String name;
	uint64_t modified_time;
	bool verified = true; // Used for checking changes.
	bool dirty = true; // The files in the current directory need to be checked.
	bool recursive = true; // Recursive checking is required.

	EditorFileSystemDirectory *parent = nullptr;
	Vector<EditorFileSystemDirectory *> subdirs;

	struct FileInfo {
		EditorFileSystemDirectory *parent = nullptr;
		String file;
		StringName type;
		StringName resource_script_class; // If any resource has script with a global class name, its found here.
		ResourceUID::ID uid = ResourceUID::INVALID_ID;
		uint64_t modified_time = 0;
		uint64_t import_modified_time = 0;
		String import_md5;
		Vector<String> import_dest_paths;
		bool import_valid = false;
		String import_group_file;
		Vector<String> deps;
		bool verified = true; // Used for checking changes.
		// This is for script resources only.
		struct ScriptClassInfo {
			String name;
			String extends;
			String lang;
			String icon_path;
			bool is_abstract = false;
			bool is_tool = false;
		};
		ScriptClassInfo class_info;
		enum FileStatus {
			NONE = 0,
			FILE_ADD = 1,
			FILE_REMOVE = 1 << 1,
			FILE_UPDATE = 1 << 2, // For files that are indeed updated, some properties of their FileInfo need to be updated.
			FILE_CHANGED = FILE_ADD | FILE_REMOVE | FILE_UPDATE,
			TYPE_ADD = 1 << 4,
			TYPE_REMOVE = 1 << 5,
			TYPE_CHANGED = TYPE_ADD | TYPE_REMOVE,
			TEMPORARY = FILE_CHANGED | TYPE_CHANGED,

			IS_ORPHAN = 1 << 12,
			HAS_CUSTOM_UID_SUPPORT = 1 << 13,

			IS_SCRIPT = 1 << 16,
			IS_PACKEDSCENE = 1 << 17,
			SPECIAL_TYPE = IS_SCRIPT | IS_PACKEDSCENE,
			IS_GLOBAL_CLASS_ALTERNATIVE = 1 << 18,
			IS_ACTIVE_GLOBAL_CLASS_ALTERNATIVE = 1 << 19,
			GLOBAL_CLASS_MASK = IS_GLOBAL_CLASS_ALTERNATIVE | IS_ACTIVE_GLOBAL_CLASS_ALTERNATIVE,

			AS_RESOURCE = 1 << 28,
			IS_IMPORTABLE = 1 << 29,
			IS_OTHER = 1 << 30,
			IS_TEXT = 1 << 31,
			CATEGORY_CHANGED = IS_IMPORTABLE | IS_OTHER | IS_TEXT,
		};
		uint32_t status = AS_RESOURCE | FILE_ADD;
	};

	Vector<FileInfo *> files;

	static void _bind_methods();

	friend class EditorFileSystem;

public:
	String get_name();
	String get_path() const;

	int get_subdir_count() const;
	EditorFileSystemDirectory *get_subdir(int p_idx);
	int get_file_count() const;
	String get_file(int p_idx) const;
	String get_file_path(int p_idx) const;
	StringName get_file_type(int p_idx) const;
	StringName get_file_resource_script_class(int p_idx) const;
	Vector<String> get_file_deps(int p_idx) const;
	bool get_file_import_is_valid(int p_idx) const;
	uint64_t get_file_modified_time(int p_idx) const;
	uint64_t get_file_import_modified_time(int p_idx) const;
	String get_file_script_class_name(int p_idx) const; //used for scripts
	String get_file_script_class_extends(int p_idx) const; //used for scripts
	String get_file_script_class_icon_path(int p_idx) const; //used for scripts
	String get_file_icon_path(int p_idx) const; //used for FileSystemDock

	EditorFileSystemDirectory *get_parent();

	int find_file_index(const String &p_file) const;
	int find_dir_index(const String &p_dir) const;

	void force_update();

	EditorFileSystemDirectory();
	~EditorFileSystemDirectory();
};

class EditorFileSystemImportFormatSupportQuery : public RefCounted {
	GDCLASS(EditorFileSystemImportFormatSupportQuery, RefCounted);

protected:
	GDVIRTUAL0RC_REQUIRED(bool, _is_active)
	GDVIRTUAL0RC_REQUIRED(Vector<String>, _get_file_extensions)
	GDVIRTUAL0RC_REQUIRED(bool, _query)
	static void _bind_methods() {
		GDVIRTUAL_BIND(_is_active);
		GDVIRTUAL_BIND(_get_file_extensions);
		GDVIRTUAL_BIND(_query);
	}

public:
	virtual bool is_active() const {
		bool ret = false;
		GDVIRTUAL_CALL(_is_active, ret);
		return ret;
	}
	virtual Vector<String> get_file_extensions() const {
		Vector<String> ret;
		GDVIRTUAL_CALL(_get_file_extensions, ret);
		return ret;
	}
	virtual bool query() {
		bool ret = false;
		GDVIRTUAL_CALL(_query, ret);
		return ret;
	}
};

class EditorFileSystem : public Node {
	GDCLASS(EditorFileSystem, Node);

	_THREAD_SAFE_CLASS_

	using EditorFileInfo = EditorFileSystemDirectory::FileInfo;
	using ScriptClassInfo = EditorFileInfo::ScriptClassInfo;

	struct ItemUIDAction {
		enum UIDAction {
			ACTION_UID_NONE,
			ACTION_UID_ADD,
			ACTION_UID_REMOVE,
			ACTION_UID_PENDING_ADD,
		};
		enum UIDStep {
			STEP_UID_VALIDATE, // Verify the validity of the uid cache. Only when the editor starts.
			STEP_UID_NEWLY_ADD, // Create and add new uids to files that do not have uids assigned to them.
			STEP_UID_REMOVE, // Remove the outdated UID from the file info cache.
			STEP_UID_PENDING_ADD, // Add the new UID obtained from the file.
			STEP_UID_REGENERATE, // Recreate and add a new uid if the uid is already taken.
			STEP_UID_MAX,
		};

		UIDAction action = ACTION_UID_NONE;
		ResourceUID::ID old_uid = ResourceUID::INVALID_ID; // Only used for UID actions. Can be used as a fallback value or a delete value.
		String path; // In order to reduce the count of path calculations.
		EditorFileInfo *file = nullptr;
	};

	struct ItemAction {
		enum Action {
			ACTION_NONE,
			ACTION_DIR_ADD,
			ACTION_DIR_REMOVE,
			ACTION_FILE_ADD,
			ACTION_FILE_REMOVE,
			ACTION_FILE_UPDATE,
			ACTION_FILE_REIMPORT,
		};
		enum Step {
			STEP_NORMAL,
			STEP_CLEAR_STATUS, // Clear the temporary flag of the file status for next use.
			STEP_FILE_REMOVE,
			STEP_DIR_REMOVE,
			STEP_MAX,
		};

		Action action = ACTION_NONE;
		String path; // In order to reduce the count of path calculations.
		EditorFileSystemDirectory *dir = nullptr;
		EditorFileInfo *file = nullptr;
	};

	struct ScannedDirectory {
		String name;
		String full_path;
		int count = 0;
		Vector<ScannedDirectory *> subdirs;
		List<String> files;

		~ScannedDirectory();
	};

	bool extensions_changed = true;
	SafeFlag scanning_done = SafeFlag(true);
	bool use_threads = false;
	Thread thread;
	static void _thread_func(void *_userdata);

	static ScannedDirectory *first_scan_root_dir;

	bool update_actions_queued = false;
	bool scanning = false;
	bool importing = false;
	bool first_scan = true;
	bool scan_changes_pending = false;
	bool full_scan_pending = false;
	float scan_total;
	String filesystem_settings_version_for_import;
	bool revalidate_import_files = false;
	static int nb_files_total;

	bool _load_filesystem_from_cache();

	void _category_validate(EditorFileInfo *p_file, const String &p_path);
	void _type_analysis(EditorFileInfo *p_file, const StringName &p_new_type);
	void _import_validate(EditorFileInfo *p_file, const String &p_path);
	void _global_script_class_info_remove(EditorFileInfo *p_file, const String &p_path);
	void _global_script_class_info_add(EditorFileInfo *p_file, const String &p_path);
	void _script_class_info_update(EditorFileInfo *p_file, const String &p_path, const ScriptClassInfo *p_sci);

	void _file_info_add(EditorFileSystemDirectory *p_parent_dir, const String &p_parent_path, const String &p_file, bool p_insert);
	void _file_info_remove(EditorFileInfo *p_file, const String &p_path, const int p_idx);
	void _file_info_update(EditorFileInfo *p_file, const String &p_path);

	int _dir_info_remove(EditorFileSystemDirectory *p_dir, const String &p_path, const int p_idx);

	void _scan_filesystem();
	void _first_scan_filesystem();
	void _first_scan_process_scripts(const ScannedDirectory *p_scan_dir, List<String> &p_gdextension_extensions, HashSet<String> &p_existing_class_names, HashSet<String> &p_extensions);

	static void _scan_for_uid_directory(const ScannedDirectory *p_scan_dir, const HashSet<String> &p_import_extensions);

	static void _load_first_scan_root_dir();

	EditorFileSystemDirectory *filesystem = nullptr;

	static EditorFileSystem *singleton;

	struct ScanProgress {
		float hi = 0;
		int current = 0;
		EditorProgressBG *progress = nullptr;
		void increment();
	};

	struct DirectoryComparator {
		bool operator()(const EditorFileSystemDirectory *p_a, const EditorFileSystemDirectory *p_b) const {
			return p_a->name.filenocasecmp_to(p_b->name) < 0;
		}
	};

	void _save_filesystem_cache();
	void _save_filesystem_cache(EditorFileSystemDirectory *p_dir, Ref<FileAccess> p_file);

	bool _find_file(const String &p_file, EditorFileSystemDirectory **r_d, int &r_file_pos) const;

	List<EditorFileSystemDirectory *> dirty_directories;

	void _scan_fs_changes(EditorFileSystemDirectory *p_dir, ScanProgress &p_progress, bool p_recursive = true);
	void scan_fs_changes(ScanProgress &p_progress);
	void _pending_scan_fs_changes(EditorFileSystemDirectory *p_dir, bool p_recursive = true);
	void _scan_dirs_changes(bool p_full_scan = true);

	void _delete_internal_files(const String &p_file);

	HashSet<String> textfile_extensions;
	HashSet<String> other_file_extensions;
	HashSet<String> valid_extensions;
	HashSet<String> import_extensions;

	static bool _validate_file_extension(const String &p_file, const HashSet<String> &p_extensions);
	static int _scan_new_dir(ScannedDirectory *p_dir, Ref<DirAccess> &da);
	void _process_file_system(const ScannedDirectory *p_scan_dir, EditorFileSystemDirectory *p_dir, ScanProgress &p_progress, HashSet<String> *p_processed_files);

	Thread thread_sources;
	bool scanning_changes = false;
	SafeFlag scanning_changes_done;

	static void _thread_func_sources(void *_userdata);

	List<String> sources_changed;
	List<ItemUIDAction> scan_uid_actions;
	List<ItemUIDAction>::Element *uid_newly_add_end;
	List<ItemUIDAction>::Element *uid_move_end;
	List<ItemAction> scan_actions;
	List<ItemAction>::Element *normal;
	List<ItemAction>::Element *remove_point;

	void _reset_uid_points();
	void _reset_points();
	void _create_uid_action(EditorFileInfo *p_fi, const String &p_path, const ItemUIDAction::UIDAction p_action, const ItemUIDAction::UIDStep p_step, const ResourceUID::ID p_old_uid = ResourceUID::INVALID_ID);
	void _create_action(EditorFileSystemDirectory *p_dir, EditorFileInfo *p_fi, const String &p_path, const ItemAction::Action p_action, const ItemAction::Step p_step = ItemAction::STEP_NORMAL, const ResourceUID::ID p_old_uid = ResourceUID::INVALID_ID);
	void _create_actions_from_uid_change(EditorFileInfo *p_fi, const String &p_path, const ResourceUID::ID p_old_uid = ResourceUID::INVALID_ID);

	bool updating_scan_actions = false;
	bool _update_scan_uid_actions();
	bool _update_scan_actions();

	Error _reimport_file(const String &p_file, const HashMap<StringName, Variant> &p_custom_options = HashMap<StringName, Variant>(), const String &p_custom_importer = String(), Variant *generator_parameters = nullptr, bool p_update_file_system = true);
	Error _reimport_group(const String &p_group_file, const Vector<String> &p_files);

	bool _test_for_reimport(const String &p_path, EditorFileInfo *p_file);
	bool _is_test_for_reimport_needed(EditorFileInfo *p_file, uint64_t p_modified_time, uint64_t p_import_modified_time);
	Vector<String> _get_import_dest_paths(const String &p_path);

	bool reimport_on_missing_imported_files;

	Vector<String> _get_dependencies(const String &p_path);

	struct ImportFile {
		String path;
		String importer;
		bool threaded = false;
		int order = 0;
		bool operator<(const ImportFile &p_if) const {
			return order == p_if.order ? (importer < p_if.importer) : (order < p_if.order);
		}
	};

	HashMap<String, EditorFileInfo *> script_file_info;
	struct ScriptClassAlternatives {
		bool active = false;
		String active_path;
		String icon_path;
		ResourceUID::ID active_uid = ResourceUID::INVALID_ID;
		HashMap<String, EditorFileInfo *> alternatives;
	};
	HashMap<StringName, ScriptClassAlternatives> global_script_class_alternatives;
	void _update_global_script_class_activation();

	bool script_classes_updated = false;
	bool loader_changed = false;
	bool saver_changed = false;
	bool need_update_extensions = false;
	void _check_loader_or_saver_changed(const ScriptClassInfo &p_class_info);
	void _reload_loader_or_saver();

	Mutex update_script_mutex;
	HashSet<String> update_script_paths_documentation;
	void _update_script_classes();
	void _update_script_documentation();
	void _process_update_pending();
	bool _should_reload_script(const String &p_path);

	Mutex update_scene_mutex;
	HashSet<String> update_scene_paths;
	void _queue_update_scene_groups(const String &p_path);
	void _update_scene_groups();
	void _update_pending_scene_groups();
	void _get_all_scenes(EditorFileSystemDirectory *p_dir, HashSet<String> &r_list);

	ScriptClassInfo _get_global_script_class(const String &p_type, const String &p_path) const;

	static Error _resource_import(const String &p_path);
	static Ref<Resource> _load_resource_on_startup(ResourceFormatImporter *p_importer, const String &p_path, Error *r_error, bool p_use_sub_threads, float *r_progress, ResourceFormatLoader::CacheMode p_cache_mode);

	bool force_detect = false;

	void _find_group_files(EditorFileSystemDirectory *efd, HashMap<String, Vector<String>> &group_files, HashSet<String> &groups_to_reimport);

	void _move_group_files(EditorFileSystemDirectory *efd, const String &p_group_file, const String &p_new_location);

	HashSet<String> group_file_cache;
	HashMap<String, String> file_icon_cache;

	bool refresh_queued = false;
	HashSet<ObjectID> folders_to_sort;

	Error _copy_file(const String &p_from, const String &p_to);
	bool _copy_directory(const String &p_from, const String &p_to, HashMap<String, String> *p_files);
	void _queue_refresh_filesystem();
	void _refresh_filesystem();

	struct ImportThreadData {
		const ImportFile *reimport_files;
		int reimport_from;
		Semaphore *imported_sem = nullptr;
	};

	void _reimport_thread(uint32_t p_index, ImportThreadData *p_import_data);

	static ResourceUID::ID _resource_saver_get_resource_id_for_path(const String &p_path, bool p_generate);

	bool _scan_extensions();
	bool _import_support_abort_scan(const Vector<String> &reimports);

	Vector<Ref<EditorFileSystemImportFormatSupportQuery>> import_support_queries;

	void _update_file_icon_path(EditorFileInfo *file_info);
	void _update_files_icon_path(EditorFileSystemDirectory *edp = nullptr);
	bool _remove_invalid_global_class_names(const HashSet<String> &p_existing_class_names);
	String _get_file_by_class_name(EditorFileSystemDirectory *p_dir, const String &p_class_name, EditorFileInfo *&r_file_info);

	void _register_global_class_script(const String &p_search_path, const String &p_target_path, const EditorFileInfo *p_fi);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static EditorFileSystem *get_singleton() { return singleton; }

	EditorFileSystemDirectory *get_filesystem();
	bool is_scanning() const;
	bool is_importing() const { return importing; }
	bool doing_first_scan() const { return first_scan; }
	float get_scanning_progress() const;
	void update_extensions();
	void scan();
	void scan_changes();
	void pending_scan_fs_changes(const String &p_dir, bool p_recursive);
	void update_file(const String &p_file);
	void update_files(const Vector<String> &p_files);
	HashSet<String> get_valid_extensions() const;
	void register_global_class_script(const String &p_search_path, const String &p_target_path);

	EditorFileSystemDirectory *get_filesystem_path(const String &p_path);
	String get_file_type(const String &p_file) const;
	EditorFileSystemDirectory *find_file(const String &p_file, int *r_index) const;
	ResourceUID::ID get_file_uid(const String &p_path) const;

	void reimport_files(const Vector<String> &p_files);
	Error reimport_append(const String &p_file, const HashMap<StringName, Variant> &p_custom_options, const String &p_custom_importer, Variant p_generator_parameters);

	void reimport_file_with_custom_parameters(const String &p_file, const String &p_importer, const HashMap<StringName, Variant> &p_custom_params);

	bool is_group_file(const String &p_path) const;
	void move_group_file(const String &p_path, const String &p_new_path);

	Error make_dir_recursive(const String &p_path, const String &p_base_path = String());
	Error copy_file(const String &p_from, const String &p_to);
	Error copy_directory(const String &p_from, const String &p_to);

	static bool _should_skip_directory(const String &p_path);

	static void scan_for_uid();

	void add_import_format_support_query(Ref<EditorFileSystemImportFormatSupportQuery> p_query);
	void remove_import_format_support_query(Ref<EditorFileSystemImportFormatSupportQuery> p_query);
	EditorFileSystem();
	~EditorFileSystem();
};
