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

#ifndef EDITOR_FILE_SYSTEM_H
#define EDITOR_FILE_SYSTEM_H

#include "core/io/dir_access.h"
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
	bool verified = false; //used for checking changes

	EditorFileSystemDirectory *parent = nullptr;
	Vector<EditorFileSystemDirectory *> subdirs;

	struct FileInfo {
		String file;
		StringName type;
		StringName resource_script_class; // If any resource has script with a global class name, its found here.
		ResourceUID::ID uid = ResourceUID::INVALID_ID;
		uint64_t modified_time = 0;
		uint64_t import_modified_time = 0;
		bool import_valid = false;
		String import_group_file;
		Vector<String> deps;
		bool verified = false; //used for checking changes
		// These are for script resources only.
		String script_class_name;
		String script_class_extends;
		String script_class_icon_path;
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
	String get_file_script_class_name(int p_idx) const; //used for scripts
	String get_file_script_class_extends(int p_idx) const; //used for scripts
	String get_file_script_class_icon_path(int p_idx) const; //used for scripts

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
	GDVIRTUAL0RC(bool, _is_active)
	GDVIRTUAL0RC(Vector<String>, _get_file_extensions)
	GDVIRTUAL0RC(bool, _query)
	static void _bind_methods() {
		GDVIRTUAL_BIND(_is_active);
		GDVIRTUAL_BIND(_get_file_extensions);
		GDVIRTUAL_BIND(_query);
	}

public:
	virtual bool is_active() const {
		bool ret = false;
		GDVIRTUAL_REQUIRED_CALL(_is_active, ret);
		return ret;
	}
	virtual Vector<String> get_file_extensions() const {
		Vector<String> ret;
		GDVIRTUAL_REQUIRED_CALL(_get_file_extensions, ret);
		return ret;
	}
	virtual bool query() {
		bool ret = false;
		GDVIRTUAL_REQUIRED_CALL(_query, ret);
		return ret;
	}
};

class EditorFileSystem : public Node {
	GDCLASS(EditorFileSystem, Node);

	_THREAD_SAFE_CLASS_

	struct ItemAction {
		enum Action {
			ACTION_NONE,
			ACTION_DIR_ADD,
			ACTION_DIR_REMOVE,
			ACTION_FILE_ADD,
			ACTION_FILE_REMOVE,
			ACTION_FILE_TEST_REIMPORT,
			ACTION_FILE_RELOAD
		};

		Action action = ACTION_NONE;
		EditorFileSystemDirectory *dir = nullptr;
		String file;
		EditorFileSystemDirectory *new_dir = nullptr;
		EditorFileSystemDirectory::FileInfo *new_file = nullptr;
	};

	bool use_threads = false;
	Thread thread;
	static void _thread_func(void *_userdata);

	EditorFileSystemDirectory *new_filesystem = nullptr;

	bool scanning = false;
	bool importing = false;
	bool first_scan = true;
	bool scan_changes_pending = false;
	float scan_total;
	String filesystem_settings_version_for_import;
	bool revalidate_import_files = false;

	void _scan_filesystem();

	HashSet<String> late_update_files;

	void _save_late_updated_files();

	EditorFileSystemDirectory *filesystem = nullptr;

	static EditorFileSystem *singleton;

	/* Used for reading the filesystem cache file */
	struct FileCache {
		String type;
		String resource_script_class;
		ResourceUID::ID uid = ResourceUID::INVALID_ID;
		uint64_t modification_time = 0;
		uint64_t import_modification_time = 0;
		Vector<String> deps;
		bool import_valid = false;
		String import_group_file;
		String script_class_name;
		String script_class_extends;
		String script_class_icon_path;
	};

	HashMap<String, FileCache> file_cache;

	struct ScanProgress {
		float low = 0;
		float hi = 0;
		mutable EditorProgressBG *progress = nullptr;
		void update(int p_current, int p_total) const;
		ScanProgress get_sub(int p_current, int p_total) const;
	};

	void _save_filesystem_cache();
	void _save_filesystem_cache(EditorFileSystemDirectory *p_dir, Ref<FileAccess> p_file);

	bool _find_file(const String &p_file, EditorFileSystemDirectory **r_d, int &r_file_pos) const;

	void _scan_fs_changes(EditorFileSystemDirectory *p_dir, const ScanProgress &p_progress);

	void _delete_internal_files(const String &p_file);

	HashSet<String> textfile_extensions;
	HashSet<String> valid_extensions;
	HashSet<String> import_extensions;

	void _scan_new_dir(EditorFileSystemDirectory *p_dir, Ref<DirAccess> &da, const ScanProgress &p_progress);

	Thread thread_sources;
	bool scanning_changes = false;
	SafeFlag scanning_changes_done;

	static void _thread_func_sources(void *_userdata);

	List<String> sources_changed;
	List<ItemAction> scan_actions;

	bool _update_scan_actions();

	void _update_extensions();

	Error _reimport_file(const String &p_file, const HashMap<StringName, Variant> &p_custom_options = HashMap<StringName, Variant>(), const String &p_custom_importer = String(), Variant *generator_parameters = nullptr);
	Error _reimport_group(const String &p_group_file, const Vector<String> &p_files);

	bool _test_for_reimport(const String &p_path, bool p_only_imported_files);

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

	Mutex update_script_mutex;
	HashSet<String> update_script_paths;
	void _queue_update_script_class(const String &p_path);
	void _update_script_classes();
	void _update_pending_script_classes();

	Mutex update_scene_mutex;
	HashSet<String> update_scene_paths;
	void _queue_update_scene_groups(const String &p_path);
	void _update_scene_groups();
	void _update_pending_scene_groups();
	HashSet<StringName> _get_scene_groups(const String &p_path);
	void _get_all_scenes(EditorFileSystemDirectory *p_dir, HashSet<String> &r_list);

	String _get_global_script_class(const String &p_type, const String &p_path, String *r_extends, String *r_icon_path) const;

	static Error _resource_import(const String &p_path);

	bool using_fat32_or_exfat; // Workaround for projects in FAT32 or exFAT filesystem (pendrives, most of the time)

	void _find_group_files(EditorFileSystemDirectory *efd, HashMap<String, Vector<String>> &group_files, HashSet<String> &groups_to_reimport);

	void _move_group_files(EditorFileSystemDirectory *efd, const String &p_group_file, const String &p_new_location);

	HashSet<String> group_file_cache;

	struct ImportThreadData {
		const ImportFile *reimport_files;
		int reimport_from;
		int max_index = 0;
	};

	void _reimport_thread(uint32_t p_index, ImportThreadData *p_import_data);

	static ResourceUID::ID _resource_saver_get_resource_id_for_path(const String &p_path, bool p_generate);

	bool _scan_extensions();
	bool _scan_import_support(const Vector<String> &reimports);

	Vector<Ref<EditorFileSystemImportFormatSupportQuery>> import_support_queries;

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
	void scan();
	void scan_changes();
	void update_file(const String &p_file);
	HashSet<String> get_valid_extensions() const;

	EditorFileSystemDirectory *get_filesystem_path(const String &p_path);
	String get_file_type(const String &p_file) const;
	EditorFileSystemDirectory *find_file(const String &p_file, int *r_index) const;

	void reimport_files(const Vector<String> &p_files);
	Error reimport_append(const String &p_file, const HashMap<StringName, Variant> &p_custom_options, const String &p_custom_importer, Variant p_generator_parameters);

	void reimport_file_with_custom_parameters(const String &p_file, const String &p_importer, const HashMap<StringName, Variant> &p_custom_params);

	bool is_group_file(const String &p_path) const;
	void move_group_file(const String &p_path, const String &p_new_path);

	static bool _should_skip_directory(const String &p_path);

	void add_import_format_support_query(Ref<EditorFileSystemImportFormatSupportQuery> p_query);
	void remove_import_format_support_query(Ref<EditorFileSystemImportFormatSupportQuery> p_query);
	EditorFileSystem();
	~EditorFileSystem();
};

#endif // EDITOR_FILE_SYSTEM_H
