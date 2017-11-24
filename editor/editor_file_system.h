/*************************************************************************/
/*  editor_file_system.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef EDITOR_FILE_SYSTEM_H
#define EDITOR_FILE_SYSTEM_H

#include "os/dir_access.h"
#include "os/thread.h"
#include "os/thread_safe.h"
#include "scene/main/node.h"
#include "set.h"
class FileAccess;

struct EditorProgressBG;
class EditorFileSystemDirectory : public Object {

	GDCLASS(EditorFileSystemDirectory, Object);

	String name;
	uint64_t modified_time;
	bool verified; //used for checking changes

	EditorFileSystemDirectory *parent;
	Vector<EditorFileSystemDirectory *> subdirs;

	struct FileInfo {
		String file;
		StringName type;
		uint64_t modified_time;
		uint64_t import_modified_time;
		bool import_valid;
		Vector<String> deps;
		bool verified; //used for checking changes
	};

	struct FileInfoSort {
		bool operator()(const FileInfo *p_a, const FileInfo *p_b) const {
			return p_a->file < p_b->file;
		}
	};

	void sort_files();

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
	Vector<String> get_file_deps(int p_idx) const;
	bool get_file_import_is_valid(int p_idx) const;

	EditorFileSystemDirectory *get_parent();

	int find_file_index(const String &p_file) const;
	int find_dir_index(const String &p_dir) const;

	EditorFileSystemDirectory();
	~EditorFileSystemDirectory();
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
			ACTION_FILE_TEST_REIMPORT
		};

		Action action;
		EditorFileSystemDirectory *dir;
		String file;
		EditorFileSystemDirectory *new_dir;
		EditorFileSystemDirectory::FileInfo *new_file;

		ItemAction() {
			action = ACTION_NONE;
			dir = NULL;
			new_dir = NULL;
			new_file = NULL;
		}
	};

	bool use_threads;
	Thread *thread;
	static void _thread_func(void *_userdata);

	EditorFileSystemDirectory *new_filesystem;

	bool abort_scan;
	bool scanning;
	bool importing;
	float scan_total;

	void _scan_filesystem();

	Set<String> late_added_files; //keep track of files that were added, these will be re-scanned
	Set<String> late_update_files;

	void _save_late_updated_files();

	EditorFileSystemDirectory *filesystem;

	static EditorFileSystem *singleton;

	/* Used for reading the filesystem cache file */
	struct FileCache {

		String type;
		uint64_t modification_time;
		uint64_t import_modification_time;
		Vector<String> deps;
		bool import_valid;
	};

	HashMap<String, FileCache> file_cache;

	struct ScanProgress {

		float low;
		float hi;
		mutable EditorProgressBG *progress;
		void update(int p_current, int p_total) const;
		ScanProgress get_sub(int p_current, int p_total) const;
	};

	void _save_filesystem_cache();
	void _save_filesystem_cache(EditorFileSystemDirectory *p_dir, FileAccess *p_file);

	bool _find_file(const String &p_file, EditorFileSystemDirectory **r_d, int &r_file_pos) const;

	void _scan_fs_changes(EditorFileSystemDirectory *p_dir, const ScanProgress &p_progress);

	void _delete_internal_files(String p_file);

	Set<String> valid_extensions;
	Set<String> import_extensions;

	void _scan_new_dir(EditorFileSystemDirectory *p_dir, DirAccess *da, const ScanProgress &p_progress);

	Thread *thread_sources;
	bool scanning_changes;
	bool scanning_changes_done;

	static void _thread_func_sources(void *_userdata);

	List<String> sources_changed;
	List<ItemAction> scan_actions;

	bool _update_scan_actions();

	static void _resource_saved(const String &p_path);

	void _update_extensions();

	void _reimport_file(const String &p_file);

	bool _test_for_reimport(const String &p_path, bool p_only_imported_files);

	bool reimport_on_missing_imported_files;

	Vector<String> _get_dependencies(const String &p_path);

	struct ImportFile {
		String path;
		int order;
		bool operator<(const ImportFile &p_if) const {
			return order < p_if.order;
		}
	};

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static EditorFileSystem *get_singleton() { return singleton; }

	EditorFileSystemDirectory *get_filesystem();
	bool is_scanning() const;
	bool is_importing() const { return importing; }
	float get_scanning_progress() const;
	void scan();
	void scan_changes();
	void get_changed_sources(List<String> *r_changed);
	void update_file(const String &p_file);

	EditorFileSystemDirectory *get_filesystem_path(const String &p_path);
	String get_file_type(const String &p_file) const;
	EditorFileSystemDirectory *find_file(const String &p_file, int *r_index) const;

	void reimport_files(const Vector<String> &p_files);

	EditorFileSystem();
	~EditorFileSystem();
};

#endif // EDITOR_FILE_SYSTEM_H
