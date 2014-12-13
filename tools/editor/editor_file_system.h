/*************************************************************************/
/*  editor_file_system.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#include "scene/main/node.h"
#include "os/thread.h"
#include "os/dir_access.h"
#include "set.h"
#include "os/thread_safe.h"
class FileAccess;

class EditorProgressBG;
class EditorFileSystemDirectory : public Object {

	OBJ_TYPE( EditorFileSystemDirectory,Object );

	String name;

	EditorFileSystemDirectory *parent;
	Vector<EditorFileSystemDirectory*> subdirs;

	struct ImportMeta {

		struct Source {

			String path;
			String md5;
			uint64_t modified_time;
			bool missing;

		};

		Vector<Source> sources;
		String import_editor;
		bool enabled;

	};

	struct FileInfo {
		String file;
		String type;
		uint64_t modified_time;

		ImportMeta meta;
	};

	Vector<FileInfo> files;

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
	String get_file_type(int p_idx) const;
	bool get_file_meta(int p_idx) const;
	bool is_missing_sources(int p_idx) const;
	Vector<String> get_missing_sources(int p_idx) const;

	EditorFileSystemDirectory *get_parent();


	EditorFileSystemDirectory();
	~EditorFileSystemDirectory();
};

class EditorFileSystem : public Node {

	OBJ_TYPE( EditorFileSystem, Node );

	_THREAD_SAFE_CLASS_

	struct SceneItem {

		String file;
		String path;
		String type;
		uint64_t modified_time;
		EditorFileSystemDirectory::ImportMeta meta;
	};

	struct DirItem {

		uint64_t modified_time;
		String path;
		String name;
		Vector<DirItem*> dirs;
		Vector<SceneItem*> files;
		~DirItem();
	};

	float total;
	bool use_threads;
	Thread *thread;
	static void _thread_func(void *_userdata);

	DirItem *scandir;
	DirItem *rootdir;

	bool abort_scan;
	bool scanning;

	EditorFileSystemDirectory* _update_tree(DirItem *p_item);

	void _scan_scenes();
	void _load_type_cache();

	EditorFileSystemDirectory *filesystem;

	static EditorFileSystem *singleton;

	struct FileCache {

		String type;
		uint64_t modification_time;
		EditorFileSystemDirectory::ImportMeta meta;
	};

	struct DirCache {

		uint64_t modification_time;
		Set<String> files;
		Set<String> subdirs;
	};


	static EditorFileSystemDirectory::ImportMeta _get_meta(const String& p_path);

	bool _check_meta_sources(EditorFileSystemDirectory::ImportMeta & p_meta,EditorProgressBG *ep=NULL);

	DirItem* _scan_dir(DirAccess *da,Set<String> &extensions,String p_name,float p_from,float p_range,const String& p_path,HashMap<String,FileCache> &file_cache,HashMap<String,DirCache> &dir_cache,EditorProgressBG& p_prog);
	void _save_type_cache_fs(DirItem *p_dir,FileAccess *p_file);

	bool _find_file(const String& p_file,EditorFileSystemDirectory ** r_d, int &r_file_pos) const;

	void _scan_sources(EditorFileSystemDirectory *p_dir,EditorProgressBG *ep);

	int md_count;


	Thread *thread_sources;
	bool scanning_sources;
	bool scanning_sources_done;
	int ss_amount;
	static void _thread_func_sources(void *_userdata);
	List<String> sources_changed;

	static void _resource_saved(const String& p_path);
	String _find_first_from_source(EditorFileSystemDirectory* p_dir,const String &p_src) const;

protected:

	void _notification(int p_what);
	static void _bind_methods();
public:


	static EditorFileSystem* get_singleton() { return singleton; }

	EditorFileSystemDirectory *get_filesystem();
	bool is_scanning() const;
	float get_scanning_progress() const;
	void scan();
	void scan_sources();
	void get_changed_sources(List<String> *r_changed);
	void update_file(const String& p_file);
	String find_resource_from_source(const String& p_path) const;
	EditorFileSystemDirectory *get_path(const String& p_path);
	String get_file_type(const String& p_file) const;
	EditorFileSystem();
	~EditorFileSystem();
};

#endif // EDITOR_FILE_SYSTEM_H
