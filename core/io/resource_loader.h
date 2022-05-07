/*************************************************************************/
/*  resource_loader.h                                                    */
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

#ifndef RESOURCE_LOADER_H
#define RESOURCE_LOADER_H

#include "core/os/thread.h"
#include "core/resource.h"

class ResourceInteractiveLoader : public Reference {
	GDCLASS(ResourceInteractiveLoader, Reference);
	friend class ResourceLoader;
	String path_loading;
	Thread::ID path_loading_thread;

protected:
	static void _bind_methods();

public:
	virtual void set_local_path(const String &p_local_path) = 0;
	virtual Ref<Resource> get_resource() = 0;
	virtual Error poll() = 0;
	virtual int get_stage() const = 0;
	virtual int get_stage_count() const = 0;
	virtual void set_translation_remapped(bool p_remapped) = 0;
	virtual Error wait();

	ResourceInteractiveLoader() {}
	~ResourceInteractiveLoader();
};

class ResourceFormatLoader : public Reference {
	GDCLASS(ResourceFormatLoader, Reference);

protected:
	static void _bind_methods();

public:
	virtual Ref<ResourceInteractiveLoader> load_interactive(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr);
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr);
	virtual bool exists(const String &p_path) const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const;
	virtual bool recognize_path(const String &p_path, const String &p_for_type = String()) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
	virtual void get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types = false);
	virtual Error rename_dependencies(const String &p_path, const Map<String, String> &p_map);
	virtual bool is_import_valid(const String &p_path) const { return true; }
	virtual bool is_imported(const String &p_path) const { return false; }
	virtual int get_import_order(const String &p_path) const { return 0; }
	virtual String get_import_group_file(const String &p_path) const { return ""; } //no group

	virtual ~ResourceFormatLoader() {}
};

typedef void (*ResourceLoadErrorNotify)(void *p_ud, const String &p_text);
typedef void (*DependencyErrorNotify)(void *p_ud, const String &p_loading, const String &p_which, const String &p_type);

typedef Error (*ResourceLoaderImport)(const String &p_path);
typedef void (*ResourceLoadedCallback)(RES p_resource, const String &p_path);

class ResourceLoader {
	enum {
		MAX_LOADERS = 64
	};

	static Ref<ResourceFormatLoader> loader[MAX_LOADERS];
	static int loader_count;
	static bool timestamp_on_load;

	static void *err_notify_ud;
	static ResourceLoadErrorNotify err_notify;
	static void *dep_err_notify_ud;
	static DependencyErrorNotify dep_err_notify;
	static bool abort_on_missing_resource;
	static HashMap<String, Vector<String>> translation_remaps;
	static HashMap<String, String> path_remaps;

	static String _path_remap(const String &p_path, bool *r_translation_remapped = nullptr);
	friend class Resource;

	static SelfList<Resource>::List remapped_list;

	friend class ResourceFormatImporter;
	friend class ResourceInteractiveLoader;
	//internal load function
	static RES _load(const String &p_path, const String &p_original_path, const String &p_type_hint, bool p_no_cache, Error *r_error);

	static ResourceLoadedCallback _loaded_callback;

	static Ref<ResourceFormatLoader> _find_custom_resource_format_loader(String path);
	static Mutex loading_map_mutex;

	//used to track paths being loaded in a thread, avoids cyclic recursion
	struct LoadingMapKey {
		String path;
		Thread::ID thread;
		bool operator==(const LoadingMapKey &p_key) const {
			return (thread == p_key.thread && path == p_key.path);
		}
	};
	struct LoadingMapKeyHasher {
		static _FORCE_INLINE_ uint32_t hash(const LoadingMapKey &p_key) { return p_key.path.hash() + HashMapHasherDefault::hash(p_key.thread); }
	};

	static HashMap<LoadingMapKey, int, LoadingMapKeyHasher> loading_map;

	static bool _add_to_loading_map(const String &p_path);
	static void _remove_from_loading_map(const String &p_path);
	static void _remove_from_loading_map_and_thread(const String &p_path, Thread::ID p_thread);

public:
	static Ref<ResourceInteractiveLoader> load_interactive(const String &p_path, const String &p_type_hint = "", bool p_no_cache = false, Error *r_error = nullptr);
	static RES load(const String &p_path, const String &p_type_hint = "", bool p_no_cache = false, Error *r_error = nullptr);
	static bool exists(const String &p_path, const String &p_type_hint = "");

	static void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions);
	static void add_resource_format_loader(Ref<ResourceFormatLoader> p_format_loader, bool p_at_front = false);
	static void remove_resource_format_loader(Ref<ResourceFormatLoader> p_format_loader);
	static String get_resource_type(const String &p_path);
	static void get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types = false);
	static Error rename_dependencies(const String &p_path, const Map<String, String> &p_map);
	static bool is_import_valid(const String &p_path);
	static String get_import_group_file(const String &p_path);
	static bool is_imported(const String &p_path);
	static int get_import_order(const String &p_path);

	static void set_timestamp_on_load(bool p_timestamp) { timestamp_on_load = p_timestamp; }
	static bool get_timestamp_on_load() { return timestamp_on_load; }

	static void notify_load_error(const String &p_err) {
		if (err_notify) {
			err_notify(err_notify_ud, p_err);
		}
	}
	static void set_error_notify_func(void *p_ud, ResourceLoadErrorNotify p_err_notify) {
		err_notify = p_err_notify;
		err_notify_ud = p_ud;
	}

	static void notify_dependency_error(const String &p_path, const String &p_dependency, const String &p_type) {
		if (dep_err_notify) {
			dep_err_notify(dep_err_notify_ud, p_path, p_dependency, p_type);
		}
	}
	static void set_dependency_error_notify_func(void *p_ud, DependencyErrorNotify p_err_notify) {
		dep_err_notify = p_err_notify;
		dep_err_notify_ud = p_ud;
	}

	static void set_abort_on_missing_resources(bool p_abort) { abort_on_missing_resource = p_abort; }
	static bool get_abort_on_missing_resources() { return abort_on_missing_resource; }

	static String path_remap(const String &p_path);
	static String import_remap(const String &p_path);

	static void load_path_remaps();
	static void clear_path_remaps();

	static void reload_translation_remaps();
	static void load_translation_remaps();
	static void clear_translation_remaps();

	static void set_load_callback(ResourceLoadedCallback p_callback);
	static ResourceLoaderImport import;

	static bool add_custom_resource_format_loader(String script_path);
	static void remove_custom_resource_format_loader(String script_path);
	static void add_custom_loaders();
	static void remove_custom_loaders();

	static void finalize();
};

#endif
