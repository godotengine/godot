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

#include "core/io/resource.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/object/script_language.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"

class ResourceFormatLoader : public RefCounted {
	GDCLASS(ResourceFormatLoader, RefCounted);

public:
	enum CacheMode {
		CACHE_MODE_IGNORE, // Resource and subresources do not use path cache, no path is set into resource.
		CACHE_MODE_REUSE, // Resource and subresources use patch cache, reuse existing loaded resources instead of loading from disk when available.
		CACHE_MODE_REPLACE, // Resource and subresource use path cache, but replace existing loaded resources when available with information from disk.
	};

protected:
	static void _bind_methods();

	GDVIRTUAL0RC(Vector<String>, _get_recognized_extensions)
	GDVIRTUAL1RC(bool, _handles_type, StringName)
	GDVIRTUAL1RC(String, _get_resource_type, String)
	GDVIRTUAL1RC(ResourceUID::ID, _get_resource_uid, String)
	GDVIRTUAL2RC(Vector<String>, _get_dependencies, String, bool)
	GDVIRTUAL2RC(int64_t, _rename_dependencies, String, Dictionary)
	GDVIRTUAL1RC(bool, _exists, String)

	GDVIRTUAL4RC(Variant, _load, String, String, bool, int)

public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
	virtual bool exists(const String &p_path) const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const;
	virtual bool recognize_path(const String &p_path, const String &p_for_type = String()) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
	virtual ResourceUID::ID get_resource_uid(const String &p_path) const;
	virtual void get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types = false);
	virtual Error rename_dependencies(const String &p_path, const Map<String, String> &p_map);
	virtual bool is_import_valid(const String &p_path) const { return true; }
	virtual bool is_imported(const String &p_path) const { return false; }
	virtual int get_import_order(const String &p_path) const { return 0; }
	virtual String get_import_group_file(const String &p_path) const { return ""; } //no group

	virtual ~ResourceFormatLoader() {}
};

VARIANT_ENUM_CAST(ResourceFormatLoader::CacheMode)

typedef void (*ResourceLoadErrorNotify)(void *p_ud, const String &p_text);
typedef void (*DependencyErrorNotify)(void *p_ud, const String &p_loading, const String &p_which, const String &p_type);

typedef Error (*ResourceLoaderImport)(const String &p_path);
typedef void (*ResourceLoadedCallback)(RES p_resource, const String &p_path);

class ResourceLoader {
	enum {
		MAX_LOADERS = 64
	};

public:
	enum ThreadLoadStatus {
		THREAD_LOAD_INVALID_RESOURCE,
		THREAD_LOAD_IN_PROGRESS,
		THREAD_LOAD_FAILED,
		THREAD_LOAD_LOADED
	};

private:
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
	// Internal load function.
	static RES _load(const String &p_path, const String &p_original_path, const String &p_type_hint, ResourceFormatLoader::CacheMode p_cache_mode, Error *r_error, bool p_use_sub_threads, float *r_progress);

	static ResourceLoadedCallback _loaded_callback;

	static Ref<ResourceFormatLoader> _find_custom_resource_format_loader(String path);

	struct ThreadLoadTask {
		Thread *thread = nullptr;
		Thread::ID loader_id = 0;
		Semaphore *semaphore = nullptr;
		String local_path;
		String remapped_path;
		String type_hint;
		float progress = 0.0;
		ThreadLoadStatus status = THREAD_LOAD_IN_PROGRESS;
		ResourceFormatLoader::CacheMode cache_mode = ResourceFormatLoader::CACHE_MODE_REUSE;
		Error error = OK;
		RES resource;
		bool xl_remapped = false;
		bool use_sub_threads = false;
		bool start_next = true;
		int requests = 0;
		int poll_requests = 0;
		Set<String> sub_tasks;
	};

	static void _thread_load_function(void *p_userdata);
	static Mutex *thread_load_mutex;
	static HashMap<String, ThreadLoadTask> thread_load_tasks;
	static Semaphore *thread_load_semaphore;
	static int thread_waiting_count;
	static int thread_loading_count;
	static int thread_suspended_count;
	static int thread_load_max;

	static float _dependency_get_progress(const String &p_path);

public:
	static Error load_threaded_request(const String &p_path, const String &p_type_hint = "", bool p_use_sub_threads = false, ResourceFormatLoader::CacheMode p_cache_mode = ResourceFormatLoader::CACHE_MODE_REUSE, const String &p_source_resource = String());
	static ThreadLoadStatus load_threaded_get_status(const String &p_path, float *r_progress = nullptr);
	static RES load_threaded_get(const String &p_path, Error *r_error = nullptr);

	static RES load(const String &p_path, const String &p_type_hint = "", ResourceFormatLoader::CacheMode p_cache_mode = ResourceFormatLoader::CACHE_MODE_REUSE, Error *r_error = nullptr);
	static bool exists(const String &p_path, const String &p_type_hint = "");

	static void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions);
	static void add_resource_format_loader(Ref<ResourceFormatLoader> p_format_loader, bool p_at_front = false);
	static void remove_resource_format_loader(Ref<ResourceFormatLoader> p_format_loader);
	static String get_resource_type(const String &p_path);
	static ResourceUID::ID get_resource_uid(const String &p_path);
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

	static void initialize();
	static void finalize();
};

#endif // RESOURCE_LOADER_H
