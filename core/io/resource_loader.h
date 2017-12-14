/*************************************************************************/
/*  resource_loader.h                                                    */
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
#ifndef RESOURCE_LOADER_H
#define RESOURCE_LOADER_H

#include "resource.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class ResourceInteractiveLoader : public Reference {

	GDCLASS(ResourceInteractiveLoader, Reference);

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
};

class ResourceFormatLoader {
public:
	virtual Ref<ResourceInteractiveLoader> load_interactive(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const = 0;
	virtual void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const;
	virtual bool recognize_path(const String &p_path, const String &p_for_type = String()) const;
	virtual bool handles_type(const String &p_type) const = 0;
	virtual String get_resource_type(const String &p_path) const = 0;
	virtual void get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types = false);
	virtual Error rename_dependencies(const String &p_path, const Map<String, String> &p_map) { return OK; }
	virtual bool is_import_valid(const String &p_path) const { return true; }
	virtual int get_import_order(const String &p_path) const { return 0; }

	virtual ~ResourceFormatLoader() {}
};

typedef void (*ResourceLoadErrorNotify)(void *p_ud, const String &p_text);
typedef void (*DependencyErrorNotify)(void *p_ud, const String &p_loading, const String &p_which, const String &p_type);

class ResourceLoader {

	enum {
		MAX_LOADERS = 64
	};

	static ResourceFormatLoader *loader[MAX_LOADERS];
	static int loader_count;
	static bool timestamp_on_load;

	static void *err_notify_ud;
	static ResourceLoadErrorNotify err_notify;
	static void *dep_err_notify_ud;
	static DependencyErrorNotify dep_err_notify;
	static bool abort_on_missing_resource;
	static HashMap<String, Vector<String> > translation_remaps;
	static HashMap<String, String> path_remaps;

	static String _path_remap(const String &p_path, bool *r_translation_remapped = NULL);
	friend class Resource;

	static SelfList<Resource>::List remapped_list;

	friend class ResourceFormatImporter;
	//internal load function
	static RES _load(const String &p_path, const String &p_original_path, const String &p_type_hint, bool p_no_cache, Error *r_error);

public:
	static Ref<ResourceInteractiveLoader> load_interactive(const String &p_path, const String &p_type_hint = "", bool p_no_cache = false, Error *r_error = NULL);
	static RES load(const String &p_path, const String &p_type_hint = "", bool p_no_cache = false, Error *r_error = NULL);

	static void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions);
	static void add_resource_format_loader(ResourceFormatLoader *p_format_loader, bool p_at_front = false);
	static String get_resource_type(const String &p_path);
	static void get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types = false);
	static Error rename_dependencies(const String &p_path, const Map<String, String> &p_map);
	static bool is_import_valid(const String &p_path);
	static int get_import_order(const String &p_path);

	static void set_timestamp_on_load(bool p_timestamp) { timestamp_on_load = p_timestamp; }

	static void notify_load_error(const String &p_err) {
		if (err_notify) err_notify(err_notify_ud, p_err);
	}
	static void set_error_notify_func(void *p_ud, ResourceLoadErrorNotify p_err_notify) {
		err_notify = p_err_notify;
		err_notify_ud = p_ud;
	}

	static void notify_dependency_error(const String &p_path, const String &p_dependency, const String &p_type) {
		if (dep_err_notify) dep_err_notify(dep_err_notify_ud, p_path, p_dependency, p_type);
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
};

#endif
