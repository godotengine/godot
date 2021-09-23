/*************************************************************************/
/*  resource_loader.cpp                                                  */
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

#include "resource_loader.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_importer.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"
#include "core/variant/variant_parser.h"

#ifdef DEBUG_LOAD_THREADED
#define print_lt(m_text) print_line(m_text)
#else
#define print_lt(m_text)
#endif

Ref<ResourceFormatLoader> ResourceLoader::loader[ResourceLoader::MAX_LOADERS];

int ResourceLoader::loader_count = 0;

bool ResourceFormatLoader::recognize_path(const String &p_path, const String &p_for_type) const {
	String extension = p_path.get_extension();

	List<String> extensions;
	if (p_for_type.is_empty()) {
		get_recognized_extensions(&extensions);
	} else {
		get_recognized_extensions_for_type(p_for_type, &extensions);
	}

	for (const String &E : extensions) {
		if (E.nocasecmp_to(extension) == 0) {
			return true;
		}
	}

	return false;
}

bool ResourceFormatLoader::handles_type(const String &p_type) const {
	bool success;
	if (GDVIRTUAL_CALL(_handles_type, p_type, success)) {
		return success;
	}

	return false;
}

String ResourceFormatLoader::get_resource_type(const String &p_path) const {
	String ret;

	if (GDVIRTUAL_CALL(_get_resource_type, p_path, ret)) {
		return ret;
	}

	return "";
}

ResourceUID::ID ResourceFormatLoader::get_resource_uid(const String &p_path) const {
	int64_t uid;
	if (GDVIRTUAL_CALL(_get_resource_uid, p_path, uid)) {
		return uid;
	}

	return ResourceUID::INVALID_ID;
}

void ResourceFormatLoader::get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const {
	if (p_type.is_empty() || handles_type(p_type)) {
		get_recognized_extensions(p_extensions);
	}
}

void ResourceLoader::get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) {
	for (int i = 0; i < loader_count; i++) {
		loader[i]->get_recognized_extensions_for_type(p_type, p_extensions);
	}
}

bool ResourceFormatLoader::exists(const String &p_path) const {
	bool success;
	if (GDVIRTUAL_CALL(_exists, p_path, success)) {
		return success;
	}
	return FileAccess::exists(p_path); //by default just check file
}

void ResourceFormatLoader::get_recognized_extensions(List<String> *p_extensions) const {
	PackedStringArray exts;
	if (GDVIRTUAL_CALL(_get_recognized_extensions, exts)) {
		const String *r = exts.ptr();
		for (int i = 0; i < exts.size(); ++i) {
			p_extensions->push_back(r[i]);
		}
	}
}

RES ResourceFormatLoader::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Variant res;
	if (GDVIRTUAL_CALL(_load, p_path, p_original_path, p_use_sub_threads, p_cache_mode, res)) {
		if (res.get_type() == Variant::INT) { // Error code, abort.
			if (r_error) {
				*r_error = (Error)res.operator int64_t();
			}
			return RES();
		} else { // Success, pass on result.
			if (r_error) {
				*r_error = OK;
			}
			return res;
		}
	}

	ERR_FAIL_V_MSG(RES(), "Failed to load resource '" + p_path + "'. ResourceFormatLoader::load was not implemented for this resource type.");
}

void ResourceFormatLoader::get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types) {
	PackedStringArray deps;
	if (GDVIRTUAL_CALL(_get_dependencies, p_path, p_add_types, deps)) {
		const String *r = deps.ptr();
		for (int i = 0; i < deps.size(); ++i) {
			p_dependencies->push_back(r[i]);
		}
	}
}

Error ResourceFormatLoader::rename_dependencies(const String &p_path, const Map<String, String> &p_map) {
	Dictionary deps_dict;
	for (KeyValue<String, String> E : p_map) {
		deps_dict[E.key] = E.value;
	}

	int64_t err;
	if (GDVIRTUAL_CALL(_rename_dependencies, p_path, deps_dict, err)) {
		return (Error)err;
	}

	return OK;
}

void ResourceFormatLoader::_bind_methods() {
	BIND_ENUM_CONSTANT(CACHE_MODE_IGNORE);
	BIND_ENUM_CONSTANT(CACHE_MODE_REUSE);
	BIND_ENUM_CONSTANT(CACHE_MODE_REPLACE);

	GDVIRTUAL_BIND(_get_recognized_extensions);
	GDVIRTUAL_BIND(_handles_type, "type");
	GDVIRTUAL_BIND(_get_resource_type, "path");
	GDVIRTUAL_BIND(_get_resource_uid, "path");
	GDVIRTUAL_BIND(_get_dependencies, "path", "add_types");
	GDVIRTUAL_BIND(_rename_dependencies, "path", "renames");
	GDVIRTUAL_BIND(_exists, "path");
	GDVIRTUAL_BIND(_load, "path", "original_path", "use_sub_threads", "cache_mode");
}

///////////////////////////////////

RES ResourceLoader::_load(const String &p_path, const String &p_original_path, const String &p_type_hint, ResourceFormatLoader::CacheMode p_cache_mode, Error *r_error, bool p_use_sub_threads, float *r_progress) {
	bool found = false;

	// Try all loaders and pick the first match for the type hint
	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(p_path, p_type_hint)) {
			continue;
		}
		found = true;
		RES res = loader[i]->load(p_path, !p_original_path.is_empty() ? p_original_path : p_path, r_error, p_use_sub_threads, r_progress, p_cache_mode);
		if (res.is_null()) {
			continue;
		}

		return res;
	}

	ERR_FAIL_COND_V_MSG(found, RES(),
			vformat("Failed loading resource: %s. Make sure resources have been imported by opening the project in the editor at least once.", p_path));

#ifdef TOOLS_ENABLED
	FileAccessRef file_check = FileAccess::create(FileAccess::ACCESS_RESOURCES);
	ERR_FAIL_COND_V_MSG(!file_check->file_exists(p_path), RES(), "Resource file not found: " + p_path + ".");
#endif

	ERR_FAIL_V_MSG(RES(), "No loader found for resource: " + p_path + ".");
}

void ResourceLoader::_thread_load_function(void *p_userdata) {
	ThreadLoadTask &load_task = *(ThreadLoadTask *)p_userdata;
	load_task.loader_id = Thread::get_caller_id();

	if (load_task.semaphore) {
		//this is an actual thread, so wait for Ok from semaphore
		thread_load_semaphore->wait(); //wait until its ok to start loading
	}
	load_task.resource = _load(load_task.remapped_path, load_task.remapped_path != load_task.local_path ? load_task.local_path : String(), load_task.type_hint, load_task.cache_mode, &load_task.error, load_task.use_sub_threads, &load_task.progress);

	load_task.progress = 1.0; //it was fully loaded at this point, so force progress to 1.0

	thread_load_mutex->lock();
	if (load_task.error != OK) {
		load_task.status = THREAD_LOAD_FAILED;
	} else {
		load_task.status = THREAD_LOAD_LOADED;
	}
	if (load_task.semaphore) {
		if (load_task.start_next && thread_waiting_count > 0) {
			thread_waiting_count--;
			//thread loading count remains constant, this ends but another one begins
			thread_load_semaphore->post();
		} else {
			thread_loading_count--; //no threads waiting, just reduce loading count
		}

		print_lt("END: load count: " + itos(thread_loading_count) + " / wait count: " + itos(thread_waiting_count) + " / suspended count: " + itos(thread_suspended_count) + " / active: " + itos(thread_loading_count - thread_suspended_count));

		for (int i = 0; i < load_task.poll_requests; i++) {
			load_task.semaphore->post();
		}
		memdelete(load_task.semaphore);
		load_task.semaphore = nullptr;
	}

	if (load_task.resource.is_valid()) {
		load_task.resource->set_path(load_task.local_path);

		if (load_task.xl_remapped) {
			load_task.resource->set_as_translation_remapped(true);
		}

#ifdef TOOLS_ENABLED

		load_task.resource->set_edited(false);
		if (timestamp_on_load) {
			uint64_t mt = FileAccess::get_modified_time(load_task.remapped_path);
			//printf("mt %s: %lli\n",remapped_path.utf8().get_data(),mt);
			load_task.resource->set_last_modified_time(mt);
		}
#endif

		if (_loaded_callback) {
			_loaded_callback(load_task.resource, load_task.local_path);
		}
	}

	thread_load_mutex->unlock();
}

static String _validate_local_path(const String &p_path) {
	ResourceUID::ID uid = ResourceUID::get_singleton()->text_to_id(p_path);
	if (uid != ResourceUID::INVALID_ID) {
		return ResourceUID::get_singleton()->get_id_path(uid);
	} else if (p_path.is_relative_path()) {
		return "res://" + p_path;
	} else {
		return ProjectSettings::get_singleton()->localize_path(p_path);
	}
}
Error ResourceLoader::load_threaded_request(const String &p_path, const String &p_type_hint, bool p_use_sub_threads, ResourceFormatLoader::CacheMode p_cache_mode, const String &p_source_resource) {
	String local_path = _validate_local_path(p_path);

	thread_load_mutex->lock();

	if (!p_source_resource.is_empty()) {
		//must be loading from this resource
		if (!thread_load_tasks.has(p_source_resource)) {
			thread_load_mutex->unlock();
			ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "There is no thread loading source resource '" + p_source_resource + "'.");
		}
		//must be loading from this thread
		if (thread_load_tasks[p_source_resource].loader_id != Thread::get_caller_id()) {
			thread_load_mutex->unlock();
			ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Threading loading resource'" + local_path + " failed: Source specified: '" + p_source_resource + "' but was not called by it.");
		}

		//must not be already added as s sub tasks
		if (thread_load_tasks[p_source_resource].sub_tasks.has(local_path)) {
			thread_load_mutex->unlock();
			ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Thread loading source resource '" + p_source_resource + "' already is loading '" + local_path + "'.");
		}
	}

	if (thread_load_tasks.has(local_path)) {
		thread_load_tasks[local_path].requests++;
		if (!p_source_resource.is_empty()) {
			thread_load_tasks[p_source_resource].sub_tasks.insert(local_path);
		}
		thread_load_mutex->unlock();
		return OK;
	}

	{
		//create load task

		ThreadLoadTask load_task;

		load_task.requests = 1;
		load_task.remapped_path = _path_remap(local_path, &load_task.xl_remapped);
		load_task.local_path = local_path;
		load_task.type_hint = p_type_hint;
		load_task.cache_mode = p_cache_mode;
		load_task.use_sub_threads = p_use_sub_threads;

		{ //must check if resource is already loaded before attempting to load it in a thread

			if (load_task.loader_id == Thread::get_caller_id()) {
				thread_load_mutex->unlock();
				ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Attempted to load a resource already being loaded from this thread, cyclic reference?");
			}
			//lock first if possible
			ResourceCache::lock.read_lock();

			//get ptr
			Resource **rptr = ResourceCache::resources.getptr(local_path);

			if (rptr) {
				RES res(*rptr);
				//it is possible this resource was just freed in a thread. If so, this referencing will not work and resource is considered not cached
				if (res.is_valid()) {
					//referencing is fine
					load_task.resource = res;
					load_task.status = THREAD_LOAD_LOADED;
					load_task.progress = 1.0;
				}
			}
			ResourceCache::lock.read_unlock();
		}

		if (!p_source_resource.is_empty()) {
			thread_load_tasks[p_source_resource].sub_tasks.insert(local_path);
		}

		thread_load_tasks[local_path] = load_task;
	}

	ThreadLoadTask &load_task = thread_load_tasks[local_path];

	if (load_task.resource.is_null()) { //needs to be loaded in thread

		load_task.semaphore = memnew(Semaphore);
		if (thread_loading_count < thread_load_max) {
			thread_loading_count++;
			thread_load_semaphore->post(); //we have free threads, so allow one
		} else {
			thread_waiting_count++;
		}

		print_lt("REQUEST: load count: " + itos(thread_loading_count) + " / wait count: " + itos(thread_waiting_count) + " / suspended count: " + itos(thread_suspended_count) + " / active: " + itos(thread_loading_count - thread_suspended_count));

		load_task.thread = memnew(Thread);
		load_task.thread->start(_thread_load_function, &thread_load_tasks[local_path]);
		load_task.loader_id = load_task.thread->get_id();
	}

	thread_load_mutex->unlock();

	return OK;
}

float ResourceLoader::_dependency_get_progress(const String &p_path) {
	if (thread_load_tasks.has(p_path)) {
		ThreadLoadTask &load_task = thread_load_tasks[p_path];
		int dep_count = load_task.sub_tasks.size();
		if (dep_count > 0) {
			float dep_progress = 0;
			for (Set<String>::Element *E = load_task.sub_tasks.front(); E; E = E->next()) {
				dep_progress += _dependency_get_progress(E->get());
			}
			dep_progress /= float(dep_count);
			dep_progress *= 0.5;
			dep_progress += load_task.progress * 0.5;
			return dep_progress;
		} else {
			return load_task.progress;
		}

	} else {
		return 1.0; //assume finished loading it so it no longer exists
	}
}

ResourceLoader::ThreadLoadStatus ResourceLoader::load_threaded_get_status(const String &p_path, float *r_progress) {
	String local_path = _validate_local_path(p_path);

	thread_load_mutex->lock();
	if (!thread_load_tasks.has(local_path)) {
		thread_load_mutex->unlock();
		return THREAD_LOAD_INVALID_RESOURCE;
	}
	ThreadLoadTask &load_task = thread_load_tasks[local_path];
	ThreadLoadStatus status;
	status = load_task.status;
	if (r_progress) {
		*r_progress = _dependency_get_progress(local_path);
	}

	thread_load_mutex->unlock();

	return status;
}

RES ResourceLoader::load_threaded_get(const String &p_path, Error *r_error) {
	String local_path = _validate_local_path(p_path);

	thread_load_mutex->lock();
	if (!thread_load_tasks.has(local_path)) {
		thread_load_mutex->unlock();
		if (r_error) {
			*r_error = ERR_INVALID_PARAMETER;
		}
		return RES();
	}

	ThreadLoadTask &load_task = thread_load_tasks[local_path];

	//semaphore still exists, meaning it's still loading, request poll
	Semaphore *semaphore = load_task.semaphore;
	if (semaphore) {
		load_task.poll_requests++;

		{
			// As we got a semaphore, this means we are going to have to wait
			// until the sub-resource is done loading
			//
			// As this thread will become 'blocked' we should "exchange" its
			// active status with a waiting one, to ensure load continues.
			//
			// This ensures loading is never blocked and that is also within
			// the maximum number of active threads.

			if (thread_waiting_count > 0) {
				thread_waiting_count--;
				thread_loading_count++;
				thread_load_semaphore->post();

				load_task.start_next = false; //do not start next since we are doing it here
			}

			thread_suspended_count++;

			print_lt("GET: load count: " + itos(thread_loading_count) + " / wait count: " + itos(thread_waiting_count) + " / suspended count: " + itos(thread_suspended_count) + " / active: " + itos(thread_loading_count - thread_suspended_count));
		}

		thread_load_mutex->unlock();
		semaphore->wait();
		thread_load_mutex->lock();

		thread_suspended_count--;

		if (!thread_load_tasks.has(local_path)) { //may have been erased during unlock and this was always an invalid call
			thread_load_mutex->unlock();
			if (r_error) {
				*r_error = ERR_INVALID_PARAMETER;
			}
			return RES();
		}
	}

	RES resource = load_task.resource;
	if (r_error) {
		*r_error = load_task.error;
	}

	load_task.requests--;

	if (load_task.requests == 0) {
		if (load_task.thread) { //thread may not have been used
			load_task.thread->wait_to_finish();
			memdelete(load_task.thread);
		}
		thread_load_tasks.erase(local_path);
	}

	thread_load_mutex->unlock();

	return resource;
}

RES ResourceLoader::load(const String &p_path, const String &p_type_hint, ResourceFormatLoader::CacheMode p_cache_mode, Error *r_error) {
	if (r_error) {
		*r_error = ERR_CANT_OPEN;
	}

	String local_path = _validate_local_path(p_path);

	if (p_cache_mode != ResourceFormatLoader::CACHE_MODE_IGNORE) {
		thread_load_mutex->lock();

		//Is it already being loaded? poll until done
		if (thread_load_tasks.has(local_path)) {
			Error err = load_threaded_request(p_path, p_type_hint);
			if (err != OK) {
				if (r_error) {
					*r_error = err;
				}
				thread_load_mutex->unlock();
				return RES();
			}
			thread_load_mutex->unlock();

			return load_threaded_get(p_path, r_error);
		}

		//Is it cached?
		ResourceCache::lock.read_lock();

		Resource **rptr = ResourceCache::resources.getptr(local_path);

		if (rptr) {
			RES res(*rptr);

			//it is possible this resource was just freed in a thread. If so, this referencing will not work and resource is considered not cached
			if (res.is_valid()) {
				ResourceCache::lock.read_unlock();
				thread_load_mutex->unlock();

				if (r_error) {
					*r_error = OK;
				}

				return res; //use cached
			}
		}

		ResourceCache::lock.read_unlock();

		//load using task (but this thread)
		ThreadLoadTask load_task;

		load_task.requests = 1;
		load_task.local_path = local_path;
		load_task.remapped_path = _path_remap(local_path, &load_task.xl_remapped);
		load_task.type_hint = p_type_hint;
		load_task.cache_mode = p_cache_mode; //ignore
		load_task.loader_id = Thread::get_caller_id();

		thread_load_tasks[local_path] = load_task;

		thread_load_mutex->unlock();

		_thread_load_function(&thread_load_tasks[local_path]);

		return load_threaded_get(p_path, r_error);

	} else {
		bool xl_remapped = false;
		String path = _path_remap(local_path, &xl_remapped);

		if (path.is_empty()) {
			ERR_FAIL_V_MSG(RES(), "Remapping '" + local_path + "' failed.");
		}

		print_verbose("Loading resource: " + path);
		float p;
		RES res = _load(path, local_path, p_type_hint, p_cache_mode, r_error, false, &p);

		if (res.is_null()) {
			print_verbose("Failed loading resource: " + path);
			return RES();
		}

		if (xl_remapped) {
			res->set_as_translation_remapped(true);
		}

#ifdef TOOLS_ENABLED

		res->set_edited(false);
		if (timestamp_on_load) {
			uint64_t mt = FileAccess::get_modified_time(path);
			//printf("mt %s: %lli\n",remapped_path.utf8().get_data(),mt);
			res->set_last_modified_time(mt);
		}
#endif

		return res;
	}
}

bool ResourceLoader::exists(const String &p_path, const String &p_type_hint) {
	String local_path = _validate_local_path(p_path);

	if (ResourceCache::has(local_path)) {
		return true; // If cached, it probably exists
	}

	bool xl_remapped = false;
	String path = _path_remap(local_path, &xl_remapped);

	// Try all loaders and pick the first match for the type hint
	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(path, p_type_hint)) {
			continue;
		}

		if (loader[i]->exists(path)) {
			return true;
		}
	}

	return false;
}

void ResourceLoader::add_resource_format_loader(Ref<ResourceFormatLoader> p_format_loader, bool p_at_front) {
	ERR_FAIL_COND(p_format_loader.is_null());
	ERR_FAIL_COND(loader_count >= MAX_LOADERS);

	if (p_at_front) {
		for (int i = loader_count; i > 0; i--) {
			loader[i] = loader[i - 1];
		}
		loader[0] = p_format_loader;
		loader_count++;
	} else {
		loader[loader_count++] = p_format_loader;
	}
}

void ResourceLoader::remove_resource_format_loader(Ref<ResourceFormatLoader> p_format_loader) {
	ERR_FAIL_COND(p_format_loader.is_null());

	// Find loader
	int i = 0;
	for (; i < loader_count; ++i) {
		if (loader[i] == p_format_loader) {
			break;
		}
	}

	ERR_FAIL_COND(i >= loader_count); // Not found

	// Shift next loaders up
	for (; i < loader_count - 1; ++i) {
		loader[i] = loader[i + 1];
	}
	loader[loader_count - 1].unref();
	--loader_count;
}

int ResourceLoader::get_import_order(const String &p_path) {
	String local_path = _path_remap(_validate_local_path(p_path));

	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(local_path)) {
			continue;
		}
		/*
		if (p_type_hint!="" && !loader[i]->handles_type(p_type_hint))
			continue;
		*/

		return loader[i]->get_import_order(p_path);
	}

	return 0;
}

String ResourceLoader::get_import_group_file(const String &p_path) {
	String local_path = _path_remap(_validate_local_path(p_path));

	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(local_path)) {
			continue;
		}
		/*
		if (p_type_hint!="" && !loader[i]->handles_type(p_type_hint))
			continue;
		*/

		return loader[i]->get_import_group_file(p_path);
	}

	return String(); //not found
}

bool ResourceLoader::is_import_valid(const String &p_path) {
	String local_path = _path_remap(_validate_local_path(p_path));

	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(local_path)) {
			continue;
		}
		/*
		if (p_type_hint!="" && !loader[i]->handles_type(p_type_hint))
			continue;
		*/

		return loader[i]->is_import_valid(p_path);
	}

	return false; //not found
}

bool ResourceLoader::is_imported(const String &p_path) {
	String local_path = _path_remap(_validate_local_path(p_path));

	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(local_path)) {
			continue;
		}
		/*
		if (p_type_hint!="" && !loader[i]->handles_type(p_type_hint))
			continue;
		*/

		return loader[i]->is_imported(p_path);
	}

	return false; //not found
}

void ResourceLoader::get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types) {
	String local_path = _path_remap(_validate_local_path(p_path));

	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(local_path)) {
			continue;
		}
		/*
		if (p_type_hint!="" && !loader[i]->handles_type(p_type_hint))
			continue;
		*/

		loader[i]->get_dependencies(local_path, p_dependencies, p_add_types);
	}
}

Error ResourceLoader::rename_dependencies(const String &p_path, const Map<String, String> &p_map) {
	String local_path = _path_remap(_validate_local_path(p_path));

	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(local_path)) {
			continue;
		}
		/*
		if (p_type_hint!="" && !loader[i]->handles_type(p_type_hint))
			continue;
		*/

		return loader[i]->rename_dependencies(local_path, p_map);
	}

	return OK; // ??
}

String ResourceLoader::get_resource_type(const String &p_path) {
	String local_path = _validate_local_path(p_path);

	for (int i = 0; i < loader_count; i++) {
		String result = loader[i]->get_resource_type(local_path);
		if (!result.is_empty()) {
			return result;
		}
	}

	return "";
}

ResourceUID::ID ResourceLoader::get_resource_uid(const String &p_path) {
	String local_path = _validate_local_path(p_path);

	for (int i = 0; i < loader_count; i++) {
		ResourceUID::ID id = loader[i]->get_resource_uid(local_path);
		if (id != ResourceUID::INVALID_ID) {
			return id;
		}
	}

	return ResourceUID::INVALID_ID;
}

String ResourceLoader::_path_remap(const String &p_path, bool *r_translation_remapped) {
	String new_path = p_path;

	if (translation_remaps.has(p_path)) {
		// translation_remaps has the following format:
		//   { "res://path.png": PackedStringArray( "res://path-ru.png:ru", "res://path-de.png:de" ) }

		// To find the path of the remapped resource, we extract the locale name after
		// the last ':' to match the project locale.

		String locale = TranslationServer::get_singleton()->get_locale();
		ERR_FAIL_COND_V_MSG(locale.length() < 2, p_path, "Could not remap path '" + p_path + "' for translation as configured locale '" + locale + "' is invalid.");

		Vector<String> &res_remaps = *translation_remaps.getptr(new_path);

		int best_score = 0;
		for (int i = 0; i < res_remaps.size(); i++) {
			int split = res_remaps[i].rfind(":");
			if (split == -1) {
				continue;
			}
			String l = res_remaps[i].substr(split + 1).strip_edges();
			int score = TranslationServer::get_singleton()->compare_locales(locale, l);
			if (score >= best_score) {
				new_path = res_remaps[i].left(split);
				best_score = score;
				if (score == 10) {
					break; // Exact match, skip the rest.
				}
			}
		}

		if (r_translation_remapped) {
			*r_translation_remapped = true;
		}
	}

	if (path_remaps.has(new_path)) {
		new_path = path_remaps[new_path];
	}

	if (new_path == p_path) { // Did not remap.
		// Try file remap.
		Error err;
		FileAccess *f = FileAccess::open(p_path + ".remap", FileAccess::READ, &err);

		if (f) {
			VariantParser::StreamFile stream;
			stream.f = f;

			String assign;
			Variant value;
			VariantParser::Tag next_tag;

			int lines = 0;
			String error_text;
			while (true) {
				assign = Variant();
				next_tag.fields.clear();
				next_tag.name = String();

				err = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, nullptr, true);
				if (err == ERR_FILE_EOF) {
					break;
				} else if (err != OK) {
					ERR_PRINT("Parse error: " + p_path + ".remap:" + itos(lines) + " error: " + error_text + ".");
					break;
				}

				if (assign == "path") {
					new_path = value;
					break;
				} else if (next_tag.name != "remap") {
					break;
				}
			}

			memdelete(f);
		}
	}

	return new_path;
}

String ResourceLoader::import_remap(const String &p_path) {
	if (ResourceFormatImporter::get_singleton()->recognize_path(p_path)) {
		return ResourceFormatImporter::get_singleton()->get_internal_resource_path(p_path);
	}

	return p_path;
}

String ResourceLoader::path_remap(const String &p_path) {
	return _path_remap(p_path);
}

void ResourceLoader::reload_translation_remaps() {
	ResourceCache::lock.read_lock();

	List<Resource *> to_reload;
	SelfList<Resource> *E = remapped_list.first();

	while (E) {
		to_reload.push_back(E->self());
		E = E->next();
	}

	ResourceCache::lock.read_unlock();

	//now just make sure to not delete any of these resources while changing locale..
	while (to_reload.front()) {
		to_reload.front()->get()->reload_from_file();
		to_reload.pop_front();
	}
}

void ResourceLoader::load_translation_remaps() {
	if (!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		return;
	}

	Dictionary remaps = ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps");
	List<Variant> keys;
	remaps.get_key_list(&keys);
	for (const Variant &E : keys) {
		Array langs = remaps[E];
		Vector<String> lang_remaps;
		lang_remaps.resize(langs.size());
		for (int i = 0; i < langs.size(); i++) {
			lang_remaps.write[i] = langs[i];
		}

		translation_remaps[String(E)] = lang_remaps;
	}
}

void ResourceLoader::clear_translation_remaps() {
	translation_remaps.clear();
	while (remapped_list.first() != nullptr) {
		remapped_list.remove(remapped_list.first());
	}
}

void ResourceLoader::load_path_remaps() {
	if (!ProjectSettings::get_singleton()->has_setting("path_remap/remapped_paths")) {
		return;
	}

	Vector<String> remaps = ProjectSettings::get_singleton()->get("path_remap/remapped_paths");
	int rc = remaps.size();
	ERR_FAIL_COND(rc & 1); //must be even
	const String *r = remaps.ptr();

	for (int i = 0; i < rc; i += 2) {
		path_remaps[r[i]] = r[i + 1];
	}
}

void ResourceLoader::clear_path_remaps() {
	path_remaps.clear();
}

void ResourceLoader::set_load_callback(ResourceLoadedCallback p_callback) {
	_loaded_callback = p_callback;
}

ResourceLoadedCallback ResourceLoader::_loaded_callback = nullptr;

Ref<ResourceFormatLoader> ResourceLoader::_find_custom_resource_format_loader(String path) {
	for (int i = 0; i < loader_count; ++i) {
		if (loader[i]->get_script_instance() && loader[i]->get_script_instance()->get_script()->get_path() == path) {
			return loader[i];
		}
	}
	return Ref<ResourceFormatLoader>();
}

bool ResourceLoader::add_custom_resource_format_loader(String script_path) {
	if (_find_custom_resource_format_loader(script_path).is_valid()) {
		return false;
	}

	Ref<Resource> res = ResourceLoader::load(script_path);
	ERR_FAIL_COND_V(res.is_null(), false);
	ERR_FAIL_COND_V(!res->is_class("Script"), false);

	Ref<Script> s = res;
	StringName ibt = s->get_instance_base_type();
	bool valid_type = ClassDB::is_parent_class(ibt, "ResourceFormatLoader");
	ERR_FAIL_COND_V_MSG(!valid_type, false, "Script does not inherit a CustomResourceLoader: " + script_path + ".");

	Object *obj = ClassDB::instantiate(ibt);

	ERR_FAIL_COND_V_MSG(obj == nullptr, false, "Cannot instance script as custom resource loader, expected 'ResourceFormatLoader' inheritance, got: " + String(ibt) + ".");

	Ref<ResourceFormatLoader> crl = Object::cast_to<ResourceFormatLoader>(obj);
	crl->set_script(s);
	ResourceLoader::add_resource_format_loader(crl);

	return true;
}

void ResourceLoader::remove_custom_resource_format_loader(String script_path) {
	Ref<ResourceFormatLoader> custom_loader = _find_custom_resource_format_loader(script_path);
	if (custom_loader.is_valid()) {
		remove_resource_format_loader(custom_loader);
	}
}

void ResourceLoader::add_custom_loaders() {
	// Custom loaders registration exploits global class names

	String custom_loader_base_class = ResourceFormatLoader::get_class_static();

	List<StringName> global_classes;
	ScriptServer::get_global_class_list(&global_classes);

	for (const StringName &class_name : global_classes) {
		StringName base_class = ScriptServer::get_global_class_native_base(class_name);

		if (base_class == custom_loader_base_class) {
			String path = ScriptServer::get_global_class_path(class_name);
			add_custom_resource_format_loader(path);
		}
	}
}

void ResourceLoader::remove_custom_loaders() {
	Vector<Ref<ResourceFormatLoader>> custom_loaders;
	for (int i = 0; i < loader_count; ++i) {
		if (loader[i]->get_script_instance()) {
			custom_loaders.push_back(loader[i]);
		}
	}

	for (int i = 0; i < custom_loaders.size(); ++i) {
		remove_resource_format_loader(custom_loaders[i]);
	}
}

void ResourceLoader::initialize() {
	thread_load_mutex = memnew(Mutex);
	thread_load_max = OS::get_singleton()->get_processor_count();
	thread_loading_count = 0;
	thread_waiting_count = 0;
	thread_suspended_count = 0;
	thread_load_semaphore = memnew(Semaphore);
}

void ResourceLoader::finalize() {
	memdelete(thread_load_mutex);
	memdelete(thread_load_semaphore);
}

ResourceLoadErrorNotify ResourceLoader::err_notify = nullptr;
void *ResourceLoader::err_notify_ud = nullptr;

DependencyErrorNotify ResourceLoader::dep_err_notify = nullptr;
void *ResourceLoader::dep_err_notify_ud = nullptr;

bool ResourceLoader::abort_on_missing_resource = true;
bool ResourceLoader::timestamp_on_load = false;

Mutex *ResourceLoader::thread_load_mutex = nullptr;
HashMap<String, ResourceLoader::ThreadLoadTask> ResourceLoader::thread_load_tasks;
Semaphore *ResourceLoader::thread_load_semaphore = nullptr;

int ResourceLoader::thread_loading_count = 0;
int ResourceLoader::thread_waiting_count = 0;
int ResourceLoader::thread_suspended_count = 0;
int ResourceLoader::thread_load_max = 0;

SelfList<Resource>::List ResourceLoader::remapped_list;
HashMap<String, Vector<String>> ResourceLoader::translation_remaps;
HashMap<String, String> ResourceLoader::path_remaps;

ResourceLoaderImport ResourceLoader::import = nullptr;
