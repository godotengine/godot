/**************************************************************************/
/*  resource_loader.cpp                                                   */
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

#include "resource_loader.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_importer.h"
#include "core/object/script_language.h"
#include "core/os/condition_variable.h"
#include "core/os/os.h"
#include "core/os/safe_binary_mutex.h"
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
	bool ret = false;
	if (GDVIRTUAL_CALL(_recognize_path, p_path, p_for_type, ret)) {
		return ret;
	}

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
	bool success = false;
	GDVIRTUAL_CALL(_handles_type, p_type, success);
	return success;
}

void ResourceFormatLoader::get_classes_used(const String &p_path, HashSet<StringName> *r_classes) {
	Vector<String> ret;
	if (GDVIRTUAL_CALL(_get_classes_used, p_path, ret)) {
		for (int i = 0; i < ret.size(); i++) {
			r_classes->insert(ret[i]);
		}
		return;
	}

	String res = get_resource_type(p_path);
	if (!res.is_empty()) {
		r_classes->insert(res);
	}
}

String ResourceFormatLoader::get_resource_type(const String &p_path) const {
	String ret;
	GDVIRTUAL_CALL(_get_resource_type, p_path, ret);
	return ret;
}

String ResourceFormatLoader::get_resource_script_class(const String &p_path) const {
	String ret;
	GDVIRTUAL_CALL(_get_resource_script_class, p_path, ret);
	return ret;
}

ResourceUID::ID ResourceFormatLoader::get_resource_uid(const String &p_path) const {
	int64_t uid = ResourceUID::INVALID_ID;
	GDVIRTUAL_CALL(_get_resource_uid, p_path, uid);
	return uid;
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
	bool success = false;
	if (GDVIRTUAL_CALL(_exists, p_path, success)) {
		return success;
	}
	return FileAccess::exists(p_path); // By default just check file.
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

Ref<Resource> ResourceFormatLoader::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Variant res;
	if (GDVIRTUAL_CALL(_load, p_path, p_original_path, p_use_sub_threads, p_cache_mode, res)) {
		if (res.get_type() == Variant::INT) { // Error code, abort.
			if (r_error) {
				*r_error = (Error)res.operator int64_t();
			}
			return Ref<Resource>();
		} else { // Success, pass on result.
			if (r_error) {
				*r_error = OK;
			}
			return res;
		}
	}

	ERR_FAIL_V_MSG(Ref<Resource>(), "Failed to load resource '" + p_path + "'. ResourceFormatLoader::load was not implemented for this resource type.");
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

Error ResourceFormatLoader::rename_dependencies(const String &p_path, const HashMap<String, String> &p_map) {
	Dictionary deps_dict;
	for (KeyValue<String, String> E : p_map) {
		deps_dict[E.key] = E.value;
	}

	Error err = OK;
	GDVIRTUAL_CALL(_rename_dependencies, p_path, deps_dict, err);
	return err;
}

void ResourceFormatLoader::_bind_methods() {
	BIND_ENUM_CONSTANT(CACHE_MODE_IGNORE);
	BIND_ENUM_CONSTANT(CACHE_MODE_REUSE);
	BIND_ENUM_CONSTANT(CACHE_MODE_REPLACE);
	BIND_ENUM_CONSTANT(CACHE_MODE_IGNORE_DEEP);
	BIND_ENUM_CONSTANT(CACHE_MODE_REPLACE_DEEP);

	GDVIRTUAL_BIND(_get_recognized_extensions);
	GDVIRTUAL_BIND(_recognize_path, "path", "type");
	GDVIRTUAL_BIND(_handles_type, "type");
	GDVIRTUAL_BIND(_get_resource_type, "path");
	GDVIRTUAL_BIND(_get_resource_script_class, "path");
	GDVIRTUAL_BIND(_get_resource_uid, "path");
	GDVIRTUAL_BIND(_get_dependencies, "path", "add_types");
	GDVIRTUAL_BIND(_rename_dependencies, "path", "renames");
	GDVIRTUAL_BIND(_exists, "path");
	GDVIRTUAL_BIND(_get_classes_used, "path");
	GDVIRTUAL_BIND(_load, "path", "original_path", "use_sub_threads", "cache_mode");
}

///////////////////////////////////

// This should be robust enough to be called redundantly without issues.
void ResourceLoader::LoadToken::clear() {
	thread_load_mutex.lock();

	WorkerThreadPool::TaskID task_to_await = 0;

	if (!local_path.is_empty()) { // Empty is used for the special case where the load task is not registered.
		DEV_ASSERT(thread_load_tasks.has(local_path));
		ThreadLoadTask &load_task = thread_load_tasks[local_path];
		if (!load_task.awaited) {
			task_to_await = load_task.task_id;
			load_task.awaited = true;
		}
		thread_load_tasks.erase(local_path);
		local_path.clear();
	}

	if (!user_path.is_empty()) {
		DEV_ASSERT(user_load_tokens.has(user_path));
		user_load_tokens.erase(user_path);
		user_path.clear();
	}

	thread_load_mutex.unlock();

	// If task is unused, await it here, locally, now the token data is consistent.
	if (task_to_await) {
		WorkerThreadPool::get_singleton()->wait_for_task_completion(task_to_await);
	}
}

ResourceLoader::LoadToken::~LoadToken() {
	clear();
}

Ref<Resource> ResourceLoader::_load(const String &p_path, const String &p_original_path, const String &p_type_hint, ResourceFormatLoader::CacheMode p_cache_mode, Error *r_error, bool p_use_sub_threads, float *r_progress) {
	const String &original_path = p_original_path.is_empty() ? p_path : p_original_path;
	load_nesting++;
	if (load_paths_stack->size()) {
		thread_load_mutex.lock();
		const String &parent_task_path = load_paths_stack->get(load_paths_stack->size() - 1);
		HashMap<String, ThreadLoadTask>::Iterator E = thread_load_tasks.find(parent_task_path);
		// Avoid double-tracking, for progress reporting, resources that boil down to a remapped path containing the real payload (e.g., imported resources).
		bool is_remapped_load = original_path == parent_task_path;
		if (E && !is_remapped_load) {
			E->value.sub_tasks.insert(p_original_path);
		}
		thread_load_mutex.unlock();
	}
	load_paths_stack->push_back(original_path);

	// Try all loaders and pick the first match for the type hint
	bool found = false;
	Ref<Resource> res;
	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(p_path, p_type_hint)) {
			continue;
		}
		found = true;
		res = loader[i]->load(p_path, original_path, r_error, p_use_sub_threads, r_progress, p_cache_mode);
		if (!res.is_null()) {
			break;
		}
	}

	load_paths_stack->resize(load_paths_stack->size() - 1);
	load_nesting--;

	if (!res.is_null()) {
		return res;
	}

	ERR_FAIL_COND_V_MSG(found, Ref<Resource>(),
			vformat("Failed loading resource: %s. Make sure resources have been imported by opening the project in the editor at least once.", p_path));

#ifdef TOOLS_ENABLED
	Ref<FileAccess> file_check = FileAccess::create(FileAccess::ACCESS_RESOURCES);
	ERR_FAIL_COND_V_MSG(!file_check->file_exists(p_path), Ref<Resource>(), vformat("Resource file not found: %s (expected type: %s)", p_path, p_type_hint));
#endif

	ERR_FAIL_V_MSG(Ref<Resource>(), vformat("No loader found for resource: %s (expected type: %s)", p_path, p_type_hint));
}

void ResourceLoader::_thread_load_function(void *p_userdata) {
	ThreadLoadTask &load_task = *(ThreadLoadTask *)p_userdata;

	thread_load_mutex.lock();
	caller_task_id = load_task.task_id;
	if (cleaning_tasks) {
		load_task.status = THREAD_LOAD_FAILED;
		thread_load_mutex.unlock();
		return;
	}
	thread_load_mutex.unlock();

	// Thread-safe either if it's the current thread or a brand new one.
	CallQueue *mq_override = nullptr;
	if (load_nesting == 0) {
		load_paths_stack = memnew(Vector<String>);

		if (!load_task.dependent_path.is_empty()) {
			load_paths_stack->push_back(load_task.dependent_path);
		}
		if (!Thread::is_main_thread()) {
			mq_override = memnew(CallQueue);
			MessageQueue::set_thread_singleton_override(mq_override);
			set_current_thread_safe_for_nodes(true);
		}
	} else {
		DEV_ASSERT(load_task.dependent_path.is_empty());
	}
	// --

	if (!Thread::is_main_thread()) {
		set_current_thread_safe_for_nodes(true);
	}

	Ref<Resource> res = _load(load_task.remapped_path, load_task.remapped_path != load_task.local_path ? load_task.local_path : String(), load_task.type_hint, load_task.cache_mode, &load_task.error, load_task.use_sub_threads, &load_task.progress);
	if (mq_override) {
		mq_override->flush();
	}

	thread_load_mutex.lock();

	load_task.resource = res;

	load_task.progress = 1.0; //it was fully loaded at this point, so force progress to 1.0
	if (load_task.error != OK) {
		load_task.status = THREAD_LOAD_FAILED;
	} else {
		load_task.status = THREAD_LOAD_LOADED;
	}

	if (load_task.cond_var) {
		load_task.cond_var->notify_all();
		memdelete(load_task.cond_var);
		load_task.cond_var = nullptr;
	}

	bool ignoring = load_task.cache_mode == ResourceFormatLoader::CACHE_MODE_IGNORE || load_task.cache_mode == ResourceFormatLoader::CACHE_MODE_IGNORE_DEEP;
	bool replacing = load_task.cache_mode == ResourceFormatLoader::CACHE_MODE_REPLACE || load_task.cache_mode == ResourceFormatLoader::CACHE_MODE_REPLACE_DEEP;
	if (load_task.resource.is_valid()) {
		if (!ignoring) {
			if (replacing) {
				Ref<Resource> old_res = ResourceCache::get_ref(load_task.local_path);
				if (old_res.is_valid() && old_res != load_task.resource) {
					// If resource is already loaded, only replace its data, to avoid existing invalidating instances.
					old_res->copy_from(load_task.resource);
					load_task.resource = old_res;
				}
			}
			load_task.resource->set_path(load_task.local_path, replacing);
		} else {
			load_task.resource->set_path_cache(load_task.local_path);
		}

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
	} else if (!ignoring) {
		Ref<Resource> existing = ResourceCache::get_ref(load_task.local_path);
		if (existing.is_valid()) {
			load_task.resource = existing;
			load_task.status = THREAD_LOAD_LOADED;
			load_task.progress = 1.0;

			if (_loaded_callback) {
				_loaded_callback(load_task.resource, load_task.local_path);
			}
		}
	}

	thread_load_mutex.unlock();

	if (load_nesting == 0) {
		if (mq_override) {
			memdelete(mq_override);
		}
		memdelete(load_paths_stack);
	}
}

static String _validate_local_path(const String &p_path) {
	ResourceUID::ID uid = ResourceUID::get_singleton()->text_to_id(p_path);
	if (uid != ResourceUID::INVALID_ID) {
		return ResourceUID::get_singleton()->get_id_path(uid);
	} else if (p_path.is_relative_path()) {
		return ("res://" + p_path).simplify_path();
	} else {
		return ProjectSettings::get_singleton()->localize_path(p_path);
	}
}

Error ResourceLoader::load_threaded_request(const String &p_path, const String &p_type_hint, bool p_use_sub_threads, ResourceFormatLoader::CacheMode p_cache_mode) {
	thread_load_mutex.lock();
	if (user_load_tokens.has(p_path)) {
		print_verbose("load_threaded_request(): Another threaded load for resource path '" + p_path + "' has been initiated. Not an error.");
		user_load_tokens[p_path]->reference(); // Additional request.
		thread_load_mutex.unlock();
		return OK;
	}
	user_load_tokens[p_path] = nullptr;
	thread_load_mutex.unlock();

	Ref<ResourceLoader::LoadToken> token = _load_start(p_path, p_type_hint, p_use_sub_threads ? LOAD_THREAD_DISTRIBUTE : LOAD_THREAD_SPAWN_SINGLE, p_cache_mode);
	if (token.is_valid()) {
		thread_load_mutex.lock();
		token->user_path = p_path;
		token->reference(); // First request.
		user_load_tokens[p_path] = token.ptr();
		print_lt("REQUEST: user load tokens: " + itos(user_load_tokens.size()));
		thread_load_mutex.unlock();
		return OK;
	} else {
		return FAILED;
	}
}

Ref<Resource> ResourceLoader::load(const String &p_path, const String &p_type_hint, ResourceFormatLoader::CacheMode p_cache_mode, Error *r_error) {
	if (r_error) {
		*r_error = OK;
	}

	Ref<LoadToken> load_token = _load_start(p_path, p_type_hint, LOAD_THREAD_FROM_CURRENT, p_cache_mode);
	if (!load_token.is_valid()) {
		if (r_error) {
			*r_error = FAILED;
		}
		return Ref<Resource>();
	}

	Ref<Resource> res = _load_complete(*load_token.ptr(), r_error);
	return res;
}

Ref<ResourceLoader::LoadToken> ResourceLoader::_load_start(const String &p_path, const String &p_type_hint, LoadThreadMode p_thread_mode, ResourceFormatLoader::CacheMode p_cache_mode) {
	String local_path = _validate_local_path(p_path);

	Ref<LoadToken> load_token;
	bool must_not_register = false;
	ThreadLoadTask unregistered_load_task; // Once set, must be valid up to the call to do the load.
	ThreadLoadTask *load_task_ptr = nullptr;
	bool run_on_current_thread = false;
	{
		MutexLock thread_load_lock(thread_load_mutex);

		if (thread_load_tasks.has(local_path)) {
			load_token = Ref<LoadToken>(thread_load_tasks[local_path].load_token);
			if (!load_token.is_valid()) {
				// The token is dying (reached 0 on another thread).
				// Ensure it's killed now so the path can be safely reused right away.
				thread_load_tasks[local_path].load_token->clear();
			} else {
				if (p_cache_mode != ResourceFormatLoader::CACHE_MODE_IGNORE) {
					return load_token;
				}
			}
		}

		load_token.instantiate();
		load_token->local_path = local_path;

		//create load task
		{
			ThreadLoadTask load_task;

			load_task.remapped_path = _path_remap(local_path, &load_task.xl_remapped);
			load_task.load_token = load_token.ptr();
			load_task.local_path = local_path;
			load_task.type_hint = p_type_hint;
			load_task.cache_mode = p_cache_mode;
			load_task.use_sub_threads = p_thread_mode == LOAD_THREAD_DISTRIBUTE;
			if (p_cache_mode == ResourceFormatLoader::CACHE_MODE_REUSE) {
				Ref<Resource> existing = ResourceCache::get_ref(local_path);
				if (existing.is_valid()) {
					//referencing is fine
					load_task.resource = existing;
					load_task.status = THREAD_LOAD_LOADED;
					load_task.progress = 1.0;
					thread_load_tasks[local_path] = load_task;
					return load_token;
				}
			}

			// If we want to ignore cache, but there's another task loading it, we can't add this one to the map and we also have to finish unconditionally synchronously.
			must_not_register = thread_load_tasks.has(local_path) && p_cache_mode == ResourceFormatLoader::CACHE_MODE_IGNORE;
			if (must_not_register) {
				load_token->local_path.clear();
				unregistered_load_task = load_task;
			} else {
				thread_load_tasks[local_path] = load_task;
			}

			load_task_ptr = must_not_register ? &unregistered_load_task : &thread_load_tasks[local_path];
		}

		run_on_current_thread = must_not_register || p_thread_mode == LOAD_THREAD_FROM_CURRENT;

		if (run_on_current_thread) {
			load_task_ptr->thread_id = Thread::get_caller_id();
		} else {
			load_task_ptr->task_id = WorkerThreadPool::get_singleton()->add_native_task(&ResourceLoader::_thread_load_function, load_task_ptr);
		}
	}

	if (run_on_current_thread) {
		_thread_load_function(load_task_ptr);
		if (must_not_register) {
			load_token->res_if_unregistered = load_task_ptr->resource;
		}
	}

	return load_token;
}

float ResourceLoader::_dependency_get_progress(const String &p_path) {
	if (thread_load_tasks.has(p_path)) {
		ThreadLoadTask &load_task = thread_load_tasks[p_path];
		float current_progress = 0.0;
		int dep_count = load_task.sub_tasks.size();
		if (dep_count > 0) {
			for (const String &E : load_task.sub_tasks) {
				current_progress += _dependency_get_progress(E);
			}
			current_progress /= float(dep_count);
			current_progress *= 0.5;
			current_progress += load_task.progress * 0.5;
		} else {
			current_progress = load_task.progress;
		}
		load_task.max_reported_progress = MAX(load_task.max_reported_progress, current_progress);
		return load_task.max_reported_progress;
	} else {
		return 1.0; //assume finished loading it so it no longer exists
	}
}

ResourceLoader::ThreadLoadStatus ResourceLoader::load_threaded_get_status(const String &p_path, float *r_progress) {
	MutexLock thread_load_lock(thread_load_mutex);

	if (!user_load_tokens.has(p_path)) {
		print_verbose("load_threaded_get_status(): No threaded load for resource path '" + p_path + "' has been initiated or its result has already been collected.");
		return THREAD_LOAD_INVALID_RESOURCE;
	}

	String local_path = _validate_local_path(p_path);
	if (!thread_load_tasks.has(local_path)) {
#ifdef DEV_ENABLED
		CRASH_NOW();
#endif
		// On non-dev, be defensive and at least avoid crashing (at this point at least).
		return THREAD_LOAD_INVALID_RESOURCE;
	}

	ThreadLoadTask &load_task = thread_load_tasks[local_path];
	ThreadLoadStatus status;
	status = load_task.status;
	if (r_progress) {
		*r_progress = _dependency_get_progress(local_path);
	}

	return status;
}

Ref<Resource> ResourceLoader::load_threaded_get(const String &p_path, Error *r_error) {
	if (r_error) {
		*r_error = OK;
	}

	Ref<Resource> res;
	{
		MutexLock thread_load_lock(thread_load_mutex);

		if (!user_load_tokens.has(p_path)) {
			print_verbose("load_threaded_get(): No threaded load for resource path '" + p_path + "' has been initiated or its result has already been collected.");
			if (r_error) {
				*r_error = ERR_INVALID_PARAMETER;
			}
			return Ref<Resource>();
		}

		LoadToken *load_token = user_load_tokens[p_path];
		if (!load_token) {
			// This happens if requested from one thread and rapidly querying from another.
			if (r_error) {
				*r_error = ERR_BUSY;
			}
			return Ref<Resource>();
		}
		res = _load_complete_inner(*load_token, r_error, thread_load_lock);
		if (load_token->unreference()) {
			memdelete(load_token);
		}
	}

	print_lt("GET: user load tokens: " + itos(user_load_tokens.size()));

	return res;
}

Ref<Resource> ResourceLoader::_load_complete(LoadToken &p_load_token, Error *r_error) {
	MutexLock thread_load_lock(thread_load_mutex);
	return _load_complete_inner(p_load_token, r_error, thread_load_lock);
}

Ref<Resource> ResourceLoader::_load_complete_inner(LoadToken &p_load_token, Error *r_error, MutexLock<SafeBinaryMutex<BINARY_MUTEX_TAG>> &p_thread_load_lock) {
	if (r_error) {
		*r_error = OK;
	}

	if (!p_load_token.local_path.is_empty()) {
		if (!thread_load_tasks.has(p_load_token.local_path)) {
#ifdef DEV_ENABLED
			CRASH_NOW();
#endif
			// On non-dev, be defensive and at least avoid crashing (at this point at least).
			if (r_error) {
				*r_error = ERR_BUG;
			}
			return Ref<Resource>();
		}

		ThreadLoadTask &load_task = thread_load_tasks[p_load_token.local_path];

		if (load_task.status == THREAD_LOAD_IN_PROGRESS) {
			DEV_ASSERT((load_task.task_id == 0) != (load_task.thread_id == 0));

			if ((load_task.task_id != 0 && load_task.task_id == caller_task_id) ||
					(load_task.thread_id != 0 && load_task.thread_id == Thread::get_caller_id())) {
				// Load is in progress, but it's precisely this thread the one in charge.
				// That means this is a cyclic load.
				if (r_error) {
					*r_error = ERR_BUSY;
				}
				return Ref<Resource>();
			}

			if (load_task.task_id != 0) {
				// Loading thread is in the worker pool.
				thread_load_mutex.unlock();
				Error err = WorkerThreadPool::get_singleton()->wait_for_task_completion(load_task.task_id);
				if (err == ERR_BUSY) {
					// The WorkerThreadPool has reported that the current task wants to await on an older one.
					// That't not allowed for safety, to avoid deadlocks. Fortunately, though, in the context of
					// resource loading that means that the task to wait for can be restarted here to break the
					// cycle, with as much recursion into this process as needed.
					// When the stack is eventually unrolled, the original load will have been notified to go on.
					// CACHE_MODE_IGNORE is needed because, otherwise, the new request would just see there's
					// an ongoing load for that resource and wait for it again. This value forces a new load.
					Ref<ResourceLoader::LoadToken> token = _load_start(load_task.local_path, load_task.type_hint, LOAD_THREAD_DISTRIBUTE, ResourceFormatLoader::CACHE_MODE_IGNORE);
					Ref<Resource> resource = _load_complete(*token.ptr(), &err);
					if (r_error) {
						*r_error = err;
					}
					thread_load_mutex.lock();
					return resource;
				} else {
					DEV_ASSERT(err == OK);
					thread_load_mutex.lock();
					load_task.awaited = true;
				}
			} else {
				// Loading thread is main or user thread.
				if (!load_task.cond_var) {
					load_task.cond_var = memnew(ConditionVariable);
				}
				do {
					load_task.cond_var->wait(p_thread_load_lock);
					DEV_ASSERT(thread_load_tasks.has(p_load_token.local_path) && p_load_token.get_reference_count());
				} while (load_task.cond_var);
			}
		}

		if (cleaning_tasks) {
			load_task.resource = Ref<Resource>();
			load_task.error = FAILED;
		}

		Ref<Resource> resource = load_task.resource;
		if (r_error) {
			*r_error = load_task.error;
		}
		return resource;
	} else {
		// Special case of an unregistered task.
		// The resource should have been loaded by now.
		Ref<Resource> resource = p_load_token.res_if_unregistered;
		if (!resource.is_valid()) {
			if (r_error) {
				*r_error = FAILED;
			}
		}
		return resource;
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

		loader[i]->get_dependencies(local_path, p_dependencies, p_add_types);
	}
}

Error ResourceLoader::rename_dependencies(const String &p_path, const HashMap<String, String> &p_map) {
	String local_path = _path_remap(_validate_local_path(p_path));

	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(local_path)) {
			continue;
		}

		return loader[i]->rename_dependencies(local_path, p_map);
	}

	return OK; // ??
}

void ResourceLoader::get_classes_used(const String &p_path, HashSet<StringName> *r_classes) {
	String local_path = _validate_local_path(p_path);

	for (int i = 0; i < loader_count; i++) {
		if (!loader[i]->recognize_path(local_path)) {
			continue;
		}

		return loader[i]->get_classes_used(p_path, r_classes);
	}
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

String ResourceLoader::get_resource_script_class(const String &p_path) {
	String local_path = _validate_local_path(p_path);

	for (int i = 0; i < loader_count; i++) {
		String result = loader[i]->get_resource_script_class(local_path);
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

		// An extra remap may still be necessary afterwards due to the text -> binary converter on export.

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
			if (score > 0 && score >= best_score) {
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

		// Fallback to p_path if new_path does not exist.
		if (!FileAccess::exists(new_path + ".import") && !FileAccess::exists(new_path)) {
			WARN_PRINT(vformat("Translation remap '%s' does not exist. Falling back to '%s'.", new_path, p_path));
			new_path = p_path;
		}
	}

	if (path_remaps.has(new_path)) {
		new_path = path_remaps[new_path];
	} else {
		// Try file remap.
		Error err;
		Ref<FileAccess> f = FileAccess::open(new_path + ".remap", FileAccess::READ, &err);
		if (f.is_valid()) {
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
	ResourceCache::lock.lock();

	List<Resource *> to_reload;
	SelfList<Resource> *E = remapped_list.first();

	while (E) {
		to_reload.push_back(E->self());
		E = E->next();
	}

	ResourceCache::lock.unlock();

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

	Dictionary remaps = GLOBAL_GET("internationalization/locale/translation_remaps");
	List<Variant> keys;
	remaps.get_key_list(&keys);
	for (const Variant &E : keys) {
		Array langs = remaps[E];
		Vector<String> lang_remaps;
		lang_remaps.resize(langs.size());
		String *lang_remaps_ptrw = lang_remaps.ptrw();
		for (const Variant &lang : langs) {
			*lang_remaps_ptrw++ = lang;
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

void ResourceLoader::clear_thread_load_tasks() {
	// Bring the thing down as quickly as possible without causing deadlocks or leaks.

	thread_load_mutex.lock();
	cleaning_tasks = true;

	while (true) {
		bool none_running = true;
		if (thread_load_tasks.size()) {
			for (KeyValue<String, ResourceLoader::ThreadLoadTask> &E : thread_load_tasks) {
				if (E.value.status == THREAD_LOAD_IN_PROGRESS) {
					if (E.value.cond_var) {
						E.value.cond_var->notify_all();
						memdelete(E.value.cond_var);
						E.value.cond_var = nullptr;
					}
					none_running = false;
				}
			}
		}
		if (none_running) {
			break;
		}
		thread_load_mutex.unlock();
		OS::get_singleton()->delay_usec(1000);
		thread_load_mutex.lock();
	}

	while (user_load_tokens.begin()) {
		// User load tokens remove themselves from the map on destruction.
		memdelete(user_load_tokens.begin()->value);
	}
	user_load_tokens.clear();

	thread_load_tasks.clear();

	cleaning_tasks = false;
	thread_load_mutex.unlock();
}

void ResourceLoader::load_path_remaps() {
	if (!ProjectSettings::get_singleton()->has_setting("path_remap/remapped_paths")) {
		return;
	}

	Vector<String> remaps = GLOBAL_GET("path_remap/remapped_paths");
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

Ref<ResourceFormatLoader> ResourceLoader::_find_custom_resource_format_loader(const String &path) {
	for (int i = 0; i < loader_count; ++i) {
		if (loader[i]->get_script_instance() && loader[i]->get_script_instance()->get_script()->get_path() == path) {
			return loader[i];
		}
	}
	return Ref<ResourceFormatLoader>();
}

bool ResourceLoader::add_custom_resource_format_loader(const String &script_path) {
	if (_find_custom_resource_format_loader(script_path).is_valid()) {
		return false;
	}

	Ref<Resource> res = ResourceLoader::load(script_path);
	ERR_FAIL_COND_V(res.is_null(), false);
	ERR_FAIL_COND_V(!res->is_class("Script"), false);

	Ref<Script> s = res;
	StringName ibt = s->get_instance_base_type();
	bool valid_type = ClassDB::is_parent_class(ibt, "ResourceFormatLoader");
	ERR_FAIL_COND_V_MSG(!valid_type, false, vformat("Failed to add a custom resource loader, script '%s' does not inherit 'ResourceFormatLoader'.", script_path));

	Object *obj = ClassDB::instantiate(ibt);
	ERR_FAIL_NULL_V_MSG(obj, false, vformat("Failed to add a custom resource loader, cannot instantiate '%s'.", ibt));

	Ref<ResourceFormatLoader> crl = Object::cast_to<ResourceFormatLoader>(obj);
	crl->set_script(s);
	ResourceLoader::add_resource_format_loader(crl);

	return true;
}

void ResourceLoader::set_create_missing_resources_if_class_unavailable(bool p_enable) {
	create_missing_resources_if_class_unavailable = p_enable;
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

bool ResourceLoader::is_cleaning_tasks() {
	MutexLock lock(thread_load_mutex);
	return cleaning_tasks;
}

void ResourceLoader::initialize() {}

void ResourceLoader::finalize() {}

ResourceLoadErrorNotify ResourceLoader::err_notify = nullptr;
DependencyErrorNotify ResourceLoader::dep_err_notify = nullptr;

bool ResourceLoader::create_missing_resources_if_class_unavailable = false;
bool ResourceLoader::abort_on_missing_resource = true;
bool ResourceLoader::timestamp_on_load = false;

thread_local int ResourceLoader::load_nesting = 0;
thread_local WorkerThreadPool::TaskID ResourceLoader::caller_task_id = 0;
thread_local Vector<String> *ResourceLoader::load_paths_stack;

template <>
thread_local uint32_t SafeBinaryMutex<ResourceLoader::BINARY_MUTEX_TAG>::count = 0;
SafeBinaryMutex<ResourceLoader::BINARY_MUTEX_TAG> ResourceLoader::thread_load_mutex;
HashMap<String, ResourceLoader::ThreadLoadTask> ResourceLoader::thread_load_tasks;
bool ResourceLoader::cleaning_tasks = false;

HashMap<String, ResourceLoader::LoadToken *> ResourceLoader::user_load_tokens;

SelfList<Resource>::List ResourceLoader::remapped_list;
HashMap<String, Vector<String>> ResourceLoader::translation_remaps;
HashMap<String, String> ResourceLoader::path_remaps;

ResourceLoaderImport ResourceLoader::import = nullptr;
