/**************************************************************************/
/*  resource_loader.hpp                                                   */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Resource;
class ResourceFormatLoader;

class ResourceLoader : public Object {
	GDEXTENSION_CLASS(ResourceLoader, Object)

	static ResourceLoader *singleton;

public:
	enum ThreadLoadStatus {
		THREAD_LOAD_INVALID_RESOURCE = 0,
		THREAD_LOAD_IN_PROGRESS = 1,
		THREAD_LOAD_FAILED = 2,
		THREAD_LOAD_LOADED = 3,
	};

	enum CacheMode {
		CACHE_MODE_IGNORE = 0,
		CACHE_MODE_REUSE = 1,
		CACHE_MODE_REPLACE = 2,
		CACHE_MODE_IGNORE_DEEP = 3,
		CACHE_MODE_REPLACE_DEEP = 4,
	};

	static ResourceLoader *get_singleton();

	Error load_threaded_request(const String &p_path, const String &p_type_hint = String(), bool p_use_sub_threads = false, ResourceLoader::CacheMode p_cache_mode = (ResourceLoader::CacheMode)1);
	ResourceLoader::ThreadLoadStatus load_threaded_get_status(const String &p_path, const Array &p_progress = Array());
	Ref<Resource> load_threaded_get(const String &p_path);
	Ref<Resource> load(const String &p_path, const String &p_type_hint = String(), ResourceLoader::CacheMode p_cache_mode = (ResourceLoader::CacheMode)1);
	PackedStringArray get_recognized_extensions_for_type(const String &p_type);
	void add_resource_format_loader(const Ref<ResourceFormatLoader> &p_format_loader, bool p_at_front = false);
	void remove_resource_format_loader(const Ref<ResourceFormatLoader> &p_format_loader);
	void set_abort_on_missing_resources(bool p_abort);
	PackedStringArray get_dependencies(const String &p_path);
	bool has_cached(const String &p_path);
	Ref<Resource> get_cached_ref(const String &p_path);
	bool exists(const String &p_path, const String &p_type_hint = String());
	int64_t get_resource_uid(const String &p_path);
	PackedStringArray list_directory(const String &p_directory_path);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~ResourceLoader();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ResourceLoader::ThreadLoadStatus);
VARIANT_ENUM_CAST(ResourceLoader::CacheMode);

