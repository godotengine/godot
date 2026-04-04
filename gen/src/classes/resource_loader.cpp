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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/resource_loader.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/classes/resource_format_loader.hpp>

namespace godot {

ResourceLoader *ResourceLoader::singleton = nullptr;

ResourceLoader *ResourceLoader::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(ResourceLoader::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<ResourceLoader *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &ResourceLoader::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(ResourceLoader::get_class_static(), singleton);
		}
	}
	return singleton;
}

ResourceLoader::~ResourceLoader() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(ResourceLoader::get_class_static());
		singleton = nullptr;
	}
}

Error ResourceLoader::load_threaded_request(const String &p_path, const String &p_type_hint, bool p_use_sub_threads, ResourceLoader::CacheMode p_cache_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("load_threaded_request")._native_ptr(), 3614384323);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_use_sub_threads_encoded;
	PtrToArg<bool>::encode(p_use_sub_threads, &p_use_sub_threads_encoded);
	int64_t p_cache_mode_encoded;
	PtrToArg<int64_t>::encode(p_cache_mode, &p_cache_mode_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_type_hint, &p_use_sub_threads_encoded, &p_cache_mode_encoded);
}

ResourceLoader::ThreadLoadStatus ResourceLoader::load_threaded_get_status(const String &p_path, const Array &p_progress) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("load_threaded_get_status")._native_ptr(), 4137685479);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ResourceLoader::ThreadLoadStatus(0)));
	return (ResourceLoader::ThreadLoadStatus)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_progress);
}

Ref<Resource> ResourceLoader::load_threaded_get(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("load_threaded_get")._native_ptr(), 1748875256);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Resource>()));
	return Ref<Resource>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Resource>(_gde_method_bind, _owner, &p_path));
}

Ref<Resource> ResourceLoader::load(const String &p_path, const String &p_type_hint, ResourceLoader::CacheMode p_cache_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("load")._native_ptr(), 3358495409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Resource>()));
	int64_t p_cache_mode_encoded;
	PtrToArg<int64_t>::encode(p_cache_mode, &p_cache_mode_encoded);
	return Ref<Resource>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Resource>(_gde_method_bind, _owner, &p_path, &p_type_hint, &p_cache_mode_encoded));
}

PackedStringArray ResourceLoader::get_recognized_extensions_for_type(const String &p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("get_recognized_extensions_for_type")._native_ptr(), 3538744774);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_type);
}

void ResourceLoader::add_resource_format_loader(const Ref<ResourceFormatLoader> &p_format_loader, bool p_at_front) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("add_resource_format_loader")._native_ptr(), 2896595483);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_at_front_encoded;
	PtrToArg<bool>::encode(p_at_front, &p_at_front_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_format_loader != nullptr ? &p_format_loader->_owner : nullptr), &p_at_front_encoded);
}

void ResourceLoader::remove_resource_format_loader(const Ref<ResourceFormatLoader> &p_format_loader) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("remove_resource_format_loader")._native_ptr(), 405397102);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_format_loader != nullptr ? &p_format_loader->_owner : nullptr));
}

void ResourceLoader::set_abort_on_missing_resources(bool p_abort) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("set_abort_on_missing_resources")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_abort_encoded;
	PtrToArg<bool>::encode(p_abort, &p_abort_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_abort_encoded);
}

PackedStringArray ResourceLoader::get_dependencies(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("get_dependencies")._native_ptr(), 3538744774);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_path);
}

bool ResourceLoader::has_cached(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("has_cached")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

Ref<Resource> ResourceLoader::get_cached_ref(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("get_cached_ref")._native_ptr(), 1748875256);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Resource>()));
	return Ref<Resource>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Resource>(_gde_method_bind, _owner, &p_path));
}

bool ResourceLoader::exists(const String &p_path, const String &p_type_hint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("exists")._native_ptr(), 4185558881);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path, &p_type_hint);
}

int64_t ResourceLoader::get_resource_uid(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("get_resource_uid")._native_ptr(), 1597066294);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

PackedStringArray ResourceLoader::list_directory(const String &p_directory_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceLoader::get_class_static()._native_ptr(), StringName("list_directory")._native_ptr(), 3538744774);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_directory_path);
}

} // namespace godot
