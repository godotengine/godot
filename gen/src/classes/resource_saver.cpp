/**************************************************************************/
/*  resource_saver.cpp                                                    */
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

#include <godot_cpp/classes/resource_saver.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/classes/resource_format_saver.hpp>

namespace godot {

ResourceSaver *ResourceSaver::singleton = nullptr;

ResourceSaver *ResourceSaver::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(ResourceSaver::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<ResourceSaver *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &ResourceSaver::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(ResourceSaver::get_class_static(), singleton);
		}
	}
	return singleton;
}

ResourceSaver::~ResourceSaver() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(ResourceSaver::get_class_static());
		singleton = nullptr;
	}
}

Error ResourceSaver::save(const Ref<Resource> &p_resource, const String &p_path, BitField<ResourceSaver::SaverFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceSaver::get_class_static()._native_ptr(), StringName("save")._native_ptr(), 2983274697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_resource != nullptr ? &p_resource->_owner : nullptr), &p_path, &p_flags);
}

Error ResourceSaver::set_uid(const String &p_resource, int64_t p_uid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceSaver::get_class_static()._native_ptr(), StringName("set_uid")._native_ptr(), 993915709);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_uid_encoded;
	PtrToArg<int64_t>::encode(p_uid, &p_uid_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_resource, &p_uid_encoded);
}

PackedStringArray ResourceSaver::get_recognized_extensions(const Ref<Resource> &p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceSaver::get_class_static()._native_ptr(), StringName("get_recognized_extensions")._native_ptr(), 4223597960);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, (p_type != nullptr ? &p_type->_owner : nullptr));
}

void ResourceSaver::add_resource_format_saver(const Ref<ResourceFormatSaver> &p_format_saver, bool p_at_front) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceSaver::get_class_static()._native_ptr(), StringName("add_resource_format_saver")._native_ptr(), 362894272);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_at_front_encoded;
	PtrToArg<bool>::encode(p_at_front, &p_at_front_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_format_saver != nullptr ? &p_format_saver->_owner : nullptr), &p_at_front_encoded);
}

void ResourceSaver::remove_resource_format_saver(const Ref<ResourceFormatSaver> &p_format_saver) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceSaver::get_class_static()._native_ptr(), StringName("remove_resource_format_saver")._native_ptr(), 3373026878);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_format_saver != nullptr ? &p_format_saver->_owner : nullptr));
}

int64_t ResourceSaver::get_resource_id_for_path(const String &p_path, bool p_generate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ResourceSaver::get_class_static()._native_ptr(), StringName("get_resource_id_for_path")._native_ptr(), 150756522);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_generate_encoded;
	PtrToArg<bool>::encode(p_generate, &p_generate_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_generate_encoded);
}

} // namespace godot
