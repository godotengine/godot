/**************************************************************************/
/*  gd_extension_manager.cpp                                              */
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

#include <godot_cpp/classes/gd_extension_manager.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/gd_extension.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

GDExtensionManager *GDExtensionManager::singleton = nullptr;

GDExtensionManager *GDExtensionManager::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(GDExtensionManager::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<GDExtensionManager *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &GDExtensionManager::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(GDExtensionManager::get_class_static(), singleton);
		}
	}
	return singleton;
}

GDExtensionManager::~GDExtensionManager() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(GDExtensionManager::get_class_static());
		singleton = nullptr;
	}
}

GDExtensionManager::LoadStatus GDExtensionManager::load_extension(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GDExtensionManager::get_class_static()._native_ptr(), StringName("load_extension")._native_ptr(), 4024158731);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GDExtensionManager::LoadStatus(0)));
	return (GDExtensionManager::LoadStatus)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

GDExtensionManager::LoadStatus GDExtensionManager::load_extension_from_function(const String &p_path, const GDExtensionInitializationFunction *p_init_func) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GDExtensionManager::get_class_static()._native_ptr(), StringName("load_extension_from_function")._native_ptr(), 1565094761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GDExtensionManager::LoadStatus(0)));
	return (GDExtensionManager::LoadStatus)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_init_func);
}

GDExtensionManager::LoadStatus GDExtensionManager::reload_extension(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GDExtensionManager::get_class_static()._native_ptr(), StringName("reload_extension")._native_ptr(), 4024158731);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GDExtensionManager::LoadStatus(0)));
	return (GDExtensionManager::LoadStatus)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

GDExtensionManager::LoadStatus GDExtensionManager::unload_extension(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GDExtensionManager::get_class_static()._native_ptr(), StringName("unload_extension")._native_ptr(), 4024158731);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GDExtensionManager::LoadStatus(0)));
	return (GDExtensionManager::LoadStatus)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

bool GDExtensionManager::is_extension_loaded(const String &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GDExtensionManager::get_class_static()._native_ptr(), StringName("is_extension_loaded")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

PackedStringArray GDExtensionManager::get_loaded_extensions() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GDExtensionManager::get_class_static()._native_ptr(), StringName("get_loaded_extensions")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

Ref<GDExtension> GDExtensionManager::get_extension(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GDExtensionManager::get_class_static()._native_ptr(), StringName("get_extension")._native_ptr(), 49743343);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GDExtension>()));
	return Ref<GDExtension>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GDExtension>(_gde_method_bind, _owner, &p_path));
}

} // namespace godot
