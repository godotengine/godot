/**************************************************************************/
/*  performance.cpp                                                       */
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

#include <godot_cpp/classes/performance.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/callable.hpp>

namespace godot {

Performance *Performance::singleton = nullptr;

Performance *Performance::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(Performance::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<Performance *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &Performance::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(Performance::get_class_static(), singleton);
		}
	}
	return singleton;
}

Performance::~Performance() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(Performance::get_class_static());
		singleton = nullptr;
	}
}

double Performance::get_monitor(Performance::Monitor p_monitor) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Performance::get_class_static()._native_ptr(), StringName("get_monitor")._native_ptr(), 1943275655);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_monitor_encoded;
	PtrToArg<int64_t>::encode(p_monitor, &p_monitor_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_monitor_encoded);
}

void Performance::add_custom_monitor(const StringName &p_id, const Callable &p_callable, const Array &p_arguments, Performance::MonitorType p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Performance::get_class_static()._native_ptr(), StringName("add_custom_monitor")._native_ptr(), 3655788610);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_callable, &p_arguments, &p_type_encoded);
}

void Performance::remove_custom_monitor(const StringName &p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Performance::get_class_static()._native_ptr(), StringName("remove_custom_monitor")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id);
}

bool Performance::has_custom_monitor(const StringName &p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Performance::get_class_static()._native_ptr(), StringName("has_custom_monitor")._native_ptr(), 2041966384);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_id);
}

Variant Performance::get_custom_monitor(const StringName &p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Performance::get_class_static()._native_ptr(), StringName("get_custom_monitor")._native_ptr(), 2138907829);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_id);
}

uint64_t Performance::get_monitor_modification_time() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Performance::get_class_static()._native_ptr(), StringName("get_monitor_modification_time")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

TypedArray<StringName> Performance::get_custom_monitor_names() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Performance::get_class_static()._native_ptr(), StringName("get_custom_monitor_names")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<StringName>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<StringName>>(_gde_method_bind, _owner);
}

PackedInt32Array Performance::get_custom_monitor_types() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Performance::get_class_static()._native_ptr(), StringName("get_custom_monitor_types")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

} // namespace godot
