/**************************************************************************/
/*  java_script_bridge.cpp                                                */
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

#include <godot_cpp/classes/java_script_bridge.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/java_script_object.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

JavaScriptBridge *JavaScriptBridge::singleton = nullptr;

JavaScriptBridge *JavaScriptBridge::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(JavaScriptBridge::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<JavaScriptBridge *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &JavaScriptBridge::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(JavaScriptBridge::get_class_static(), singleton);
		}
	}
	return singleton;
}

JavaScriptBridge::~JavaScriptBridge() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(JavaScriptBridge::get_class_static());
		singleton = nullptr;
	}
}

Variant JavaScriptBridge::eval(const String &p_code, bool p_use_global_execution_context) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("eval")._native_ptr(), 218087648);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int8_t p_use_global_execution_context_encoded;
	PtrToArg<bool>::encode(p_use_global_execution_context, &p_use_global_execution_context_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_code, &p_use_global_execution_context_encoded);
}

Ref<JavaScriptObject> JavaScriptBridge::get_interface(const String &p_interface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("get_interface")._native_ptr(), 1355533281);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<JavaScriptObject>()));
	return Ref<JavaScriptObject>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<JavaScriptObject>(_gde_method_bind, _owner, &p_interface));
}

Ref<JavaScriptObject> JavaScriptBridge::create_callback(const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("create_callback")._native_ptr(), 422818440);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<JavaScriptObject>()));
	return Ref<JavaScriptObject>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<JavaScriptObject>(_gde_method_bind, _owner, &p_callable));
}

bool JavaScriptBridge::is_js_buffer(const Ref<JavaScriptObject> &p_javascript_object) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("is_js_buffer")._native_ptr(), 821968997);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_javascript_object != nullptr ? &p_javascript_object->_owner : nullptr));
}

PackedByteArray JavaScriptBridge::js_buffer_to_packed_byte_array(const Ref<JavaScriptObject> &p_javascript_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("js_buffer_to_packed_byte_array")._native_ptr(), 64409880);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, (p_javascript_buffer != nullptr ? &p_javascript_buffer->_owner : nullptr));
}

Variant JavaScriptBridge::create_object_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("create_object")._native_ptr(), 3093893586);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	GDExtensionCallError error;
	Variant ret;
	::godot::gdextension_interface::object_method_bind_call(_gde_method_bind, _owner, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, &ret, &error);
	return ret;
}

void JavaScriptBridge::download_buffer(const PackedByteArray &p_buffer, const String &p_name, const String &p_mime) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("download_buffer")._native_ptr(), 3352272093);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_buffer, &p_name, &p_mime);
}

bool JavaScriptBridge::pwa_needs_update() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("pwa_needs_update")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Error JavaScriptBridge::pwa_update() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("pwa_update")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void JavaScriptBridge::force_fs_sync() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JavaScriptBridge::get_class_static()._native_ptr(), StringName("force_fs_sync")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
