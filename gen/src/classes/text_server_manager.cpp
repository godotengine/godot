/**************************************************************************/
/*  text_server_manager.cpp                                               */
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

#include <godot_cpp/classes/text_server_manager.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

TextServerManager *TextServerManager::singleton = nullptr;

TextServerManager *TextServerManager::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(TextServerManager::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<TextServerManager *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &TextServerManager::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(TextServerManager::get_class_static(), singleton);
		}
	}
	return singleton;
}

TextServerManager::~TextServerManager() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(TextServerManager::get_class_static());
		singleton = nullptr;
	}
}

void TextServerManager::add_interface(const Ref<TextServer> &p_interface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServerManager::get_class_static()._native_ptr(), StringName("add_interface")._native_ptr(), 1799689403);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_interface != nullptr ? &p_interface->_owner : nullptr));
}

int32_t TextServerManager::get_interface_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServerManager::get_class_static()._native_ptr(), StringName("get_interface_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextServerManager::remove_interface(const Ref<TextServer> &p_interface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServerManager::get_class_static()._native_ptr(), StringName("remove_interface")._native_ptr(), 1799689403);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_interface != nullptr ? &p_interface->_owner : nullptr));
}

Ref<TextServer> TextServerManager::get_interface(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServerManager::get_class_static()._native_ptr(), StringName("get_interface")._native_ptr(), 1672475555);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TextServer>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<TextServer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TextServer>(_gde_method_bind, _owner, &p_idx_encoded));
}

TypedArray<Dictionary> TextServerManager::get_interfaces() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServerManager::get_class_static()._native_ptr(), StringName("get_interfaces")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner);
}

Ref<TextServer> TextServerManager::find_interface(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServerManager::get_class_static()._native_ptr(), StringName("find_interface")._native_ptr(), 2240905781);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TextServer>()));
	return Ref<TextServer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TextServer>(_gde_method_bind, _owner, &p_name));
}

void TextServerManager::set_primary_interface(const Ref<TextServer> &p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServerManager::get_class_static()._native_ptr(), StringName("set_primary_interface")._native_ptr(), 1799689403);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_index != nullptr ? &p_index->_owner : nullptr));
}

Ref<TextServer> TextServerManager::get_primary_interface() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextServerManager::get_class_static()._native_ptr(), StringName("get_primary_interface")._native_ptr(), 905850878);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TextServer>()));
	return Ref<TextServer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TextServer>(_gde_method_bind, _owner));
}

} // namespace godot
