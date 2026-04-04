/**************************************************************************/
/*  open_xr_interaction_profile.cpp                                       */
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

#include <godot_cpp/classes/open_xr_interaction_profile.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/open_xrip_binding.hpp>
#include <godot_cpp/classes/open_xrip_binding_modifier.hpp>

namespace godot {

void OpenXRInteractionProfile::set_interaction_profile_path(const String &p_interaction_profile_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("set_interaction_profile_path")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_interaction_profile_path);
}

String OpenXRInteractionProfile::get_interaction_profile_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("get_interaction_profile_path")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

int32_t OpenXRInteractionProfile::get_binding_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("get_binding_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<OpenXRIPBinding> OpenXRInteractionProfile::get_binding(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("get_binding")._native_ptr(), 3934429652);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRIPBinding>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<OpenXRIPBinding>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRIPBinding>(_gde_method_bind, _owner, &p_index_encoded));
}

void OpenXRInteractionProfile::set_bindings(const Array &p_bindings) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("set_bindings")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bindings);
}

Array OpenXRInteractionProfile::get_bindings() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("get_bindings")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

int32_t OpenXRInteractionProfile::get_binding_modifier_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("get_binding_modifier_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<OpenXRIPBindingModifier> OpenXRInteractionProfile::get_binding_modifier(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("get_binding_modifier")._native_ptr(), 2419896583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRIPBindingModifier>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<OpenXRIPBindingModifier>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRIPBindingModifier>(_gde_method_bind, _owner, &p_index_encoded));
}

void OpenXRInteractionProfile::set_binding_modifiers(const Array &p_binding_modifiers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("set_binding_modifiers")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_binding_modifiers);
}

Array OpenXRInteractionProfile::get_binding_modifiers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRInteractionProfile::get_class_static()._native_ptr(), StringName("get_binding_modifiers")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

} // namespace godot
