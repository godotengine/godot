/**************************************************************************/
/*  open_xr_action_map.cpp                                                */
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

#include <godot_cpp/classes/open_xr_action_map.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/open_xr_action_set.hpp>
#include <godot_cpp/classes/open_xr_interaction_profile.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

void OpenXRActionMap::set_action_sets(const Array &p_action_sets) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("set_action_sets")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action_sets);
}

Array OpenXRActionMap::get_action_sets() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("get_action_sets")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

int32_t OpenXRActionMap::get_action_set_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("get_action_set_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<OpenXRActionSet> OpenXRActionMap::find_action_set(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("find_action_set")._native_ptr(), 1888809267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRActionSet>()));
	return Ref<OpenXRActionSet>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRActionSet>(_gde_method_bind, _owner, &p_name));
}

Ref<OpenXRActionSet> OpenXRActionMap::get_action_set(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("get_action_set")._native_ptr(), 1789580336);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRActionSet>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<OpenXRActionSet>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRActionSet>(_gde_method_bind, _owner, &p_idx_encoded));
}

void OpenXRActionMap::add_action_set(const Ref<OpenXRActionSet> &p_action_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("add_action_set")._native_ptr(), 2093310581);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_action_set != nullptr ? &p_action_set->_owner : nullptr));
}

void OpenXRActionMap::remove_action_set(const Ref<OpenXRActionSet> &p_action_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("remove_action_set")._native_ptr(), 2093310581);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_action_set != nullptr ? &p_action_set->_owner : nullptr));
}

void OpenXRActionMap::set_interaction_profiles(const Array &p_interaction_profiles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("set_interaction_profiles")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_interaction_profiles);
}

Array OpenXRActionMap::get_interaction_profiles() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("get_interaction_profiles")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

int32_t OpenXRActionMap::get_interaction_profile_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("get_interaction_profile_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<OpenXRInteractionProfile> OpenXRActionMap::find_interaction_profile(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("find_interaction_profile")._native_ptr(), 3095875538);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRInteractionProfile>()));
	return Ref<OpenXRInteractionProfile>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRInteractionProfile>(_gde_method_bind, _owner, &p_name));
}

Ref<OpenXRInteractionProfile> OpenXRActionMap::get_interaction_profile(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("get_interaction_profile")._native_ptr(), 2546151210);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRInteractionProfile>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<OpenXRInteractionProfile>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRInteractionProfile>(_gde_method_bind, _owner, &p_idx_encoded));
}

void OpenXRActionMap::add_interaction_profile(const Ref<OpenXRInteractionProfile> &p_interaction_profile) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("add_interaction_profile")._native_ptr(), 2697953512);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_interaction_profile != nullptr ? &p_interaction_profile->_owner : nullptr));
}

void OpenXRActionMap::remove_interaction_profile(const Ref<OpenXRInteractionProfile> &p_interaction_profile) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("remove_interaction_profile")._native_ptr(), 2697953512);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_interaction_profile != nullptr ? &p_interaction_profile->_owner : nullptr));
}

void OpenXRActionMap::create_default_action_sets() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRActionMap::get_class_static()._native_ptr(), StringName("create_default_action_sets")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
