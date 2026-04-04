/**************************************************************************/
/*  bone_map.cpp                                                          */
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

#include <godot_cpp/classes/bone_map.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/skeleton_profile.hpp>

namespace godot {

Ref<SkeletonProfile> BoneMap::get_profile() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneMap::get_class_static()._native_ptr(), StringName("get_profile")._native_ptr(), 4291782652);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SkeletonProfile>()));
	return Ref<SkeletonProfile>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SkeletonProfile>(_gde_method_bind, _owner));
}

void BoneMap::set_profile(const Ref<SkeletonProfile> &p_profile) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneMap::get_class_static()._native_ptr(), StringName("set_profile")._native_ptr(), 3870374136);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_profile != nullptr ? &p_profile->_owner : nullptr));
}

StringName BoneMap::get_skeleton_bone_name(const StringName &p_profile_bone_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneMap::get_class_static()._native_ptr(), StringName("get_skeleton_bone_name")._native_ptr(), 1965194235);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_profile_bone_name);
}

void BoneMap::set_skeleton_bone_name(const StringName &p_profile_bone_name, const StringName &p_skeleton_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneMap::get_class_static()._native_ptr(), StringName("set_skeleton_bone_name")._native_ptr(), 3740211285);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_profile_bone_name, &p_skeleton_bone_name);
}

StringName BoneMap::find_profile_bone_name(const StringName &p_skeleton_bone_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneMap::get_class_static()._native_ptr(), StringName("find_profile_bone_name")._native_ptr(), 1965194235);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_skeleton_bone_name);
}

} // namespace godot
