/**************************************************************************/
/*  skeleton2d.cpp                                                        */
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

#include <godot_cpp/classes/skeleton2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/bone2d.hpp>
#include <godot_cpp/classes/skeleton_modification_stack2d.hpp>

namespace godot {

int32_t Skeleton2D::get_bone_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Skeleton2D::get_class_static()._native_ptr(), StringName("get_bone_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Bone2D *Skeleton2D::get_bone(int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Skeleton2D::get_class_static()._native_ptr(), StringName("get_bone")._native_ptr(), 2556267111);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Bone2D>(_gde_method_bind, _owner, &p_idx_encoded);
}

RID Skeleton2D::get_skeleton() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Skeleton2D::get_class_static()._native_ptr(), StringName("get_skeleton")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void Skeleton2D::set_modification_stack(const Ref<SkeletonModificationStack2D> &p_modification_stack) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Skeleton2D::get_class_static()._native_ptr(), StringName("set_modification_stack")._native_ptr(), 3907307132);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_modification_stack != nullptr ? &p_modification_stack->_owner : nullptr));
}

Ref<SkeletonModificationStack2D> Skeleton2D::get_modification_stack() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Skeleton2D::get_class_static()._native_ptr(), StringName("get_modification_stack")._native_ptr(), 2107508396);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SkeletonModificationStack2D>()));
	return Ref<SkeletonModificationStack2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SkeletonModificationStack2D>(_gde_method_bind, _owner));
}

void Skeleton2D::execute_modifications(float p_delta, int32_t p_execution_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Skeleton2D::get_class_static()._native_ptr(), StringName("execute_modifications")._native_ptr(), 1005356550);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	int64_t p_execution_mode_encoded;
	PtrToArg<int64_t>::encode(p_execution_mode, &p_execution_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_delta_encoded, &p_execution_mode_encoded);
}

void Skeleton2D::set_bone_local_pose_override(int32_t p_bone_idx, const Transform2D &p_override_pose, float p_strength, bool p_persistent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Skeleton2D::get_class_static()._native_ptr(), StringName("set_bone_local_pose_override")._native_ptr(), 555457532);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	int8_t p_persistent_encoded;
	PtrToArg<bool>::encode(p_persistent, &p_persistent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_override_pose, &p_strength_encoded, &p_persistent_encoded);
}

Transform2D Skeleton2D::get_bone_local_pose_override(int32_t p_bone_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Skeleton2D::get_class_static()._native_ptr(), StringName("get_bone_local_pose_override")._native_ptr(), 2995540667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

} // namespace godot
