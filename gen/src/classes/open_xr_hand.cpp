/**************************************************************************/
/*  open_xr_hand.cpp                                                      */
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

#include <godot_cpp/classes/open_xr_hand.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void OpenXRHand::set_hand(OpenXRHand::Hands p_hand) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("set_hand")._native_ptr(), 1849328560);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_hand_encoded;
	PtrToArg<int64_t>::encode(p_hand, &p_hand_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hand_encoded);
}

OpenXRHand::Hands OpenXRHand::get_hand() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("get_hand")._native_ptr(), 2850644561);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRHand::Hands(0)));
	return (OpenXRHand::Hands)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRHand::set_hand_skeleton(const NodePath &p_hand_skeleton) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("set_hand_skeleton")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hand_skeleton);
}

NodePath OpenXRHand::get_hand_skeleton() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("get_hand_skeleton")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void OpenXRHand::set_motion_range(OpenXRHand::MotionRange p_motion_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("set_motion_range")._native_ptr(), 3326516003);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_motion_range_encoded;
	PtrToArg<int64_t>::encode(p_motion_range, &p_motion_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_motion_range_encoded);
}

OpenXRHand::MotionRange OpenXRHand::get_motion_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("get_motion_range")._native_ptr(), 2191822314);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRHand::MotionRange(0)));
	return (OpenXRHand::MotionRange)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRHand::set_skeleton_rig(OpenXRHand::SkeletonRig p_skeleton_rig) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("set_skeleton_rig")._native_ptr(), 1528072213);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_skeleton_rig_encoded;
	PtrToArg<int64_t>::encode(p_skeleton_rig, &p_skeleton_rig_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton_rig_encoded);
}

OpenXRHand::SkeletonRig OpenXRHand::get_skeleton_rig() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("get_skeleton_rig")._native_ptr(), 968409338);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRHand::SkeletonRig(0)));
	return (OpenXRHand::SkeletonRig)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRHand::set_bone_update(OpenXRHand::BoneUpdate p_bone_update) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("set_bone_update")._native_ptr(), 3144625444);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_update_encoded;
	PtrToArg<int64_t>::encode(p_bone_update, &p_bone_update_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_update_encoded);
}

OpenXRHand::BoneUpdate OpenXRHand::get_bone_update() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRHand::get_class_static()._native_ptr(), StringName("get_bone_update")._native_ptr(), 1310695248);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRHand::BoneUpdate(0)));
	return (OpenXRHand::BoneUpdate)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
