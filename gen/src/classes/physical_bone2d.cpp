/**************************************************************************/
/*  physical_bone2d.cpp                                                   */
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

#include <godot_cpp/classes/physical_bone2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/joint2d.hpp>

namespace godot {

Joint2D *PhysicalBone2D::get_joint() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("get_joint")._native_ptr(), 3582132112);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Joint2D>(_gde_method_bind, _owner);
}

bool PhysicalBone2D::get_auto_configure_joint() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("get_auto_configure_joint")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PhysicalBone2D::set_auto_configure_joint(bool p_auto_configure_joint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("set_auto_configure_joint")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_auto_configure_joint_encoded;
	PtrToArg<bool>::encode(p_auto_configure_joint, &p_auto_configure_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_auto_configure_joint_encoded);
}

void PhysicalBone2D::set_simulate_physics(bool p_simulate_physics) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("set_simulate_physics")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_simulate_physics_encoded;
	PtrToArg<bool>::encode(p_simulate_physics, &p_simulate_physics_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_simulate_physics_encoded);
}

bool PhysicalBone2D::get_simulate_physics() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("get_simulate_physics")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool PhysicalBone2D::is_simulating_physics() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("is_simulating_physics")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PhysicalBone2D::set_bone2d_nodepath(const NodePath &p_nodepath) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("set_bone2d_nodepath")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_nodepath);
}

NodePath PhysicalBone2D::get_bone2d_nodepath() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("get_bone2d_nodepath")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void PhysicalBone2D::set_bone2d_index(int32_t p_bone_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("set_bone2d_index")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_index_encoded;
	PtrToArg<int64_t>::encode(p_bone_index, &p_bone_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_index_encoded);
}

int32_t PhysicalBone2D::get_bone2d_index() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("get_bone2d_index")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PhysicalBone2D::set_follow_bone_when_simulating(bool p_follow_bone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("set_follow_bone_when_simulating")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_follow_bone_encoded;
	PtrToArg<bool>::encode(p_follow_bone, &p_follow_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_follow_bone_encoded);
}

bool PhysicalBone2D::get_follow_bone_when_simulating() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone2D::get_class_static()._native_ptr(), StringName("get_follow_bone_when_simulating")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
