/**************************************************************************/
/*  gltf_skin.cpp                                                         */
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

#include <godot_cpp/classes/gltf_skin.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/skin.hpp>

namespace godot {

int32_t GLTFSkin::get_skin_root() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_skin_root")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFSkin::set_skin_root(int32_t p_skin_root) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_skin_root")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_skin_root_encoded;
	PtrToArg<int64_t>::encode(p_skin_root, &p_skin_root_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skin_root_encoded);
}

PackedInt32Array GLTFSkin::get_joints_original() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_joints_original")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void GLTFSkin::set_joints_original(const PackedInt32Array &p_joints_original) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_joints_original")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joints_original);
}

TypedArray<Transform3D> GLTFSkin::get_inverse_binds() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_inverse_binds")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Transform3D>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Transform3D>>(_gde_method_bind, _owner);
}

void GLTFSkin::set_inverse_binds(const TypedArray<Transform3D> &p_inverse_binds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_inverse_binds")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_inverse_binds);
}

PackedInt32Array GLTFSkin::get_joints() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_joints")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void GLTFSkin::set_joints(const PackedInt32Array &p_joints) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_joints")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joints);
}

PackedInt32Array GLTFSkin::get_non_joints() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_non_joints")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void GLTFSkin::set_non_joints(const PackedInt32Array &p_non_joints) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_non_joints")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_non_joints);
}

PackedInt32Array GLTFSkin::get_roots() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_roots")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void GLTFSkin::set_roots(const PackedInt32Array &p_roots) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_roots")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_roots);
}

int32_t GLTFSkin::get_skeleton() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_skeleton")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFSkin::set_skeleton(int32_t p_skeleton) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_skeleton")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_skeleton_encoded;
	PtrToArg<int64_t>::encode(p_skeleton, &p_skeleton_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton_encoded);
}

Dictionary GLTFSkin::get_joint_i_to_bone_i() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_joint_i_to_bone_i")._native_ptr(), 2382534195);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

void GLTFSkin::set_joint_i_to_bone_i(const Dictionary &p_joint_i_to_bone_i) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_joint_i_to_bone_i")._native_ptr(), 4155329257);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_i_to_bone_i);
}

Dictionary GLTFSkin::get_joint_i_to_name() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_joint_i_to_name")._native_ptr(), 2382534195);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

void GLTFSkin::set_joint_i_to_name(const Dictionary &p_joint_i_to_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_joint_i_to_name")._native_ptr(), 4155329257);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_i_to_name);
}

Ref<Skin> GLTFSkin::get_godot_skin() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("get_godot_skin")._native_ptr(), 1032037385);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Skin>()));
	return Ref<Skin>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Skin>(_gde_method_bind, _owner));
}

void GLTFSkin::set_godot_skin(const Ref<Skin> &p_godot_skin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkin::get_class_static()._native_ptr(), StringName("set_godot_skin")._native_ptr(), 3971435618);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_godot_skin != nullptr ? &p_godot_skin->_owner : nullptr));
}

} // namespace godot
