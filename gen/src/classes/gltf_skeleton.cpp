/**************************************************************************/
/*  gltf_skeleton.cpp                                                     */
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

#include <godot_cpp/classes/gltf_skeleton.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/bone_attachment3d.hpp>
#include <godot_cpp/classes/skeleton3d.hpp>

namespace godot {

PackedInt32Array GLTFSkeleton::get_joints() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("get_joints")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void GLTFSkeleton::set_joints(const PackedInt32Array &p_joints) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("set_joints")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joints);
}

PackedInt32Array GLTFSkeleton::get_roots() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("get_roots")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void GLTFSkeleton::set_roots(const PackedInt32Array &p_roots) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("set_roots")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_roots);
}

Skeleton3D *GLTFSkeleton::get_godot_skeleton() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("get_godot_skeleton")._native_ptr(), 1814733083);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Skeleton3D>(_gde_method_bind, _owner);
}

TypedArray<String> GLTFSkeleton::get_unique_names() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("get_unique_names")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<String>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<String>>(_gde_method_bind, _owner);
}

void GLTFSkeleton::set_unique_names(const TypedArray<String> &p_unique_names) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("set_unique_names")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_unique_names);
}

Dictionary GLTFSkeleton::get_godot_bone_node() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("get_godot_bone_node")._native_ptr(), 2382534195);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

void GLTFSkeleton::set_godot_bone_node(const Dictionary &p_godot_bone_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("set_godot_bone_node")._native_ptr(), 4155329257);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_godot_bone_node);
}

int32_t GLTFSkeleton::get_bone_attachment_count() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("get_bone_attachment_count")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

BoneAttachment3D *GLTFSkeleton::get_bone_attachment(int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSkeleton::get_class_static()._native_ptr(), StringName("get_bone_attachment")._native_ptr(), 945440495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<BoneAttachment3D>(_gde_method_bind, _owner, &p_idx_encoded);
}

} // namespace godot
