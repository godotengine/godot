/**************************************************************************/
/*  skeleton_profile.cpp                                                  */
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

#include <godot_cpp/classes/skeleton_profile.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void SkeletonProfile::set_root_bone(const StringName &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_root_bone")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_name);
}

StringName SkeletonProfile::get_root_bone() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_root_bone")._native_ptr(), 2737447660);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void SkeletonProfile::set_scale_base_bone(const StringName &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_scale_base_bone")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_name);
}

StringName SkeletonProfile::get_scale_base_bone() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_scale_base_bone")._native_ptr(), 2737447660);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void SkeletonProfile::set_group_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_group_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t SkeletonProfile::get_group_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_group_size")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

StringName SkeletonProfile::get_group_name(int32_t p_group_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_group_name")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_group_idx_encoded;
	PtrToArg<int64_t>::encode(p_group_idx, &p_group_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_group_idx_encoded);
}

void SkeletonProfile::set_group_name(int32_t p_group_idx, const StringName &p_group_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_group_name")._native_ptr(), 3780747571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_group_idx_encoded;
	PtrToArg<int64_t>::encode(p_group_idx, &p_group_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_group_idx_encoded, &p_group_name);
}

Ref<Texture2D> SkeletonProfile::get_texture(int32_t p_group_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_texture")._native_ptr(), 3536238170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_group_idx_encoded;
	PtrToArg<int64_t>::encode(p_group_idx, &p_group_idx_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_group_idx_encoded));
}

void SkeletonProfile::set_texture(int32_t p_group_idx, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_texture")._native_ptr(), 666127730);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_group_idx_encoded;
	PtrToArg<int64_t>::encode(p_group_idx, &p_group_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_group_idx_encoded, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

void SkeletonProfile::set_bone_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_bone_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t SkeletonProfile::get_bone_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_bone_size")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t SkeletonProfile::find_bone(const StringName &p_bone_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("find_bone")._native_ptr(), 2458036349);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_bone_name);
}

StringName SkeletonProfile::get_bone_name(int32_t p_bone_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_bone_name")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

void SkeletonProfile::set_bone_name(int32_t p_bone_idx, const StringName &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_bone_name")._native_ptr(), 3780747571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_bone_name);
}

StringName SkeletonProfile::get_bone_parent(int32_t p_bone_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_bone_parent")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

void SkeletonProfile::set_bone_parent(int32_t p_bone_idx, const StringName &p_bone_parent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_bone_parent")._native_ptr(), 3780747571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_bone_parent);
}

SkeletonProfile::TailDirection SkeletonProfile::get_tail_direction(int32_t p_bone_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_tail_direction")._native_ptr(), 2675997574);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonProfile::TailDirection(0)));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return (SkeletonProfile::TailDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

void SkeletonProfile::set_tail_direction(int32_t p_bone_idx, SkeletonProfile::TailDirection p_tail_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_tail_direction")._native_ptr(), 1231951015);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	int64_t p_tail_direction_encoded;
	PtrToArg<int64_t>::encode(p_tail_direction, &p_tail_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_tail_direction_encoded);
}

StringName SkeletonProfile::get_bone_tail(int32_t p_bone_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_bone_tail")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

void SkeletonProfile::set_bone_tail(int32_t p_bone_idx, const StringName &p_bone_tail) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_bone_tail")._native_ptr(), 3780747571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_bone_tail);
}

Transform3D SkeletonProfile::get_reference_pose(int32_t p_bone_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_reference_pose")._native_ptr(), 1965739696);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

void SkeletonProfile::set_reference_pose(int32_t p_bone_idx, const Transform3D &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_reference_pose")._native_ptr(), 3616898986);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_bone_name);
}

Vector2 SkeletonProfile::get_handle_offset(int32_t p_bone_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_handle_offset")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

void SkeletonProfile::set_handle_offset(int32_t p_bone_idx, const Vector2 &p_handle_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_handle_offset")._native_ptr(), 163021252);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_handle_offset);
}

StringName SkeletonProfile::get_group(int32_t p_bone_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("get_group")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

void SkeletonProfile::set_group(int32_t p_bone_idx, const StringName &p_group) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_group")._native_ptr(), 3780747571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_group);
}

bool SkeletonProfile::is_required(int32_t p_bone_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("is_required")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

void SkeletonProfile::set_required(int32_t p_bone_idx, bool p_required) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonProfile::get_class_static()._native_ptr(), StringName("set_required")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	int8_t p_required_encoded;
	PtrToArg<bool>::encode(p_required, &p_required_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded, &p_required_encoded);
}

} // namespace godot
