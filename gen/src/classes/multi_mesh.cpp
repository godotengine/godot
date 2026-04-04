/**************************************************************************/
/*  multi_mesh.cpp                                                        */
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

#include <godot_cpp/classes/multi_mesh.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/mesh.hpp>

namespace godot {

void MultiMesh::set_mesh(const Ref<Mesh> &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_mesh")._native_ptr(), 194775623);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mesh != nullptr ? &p_mesh->_owner : nullptr));
}

Ref<Mesh> MultiMesh::get_mesh() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_mesh")._native_ptr(), 1808005922);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Mesh>()));
	return Ref<Mesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Mesh>(_gde_method_bind, _owner));
}

void MultiMesh::set_use_colors(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_use_colors")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool MultiMesh::is_using_colors() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("is_using_colors")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void MultiMesh::set_use_custom_data(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_use_custom_data")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool MultiMesh::is_using_custom_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("is_using_custom_data")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void MultiMesh::set_transform_format(MultiMesh::TransformFormat p_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_transform_format")._native_ptr(), 2404750322);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_format_encoded);
}

MultiMesh::TransformFormat MultiMesh::get_transform_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_transform_format")._native_ptr(), 2444156481);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (MultiMesh::TransformFormat(0)));
	return (MultiMesh::TransformFormat)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiMesh::set_instance_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_instance_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t MultiMesh::get_instance_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_instance_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiMesh::set_visible_instance_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_visible_instance_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t MultiMesh::get_visible_instance_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_visible_instance_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiMesh::set_physics_interpolation_quality(MultiMesh::PhysicsInterpolationQuality p_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_physics_interpolation_quality")._native_ptr(), 1819488408);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded);
}

MultiMesh::PhysicsInterpolationQuality MultiMesh::get_physics_interpolation_quality() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_physics_interpolation_quality")._native_ptr(), 1465701882);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (MultiMesh::PhysicsInterpolationQuality(0)));
	return (MultiMesh::PhysicsInterpolationQuality)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiMesh::set_instance_transform(int32_t p_instance, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_instance_transform")._native_ptr(), 3616898986);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance_encoded, &p_transform);
}

void MultiMesh::set_instance_transform_2d(int32_t p_instance, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_instance_transform_2d")._native_ptr(), 30160968);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance_encoded, &p_transform);
}

Transform3D MultiMesh::get_instance_transform(int32_t p_instance) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_instance_transform")._native_ptr(), 1965739696);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_instance_encoded);
}

Transform2D MultiMesh::get_instance_transform_2d(int32_t p_instance) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_instance_transform_2d")._native_ptr(), 3836996910);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_instance_encoded);
}

void MultiMesh::set_instance_color(int32_t p_instance, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_instance_color")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance_encoded, &p_color);
}

Color MultiMesh::get_instance_color(int32_t p_instance) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_instance_color")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_instance_encoded);
}

void MultiMesh::set_instance_custom_data(int32_t p_instance, const Color &p_custom_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_instance_custom_data")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance_encoded, &p_custom_data);
}

Color MultiMesh::get_instance_custom_data(int32_t p_instance) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_instance_custom_data")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_instance_encoded);
}

void MultiMesh::reset_instance_physics_interpolation(int32_t p_instance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("reset_instance_physics_interpolation")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_instance_encoded;
	PtrToArg<int64_t>::encode(p_instance, &p_instance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance_encoded);
}

void MultiMesh::reset_instances_physics_interpolation() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("reset_instances_physics_interpolation")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void MultiMesh::set_custom_aabb(const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_custom_aabb")._native_ptr(), 259215842);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_aabb);
}

AABB MultiMesh::get_custom_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_custom_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

AABB MultiMesh::get_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

PackedFloat32Array MultiMesh::get_buffer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("get_buffer")._native_ptr(), 675695659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner);
}

void MultiMesh::set_buffer(const PackedFloat32Array &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_buffer")._native_ptr(), 2899603908);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_buffer);
}

void MultiMesh::set_buffer_interpolated(const PackedFloat32Array &p_buffer_curr, const PackedFloat32Array &p_buffer_prev) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiMesh::get_class_static()._native_ptr(), StringName("set_buffer_interpolated")._native_ptr(), 3514430332);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_buffer_curr, &p_buffer_prev);
}

} // namespace godot
