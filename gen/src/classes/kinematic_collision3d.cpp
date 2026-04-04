/**************************************************************************/
/*  kinematic_collision3d.cpp                                             */
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

#include <godot_cpp/classes/kinematic_collision3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/core/object.hpp>

namespace godot {

Vector3 KinematicCollision3D::get_travel() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_travel")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 KinematicCollision3D::get_remainder() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_remainder")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

float KinematicCollision3D::get_depth() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_depth")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

int32_t KinematicCollision3D::get_collision_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_collision_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Vector3 KinematicCollision3D::get_position(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_position")._native_ptr(), 1914908202);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

Vector3 KinematicCollision3D::get_normal(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_normal")._native_ptr(), 1914908202);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

float KinematicCollision3D::get_angle(int32_t p_collision_index, const Vector3 &p_up_direction) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_angle")._native_ptr(), 1242741860);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_collision_index_encoded, &p_up_direction);
}

Object *KinematicCollision3D::get_local_shape(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_local_shape")._native_ptr(), 2639523548);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Object>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

Object *KinematicCollision3D::get_collider(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_collider")._native_ptr(), 2639523548);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Object>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

uint64_t KinematicCollision3D::get_collider_id(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_collider_id")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

RID KinematicCollision3D::get_collider_rid(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_collider_rid")._native_ptr(), 1231817359);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

Object *KinematicCollision3D::get_collider_shape(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_collider_shape")._native_ptr(), 2639523548);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Object>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

int32_t KinematicCollision3D::get_collider_shape_index(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_collider_shape_index")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

Vector3 KinematicCollision3D::get_collider_velocity(int32_t p_collision_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(KinematicCollision3D::get_class_static()._native_ptr(), StringName("get_collider_velocity")._native_ptr(), 1914908202);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_collision_index_encoded;
	PtrToArg<int64_t>::encode(p_collision_index, &p_collision_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_collision_index_encoded);
}

} // namespace godot
