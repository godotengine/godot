/**************************************************************************/
/*  physics_direct_space_state3d.cpp                                      */
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

#include <godot_cpp/classes/physics_direct_space_state3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_point_query_parameters3d.hpp>
#include <godot_cpp/classes/physics_ray_query_parameters3d.hpp>
#include <godot_cpp/classes/physics_shape_query_parameters3d.hpp>

namespace godot {

TypedArray<Dictionary> PhysicsDirectSpaceState3D::intersect_point(const Ref<PhysicsPointQueryParameters3D> &p_parameters, int32_t p_max_results) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectSpaceState3D::get_class_static()._native_ptr(), StringName("intersect_point")._native_ptr(), 975173756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	int64_t p_max_results_encoded;
	PtrToArg<int64_t>::encode(p_max_results, &p_max_results_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, (p_parameters != nullptr ? &p_parameters->_owner : nullptr), &p_max_results_encoded);
}

Dictionary PhysicsDirectSpaceState3D::intersect_ray(const Ref<PhysicsRayQueryParameters3D> &p_parameters) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectSpaceState3D::get_class_static()._native_ptr(), StringName("intersect_ray")._native_ptr(), 3957970750);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, (p_parameters != nullptr ? &p_parameters->_owner : nullptr));
}

TypedArray<Dictionary> PhysicsDirectSpaceState3D::intersect_shape(const Ref<PhysicsShapeQueryParameters3D> &p_parameters, int32_t p_max_results) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectSpaceState3D::get_class_static()._native_ptr(), StringName("intersect_shape")._native_ptr(), 3762137681);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	int64_t p_max_results_encoded;
	PtrToArg<int64_t>::encode(p_max_results, &p_max_results_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, (p_parameters != nullptr ? &p_parameters->_owner : nullptr), &p_max_results_encoded);
}

PackedFloat32Array PhysicsDirectSpaceState3D::cast_motion(const Ref<PhysicsShapeQueryParameters3D> &p_parameters) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectSpaceState3D::get_class_static()._native_ptr(), StringName("cast_motion")._native_ptr(), 1778757334);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner, (p_parameters != nullptr ? &p_parameters->_owner : nullptr));
}

TypedArray<Vector3> PhysicsDirectSpaceState3D::collide_shape(const Ref<PhysicsShapeQueryParameters3D> &p_parameters, int32_t p_max_results) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectSpaceState3D::get_class_static()._native_ptr(), StringName("collide_shape")._native_ptr(), 3762137681);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector3>()));
	int64_t p_max_results_encoded;
	PtrToArg<int64_t>::encode(p_max_results, &p_max_results_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector3>>(_gde_method_bind, _owner, (p_parameters != nullptr ? &p_parameters->_owner : nullptr), &p_max_results_encoded);
}

Dictionary PhysicsDirectSpaceState3D::get_rest_info(const Ref<PhysicsShapeQueryParameters3D> &p_parameters) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectSpaceState3D::get_class_static()._native_ptr(), StringName("get_rest_info")._native_ptr(), 1376751592);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, (p_parameters != nullptr ? &p_parameters->_owner : nullptr));
}

} // namespace godot
