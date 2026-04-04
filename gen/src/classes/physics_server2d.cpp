/**************************************************************************/
/*  physics_server2d.cpp                                                  */
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

#include <godot_cpp/classes/physics_server2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_direct_body_state2d.hpp>
#include <godot_cpp/classes/physics_direct_space_state2d.hpp>
#include <godot_cpp/classes/physics_test_motion_parameters2d.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

PhysicsServer2D *PhysicsServer2D::singleton = nullptr;

PhysicsServer2D *PhysicsServer2D::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(PhysicsServer2D::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<PhysicsServer2D *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &PhysicsServer2D::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(PhysicsServer2D::get_class_static(), singleton);
		}
	}
	return singleton;
}

PhysicsServer2D::~PhysicsServer2D() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(PhysicsServer2D::get_class_static());
		singleton = nullptr;
	}
}

RID PhysicsServer2D::world_boundary_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("world_boundary_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer2D::separation_ray_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("separation_ray_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer2D::segment_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("segment_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer2D::circle_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("circle_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer2D::rectangle_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("rectangle_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer2D::capsule_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("capsule_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer2D::convex_polygon_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("convex_polygon_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer2D::concave_polygon_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("concave_polygon_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer2D::shape_set_data(const RID &p_shape, const Variant &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("shape_set_data")._native_ptr(), 3175752987);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape, &p_data);
}

PhysicsServer2D::ShapeType PhysicsServer2D::shape_get_type(const RID &p_shape) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("shape_get_type")._native_ptr(), 1240598777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicsServer2D::ShapeType(0)));
	return (PhysicsServer2D::ShapeType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shape);
}

Variant PhysicsServer2D::shape_get_data(const RID &p_shape) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("shape_get_data")._native_ptr(), 4171304767);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_shape);
}

RID PhysicsServer2D::space_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("space_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer2D::space_set_active(const RID &p_space, bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("space_set_active")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_space, &p_active_encoded);
}

bool PhysicsServer2D::space_is_active(const RID &p_space) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("space_is_active")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_space);
}

void PhysicsServer2D::space_set_param(const RID &p_space, PhysicsServer2D::SpaceParameter p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("space_set_param")._native_ptr(), 949194586);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_space, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer2D::space_get_param(const RID &p_space, PhysicsServer2D::SpaceParameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("space_get_param")._native_ptr(), 874111783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_space, &p_param_encoded);
}

PhysicsDirectSpaceState2D *PhysicsServer2D::space_get_direct_state(const RID &p_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("space_get_direct_state")._native_ptr(), 3160173886);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<PhysicsDirectSpaceState2D>(_gde_method_bind, _owner, &p_space);
}

RID PhysicsServer2D::area_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer2D::area_set_space(const RID &p_area, const RID &p_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_space")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_space);
}

RID PhysicsServer2D::area_get_space(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_space")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer2D::area_add_shape(const RID &p_area, const RID &p_shape, const Transform2D &p_transform, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_add_shape")._native_ptr(), 339056240);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape, &p_transform, &p_disabled_encoded);
}

void PhysicsServer2D::area_set_shape(const RID &p_area, int32_t p_shape_idx, const RID &p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_shape")._native_ptr(), 2310537182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded, &p_shape);
}

void PhysicsServer2D::area_set_shape_transform(const RID &p_area, int32_t p_shape_idx, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_shape_transform")._native_ptr(), 736082694);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded, &p_transform);
}

void PhysicsServer2D::area_set_shape_disabled(const RID &p_area, int32_t p_shape_idx, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_shape_disabled")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded, &p_disabled_encoded);
}

int32_t PhysicsServer2D::area_get_shape_count(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_shape_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_area);
}

RID PhysicsServer2D::area_get_shape(const RID &p_area, int32_t p_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_shape")._native_ptr(), 1066463050);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded);
}

Transform2D PhysicsServer2D::area_get_shape_transform(const RID &p_area, int32_t p_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_shape_transform")._native_ptr(), 1324854622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded);
}

void PhysicsServer2D::area_remove_shape(const RID &p_area, int32_t p_shape_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_remove_shape")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded);
}

void PhysicsServer2D::area_clear_shapes(const RID &p_area) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_clear_shapes")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer2D::area_set_collision_layer(const RID &p_area, uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_collision_layer")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_layer_encoded);
}

uint32_t PhysicsServer2D::area_get_collision_layer(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_collision_layer")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer2D::area_set_collision_mask(const RID &p_area, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_collision_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_mask_encoded);
}

uint32_t PhysicsServer2D::area_get_collision_mask(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_collision_mask")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer2D::area_set_param(const RID &p_area, PhysicsServer2D::AreaParameter p_param, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_param")._native_ptr(), 1257146028);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_param_encoded, &p_value);
}

void PhysicsServer2D::area_set_transform(const RID &p_area, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_transform);
}

Variant PhysicsServer2D::area_get_param(const RID &p_area, PhysicsServer2D::AreaParameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_param")._native_ptr(), 3047435120);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_area, &p_param_encoded);
}

Transform2D PhysicsServer2D::area_get_transform(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_transform")._native_ptr(), 213527486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer2D::area_attach_object_instance_id(const RID &p_area, uint64_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_attach_object_instance_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_id_encoded);
}

uint64_t PhysicsServer2D::area_get_object_instance_id(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_object_instance_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer2D::area_attach_canvas_instance_id(const RID &p_area, uint64_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_attach_canvas_instance_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_id_encoded);
}

uint64_t PhysicsServer2D::area_get_canvas_instance_id(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_get_canvas_instance_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer2D::area_set_monitor_callback(const RID &p_area, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_monitor_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_callback);
}

void PhysicsServer2D::area_set_area_monitor_callback(const RID &p_area, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_area_monitor_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_callback);
}

void PhysicsServer2D::area_set_monitorable(const RID &p_area, bool p_monitorable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("area_set_monitorable")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_monitorable_encoded;
	PtrToArg<bool>::encode(p_monitorable, &p_monitorable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_monitorable_encoded);
}

RID PhysicsServer2D::body_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer2D::body_set_space(const RID &p_body, const RID &p_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_space")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_space);
}

RID PhysicsServer2D::body_get_space(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_space")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_mode(const RID &p_body, PhysicsServer2D::BodyMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_mode")._native_ptr(), 1658067650);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_mode_encoded);
}

PhysicsServer2D::BodyMode PhysicsServer2D::body_get_mode(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_mode")._native_ptr(), 3261702585);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicsServer2D::BodyMode(0)));
	return (PhysicsServer2D::BodyMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_add_shape(const RID &p_body, const RID &p_shape, const Transform2D &p_transform, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_add_shape")._native_ptr(), 339056240);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape, &p_transform, &p_disabled_encoded);
}

void PhysicsServer2D::body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_shape")._native_ptr(), 2310537182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded, &p_shape);
}

void PhysicsServer2D::body_set_shape_transform(const RID &p_body, int32_t p_shape_idx, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_shape_transform")._native_ptr(), 736082694);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded, &p_transform);
}

int32_t PhysicsServer2D::body_get_shape_count(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_shape_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

RID PhysicsServer2D::body_get_shape(const RID &p_body, int32_t p_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_shape")._native_ptr(), 1066463050);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded);
}

Transform2D PhysicsServer2D::body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_shape_transform")._native_ptr(), 1324854622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded);
}

void PhysicsServer2D::body_remove_shape(const RID &p_body, int32_t p_shape_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_remove_shape")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded);
}

void PhysicsServer2D::body_clear_shapes(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_clear_shapes")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_shape_disabled")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded, &p_disabled_encoded);
}

void PhysicsServer2D::body_set_shape_as_one_way_collision(const RID &p_body, int32_t p_shape_idx, bool p_enable, float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_shape_as_one_way_collision")._native_ptr(), 2556489974);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded, &p_enable_encoded, &p_margin_encoded);
}

void PhysicsServer2D::body_attach_object_instance_id(const RID &p_body, uint64_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_attach_object_instance_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_id_encoded);
}

uint64_t PhysicsServer2D::body_get_object_instance_id(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_object_instance_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_attach_canvas_instance_id(const RID &p_body, uint64_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_attach_canvas_instance_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_id_encoded);
}

uint64_t PhysicsServer2D::body_get_canvas_instance_id(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_canvas_instance_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_continuous_collision_detection_mode(const RID &p_body, PhysicsServer2D::CCDMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_continuous_collision_detection_mode")._native_ptr(), 1882257015);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_mode_encoded);
}

PhysicsServer2D::CCDMode PhysicsServer2D::body_get_continuous_collision_detection_mode(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_continuous_collision_detection_mode")._native_ptr(), 2661282217);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicsServer2D::CCDMode(0)));
	return (PhysicsServer2D::CCDMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_collision_layer(const RID &p_body, uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_collision_layer")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_layer_encoded);
}

uint32_t PhysicsServer2D::body_get_collision_layer(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_collision_layer")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_collision_mask(const RID &p_body, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_collision_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_mask_encoded);
}

uint32_t PhysicsServer2D::body_get_collision_mask(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_collision_mask")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_collision_priority(const RID &p_body, float p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_collision_priority")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_priority_encoded;
	PtrToArg<double>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_priority_encoded);
}

float PhysicsServer2D::body_get_collision_priority(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_collision_priority")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_param(const RID &p_body, PhysicsServer2D::BodyParameter p_param, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_param")._native_ptr(), 2715630609);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_param_encoded, &p_value);
}

Variant PhysicsServer2D::body_get_param(const RID &p_body, PhysicsServer2D::BodyParameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_param")._native_ptr(), 3208033526);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_body, &p_param_encoded);
}

void PhysicsServer2D::body_reset_mass_properties(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_reset_mass_properties")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_state(const RID &p_body, PhysicsServer2D::BodyState p_state, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_state")._native_ptr(), 1706355209);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_state_encoded, &p_value);
}

Variant PhysicsServer2D::body_get_state(const RID &p_body, PhysicsServer2D::BodyState p_state) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_state")._native_ptr(), 4036367961);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_body, &p_state_encoded);
}

void PhysicsServer2D::body_apply_central_impulse(const RID &p_body, const Vector2 &p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_apply_central_impulse")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_impulse);
}

void PhysicsServer2D::body_apply_torque_impulse(const RID &p_body, float p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_apply_torque_impulse")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_impulse_encoded;
	PtrToArg<double>::encode(p_impulse, &p_impulse_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_impulse_encoded);
}

void PhysicsServer2D::body_apply_impulse(const RID &p_body, const Vector2 &p_impulse, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_apply_impulse")._native_ptr(), 205485391);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_impulse, &p_position);
}

void PhysicsServer2D::body_apply_central_force(const RID &p_body, const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_apply_central_force")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force);
}

void PhysicsServer2D::body_apply_force(const RID &p_body, const Vector2 &p_force, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_apply_force")._native_ptr(), 205485391);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force, &p_position);
}

void PhysicsServer2D::body_apply_torque(const RID &p_body, float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_apply_torque")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_torque_encoded);
}

void PhysicsServer2D::body_add_constant_central_force(const RID &p_body, const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_add_constant_central_force")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force);
}

void PhysicsServer2D::body_add_constant_force(const RID &p_body, const Vector2 &p_force, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_add_constant_force")._native_ptr(), 205485391);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force, &p_position);
}

void PhysicsServer2D::body_add_constant_torque(const RID &p_body, float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_add_constant_torque")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_torque_encoded);
}

void PhysicsServer2D::body_set_constant_force(const RID &p_body, const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_constant_force")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force);
}

Vector2 PhysicsServer2D::body_get_constant_force(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_constant_force")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_constant_torque(const RID &p_body, float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_constant_torque")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_torque_encoded);
}

float PhysicsServer2D::body_get_constant_torque(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_constant_torque")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_axis_velocity(const RID &p_body, const Vector2 &p_axis_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_axis_velocity")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_axis_velocity);
}

void PhysicsServer2D::body_add_collision_exception(const RID &p_body, const RID &p_excepted_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_add_collision_exception")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_excepted_body);
}

void PhysicsServer2D::body_remove_collision_exception(const RID &p_body, const RID &p_excepted_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_remove_collision_exception")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_excepted_body);
}

void PhysicsServer2D::body_set_max_contacts_reported(const RID &p_body, int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_max_contacts_reported")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_amount_encoded);
}

int32_t PhysicsServer2D::body_get_max_contacts_reported(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_max_contacts_reported")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_omit_force_integration(const RID &p_body, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_omit_force_integration")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_enable_encoded);
}

bool PhysicsServer2D::body_is_omitting_force_integration(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_is_omitting_force_integration")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer2D::body_set_state_sync_callback(const RID &p_body, const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_state_sync_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_callable);
}

void PhysicsServer2D::body_set_force_integration_callback(const RID &p_body, const Callable &p_callable, const Variant &p_userdata) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_set_force_integration_callback")._native_ptr(), 3059434249);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_callable, &p_userdata);
}

bool PhysicsServer2D::body_test_motion(const RID &p_body, const Ref<PhysicsTestMotionParameters2D> &p_parameters, const Ref<PhysicsTestMotionResult2D> &p_result) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_test_motion")._native_ptr(), 1699844009);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body, (p_parameters != nullptr ? &p_parameters->_owner : nullptr), (p_result != nullptr ? &p_result->_owner : nullptr));
}

PhysicsDirectBodyState2D *PhysicsServer2D::body_get_direct_state(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("body_get_direct_state")._native_ptr(), 1191931871);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<PhysicsDirectBodyState2D>(_gde_method_bind, _owner, &p_body);
}

RID PhysicsServer2D::joint_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer2D::joint_clear(const RID &p_joint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_clear")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer2D::joint_set_param(const RID &p_joint, PhysicsServer2D::JointParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_set_param")._native_ptr(), 3972556514);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer2D::joint_get_param(const RID &p_joint, PhysicsServer2D::JointParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_get_param")._native_ptr(), 4016448949);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint, &p_param_encoded);
}

void PhysicsServer2D::joint_disable_collisions_between_bodies(const RID &p_joint, bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_disable_collisions_between_bodies")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_disable_encoded);
}

bool PhysicsServer2D::joint_is_disabled_collisions_between_bodies(const RID &p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_is_disabled_collisions_between_bodies")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer2D::joint_make_pin(const RID &p_joint, const Vector2 &p_anchor, const RID &p_body_a, const RID &p_body_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_make_pin")._native_ptr(), 1612646186);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_anchor, &p_body_a, &p_body_b);
}

void PhysicsServer2D::joint_make_groove(const RID &p_joint, const Vector2 &p_groove1_a, const Vector2 &p_groove2_a, const Vector2 &p_anchor_b, const RID &p_body_a, const RID &p_body_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_make_groove")._native_ptr(), 481430435);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_groove1_a, &p_groove2_a, &p_anchor_b, &p_body_a, &p_body_b);
}

void PhysicsServer2D::joint_make_damped_spring(const RID &p_joint, const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, const RID &p_body_a, const RID &p_body_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_make_damped_spring")._native_ptr(), 1994657646);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_anchor_a, &p_anchor_b, &p_body_a, &p_body_b);
}

void PhysicsServer2D::pin_joint_set_flag(const RID &p_joint, PhysicsServer2D::PinJointFlag p_flag, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("pin_joint_set_flag")._native_ptr(), 3520002352);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_flag_encoded, &p_enabled_encoded);
}

bool PhysicsServer2D::pin_joint_get_flag(const RID &p_joint, PhysicsServer2D::PinJointFlag p_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("pin_joint_get_flag")._native_ptr(), 2647867364);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_joint, &p_flag_encoded);
}

void PhysicsServer2D::pin_joint_set_param(const RID &p_joint, PhysicsServer2D::PinJointParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("pin_joint_set_param")._native_ptr(), 550574241);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer2D::pin_joint_get_param(const RID &p_joint, PhysicsServer2D::PinJointParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("pin_joint_get_param")._native_ptr(), 348281383);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint, &p_param_encoded);
}

void PhysicsServer2D::damped_spring_joint_set_param(const RID &p_joint, PhysicsServer2D::DampedSpringParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("damped_spring_joint_set_param")._native_ptr(), 220564071);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer2D::damped_spring_joint_get_param(const RID &p_joint, PhysicsServer2D::DampedSpringParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("damped_spring_joint_get_param")._native_ptr(), 2075871277);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint, &p_param_encoded);
}

PhysicsServer2D::JointType PhysicsServer2D::joint_get_type(const RID &p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("joint_get_type")._native_ptr(), 4262502231);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicsServer2D::JointType(0)));
	return (PhysicsServer2D::JointType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer2D::free_rid(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("free_rid")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

void PhysicsServer2D::set_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("set_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

int32_t PhysicsServer2D::get_process_info(PhysicsServer2D::ProcessInfo p_process_info) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2D::get_class_static()._native_ptr(), StringName("get_process_info")._native_ptr(), 576496006);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_process_info_encoded;
	PtrToArg<int64_t>::encode(p_process_info, &p_process_info_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_process_info_encoded);
}

} // namespace godot
