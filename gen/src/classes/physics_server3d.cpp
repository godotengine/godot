/**************************************************************************/
/*  physics_server3d.cpp                                                  */
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

#include <godot_cpp/classes/physics_server3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_direct_body_state3d.hpp>
#include <godot_cpp/classes/physics_direct_space_state3d.hpp>
#include <godot_cpp/classes/physics_server3d_rendering_server_handler.hpp>
#include <godot_cpp/classes/physics_test_motion_parameters3d.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

PhysicsServer3D *PhysicsServer3D::singleton = nullptr;

PhysicsServer3D *PhysicsServer3D::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(PhysicsServer3D::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<PhysicsServer3D *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &PhysicsServer3D::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(PhysicsServer3D::get_class_static(), singleton);
		}
	}
	return singleton;
}

PhysicsServer3D::~PhysicsServer3D() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(PhysicsServer3D::get_class_static());
		singleton = nullptr;
	}
}

RID PhysicsServer3D::world_boundary_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("world_boundary_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::separation_ray_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("separation_ray_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::sphere_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("sphere_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::box_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("box_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::capsule_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("capsule_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::cylinder_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("cylinder_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::convex_polygon_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("convex_polygon_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::concave_polygon_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("concave_polygon_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::heightmap_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("heightmap_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID PhysicsServer3D::custom_shape_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("custom_shape_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer3D::shape_set_data(const RID &p_shape, const Variant &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("shape_set_data")._native_ptr(), 3175752987);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape, &p_data);
}

void PhysicsServer3D::shape_set_margin(const RID &p_shape, float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("shape_set_margin")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape, &p_margin_encoded);
}

PhysicsServer3D::ShapeType PhysicsServer3D::shape_get_type(const RID &p_shape) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("shape_get_type")._native_ptr(), 3418923367);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicsServer3D::ShapeType(0)));
	return (PhysicsServer3D::ShapeType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_shape);
}

Variant PhysicsServer3D::shape_get_data(const RID &p_shape) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("shape_get_data")._native_ptr(), 4171304767);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_shape);
}

float PhysicsServer3D::shape_get_margin(const RID &p_shape) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("shape_get_margin")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_shape);
}

RID PhysicsServer3D::space_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("space_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer3D::space_set_active(const RID &p_space, bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("space_set_active")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_space, &p_active_encoded);
}

bool PhysicsServer3D::space_is_active(const RID &p_space) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("space_is_active")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_space);
}

void PhysicsServer3D::space_set_param(const RID &p_space, PhysicsServer3D::SpaceParameter p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("space_set_param")._native_ptr(), 2406017470);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_space, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer3D::space_get_param(const RID &p_space, PhysicsServer3D::SpaceParameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("space_get_param")._native_ptr(), 1523206731);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_space, &p_param_encoded);
}

PhysicsDirectSpaceState3D *PhysicsServer3D::space_get_direct_state(const RID &p_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("space_get_direct_state")._native_ptr(), 2048616813);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<PhysicsDirectSpaceState3D>(_gde_method_bind, _owner, &p_space);
}

RID PhysicsServer3D::area_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer3D::area_set_space(const RID &p_area, const RID &p_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_space")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_space);
}

RID PhysicsServer3D::area_get_space(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_space")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer3D::area_add_shape(const RID &p_area, const RID &p_shape, const Transform3D &p_transform, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_add_shape")._native_ptr(), 3711419014);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape, &p_transform, &p_disabled_encoded);
}

void PhysicsServer3D::area_set_shape(const RID &p_area, int32_t p_shape_idx, const RID &p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_shape")._native_ptr(), 2310537182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded, &p_shape);
}

void PhysicsServer3D::area_set_shape_transform(const RID &p_area, int32_t p_shape_idx, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_shape_transform")._native_ptr(), 675327471);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded, &p_transform);
}

void PhysicsServer3D::area_set_shape_disabled(const RID &p_area, int32_t p_shape_idx, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_shape_disabled")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded, &p_disabled_encoded);
}

int32_t PhysicsServer3D::area_get_shape_count(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_shape_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_area);
}

RID PhysicsServer3D::area_get_shape(const RID &p_area, int32_t p_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_shape")._native_ptr(), 1066463050);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded);
}

Transform3D PhysicsServer3D::area_get_shape_transform(const RID &p_area, int32_t p_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_shape_transform")._native_ptr(), 1050775521);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded);
}

void PhysicsServer3D::area_remove_shape(const RID &p_area, int32_t p_shape_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_remove_shape")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_shape_idx_encoded);
}

void PhysicsServer3D::area_clear_shapes(const RID &p_area) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_clear_shapes")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer3D::area_set_collision_layer(const RID &p_area, uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_collision_layer")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_layer_encoded);
}

uint32_t PhysicsServer3D::area_get_collision_layer(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_collision_layer")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer3D::area_set_collision_mask(const RID &p_area, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_collision_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_mask_encoded);
}

uint32_t PhysicsServer3D::area_get_collision_mask(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_collision_mask")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer3D::area_set_param(const RID &p_area, PhysicsServer3D::AreaParameter p_param, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_param")._native_ptr(), 2980114638);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_param_encoded, &p_value);
}

void PhysicsServer3D::area_set_transform(const RID &p_area, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_transform")._native_ptr(), 3935195649);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_transform);
}

Variant PhysicsServer3D::area_get_param(const RID &p_area, PhysicsServer3D::AreaParameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_param")._native_ptr(), 890056067);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_area, &p_param_encoded);
}

Transform3D PhysicsServer3D::area_get_transform(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_transform")._native_ptr(), 1128465797);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer3D::area_attach_object_instance_id(const RID &p_area, uint64_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_attach_object_instance_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_id_encoded);
}

uint64_t PhysicsServer3D::area_get_object_instance_id(const RID &p_area) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_get_object_instance_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_area);
}

void PhysicsServer3D::area_set_monitor_callback(const RID &p_area, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_monitor_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_callback);
}

void PhysicsServer3D::area_set_area_monitor_callback(const RID &p_area, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_area_monitor_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_callback);
}

void PhysicsServer3D::area_set_monitorable(const RID &p_area, bool p_monitorable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_monitorable")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_monitorable_encoded;
	PtrToArg<bool>::encode(p_monitorable, &p_monitorable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_monitorable_encoded);
}

void PhysicsServer3D::area_set_ray_pickable(const RID &p_area, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("area_set_ray_pickable")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area, &p_enable_encoded);
}

RID PhysicsServer3D::body_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer3D::body_set_space(const RID &p_body, const RID &p_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_space")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_space);
}

RID PhysicsServer3D::body_get_space(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_space")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_mode(const RID &p_body, PhysicsServer3D::BodyMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_mode")._native_ptr(), 606803466);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_mode_encoded);
}

PhysicsServer3D::BodyMode PhysicsServer3D::body_get_mode(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_mode")._native_ptr(), 2488819728);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicsServer3D::BodyMode(0)));
	return (PhysicsServer3D::BodyMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_collision_layer(const RID &p_body, uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_collision_layer")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_layer_encoded);
}

uint32_t PhysicsServer3D::body_get_collision_layer(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_collision_layer")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_collision_mask(const RID &p_body, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_collision_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_mask_encoded);
}

uint32_t PhysicsServer3D::body_get_collision_mask(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_collision_mask")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_collision_priority(const RID &p_body, float p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_collision_priority")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_priority_encoded;
	PtrToArg<double>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_priority_encoded);
}

float PhysicsServer3D::body_get_collision_priority(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_collision_priority")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_add_shape(const RID &p_body, const RID &p_shape, const Transform3D &p_transform, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_add_shape")._native_ptr(), 3711419014);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape, &p_transform, &p_disabled_encoded);
}

void PhysicsServer3D::body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_shape")._native_ptr(), 2310537182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded, &p_shape);
}

void PhysicsServer3D::body_set_shape_transform(const RID &p_body, int32_t p_shape_idx, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_shape_transform")._native_ptr(), 675327471);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded, &p_transform);
}

void PhysicsServer3D::body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_shape_disabled")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded, &p_disabled_encoded);
}

int32_t PhysicsServer3D::body_get_shape_count(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_shape_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

RID PhysicsServer3D::body_get_shape(const RID &p_body, int32_t p_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_shape")._native_ptr(), 1066463050);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded);
}

Transform3D PhysicsServer3D::body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_shape_transform")._native_ptr(), 1050775521);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded);
}

void PhysicsServer3D::body_remove_shape(const RID &p_body, int32_t p_shape_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_remove_shape")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_shape_idx, &p_shape_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shape_idx_encoded);
}

void PhysicsServer3D::body_clear_shapes(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_clear_shapes")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_attach_object_instance_id(const RID &p_body, uint64_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_attach_object_instance_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_id_encoded);
}

uint64_t PhysicsServer3D::body_get_object_instance_id(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_object_instance_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_enable_continuous_collision_detection(const RID &p_body, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_enable_continuous_collision_detection")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_enable_encoded);
}

bool PhysicsServer3D::body_is_continuous_collision_detection_enabled(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_is_continuous_collision_detection_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_param")._native_ptr(), 910941953);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_param_encoded, &p_value);
}

Variant PhysicsServer3D::body_get_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_param")._native_ptr(), 3385027841);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_body, &p_param_encoded);
}

void PhysicsServer3D::body_reset_mass_properties(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_reset_mass_properties")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_state")._native_ptr(), 599977762);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_state_encoded, &p_value);
}

Variant PhysicsServer3D::body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_state")._native_ptr(), 1850449534);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_body, &p_state_encoded);
}

void PhysicsServer3D::body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_apply_central_impulse")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_impulse);
}

void PhysicsServer3D::body_apply_impulse(const RID &p_body, const Vector3 &p_impulse, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_apply_impulse")._native_ptr(), 390416203);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_impulse, &p_position);
}

void PhysicsServer3D::body_apply_torque_impulse(const RID &p_body, const Vector3 &p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_apply_torque_impulse")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_impulse);
}

void PhysicsServer3D::body_apply_central_force(const RID &p_body, const Vector3 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_apply_central_force")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force);
}

void PhysicsServer3D::body_apply_force(const RID &p_body, const Vector3 &p_force, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_apply_force")._native_ptr(), 390416203);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force, &p_position);
}

void PhysicsServer3D::body_apply_torque(const RID &p_body, const Vector3 &p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_apply_torque")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_torque);
}

void PhysicsServer3D::body_add_constant_central_force(const RID &p_body, const Vector3 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_add_constant_central_force")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force);
}

void PhysicsServer3D::body_add_constant_force(const RID &p_body, const Vector3 &p_force, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_add_constant_force")._native_ptr(), 390416203);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force, &p_position);
}

void PhysicsServer3D::body_add_constant_torque(const RID &p_body, const Vector3 &p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_add_constant_torque")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_torque);
}

void PhysicsServer3D::body_set_constant_force(const RID &p_body, const Vector3 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_constant_force")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force);
}

Vector3 PhysicsServer3D::body_get_constant_force(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_constant_force")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_constant_torque(const RID &p_body, const Vector3 &p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_constant_torque")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_torque);
}

Vector3 PhysicsServer3D::body_get_constant_torque(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_constant_torque")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_axis_velocity(const RID &p_body, const Vector3 &p_axis_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_axis_velocity")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_axis_velocity);
}

void PhysicsServer3D::body_set_axis_lock(const RID &p_body, PhysicsServer3D::BodyAxis p_axis, bool p_lock) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_axis_lock")._native_ptr(), 2020836892);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	int8_t p_lock_encoded;
	PtrToArg<bool>::encode(p_lock, &p_lock_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_axis_encoded, &p_lock_encoded);
}

bool PhysicsServer3D::body_is_axis_locked(const RID &p_body, PhysicsServer3D::BodyAxis p_axis) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_is_axis_locked")._native_ptr(), 587853580);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body, &p_axis_encoded);
}

void PhysicsServer3D::body_add_collision_exception(const RID &p_body, const RID &p_excepted_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_add_collision_exception")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_excepted_body);
}

void PhysicsServer3D::body_remove_collision_exception(const RID &p_body, const RID &p_excepted_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_remove_collision_exception")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_excepted_body);
}

void PhysicsServer3D::body_set_max_contacts_reported(const RID &p_body, int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_max_contacts_reported")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_amount_encoded);
}

int32_t PhysicsServer3D::body_get_max_contacts_reported(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_max_contacts_reported")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_omit_force_integration(const RID &p_body, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_omit_force_integration")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_enable_encoded);
}

bool PhysicsServer3D::body_is_omitting_force_integration(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_is_omitting_force_integration")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::body_set_state_sync_callback(const RID &p_body, const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_state_sync_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_callable);
}

void PhysicsServer3D::body_set_force_integration_callback(const RID &p_body, const Callable &p_callable, const Variant &p_userdata) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_force_integration_callback")._native_ptr(), 3059434249);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_callable, &p_userdata);
}

void PhysicsServer3D::body_set_ray_pickable(const RID &p_body, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_set_ray_pickable")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_enable_encoded);
}

bool PhysicsServer3D::body_test_motion(const RID &p_body, const Ref<PhysicsTestMotionParameters3D> &p_parameters, const Ref<PhysicsTestMotionResult3D> &p_result) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_test_motion")._native_ptr(), 1944921792);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body, (p_parameters != nullptr ? &p_parameters->_owner : nullptr), (p_result != nullptr ? &p_result->_owner : nullptr));
}

PhysicsDirectBodyState3D *PhysicsServer3D::body_get_direct_state(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("body_get_direct_state")._native_ptr(), 3029727957);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<PhysicsDirectBodyState3D>(_gde_method_bind, _owner, &p_body);
}

RID PhysicsServer3D::soft_body_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer3D::soft_body_update_rendering_server(const RID &p_body, PhysicsServer3DRenderingServerHandler *p_rendering_server_handler) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_update_rendering_server")._native_ptr(), 2218179753);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, (p_rendering_server_handler != nullptr ? &p_rendering_server_handler->_owner : nullptr));
}

void PhysicsServer3D::soft_body_set_space(const RID &p_body, const RID &p_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_space")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_space);
}

RID PhysicsServer3D::soft_body_get_space(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_space")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_mesh(const RID &p_body, const RID &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_mesh")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_mesh);
}

AABB PhysicsServer3D::soft_body_get_bounds(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_bounds")._native_ptr(), 974181306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_collision_layer(const RID &p_body, uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_collision_layer")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_layer_encoded);
}

uint32_t PhysicsServer3D::soft_body_get_collision_layer(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_collision_layer")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_collision_mask(const RID &p_body, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_collision_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_mask_encoded);
}

uint32_t PhysicsServer3D::soft_body_get_collision_mask(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_collision_mask")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_add_collision_exception(const RID &p_body, const RID &p_body_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_add_collision_exception")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_body_b);
}

void PhysicsServer3D::soft_body_remove_collision_exception(const RID &p_body, const RID &p_body_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_remove_collision_exception")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_body_b);
}

void PhysicsServer3D::soft_body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state, const Variant &p_variant) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_state")._native_ptr(), 599977762);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_state_encoded, &p_variant);
}

Variant PhysicsServer3D::soft_body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_state")._native_ptr(), 1850449534);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_body, &p_state_encoded);
}

void PhysicsServer3D::soft_body_set_transform(const RID &p_body, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_transform")._native_ptr(), 3935195649);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_transform);
}

void PhysicsServer3D::soft_body_set_ray_pickable(const RID &p_body, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_ray_pickable")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_enable_encoded);
}

void PhysicsServer3D::soft_body_set_simulation_precision(const RID &p_body, int32_t p_simulation_precision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_simulation_precision")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_simulation_precision_encoded;
	PtrToArg<int64_t>::encode(p_simulation_precision, &p_simulation_precision_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_simulation_precision_encoded);
}

int32_t PhysicsServer3D::soft_body_get_simulation_precision(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_simulation_precision")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_total_mass(const RID &p_body, float p_total_mass) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_total_mass")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_total_mass_encoded;
	PtrToArg<double>::encode(p_total_mass, &p_total_mass_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_total_mass_encoded);
}

float PhysicsServer3D::soft_body_get_total_mass(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_total_mass")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_linear_stiffness(const RID &p_body, float p_stiffness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_linear_stiffness")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_stiffness_encoded;
	PtrToArg<double>::encode(p_stiffness, &p_stiffness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_stiffness_encoded);
}

float PhysicsServer3D::soft_body_get_linear_stiffness(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_linear_stiffness")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_shrinking_factor(const RID &p_body, float p_shrinking_factor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_shrinking_factor")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_shrinking_factor_encoded;
	PtrToArg<double>::encode(p_shrinking_factor, &p_shrinking_factor_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_shrinking_factor_encoded);
}

float PhysicsServer3D::soft_body_get_shrinking_factor(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_shrinking_factor")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_pressure_coefficient(const RID &p_body, float p_pressure_coefficient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_pressure_coefficient")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pressure_coefficient_encoded;
	PtrToArg<double>::encode(p_pressure_coefficient, &p_pressure_coefficient_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_pressure_coefficient_encoded);
}

float PhysicsServer3D::soft_body_get_pressure_coefficient(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_pressure_coefficient")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_damping_coefficient(const RID &p_body, float p_damping_coefficient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_damping_coefficient")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_damping_coefficient_encoded;
	PtrToArg<double>::encode(p_damping_coefficient, &p_damping_coefficient_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_damping_coefficient_encoded);
}

float PhysicsServer3D::soft_body_get_damping_coefficient(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_damping_coefficient")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_set_drag_coefficient(const RID &p_body, float p_drag_coefficient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_set_drag_coefficient")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_drag_coefficient_encoded;
	PtrToArg<double>::encode(p_drag_coefficient, &p_drag_coefficient_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_drag_coefficient_encoded);
}

float PhysicsServer3D::soft_body_get_drag_coefficient(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_drag_coefficient")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_move_point(const RID &p_body, int32_t p_point_index, const Vector3 &p_global_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_move_point")._native_ptr(), 831953689);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_point_index_encoded;
	PtrToArg<int64_t>::encode(p_point_index, &p_point_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_point_index_encoded, &p_global_position);
}

Vector3 PhysicsServer3D::soft_body_get_point_global_position(const RID &p_body, int32_t p_point_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_get_point_global_position")._native_ptr(), 3440143363);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_point_index_encoded;
	PtrToArg<int64_t>::encode(p_point_index, &p_point_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_body, &p_point_index_encoded);
}

void PhysicsServer3D::soft_body_remove_all_pinned_points(const RID &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_remove_all_pinned_points")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body);
}

void PhysicsServer3D::soft_body_pin_point(const RID &p_body, int32_t p_point_index, bool p_pin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_pin_point")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_point_index_encoded;
	PtrToArg<int64_t>::encode(p_point_index, &p_point_index_encoded);
	int8_t p_pin_encoded;
	PtrToArg<bool>::encode(p_pin, &p_pin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_point_index_encoded, &p_pin_encoded);
}

bool PhysicsServer3D::soft_body_is_point_pinned(const RID &p_body, int32_t p_point_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_is_point_pinned")._native_ptr(), 3120086654);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_point_index_encoded;
	PtrToArg<int64_t>::encode(p_point_index, &p_point_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body, &p_point_index_encoded);
}

void PhysicsServer3D::soft_body_apply_point_impulse(const RID &p_body, int32_t p_point_index, const Vector3 &p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_apply_point_impulse")._native_ptr(), 831953689);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_point_index_encoded;
	PtrToArg<int64_t>::encode(p_point_index, &p_point_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_point_index_encoded, &p_impulse);
}

void PhysicsServer3D::soft_body_apply_point_force(const RID &p_body, int32_t p_point_index, const Vector3 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_apply_point_force")._native_ptr(), 831953689);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_point_index_encoded;
	PtrToArg<int64_t>::encode(p_point_index, &p_point_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_point_index_encoded, &p_force);
}

void PhysicsServer3D::soft_body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_apply_central_impulse")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_impulse);
}

void PhysicsServer3D::soft_body_apply_central_force(const RID &p_body, const Vector3 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("soft_body_apply_central_force")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body, &p_force);
}

RID PhysicsServer3D::joint_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void PhysicsServer3D::joint_clear(const RID &p_joint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_clear")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer3D::joint_make_pin(const RID &p_joint, const RID &p_body_A, const Vector3 &p_local_A, const RID &p_body_B, const Vector3 &p_local_B) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_make_pin")._native_ptr(), 4280171926);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_body_A, &p_local_A, &p_body_B, &p_local_B);
}

void PhysicsServer3D::pin_joint_set_param(const RID &p_joint, PhysicsServer3D::PinJointParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("pin_joint_set_param")._native_ptr(), 810685294);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer3D::pin_joint_get_param(const RID &p_joint, PhysicsServer3D::PinJointParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("pin_joint_get_param")._native_ptr(), 2817972347);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint, &p_param_encoded);
}

void PhysicsServer3D::pin_joint_set_local_a(const RID &p_joint, const Vector3 &p_local_A) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("pin_joint_set_local_a")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_local_A);
}

Vector3 PhysicsServer3D::pin_joint_get_local_a(const RID &p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("pin_joint_get_local_a")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer3D::pin_joint_set_local_b(const RID &p_joint, const Vector3 &p_local_B) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("pin_joint_set_local_b")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_local_B);
}

Vector3 PhysicsServer3D::pin_joint_get_local_b(const RID &p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("pin_joint_get_local_b")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer3D::joint_make_hinge(const RID &p_joint, const RID &p_body_A, const Transform3D &p_hinge_A, const RID &p_body_B, const Transform3D &p_hinge_B) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_make_hinge")._native_ptr(), 1684107643);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_body_A, &p_hinge_A, &p_body_B, &p_hinge_B);
}

void PhysicsServer3D::hinge_joint_set_param(const RID &p_joint, PhysicsServer3D::HingeJointParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("hinge_joint_set_param")._native_ptr(), 3165502333);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer3D::hinge_joint_get_param(const RID &p_joint, PhysicsServer3D::HingeJointParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("hinge_joint_get_param")._native_ptr(), 2129207581);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint, &p_param_encoded);
}

void PhysicsServer3D::hinge_joint_set_flag(const RID &p_joint, PhysicsServer3D::HingeJointFlag p_flag, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("hinge_joint_set_flag")._native_ptr(), 1601626188);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_flag_encoded, &p_enabled_encoded);
}

bool PhysicsServer3D::hinge_joint_get_flag(const RID &p_joint, PhysicsServer3D::HingeJointFlag p_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("hinge_joint_get_flag")._native_ptr(), 4165147865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_joint, &p_flag_encoded);
}

void PhysicsServer3D::joint_make_slider(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_make_slider")._native_ptr(), 1684107643);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_body_A, &p_local_ref_A, &p_body_B, &p_local_ref_B);
}

void PhysicsServer3D::slider_joint_set_param(const RID &p_joint, PhysicsServer3D::SliderJointParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("slider_joint_set_param")._native_ptr(), 2264833593);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer3D::slider_joint_get_param(const RID &p_joint, PhysicsServer3D::SliderJointParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("slider_joint_get_param")._native_ptr(), 3498644957);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint, &p_param_encoded);
}

void PhysicsServer3D::joint_make_cone_twist(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_make_cone_twist")._native_ptr(), 1684107643);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_body_A, &p_local_ref_A, &p_body_B, &p_local_ref_B);
}

void PhysicsServer3D::cone_twist_joint_set_param(const RID &p_joint, PhysicsServer3D::ConeTwistJointParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("cone_twist_joint_set_param")._native_ptr(), 808587618);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer3D::cone_twist_joint_get_param(const RID &p_joint, PhysicsServer3D::ConeTwistJointParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("cone_twist_joint_get_param")._native_ptr(), 1134789658);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint, &p_param_encoded);
}

PhysicsServer3D::JointType PhysicsServer3D::joint_get_type(const RID &p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_get_type")._native_ptr(), 4290791900);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicsServer3D::JointType(0)));
	return (PhysicsServer3D::JointType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer3D::joint_set_solver_priority(const RID &p_joint, int32_t p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_set_solver_priority")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_encoded;
	PtrToArg<int64_t>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_priority_encoded);
}

int32_t PhysicsServer3D::joint_get_solver_priority(const RID &p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_get_solver_priority")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer3D::joint_disable_collisions_between_bodies(const RID &p_joint, bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_disable_collisions_between_bodies")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_disable_encoded);
}

bool PhysicsServer3D::joint_is_disabled_collisions_between_bodies(const RID &p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_is_disabled_collisions_between_bodies")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_joint);
}

void PhysicsServer3D::joint_make_generic_6dof(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("joint_make_generic_6dof")._native_ptr(), 1684107643);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_body_A, &p_local_ref_A, &p_body_B, &p_local_ref_B);
}

void PhysicsServer3D::generic_6dof_joint_set_param(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("generic_6dof_joint_set_param")._native_ptr(), 2600081391);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_axis_encoded, &p_param_encoded, &p_value_encoded);
}

float PhysicsServer3D::generic_6dof_joint_get_param(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("generic_6dof_joint_get_param")._native_ptr(), 467122058);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_joint, &p_axis_encoded, &p_param_encoded);
}

void PhysicsServer3D::generic_6dof_joint_set_flag(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("generic_6dof_joint_set_flag")._native_ptr(), 3570926903);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint, &p_axis_encoded, &p_flag_encoded, &p_enable_encoded);
}

bool PhysicsServer3D::generic_6dof_joint_get_flag(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("generic_6dof_joint_get_flag")._native_ptr(), 4158090196);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_joint, &p_axis_encoded, &p_flag_encoded);
}

void PhysicsServer3D::free_rid(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("free_rid")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

void PhysicsServer3D::set_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("set_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

int32_t PhysicsServer3D::get_process_info(PhysicsServer3D::ProcessInfo p_process_info) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3D::get_class_static()._native_ptr(), StringName("get_process_info")._native_ptr(), 1332958745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_process_info_encoded;
	PtrToArg<int64_t>::encode(p_process_info, &p_process_info_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_process_info_encoded);
}

} // namespace godot
