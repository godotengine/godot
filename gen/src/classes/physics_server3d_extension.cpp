/**************************************************************************/
/*  physics_server3d_extension.cpp                                        */
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

#include <godot_cpp/classes/physics_server3d_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_direct_body_state3d.hpp>
#include <godot_cpp/classes/physics_direct_space_state3d.hpp>
#include <godot_cpp/classes/physics_server3d_rendering_server_handler.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

bool PhysicsServer3DExtension::body_test_motion_is_excluding_body(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3DExtension::get_class_static()._native_ptr(), StringName("body_test_motion_is_excluding_body")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body);
}

bool PhysicsServer3DExtension::body_test_motion_is_excluding_object(uint64_t p_object) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer3DExtension::get_class_static()._native_ptr(), StringName("body_test_motion_is_excluding_object")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_object_encoded;
	PtrToArg<int64_t>::encode(p_object, &p_object_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_object_encoded);
}

RID PhysicsServer3DExtension::_world_boundary_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_separation_ray_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_sphere_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_box_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_capsule_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_cylinder_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_convex_polygon_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_concave_polygon_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_heightmap_shape_create() {
	return RID();
}

RID PhysicsServer3DExtension::_custom_shape_create() {
	return RID();
}

void PhysicsServer3DExtension::_shape_set_data(const RID &p_shape, const Variant &p_data) {}

void PhysicsServer3DExtension::_shape_set_custom_solver_bias(const RID &p_shape, float p_bias) {}

void PhysicsServer3DExtension::_shape_set_margin(const RID &p_shape, float p_margin) {}

float PhysicsServer3DExtension::_shape_get_margin(const RID &p_shape) const {
	return 0.0;
}

PhysicsServer3D::ShapeType PhysicsServer3DExtension::_shape_get_type(const RID &p_shape) const {
	return PhysicsServer3D::ShapeType(0);
}

Variant PhysicsServer3DExtension::_shape_get_data(const RID &p_shape) const {
	return Variant();
}

float PhysicsServer3DExtension::_shape_get_custom_solver_bias(const RID &p_shape) const {
	return 0.0;
}

RID PhysicsServer3DExtension::_space_create() {
	return RID();
}

void PhysicsServer3DExtension::_space_set_active(const RID &p_space, bool p_active) {}

bool PhysicsServer3DExtension::_space_is_active(const RID &p_space) const {
	return false;
}

void PhysicsServer3DExtension::_space_set_param(const RID &p_space, PhysicsServer3D::SpaceParameter p_param, float p_value) {}

float PhysicsServer3DExtension::_space_get_param(const RID &p_space, PhysicsServer3D::SpaceParameter p_param) const {
	return 0.0;
}

PhysicsDirectSpaceState3D *PhysicsServer3DExtension::_space_get_direct_state(const RID &p_space) {
	return nullptr;
}

void PhysicsServer3DExtension::_space_set_debug_contacts(const RID &p_space, int32_t p_max_contacts) {}

PackedVector3Array PhysicsServer3DExtension::_space_get_contacts(const RID &p_space) const {
	return PackedVector3Array();
}

int32_t PhysicsServer3DExtension::_space_get_contact_count(const RID &p_space) const {
	return 0;
}

RID PhysicsServer3DExtension::_area_create() {
	return RID();
}

void PhysicsServer3DExtension::_area_set_space(const RID &p_area, const RID &p_space) {}

RID PhysicsServer3DExtension::_area_get_space(const RID &p_area) const {
	return RID();
}

void PhysicsServer3DExtension::_area_add_shape(const RID &p_area, const RID &p_shape, const Transform3D &p_transform, bool p_disabled) {}

void PhysicsServer3DExtension::_area_set_shape(const RID &p_area, int32_t p_shape_idx, const RID &p_shape) {}

void PhysicsServer3DExtension::_area_set_shape_transform(const RID &p_area, int32_t p_shape_idx, const Transform3D &p_transform) {}

void PhysicsServer3DExtension::_area_set_shape_disabled(const RID &p_area, int32_t p_shape_idx, bool p_disabled) {}

int32_t PhysicsServer3DExtension::_area_get_shape_count(const RID &p_area) const {
	return 0;
}

RID PhysicsServer3DExtension::_area_get_shape(const RID &p_area, int32_t p_shape_idx) const {
	return RID();
}

Transform3D PhysicsServer3DExtension::_area_get_shape_transform(const RID &p_area, int32_t p_shape_idx) const {
	return Transform3D();
}

void PhysicsServer3DExtension::_area_remove_shape(const RID &p_area, int32_t p_shape_idx) {}

void PhysicsServer3DExtension::_area_clear_shapes(const RID &p_area) {}

void PhysicsServer3DExtension::_area_attach_object_instance_id(const RID &p_area, uint64_t p_id) {}

uint64_t PhysicsServer3DExtension::_area_get_object_instance_id(const RID &p_area) const {
	return 0;
}

void PhysicsServer3DExtension::_area_set_param(const RID &p_area, PhysicsServer3D::AreaParameter p_param, const Variant &p_value) {}

void PhysicsServer3DExtension::_area_set_transform(const RID &p_area, const Transform3D &p_transform) {}

Variant PhysicsServer3DExtension::_area_get_param(const RID &p_area, PhysicsServer3D::AreaParameter p_param) const {
	return Variant();
}

Transform3D PhysicsServer3DExtension::_area_get_transform(const RID &p_area) const {
	return Transform3D();
}

void PhysicsServer3DExtension::_area_set_collision_layer(const RID &p_area, uint32_t p_layer) {}

uint32_t PhysicsServer3DExtension::_area_get_collision_layer(const RID &p_area) const {
	return 0;
}

void PhysicsServer3DExtension::_area_set_collision_mask(const RID &p_area, uint32_t p_mask) {}

uint32_t PhysicsServer3DExtension::_area_get_collision_mask(const RID &p_area) const {
	return 0;
}

void PhysicsServer3DExtension::_area_set_monitorable(const RID &p_area, bool p_monitorable) {}

void PhysicsServer3DExtension::_area_set_ray_pickable(const RID &p_area, bool p_enable) {}

void PhysicsServer3DExtension::_area_set_monitor_callback(const RID &p_area, const Callable &p_callback) {}

void PhysicsServer3DExtension::_area_set_area_monitor_callback(const RID &p_area, const Callable &p_callback) {}

RID PhysicsServer3DExtension::_body_create() {
	return RID();
}

void PhysicsServer3DExtension::_body_set_space(const RID &p_body, const RID &p_space) {}

RID PhysicsServer3DExtension::_body_get_space(const RID &p_body) const {
	return RID();
}

void PhysicsServer3DExtension::_body_set_mode(const RID &p_body, PhysicsServer3D::BodyMode p_mode) {}

PhysicsServer3D::BodyMode PhysicsServer3DExtension::_body_get_mode(const RID &p_body) const {
	return PhysicsServer3D::BodyMode(0);
}

void PhysicsServer3DExtension::_body_add_shape(const RID &p_body, const RID &p_shape, const Transform3D &p_transform, bool p_disabled) {}

void PhysicsServer3DExtension::_body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape) {}

void PhysicsServer3DExtension::_body_set_shape_transform(const RID &p_body, int32_t p_shape_idx, const Transform3D &p_transform) {}

void PhysicsServer3DExtension::_body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled) {}

int32_t PhysicsServer3DExtension::_body_get_shape_count(const RID &p_body) const {
	return 0;
}

RID PhysicsServer3DExtension::_body_get_shape(const RID &p_body, int32_t p_shape_idx) const {
	return RID();
}

Transform3D PhysicsServer3DExtension::_body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const {
	return Transform3D();
}

void PhysicsServer3DExtension::_body_remove_shape(const RID &p_body, int32_t p_shape_idx) {}

void PhysicsServer3DExtension::_body_clear_shapes(const RID &p_body) {}

void PhysicsServer3DExtension::_body_attach_object_instance_id(const RID &p_body, uint64_t p_id) {}

uint64_t PhysicsServer3DExtension::_body_get_object_instance_id(const RID &p_body) const {
	return 0;
}

void PhysicsServer3DExtension::_body_set_enable_continuous_collision_detection(const RID &p_body, bool p_enable) {}

bool PhysicsServer3DExtension::_body_is_continuous_collision_detection_enabled(const RID &p_body) const {
	return false;
}

void PhysicsServer3DExtension::_body_set_collision_layer(const RID &p_body, uint32_t p_layer) {}

uint32_t PhysicsServer3DExtension::_body_get_collision_layer(const RID &p_body) const {
	return 0;
}

void PhysicsServer3DExtension::_body_set_collision_mask(const RID &p_body, uint32_t p_mask) {}

uint32_t PhysicsServer3DExtension::_body_get_collision_mask(const RID &p_body) const {
	return 0;
}

void PhysicsServer3DExtension::_body_set_collision_priority(const RID &p_body, float p_priority) {}

float PhysicsServer3DExtension::_body_get_collision_priority(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer3DExtension::_body_set_user_flags(const RID &p_body, uint32_t p_flags) {}

uint32_t PhysicsServer3DExtension::_body_get_user_flags(const RID &p_body) const {
	return 0;
}

void PhysicsServer3DExtension::_body_set_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param, const Variant &p_value) {}

Variant PhysicsServer3DExtension::_body_get_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param) const {
	return Variant();
}

void PhysicsServer3DExtension::_body_reset_mass_properties(const RID &p_body) {}

void PhysicsServer3DExtension::_body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state, const Variant &p_value) {}

Variant PhysicsServer3DExtension::_body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const {
	return Variant();
}

void PhysicsServer3DExtension::_body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse) {}

void PhysicsServer3DExtension::_body_apply_impulse(const RID &p_body, const Vector3 &p_impulse, const Vector3 &p_position) {}

void PhysicsServer3DExtension::_body_apply_torque_impulse(const RID &p_body, const Vector3 &p_impulse) {}

void PhysicsServer3DExtension::_body_apply_central_force(const RID &p_body, const Vector3 &p_force) {}

void PhysicsServer3DExtension::_body_apply_force(const RID &p_body, const Vector3 &p_force, const Vector3 &p_position) {}

void PhysicsServer3DExtension::_body_apply_torque(const RID &p_body, const Vector3 &p_torque) {}

void PhysicsServer3DExtension::_body_add_constant_central_force(const RID &p_body, const Vector3 &p_force) {}

void PhysicsServer3DExtension::_body_add_constant_force(const RID &p_body, const Vector3 &p_force, const Vector3 &p_position) {}

void PhysicsServer3DExtension::_body_add_constant_torque(const RID &p_body, const Vector3 &p_torque) {}

void PhysicsServer3DExtension::_body_set_constant_force(const RID &p_body, const Vector3 &p_force) {}

Vector3 PhysicsServer3DExtension::_body_get_constant_force(const RID &p_body) const {
	return Vector3();
}

void PhysicsServer3DExtension::_body_set_constant_torque(const RID &p_body, const Vector3 &p_torque) {}

Vector3 PhysicsServer3DExtension::_body_get_constant_torque(const RID &p_body) const {
	return Vector3();
}

void PhysicsServer3DExtension::_body_set_axis_velocity(const RID &p_body, const Vector3 &p_axis_velocity) {}

void PhysicsServer3DExtension::_body_set_axis_lock(const RID &p_body, PhysicsServer3D::BodyAxis p_axis, bool p_lock) {}

bool PhysicsServer3DExtension::_body_is_axis_locked(const RID &p_body, PhysicsServer3D::BodyAxis p_axis) const {
	return false;
}

void PhysicsServer3DExtension::_body_add_collision_exception(const RID &p_body, const RID &p_excepted_body) {}

void PhysicsServer3DExtension::_body_remove_collision_exception(const RID &p_body, const RID &p_excepted_body) {}

TypedArray<RID> PhysicsServer3DExtension::_body_get_collision_exceptions(const RID &p_body) const {
	return TypedArray<RID>();
}

void PhysicsServer3DExtension::_body_set_max_contacts_reported(const RID &p_body, int32_t p_amount) {}

int32_t PhysicsServer3DExtension::_body_get_max_contacts_reported(const RID &p_body) const {
	return 0;
}

void PhysicsServer3DExtension::_body_set_contacts_reported_depth_threshold(const RID &p_body, float p_threshold) {}

float PhysicsServer3DExtension::_body_get_contacts_reported_depth_threshold(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer3DExtension::_body_set_omit_force_integration(const RID &p_body, bool p_enable) {}

bool PhysicsServer3DExtension::_body_is_omitting_force_integration(const RID &p_body) const {
	return false;
}

void PhysicsServer3DExtension::_body_set_state_sync_callback(const RID &p_body, const Callable &p_callable) {}

void PhysicsServer3DExtension::_body_set_force_integration_callback(const RID &p_body, const Callable &p_callable, const Variant &p_userdata) {}

void PhysicsServer3DExtension::_body_set_ray_pickable(const RID &p_body, bool p_enable) {}

bool PhysicsServer3DExtension::_body_test_motion(const RID &p_body, const Transform3D &p_from, const Vector3 &p_motion, float p_margin, int32_t p_max_collisions, bool p_collide_separation_ray, bool p_recovery_as_collision, PhysicsServer3DExtensionMotionResult *p_result) const {
	return false;
}

PhysicsDirectBodyState3D *PhysicsServer3DExtension::_body_get_direct_state(const RID &p_body) {
	return nullptr;
}

RID PhysicsServer3DExtension::_soft_body_create() {
	return RID();
}

void PhysicsServer3DExtension::_soft_body_update_rendering_server(const RID &p_body, PhysicsServer3DRenderingServerHandler *p_rendering_server_handler) {}

void PhysicsServer3DExtension::_soft_body_set_space(const RID &p_body, const RID &p_space) {}

RID PhysicsServer3DExtension::_soft_body_get_space(const RID &p_body) const {
	return RID();
}

void PhysicsServer3DExtension::_soft_body_set_ray_pickable(const RID &p_body, bool p_enable) {}

void PhysicsServer3DExtension::_soft_body_set_collision_layer(const RID &p_body, uint32_t p_layer) {}

uint32_t PhysicsServer3DExtension::_soft_body_get_collision_layer(const RID &p_body) const {
	return 0;
}

void PhysicsServer3DExtension::_soft_body_set_collision_mask(const RID &p_body, uint32_t p_mask) {}

uint32_t PhysicsServer3DExtension::_soft_body_get_collision_mask(const RID &p_body) const {
	return 0;
}

void PhysicsServer3DExtension::_soft_body_add_collision_exception(const RID &p_body, const RID &p_body_b) {}

void PhysicsServer3DExtension::_soft_body_remove_collision_exception(const RID &p_body, const RID &p_body_b) {}

TypedArray<RID> PhysicsServer3DExtension::_soft_body_get_collision_exceptions(const RID &p_body) const {
	return TypedArray<RID>();
}

void PhysicsServer3DExtension::_soft_body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state, const Variant &p_variant) {}

Variant PhysicsServer3DExtension::_soft_body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const {
	return Variant();
}

void PhysicsServer3DExtension::_soft_body_set_transform(const RID &p_body, const Transform3D &p_transform) {}

void PhysicsServer3DExtension::_soft_body_set_simulation_precision(const RID &p_body, int32_t p_simulation_precision) {}

int32_t PhysicsServer3DExtension::_soft_body_get_simulation_precision(const RID &p_body) const {
	return 0;
}

void PhysicsServer3DExtension::_soft_body_set_total_mass(const RID &p_body, float p_total_mass) {}

float PhysicsServer3DExtension::_soft_body_get_total_mass(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer3DExtension::_soft_body_set_linear_stiffness(const RID &p_body, float p_linear_stiffness) {}

float PhysicsServer3DExtension::_soft_body_get_linear_stiffness(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer3DExtension::_soft_body_set_shrinking_factor(const RID &p_body, float p_shrinking_factor) {}

float PhysicsServer3DExtension::_soft_body_get_shrinking_factor(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer3DExtension::_soft_body_set_pressure_coefficient(const RID &p_body, float p_pressure_coefficient) {}

float PhysicsServer3DExtension::_soft_body_get_pressure_coefficient(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer3DExtension::_soft_body_set_damping_coefficient(const RID &p_body, float p_damping_coefficient) {}

float PhysicsServer3DExtension::_soft_body_get_damping_coefficient(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer3DExtension::_soft_body_set_drag_coefficient(const RID &p_body, float p_drag_coefficient) {}

float PhysicsServer3DExtension::_soft_body_get_drag_coefficient(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer3DExtension::_soft_body_set_mesh(const RID &p_body, const RID &p_mesh) {}

AABB PhysicsServer3DExtension::_soft_body_get_bounds(const RID &p_body) const {
	return AABB();
}

void PhysicsServer3DExtension::_soft_body_move_point(const RID &p_body, int32_t p_point_index, const Vector3 &p_global_position) {}

Vector3 PhysicsServer3DExtension::_soft_body_get_point_global_position(const RID &p_body, int32_t p_point_index) const {
	return Vector3();
}

void PhysicsServer3DExtension::_soft_body_remove_all_pinned_points(const RID &p_body) {}

void PhysicsServer3DExtension::_soft_body_pin_point(const RID &p_body, int32_t p_point_index, bool p_pin) {}

bool PhysicsServer3DExtension::_soft_body_is_point_pinned(const RID &p_body, int32_t p_point_index) const {
	return false;
}

void PhysicsServer3DExtension::_soft_body_apply_point_impulse(const RID &p_body, int32_t p_point_index, const Vector3 &p_impulse) {}

void PhysicsServer3DExtension::_soft_body_apply_point_force(const RID &p_body, int32_t p_point_index, const Vector3 &p_force) {}

void PhysicsServer3DExtension::_soft_body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse) {}

void PhysicsServer3DExtension::_soft_body_apply_central_force(const RID &p_body, const Vector3 &p_force) {}

RID PhysicsServer3DExtension::_joint_create() {
	return RID();
}

void PhysicsServer3DExtension::_joint_clear(const RID &p_joint) {}

void PhysicsServer3DExtension::_joint_make_pin(const RID &p_joint, const RID &p_body_A, const Vector3 &p_local_A, const RID &p_body_B, const Vector3 &p_local_B) {}

void PhysicsServer3DExtension::_pin_joint_set_param(const RID &p_joint, PhysicsServer3D::PinJointParam p_param, float p_value) {}

float PhysicsServer3DExtension::_pin_joint_get_param(const RID &p_joint, PhysicsServer3D::PinJointParam p_param) const {
	return 0.0;
}

void PhysicsServer3DExtension::_pin_joint_set_local_a(const RID &p_joint, const Vector3 &p_local_A) {}

Vector3 PhysicsServer3DExtension::_pin_joint_get_local_a(const RID &p_joint) const {
	return Vector3();
}

void PhysicsServer3DExtension::_pin_joint_set_local_b(const RID &p_joint, const Vector3 &p_local_B) {}

Vector3 PhysicsServer3DExtension::_pin_joint_get_local_b(const RID &p_joint) const {
	return Vector3();
}

void PhysicsServer3DExtension::_joint_make_hinge(const RID &p_joint, const RID &p_body_A, const Transform3D &p_hinge_A, const RID &p_body_B, const Transform3D &p_hinge_B) {}

void PhysicsServer3DExtension::_joint_make_hinge_simple(const RID &p_joint, const RID &p_body_A, const Vector3 &p_pivot_A, const Vector3 &p_axis_A, const RID &p_body_B, const Vector3 &p_pivot_B, const Vector3 &p_axis_B) {}

void PhysicsServer3DExtension::_hinge_joint_set_param(const RID &p_joint, PhysicsServer3D::HingeJointParam p_param, float p_value) {}

float PhysicsServer3DExtension::_hinge_joint_get_param(const RID &p_joint, PhysicsServer3D::HingeJointParam p_param) const {
	return 0.0;
}

void PhysicsServer3DExtension::_hinge_joint_set_flag(const RID &p_joint, PhysicsServer3D::HingeJointFlag p_flag, bool p_enabled) {}

bool PhysicsServer3DExtension::_hinge_joint_get_flag(const RID &p_joint, PhysicsServer3D::HingeJointFlag p_flag) const {
	return false;
}

void PhysicsServer3DExtension::_joint_make_slider(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B) {}

void PhysicsServer3DExtension::_slider_joint_set_param(const RID &p_joint, PhysicsServer3D::SliderJointParam p_param, float p_value) {}

float PhysicsServer3DExtension::_slider_joint_get_param(const RID &p_joint, PhysicsServer3D::SliderJointParam p_param) const {
	return 0.0;
}

void PhysicsServer3DExtension::_joint_make_cone_twist(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B) {}

void PhysicsServer3DExtension::_cone_twist_joint_set_param(const RID &p_joint, PhysicsServer3D::ConeTwistJointParam p_param, float p_value) {}

float PhysicsServer3DExtension::_cone_twist_joint_get_param(const RID &p_joint, PhysicsServer3D::ConeTwistJointParam p_param) const {
	return 0.0;
}

void PhysicsServer3DExtension::_joint_make_generic_6dof(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B) {}

void PhysicsServer3DExtension::_generic_6dof_joint_set_param(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param, float p_value) {}

float PhysicsServer3DExtension::_generic_6dof_joint_get_param(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param) const {
	return 0.0;
}

void PhysicsServer3DExtension::_generic_6dof_joint_set_flag(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag, bool p_enable) {}

bool PhysicsServer3DExtension::_generic_6dof_joint_get_flag(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag) const {
	return false;
}

PhysicsServer3D::JointType PhysicsServer3DExtension::_joint_get_type(const RID &p_joint) const {
	return PhysicsServer3D::JointType(0);
}

void PhysicsServer3DExtension::_joint_set_solver_priority(const RID &p_joint, int32_t p_priority) {}

int32_t PhysicsServer3DExtension::_joint_get_solver_priority(const RID &p_joint) const {
	return 0;
}

void PhysicsServer3DExtension::_joint_disable_collisions_between_bodies(const RID &p_joint, bool p_disable) {}

bool PhysicsServer3DExtension::_joint_is_disabled_collisions_between_bodies(const RID &p_joint) const {
	return false;
}

void PhysicsServer3DExtension::_free_rid(const RID &p_rid) {}

void PhysicsServer3DExtension::_set_active(bool p_active) {}

void PhysicsServer3DExtension::_init() {}

void PhysicsServer3DExtension::_step(float p_step) {}

void PhysicsServer3DExtension::_sync() {}

void PhysicsServer3DExtension::_flush_queries() {}

void PhysicsServer3DExtension::_end_sync() {}

void PhysicsServer3DExtension::_finish() {}

bool PhysicsServer3DExtension::_is_flushing_queries() const {
	return false;
}

int32_t PhysicsServer3DExtension::_get_process_info(PhysicsServer3D::ProcessInfo p_process_info) {
	return 0;
}

} // namespace godot
