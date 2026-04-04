/**************************************************************************/
/*  physics_server2d_extension.cpp                                        */
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

#include <godot_cpp/classes/physics_server2d_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_direct_body_state2d.hpp>
#include <godot_cpp/classes/physics_direct_space_state2d.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

bool PhysicsServer2DExtension::body_test_motion_is_excluding_body(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2DExtension::get_class_static()._native_ptr(), StringName("body_test_motion_is_excluding_body")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body);
}

bool PhysicsServer2DExtension::body_test_motion_is_excluding_object(uint64_t p_object) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsServer2DExtension::get_class_static()._native_ptr(), StringName("body_test_motion_is_excluding_object")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_object_encoded;
	PtrToArg<int64_t>::encode(p_object, &p_object_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_object_encoded);
}

RID PhysicsServer2DExtension::_world_boundary_shape_create() {
	return RID();
}

RID PhysicsServer2DExtension::_separation_ray_shape_create() {
	return RID();
}

RID PhysicsServer2DExtension::_segment_shape_create() {
	return RID();
}

RID PhysicsServer2DExtension::_circle_shape_create() {
	return RID();
}

RID PhysicsServer2DExtension::_rectangle_shape_create() {
	return RID();
}

RID PhysicsServer2DExtension::_capsule_shape_create() {
	return RID();
}

RID PhysicsServer2DExtension::_convex_polygon_shape_create() {
	return RID();
}

RID PhysicsServer2DExtension::_concave_polygon_shape_create() {
	return RID();
}

void PhysicsServer2DExtension::_shape_set_data(const RID &p_shape, const Variant &p_data) {}

void PhysicsServer2DExtension::_shape_set_custom_solver_bias(const RID &p_shape, float p_bias) {}

PhysicsServer2D::ShapeType PhysicsServer2DExtension::_shape_get_type(const RID &p_shape) const {
	return PhysicsServer2D::ShapeType(0);
}

Variant PhysicsServer2DExtension::_shape_get_data(const RID &p_shape) const {
	return Variant();
}

float PhysicsServer2DExtension::_shape_get_custom_solver_bias(const RID &p_shape) const {
	return 0.0;
}

bool PhysicsServer2DExtension::_shape_collide(const RID &p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, const RID &p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, void *p_results, int32_t p_result_max, int32_t *p_result_count) {
	return false;
}

RID PhysicsServer2DExtension::_space_create() {
	return RID();
}

void PhysicsServer2DExtension::_space_set_active(const RID &p_space, bool p_active) {}

bool PhysicsServer2DExtension::_space_is_active(const RID &p_space) const {
	return false;
}

void PhysicsServer2DExtension::_space_set_param(const RID &p_space, PhysicsServer2D::SpaceParameter p_param, float p_value) {}

float PhysicsServer2DExtension::_space_get_param(const RID &p_space, PhysicsServer2D::SpaceParameter p_param) const {
	return 0.0;
}

PhysicsDirectSpaceState2D *PhysicsServer2DExtension::_space_get_direct_state(const RID &p_space) {
	return nullptr;
}

void PhysicsServer2DExtension::_space_set_debug_contacts(const RID &p_space, int32_t p_max_contacts) {}

PackedVector2Array PhysicsServer2DExtension::_space_get_contacts(const RID &p_space) const {
	return PackedVector2Array();
}

int32_t PhysicsServer2DExtension::_space_get_contact_count(const RID &p_space) const {
	return 0;
}

RID PhysicsServer2DExtension::_area_create() {
	return RID();
}

void PhysicsServer2DExtension::_area_set_space(const RID &p_area, const RID &p_space) {}

RID PhysicsServer2DExtension::_area_get_space(const RID &p_area) const {
	return RID();
}

void PhysicsServer2DExtension::_area_add_shape(const RID &p_area, const RID &p_shape, const Transform2D &p_transform, bool p_disabled) {}

void PhysicsServer2DExtension::_area_set_shape(const RID &p_area, int32_t p_shape_idx, const RID &p_shape) {}

void PhysicsServer2DExtension::_area_set_shape_transform(const RID &p_area, int32_t p_shape_idx, const Transform2D &p_transform) {}

void PhysicsServer2DExtension::_area_set_shape_disabled(const RID &p_area, int32_t p_shape_idx, bool p_disabled) {}

int32_t PhysicsServer2DExtension::_area_get_shape_count(const RID &p_area) const {
	return 0;
}

RID PhysicsServer2DExtension::_area_get_shape(const RID &p_area, int32_t p_shape_idx) const {
	return RID();
}

Transform2D PhysicsServer2DExtension::_area_get_shape_transform(const RID &p_area, int32_t p_shape_idx) const {
	return Transform2D();
}

void PhysicsServer2DExtension::_area_remove_shape(const RID &p_area, int32_t p_shape_idx) {}

void PhysicsServer2DExtension::_area_clear_shapes(const RID &p_area) {}

void PhysicsServer2DExtension::_area_attach_object_instance_id(const RID &p_area, uint64_t p_id) {}

uint64_t PhysicsServer2DExtension::_area_get_object_instance_id(const RID &p_area) const {
	return 0;
}

void PhysicsServer2DExtension::_area_attach_canvas_instance_id(const RID &p_area, uint64_t p_id) {}

uint64_t PhysicsServer2DExtension::_area_get_canvas_instance_id(const RID &p_area) const {
	return 0;
}

void PhysicsServer2DExtension::_area_set_param(const RID &p_area, PhysicsServer2D::AreaParameter p_param, const Variant &p_value) {}

void PhysicsServer2DExtension::_area_set_transform(const RID &p_area, const Transform2D &p_transform) {}

Variant PhysicsServer2DExtension::_area_get_param(const RID &p_area, PhysicsServer2D::AreaParameter p_param) const {
	return Variant();
}

Transform2D PhysicsServer2DExtension::_area_get_transform(const RID &p_area) const {
	return Transform2D();
}

void PhysicsServer2DExtension::_area_set_collision_layer(const RID &p_area, uint32_t p_layer) {}

uint32_t PhysicsServer2DExtension::_area_get_collision_layer(const RID &p_area) const {
	return 0;
}

void PhysicsServer2DExtension::_area_set_collision_mask(const RID &p_area, uint32_t p_mask) {}

uint32_t PhysicsServer2DExtension::_area_get_collision_mask(const RID &p_area) const {
	return 0;
}

void PhysicsServer2DExtension::_area_set_monitorable(const RID &p_area, bool p_monitorable) {}

void PhysicsServer2DExtension::_area_set_pickable(const RID &p_area, bool p_pickable) {}

void PhysicsServer2DExtension::_area_set_monitor_callback(const RID &p_area, const Callable &p_callback) {}

void PhysicsServer2DExtension::_area_set_area_monitor_callback(const RID &p_area, const Callable &p_callback) {}

RID PhysicsServer2DExtension::_body_create() {
	return RID();
}

void PhysicsServer2DExtension::_body_set_space(const RID &p_body, const RID &p_space) {}

RID PhysicsServer2DExtension::_body_get_space(const RID &p_body) const {
	return RID();
}

void PhysicsServer2DExtension::_body_set_mode(const RID &p_body, PhysicsServer2D::BodyMode p_mode) {}

PhysicsServer2D::BodyMode PhysicsServer2DExtension::_body_get_mode(const RID &p_body) const {
	return PhysicsServer2D::BodyMode(0);
}

void PhysicsServer2DExtension::_body_add_shape(const RID &p_body, const RID &p_shape, const Transform2D &p_transform, bool p_disabled) {}

void PhysicsServer2DExtension::_body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape) {}

void PhysicsServer2DExtension::_body_set_shape_transform(const RID &p_body, int32_t p_shape_idx, const Transform2D &p_transform) {}

int32_t PhysicsServer2DExtension::_body_get_shape_count(const RID &p_body) const {
	return 0;
}

RID PhysicsServer2DExtension::_body_get_shape(const RID &p_body, int32_t p_shape_idx) const {
	return RID();
}

Transform2D PhysicsServer2DExtension::_body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const {
	return Transform2D();
}

void PhysicsServer2DExtension::_body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled) {}

void PhysicsServer2DExtension::_body_set_shape_as_one_way_collision(const RID &p_body, int32_t p_shape_idx, bool p_enable, float p_margin) {}

void PhysicsServer2DExtension::_body_remove_shape(const RID &p_body, int32_t p_shape_idx) {}

void PhysicsServer2DExtension::_body_clear_shapes(const RID &p_body) {}

void PhysicsServer2DExtension::_body_attach_object_instance_id(const RID &p_body, uint64_t p_id) {}

uint64_t PhysicsServer2DExtension::_body_get_object_instance_id(const RID &p_body) const {
	return 0;
}

void PhysicsServer2DExtension::_body_attach_canvas_instance_id(const RID &p_body, uint64_t p_id) {}

uint64_t PhysicsServer2DExtension::_body_get_canvas_instance_id(const RID &p_body) const {
	return 0;
}

void PhysicsServer2DExtension::_body_set_continuous_collision_detection_mode(const RID &p_body, PhysicsServer2D::CCDMode p_mode) {}

PhysicsServer2D::CCDMode PhysicsServer2DExtension::_body_get_continuous_collision_detection_mode(const RID &p_body) const {
	return PhysicsServer2D::CCDMode(0);
}

void PhysicsServer2DExtension::_body_set_collision_layer(const RID &p_body, uint32_t p_layer) {}

uint32_t PhysicsServer2DExtension::_body_get_collision_layer(const RID &p_body) const {
	return 0;
}

void PhysicsServer2DExtension::_body_set_collision_mask(const RID &p_body, uint32_t p_mask) {}

uint32_t PhysicsServer2DExtension::_body_get_collision_mask(const RID &p_body) const {
	return 0;
}

void PhysicsServer2DExtension::_body_set_collision_priority(const RID &p_body, float p_priority) {}

float PhysicsServer2DExtension::_body_get_collision_priority(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer2DExtension::_body_set_param(const RID &p_body, PhysicsServer2D::BodyParameter p_param, const Variant &p_value) {}

Variant PhysicsServer2DExtension::_body_get_param(const RID &p_body, PhysicsServer2D::BodyParameter p_param) const {
	return Variant();
}

void PhysicsServer2DExtension::_body_reset_mass_properties(const RID &p_body) {}

void PhysicsServer2DExtension::_body_set_state(const RID &p_body, PhysicsServer2D::BodyState p_state, const Variant &p_value) {}

Variant PhysicsServer2DExtension::_body_get_state(const RID &p_body, PhysicsServer2D::BodyState p_state) const {
	return Variant();
}

void PhysicsServer2DExtension::_body_apply_central_impulse(const RID &p_body, const Vector2 &p_impulse) {}

void PhysicsServer2DExtension::_body_apply_torque_impulse(const RID &p_body, float p_impulse) {}

void PhysicsServer2DExtension::_body_apply_impulse(const RID &p_body, const Vector2 &p_impulse, const Vector2 &p_position) {}

void PhysicsServer2DExtension::_body_apply_central_force(const RID &p_body, const Vector2 &p_force) {}

void PhysicsServer2DExtension::_body_apply_force(const RID &p_body, const Vector2 &p_force, const Vector2 &p_position) {}

void PhysicsServer2DExtension::_body_apply_torque(const RID &p_body, float p_torque) {}

void PhysicsServer2DExtension::_body_add_constant_central_force(const RID &p_body, const Vector2 &p_force) {}

void PhysicsServer2DExtension::_body_add_constant_force(const RID &p_body, const Vector2 &p_force, const Vector2 &p_position) {}

void PhysicsServer2DExtension::_body_add_constant_torque(const RID &p_body, float p_torque) {}

void PhysicsServer2DExtension::_body_set_constant_force(const RID &p_body, const Vector2 &p_force) {}

Vector2 PhysicsServer2DExtension::_body_get_constant_force(const RID &p_body) const {
	return Vector2();
}

void PhysicsServer2DExtension::_body_set_constant_torque(const RID &p_body, float p_torque) {}

float PhysicsServer2DExtension::_body_get_constant_torque(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer2DExtension::_body_set_axis_velocity(const RID &p_body, const Vector2 &p_axis_velocity) {}

void PhysicsServer2DExtension::_body_add_collision_exception(const RID &p_body, const RID &p_excepted_body) {}

void PhysicsServer2DExtension::_body_remove_collision_exception(const RID &p_body, const RID &p_excepted_body) {}

TypedArray<RID> PhysicsServer2DExtension::_body_get_collision_exceptions(const RID &p_body) const {
	return TypedArray<RID>();
}

void PhysicsServer2DExtension::_body_set_max_contacts_reported(const RID &p_body, int32_t p_amount) {}

int32_t PhysicsServer2DExtension::_body_get_max_contacts_reported(const RID &p_body) const {
	return 0;
}

void PhysicsServer2DExtension::_body_set_contacts_reported_depth_threshold(const RID &p_body, float p_threshold) {}

float PhysicsServer2DExtension::_body_get_contacts_reported_depth_threshold(const RID &p_body) const {
	return 0.0;
}

void PhysicsServer2DExtension::_body_set_omit_force_integration(const RID &p_body, bool p_enable) {}

bool PhysicsServer2DExtension::_body_is_omitting_force_integration(const RID &p_body) const {
	return false;
}

void PhysicsServer2DExtension::_body_set_state_sync_callback(const RID &p_body, const Callable &p_callable) {}

void PhysicsServer2DExtension::_body_set_force_integration_callback(const RID &p_body, const Callable &p_callable, const Variant &p_userdata) {}

bool PhysicsServer2DExtension::_body_collide_shape(const RID &p_body, int32_t p_body_shape, const RID &p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, void *p_results, int32_t p_result_max, int32_t *p_result_count) {
	return false;
}

void PhysicsServer2DExtension::_body_set_pickable(const RID &p_body, bool p_pickable) {}

PhysicsDirectBodyState2D *PhysicsServer2DExtension::_body_get_direct_state(const RID &p_body) {
	return nullptr;
}

bool PhysicsServer2DExtension::_body_test_motion(const RID &p_body, const Transform2D &p_from, const Vector2 &p_motion, float p_margin, bool p_collide_separation_ray, bool p_recovery_as_collision, PhysicsServer2DExtensionMotionResult *p_result) const {
	return false;
}

RID PhysicsServer2DExtension::_joint_create() {
	return RID();
}

void PhysicsServer2DExtension::_joint_clear(const RID &p_joint) {}

void PhysicsServer2DExtension::_joint_set_param(const RID &p_joint, PhysicsServer2D::JointParam p_param, float p_value) {}

float PhysicsServer2DExtension::_joint_get_param(const RID &p_joint, PhysicsServer2D::JointParam p_param) const {
	return 0.0;
}

void PhysicsServer2DExtension::_joint_disable_collisions_between_bodies(const RID &p_joint, bool p_disable) {}

bool PhysicsServer2DExtension::_joint_is_disabled_collisions_between_bodies(const RID &p_joint) const {
	return false;
}

void PhysicsServer2DExtension::_joint_make_pin(const RID &p_joint, const Vector2 &p_anchor, const RID &p_body_a, const RID &p_body_b) {}

void PhysicsServer2DExtension::_joint_make_groove(const RID &p_joint, const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, const RID &p_body_a, const RID &p_body_b) {}

void PhysicsServer2DExtension::_joint_make_damped_spring(const RID &p_joint, const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, const RID &p_body_a, const RID &p_body_b) {}

void PhysicsServer2DExtension::_pin_joint_set_flag(const RID &p_joint, PhysicsServer2D::PinJointFlag p_flag, bool p_enabled) {}

bool PhysicsServer2DExtension::_pin_joint_get_flag(const RID &p_joint, PhysicsServer2D::PinJointFlag p_flag) const {
	return false;
}

void PhysicsServer2DExtension::_pin_joint_set_param(const RID &p_joint, PhysicsServer2D::PinJointParam p_param, float p_value) {}

float PhysicsServer2DExtension::_pin_joint_get_param(const RID &p_joint, PhysicsServer2D::PinJointParam p_param) const {
	return 0.0;
}

void PhysicsServer2DExtension::_damped_spring_joint_set_param(const RID &p_joint, PhysicsServer2D::DampedSpringParam p_param, float p_value) {}

float PhysicsServer2DExtension::_damped_spring_joint_get_param(const RID &p_joint, PhysicsServer2D::DampedSpringParam p_param) const {
	return 0.0;
}

PhysicsServer2D::JointType PhysicsServer2DExtension::_joint_get_type(const RID &p_joint) const {
	return PhysicsServer2D::JointType(0);
}

void PhysicsServer2DExtension::_free_rid(const RID &p_rid) {}

void PhysicsServer2DExtension::_set_active(bool p_active) {}

void PhysicsServer2DExtension::_init() {}

void PhysicsServer2DExtension::_step(float p_step) {}

void PhysicsServer2DExtension::_sync() {}

void PhysicsServer2DExtension::_flush_queries() {}

void PhysicsServer2DExtension::_end_sync() {}

void PhysicsServer2DExtension::_finish() {}

bool PhysicsServer2DExtension::_is_flushing_queries() const {
	return false;
}

int32_t PhysicsServer2DExtension::_get_process_info(PhysicsServer2D::ProcessInfo p_process_info) {
	return 0;
}

} // namespace godot
