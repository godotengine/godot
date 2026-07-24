/**************************************************************************/
/*  physics_server_3d.h                                                   */
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

#pragma once

#include "servers/physics_3d/direct_states/physics_direct_body_state_3d.h"
#include "servers/physics_3d/direct_states/physics_direct_space_state_3d.h"
#include "servers/physics_3d/physics_server_3d_enums.h"
#include "servers/physics_3d/physics_server_3d_rendering_server_handler.h"
#include "servers/physics_3d/physics_server_3d_types.h"
#include "servers/physics_3d/queries/physics_testmotion_query_parameters_3d.h"
#include "servers/physics_3d/queries/physics_testmotion_query_result_3d.h"

class PhysicsDirectSpaceState3D;
template <typename T>
class TypedArray;

template <typename T>
class Ref;

class PhysicsTestMotionParameters3D;
class PhysicsTestMotionResult3D;

class PhysicsServer3D : public Object {
	GDCLASS(PhysicsServer3D, Object);

	static PhysicsServer3D *singleton;

	virtual bool _body_test_motion(RID p_body, RequiredParam<PhysicsTestMotionParameters3D> rp_parameters, const Ref<PhysicsTestMotionResult3D> &p_result = Ref<PhysicsTestMotionResult3D>());

protected:
	static void _bind_methods();

public:
	static PhysicsServer3D *get_singleton();

	RID shape_create(PS3DE::ShapeType p_shape);

	virtual RID world_boundary_shape_create() = 0;
	virtual RID separation_ray_shape_create() = 0;
	virtual RID sphere_shape_create() = 0;
	virtual RID box_shape_create() = 0;
	virtual RID capsule_shape_create() = 0;
	virtual RID cylinder_shape_create() = 0;
	virtual RID convex_polygon_shape_create() = 0;
	virtual RID concave_polygon_shape_create() = 0;
	virtual RID heightmap_shape_create() = 0;
	virtual RID custom_shape_create() = 0;

	virtual void shape_set_data(RID p_shape, const Variant &p_data) = 0;
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias) = 0;

	virtual PS3DE::ShapeType shape_get_type(RID p_shape) const = 0;
	virtual Variant shape_get_data(RID p_shape) const = 0;

	virtual void shape_set_margin(RID p_shape, real_t p_margin) = 0;
	virtual real_t shape_get_margin(RID p_shape) const = 0;

	virtual real_t shape_get_custom_solver_bias(RID p_shape) const = 0;

	/* SPACE API */

	virtual RID space_create() = 0;
	virtual void space_set_active(RID p_space, bool p_active) = 0;
	virtual bool space_is_active(RID p_space) const = 0;

	virtual void space_set_param(RID p_space, PS3DE::SpaceParameter p_param, real_t p_value) = 0;
	virtual real_t space_get_param(RID p_space, PS3DE::SpaceParameter p_param) const = 0;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectSpaceState3D *space_get_direct_state(RID p_space) = 0;

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts) = 0;
	virtual Vector<Vector3> space_get_contacts(RID p_space) const = 0;
	virtual int space_get_contact_count(RID p_space) const = 0;

	//missing space parameters

	/* AREA API */

	//missing attenuation? missing better override?

	virtual RID area_create() = 0;

	virtual void area_set_space(RID p_area, RID p_space) = 0;
	virtual RID area_get_space(RID p_area) const = 0;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform3D &p_transform = Transform3D(), bool p_disabled = false) = 0;
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape) = 0;
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform3D &p_transform) = 0;

	virtual int area_get_shape_count(RID p_area) const = 0;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const = 0;
	virtual Transform3D area_get_shape_transform(RID p_area, int p_shape_idx) const = 0;

	virtual void area_remove_shape(RID p_area, int p_shape_idx) = 0;
	virtual void area_clear_shapes(RID p_area) = 0;

	virtual void area_set_shape_disabled(RID p_area, int p_shape_idx, bool p_disabled) = 0;

	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_id) = 0;
	virtual ObjectID area_get_object_instance_id(RID p_area) const = 0;

	virtual void area_set_param(RID p_area, PS3DE::AreaParameter p_param, const Variant &p_value) = 0;
	virtual void area_set_transform(RID p_area, const Transform3D &p_transform) = 0;

	virtual Variant area_get_param(RID p_parea, PS3DE::AreaParameter p_param) const = 0;
	virtual Transform3D area_get_transform(RID p_area) const = 0;

	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer) = 0;
	virtual uint32_t area_get_collision_layer(RID p_area) const = 0;

	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask) = 0;
	virtual uint32_t area_get_collision_mask(RID p_area) const = 0;

	virtual void area_set_monitorable(RID p_area, bool p_monitorable) = 0;

	virtual void area_set_monitor_callback(RID p_area, const Callable &p_callback) = 0;
	virtual void area_set_area_monitor_callback(RID p_area, const Callable &p_callback) = 0;

	virtual void area_set_ray_pickable(RID p_area, bool p_enable) = 0;

	/* BODY API */

	//missing ccd?

	virtual RID body_create() = 0;

	virtual void body_set_space(RID p_body, RID p_space) = 0;
	virtual RID body_get_space(RID p_body) const = 0;

	virtual void body_set_mode(RID p_body, PS3DE::BodyMode p_mode) = 0;
	virtual PS3DE::BodyMode body_get_mode(RID p_body) const = 0;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform3D &p_transform = Transform3D(), bool p_disabled = false) = 0;
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) = 0;
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform3D &p_transform) = 0;

	virtual int body_get_shape_count(RID p_body) const = 0;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const = 0;
	virtual Transform3D body_get_shape_transform(RID p_body, int p_shape_idx) const = 0;

	virtual void body_remove_shape(RID p_body, int p_shape_idx) = 0;
	virtual void body_clear_shapes(RID p_body) = 0;

	virtual void body_set_shape_disabled(RID p_body, int p_shape_idx, bool p_disabled) = 0;

	virtual void body_attach_object_instance_id(RID p_body, ObjectID p_id) = 0;
	virtual ObjectID body_get_object_instance_id(RID p_body) const = 0;

	virtual void body_set_enable_continuous_collision_detection(RID p_body, bool p_enable) = 0;
	virtual bool body_is_continuous_collision_detection_enabled(RID p_body) const = 0;

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer) = 0;
	virtual uint32_t body_get_collision_layer(RID p_body) const = 0;

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask) = 0;
	virtual uint32_t body_get_collision_mask(RID p_body) const = 0;

	virtual void body_set_collision_priority(RID p_body, real_t p_priority) = 0;
	virtual real_t body_get_collision_priority(RID p_body) const = 0;

	virtual void body_set_user_flags(RID p_body, uint32_t p_flags) = 0;
	virtual uint32_t body_get_user_flags(RID p_body) const = 0;

	// common body variables

	virtual void body_set_param(RID p_body, PS3DE::BodyParameter p_param, const Variant &p_value) = 0;
	virtual Variant body_get_param(RID p_body, PS3DE::BodyParameter p_param) const = 0;

	virtual void body_reset_mass_properties(RID p_body) = 0;

	//state

	virtual void body_set_state(RID p_body, PS3DE::BodyState p_state, const Variant &p_variant) = 0;
	virtual Variant body_get_state(RID p_body, PS3DE::BodyState p_state) const = 0;

	virtual void body_apply_central_impulse(RID p_body, const Vector3 &p_impulse) = 0;
	virtual void body_apply_impulse(RID p_body, const Vector3 &p_impulse, const Vector3 &p_position = Vector3()) = 0;
	virtual void body_apply_impulse_at_position(RID p_body, const Vector3 &p_impulse, const Vector3 &p_global_position = Vector3()) = 0;
	virtual void body_apply_torque_impulse(RID p_body, const Vector3 &p_impulse) = 0;

	virtual void body_apply_central_force(RID p_body, const Vector3 &p_force) = 0;
	virtual void body_apply_force(RID p_body, const Vector3 &p_force, const Vector3 &p_position = Vector3()) = 0;
	virtual void body_apply_force_at_position(RID p_body, const Vector3 &p_force, const Vector3 &p_global_position = Vector3()) = 0;
	virtual void body_apply_torque(RID p_body, const Vector3 &p_torque) = 0;

	virtual void body_add_constant_central_force(RID p_body, const Vector3 &p_force) = 0;
	virtual void body_add_constant_force(RID p_body, const Vector3 &p_force, const Vector3 &p_position = Vector3()) = 0;
	virtual void body_add_constant_torque(RID p_body, const Vector3 &p_torque) = 0;

	virtual void body_set_constant_force(RID p_body, const Vector3 &p_force) = 0;
	virtual Vector3 body_get_constant_force(RID p_body) const = 0;

	virtual void body_set_constant_torque(RID p_body, const Vector3 &p_torque) = 0;
	virtual Vector3 body_get_constant_torque(RID p_body) const = 0;

	virtual void body_set_axis_velocity(RID p_body, const Vector3 &p_axis_velocity) = 0;

	virtual void body_set_axis_lock(RID p_body, PS3DE::BodyAxis p_axis, bool p_lock) = 0;
	virtual bool body_is_axis_locked(RID p_body, PS3DE::BodyAxis p_axis) const = 0;

	//fix
	virtual void body_add_collision_exception(RID p_body, RID p_body_b) = 0;
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b) = 0;
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) = 0;

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts) = 0;
	virtual int body_get_max_contacts_reported(RID p_body) const = 0;

	//missing remove
	virtual void body_set_contacts_reported_depth_threshold(RID p_body, real_t p_threshold) = 0;
	virtual real_t body_get_contacts_reported_depth_threshold(RID p_body) const = 0;

	virtual void body_set_omit_force_integration(RID p_body, bool p_omit) = 0;
	virtual bool body_is_omitting_force_integration(RID p_body) const = 0;

	virtual void body_set_state_sync_callback(RID p_body, const Callable &p_callable) = 0;
	virtual void body_set_force_integration_callback(RID p_body, const Callable &p_callable, const Variant &p_udata = Variant()) = 0;

	virtual void body_set_ray_pickable(RID p_body, bool p_enable) = 0;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectBodyState3D *body_get_direct_state(RID p_body) = 0;

	virtual bool body_test_motion(RID p_body, const PS3DT::MotionParameters &p_parameters, PS3DT::MotionResult *r_result = nullptr) = 0;

	/* SOFT BODY */

	virtual RID soft_body_create() = 0;

	virtual void soft_body_update_rendering_server(RID p_body, RequiredParam<PhysicsServer3DRenderingServerHandler> rp_rendering_server_handler) = 0;

	virtual void soft_body_set_space(RID p_body, RID p_space) = 0;
	virtual RID soft_body_get_space(RID p_body) const = 0;

	virtual void soft_body_set_mesh(RID p_body, RID p_mesh) = 0;

	virtual AABB soft_body_get_bounds(RID p_body) const = 0;

	virtual void soft_body_set_collision_layer(RID p_body, uint32_t p_layer) = 0;
	virtual uint32_t soft_body_get_collision_layer(RID p_body) const = 0;

	virtual void soft_body_set_collision_mask(RID p_body, uint32_t p_mask) = 0;
	virtual uint32_t soft_body_get_collision_mask(RID p_body) const = 0;

	virtual void soft_body_add_collision_exception(RID p_body, RID p_body_b) = 0;
	virtual void soft_body_remove_collision_exception(RID p_body, RID p_body_b) = 0;
	virtual void soft_body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) = 0;

	virtual void soft_body_set_state(RID p_body, PS3DE::BodyState p_state, const Variant &p_variant) = 0;
	virtual Variant soft_body_get_state(RID p_body, PS3DE::BodyState p_state) const = 0;

	virtual void soft_body_set_transform(RID p_body, const Transform3D &p_transform) = 0;

	virtual void soft_body_set_ray_pickable(RID p_body, bool p_enable) = 0;

	virtual void soft_body_set_simulation_precision(RID p_body, int p_simulation_precision) = 0;
	virtual int soft_body_get_simulation_precision(RID p_body) const = 0;

	virtual void soft_body_set_total_mass(RID p_body, real_t p_total_mass) = 0;
	virtual real_t soft_body_get_total_mass(RID p_body) const = 0;

	virtual void soft_body_set_linear_stiffness(RID p_body, real_t p_stiffness) = 0;
	virtual real_t soft_body_get_linear_stiffness(RID p_body) const = 0;

	virtual void soft_body_set_shrinking_factor(RID p_body, real_t p_shrinking_factor) = 0;
	virtual real_t soft_body_get_shrinking_factor(RID p_body) const = 0;

	virtual void soft_body_set_pressure_coefficient(RID p_body, real_t p_pressure_coefficient) = 0;
	virtual real_t soft_body_get_pressure_coefficient(RID p_body) const = 0;

	virtual void soft_body_set_damping_coefficient(RID p_body, real_t p_damping_coefficient) = 0;
	virtual real_t soft_body_get_damping_coefficient(RID p_body) const = 0;

	virtual void soft_body_set_drag_coefficient(RID p_body, real_t p_drag_coefficient) = 0;
	virtual real_t soft_body_get_drag_coefficient(RID p_body) const = 0;

	virtual void soft_body_move_point(RID p_body, int p_point_index, const Vector3 &p_global_position) = 0;
	virtual Vector3 soft_body_get_point_global_position(RID p_body, int p_point_index) const = 0;

	virtual void soft_body_apply_point_impulse(RID p_body, int p_point_index, const Vector3 &p_impulse) = 0;
	virtual void soft_body_apply_point_force(RID p_body, int p_point_index, const Vector3 &p_force) = 0;
	virtual void soft_body_apply_central_impulse(RID p_body, const Vector3 &p_impulse) = 0;
	virtual void soft_body_apply_central_force(RID p_body, const Vector3 &p_force) = 0;

	virtual void soft_body_remove_all_pinned_points(RID p_body) = 0;
	virtual void soft_body_pin_point(RID p_body, int p_point_index, bool p_pin) = 0;
	virtual bool soft_body_is_point_pinned(RID p_body, int p_point_index) const = 0;

	/* JOINT API */

	virtual RID joint_create() = 0;

	virtual void joint_clear(RID p_joint) = 0;

	virtual PS3DE::JointType joint_get_type(RID p_joint) const = 0;

	virtual void joint_set_solver_priority(RID p_joint, int p_priority) = 0;
	virtual int joint_get_solver_priority(RID p_joint) const = 0;

	virtual void joint_disable_collisions_between_bodies(RID p_joint, bool p_disable) = 0;
	virtual bool joint_is_disabled_collisions_between_bodies(RID p_joint) const = 0;

	virtual void joint_make_pin(RID p_joint, RID p_body_A, const Vector3 &p_local_A, RID p_body_B, const Vector3 &p_local_B) = 0;

	virtual void pin_joint_set_param(RID p_joint, PS3DE::PinJointParam p_param, real_t p_value) = 0;
	virtual real_t pin_joint_get_param(RID p_joint, PS3DE::PinJointParam p_param) const = 0;

	virtual void pin_joint_set_local_a(RID p_joint, const Vector3 &p_A) = 0;
	virtual Vector3 pin_joint_get_local_a(RID p_joint) const = 0;

	virtual void pin_joint_set_local_b(RID p_joint, const Vector3 &p_B) = 0;
	virtual Vector3 pin_joint_get_local_b(RID p_joint) const = 0;

	virtual void joint_make_hinge(RID p_joint, RID p_body_A, const Transform3D &p_hinge_A, RID p_body_B, const Transform3D &p_hinge_B) = 0;
	virtual void joint_make_hinge_simple(RID p_joint, RID p_body_A, const Vector3 &p_pivot_A, const Vector3 &p_axis_A, RID p_body_B, const Vector3 &p_pivot_B, const Vector3 &p_axis_B) = 0;

	virtual void hinge_joint_set_param(RID p_joint, PS3DE::HingeJointParam p_param, real_t p_value) = 0;
	virtual real_t hinge_joint_get_param(RID p_joint, PS3DE::HingeJointParam p_param) const = 0;

	virtual void hinge_joint_set_flag(RID p_joint, PS3DE::HingeJointFlag p_flag, bool p_enabled) = 0;
	virtual bool hinge_joint_get_flag(RID p_joint, PS3DE::HingeJointFlag p_flag) const = 0;

	virtual void joint_make_slider(RID p_joint, RID p_body_A, const Transform3D &p_local_frame_A, RID p_body_B, const Transform3D &p_local_frame_B) = 0; //reference frame is A

	virtual void slider_joint_set_param(RID p_joint, PS3DE::SliderJointParam p_param, real_t p_value) = 0;
	virtual real_t slider_joint_get_param(RID p_joint, PS3DE::SliderJointParam p_param) const = 0;

	virtual void joint_make_cone_twist(RID p_joint, RID p_body_A, const Transform3D &p_local_frame_A, RID p_body_B, const Transform3D &p_local_frame_B) = 0; //reference frame is A

	virtual void cone_twist_joint_set_param(RID p_joint, PS3DE::ConeTwistJointParam p_param, real_t p_value) = 0;
	virtual real_t cone_twist_joint_get_param(RID p_joint, PS3DE::ConeTwistJointParam p_param) const = 0;

	virtual void joint_make_generic_6dof(RID p_joint, RID p_body_A, const Transform3D &p_local_frame_A, RID p_body_B, const Transform3D &p_local_frame_B) = 0; //reference frame is A

	virtual void generic_6dof_joint_set_param(RID p_joint, Vector3::Axis, PS3DE::G6DOFJointAxisParam p_param, real_t p_value) = 0;
	virtual real_t generic_6dof_joint_get_param(RID p_joint, Vector3::Axis, PS3DE::G6DOFJointAxisParam p_param) const = 0;

	virtual void generic_6dof_joint_set_flag(RID p_joint, Vector3::Axis, PS3DE::G6DOFJointAxisFlag p_flag, bool p_enable) = 0;
	virtual bool generic_6dof_joint_get_flag(RID p_joint, Vector3::Axis, PS3DE::G6DOFJointAxisFlag p_flag) const = 0;

	virtual void generic_6dof_joint_set_angular_target_rotation(RID p_joint, const Quaternion &p_target_rotation) = 0;
	virtual Quaternion generic_6dof_joint_get_angular_target_rotation(RID p_joint) const = 0;

	/* QUERY API */

	/* MISC */

	virtual void free_rid(RID p_rid) = 0;
#ifndef DISABLE_DEPRECATED
	[[deprecated("Use `free_rid()` instead.")]] void free(RID p_rid) {
		free_rid(p_rid);
	}
#endif // DISABLE_DEPRECATED

	virtual void set_active(bool p_active) = 0;
	virtual void init() = 0;
	virtual void step(real_t p_step) = 0;
	virtual void sync() = 0;
	virtual void flush_queries() = 0;
	virtual void end_sync() = 0;
	virtual void finish() = 0;

	virtual bool is_flushing_queries() const = 0;

	virtual int get_process_info(PS3DE::ProcessInfo p_info) = 0;

	PhysicsServer3D();
	~PhysicsServer3D();
};

VARIANT_ENUM_CAST_EXT(PS3DE::ShapeType, PhysicsServer3D::ShapeType);
VARIANT_ENUM_CAST_EXT(PS3DE::SpaceParameter, PhysicsServer3D::SpaceParameter);
VARIANT_ENUM_CAST_EXT(PS3DE::AreaParameter, PhysicsServer3D::AreaParameter);
VARIANT_ENUM_CAST_EXT(PS3DE::AreaSpaceOverrideMode, PhysicsServer3D::AreaSpaceOverrideMode);
VARIANT_ENUM_CAST_EXT(PS3DE::BodyMode, PhysicsServer3D::BodyMode);
VARIANT_ENUM_CAST_EXT(PS3DE::BodyParameter, PhysicsServer3D::BodyParameter);
VARIANT_ENUM_CAST_EXT(PS3DE::BodyDampMode, PhysicsServer3D::BodyDampMode);
VARIANT_ENUM_CAST_EXT(PS3DE::BodyState, PhysicsServer3D::BodyState);
VARIANT_ENUM_CAST_EXT(PS3DE::BodyAxis, PhysicsServer3D::BodyAxis);
VARIANT_ENUM_CAST_EXT(PS3DE::PinJointParam, PhysicsServer3D::PinJointParam);
VARIANT_ENUM_CAST_EXT(PS3DE::JointType, PhysicsServer3D::JointType);
VARIANT_ENUM_CAST_EXT(PS3DE::HingeJointParam, PhysicsServer3D::HingeJointParam);
VARIANT_ENUM_CAST_EXT(PS3DE::HingeJointFlag, PhysicsServer3D::HingeJointFlag);
VARIANT_ENUM_CAST_EXT(PS3DE::SliderJointParam, PhysicsServer3D::SliderJointParam);
VARIANT_ENUM_CAST_EXT(PS3DE::ConeTwistJointParam, PhysicsServer3D::ConeTwistJointParam);
VARIANT_ENUM_CAST_EXT(PS3DE::G6DOFJointAxisParam, PhysicsServer3D::G6DOFJointAxisParam);
VARIANT_ENUM_CAST_EXT(PS3DE::G6DOFJointAxisFlag, PhysicsServer3D::G6DOFJointAxisFlag);
VARIANT_ENUM_CAST_EXT(PS3DE::AreaBodyStatus, PhysicsServer3D::AreaBodyStatus);
VARIANT_ENUM_CAST_EXT(PS3DE::ProcessInfo, PhysicsServer3D::ProcessInfo);
