/**************************************************************************/
/*  physics_server_2d.h                                                   */
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

#include "core/object/ref_counted.h"
#include "servers/physics_2d/direct_states/physics_direct_body_state_2d.h"
#include "servers/physics_2d/direct_states/physics_direct_space_state_2d.h"
#include "servers/physics_2d/physics_server_2d_enums.h"
#include "servers/physics_2d/physics_server_2d_types.h"
#include "servers/physics_2d/queries/physics_shape_query_parameters_2d.h"
#include "servers/physics_2d/queries/physics_testmotion_query_parameters_2d.h"
#include "servers/physics_2d/queries/physics_testmotion_query_result_2d.h"

class PhysicsServer2D : public Object {
	GDCLASS(PhysicsServer2D, Object);

	static PhysicsServer2D *singleton;

	virtual bool _body_test_motion(RID p_body, RequiredParam<PhysicsTestMotionParameters2D> rp_parameters, const Ref<PhysicsTestMotionResult2D> &p_result = Ref<PhysicsTestMotionResult2D>());

protected:
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	void _body_set_shape_as_one_way_collision_bind_compat_104736(RID p_body, int p_shape_idx, bool p_enable, real_t p_margin);
	static void _bind_compatibility_methods();
#endif

public:
	static PhysicsServer2D *get_singleton();

	virtual RID world_boundary_shape_create() = 0;
	virtual RID separation_ray_shape_create() = 0;
	virtual RID segment_shape_create() = 0;
	virtual RID circle_shape_create() = 0;
	virtual RID rectangle_shape_create() = 0;
	virtual RID capsule_shape_create() = 0;
	virtual RID convex_polygon_shape_create() = 0;
	virtual RID concave_polygon_shape_create() = 0;

	virtual void shape_set_data(RID p_shape, const Variant &p_data) = 0;
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias) = 0;

	virtual PS2DE::ShapeType shape_get_type(RID p_shape) const = 0;
	virtual Variant shape_get_data(RID p_shape) const = 0;
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const = 0;

	//these work well, but should be used from the main thread only
	virtual bool shape_collide(RID p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, RID p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, Vector2 *r_results, int p_result_max, int &r_result_count) = 0;

	/* SPACE API */

	virtual RID space_create() = 0;
	virtual void space_set_active(RID p_space, bool p_active) = 0;
	virtual bool space_is_active(RID p_space) const = 0;

	virtual void space_set_param(RID p_space, PS2DE::SpaceParameter p_param, real_t p_value) = 0;
	virtual real_t space_get_param(RID p_space, PS2DE::SpaceParameter p_param) const = 0;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectSpaceState2D *space_get_direct_state(RID p_space) = 0;

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts) = 0;
	virtual Vector<Vector2> space_get_contacts(RID p_space) const = 0;
	virtual int space_get_contact_count(RID p_space) const = 0;

	//missing space parameters

	/* AREA API */

	//missing attenuation? missing better override?

	virtual RID area_create() = 0;

	virtual void area_set_space(RID p_area, RID p_space) = 0;
	virtual RID area_get_space(RID p_area) const = 0;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false) = 0;
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape) = 0;
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform2D &p_transform) = 0;

	virtual int area_get_shape_count(RID p_area) const = 0;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const = 0;
	virtual Transform2D area_get_shape_transform(RID p_area, int p_shape_idx) const = 0;

	virtual void area_remove_shape(RID p_area, int p_shape_idx) = 0;
	virtual void area_clear_shapes(RID p_area) = 0;

	virtual void area_set_shape_disabled(RID p_area, int p_shape, bool p_disabled) = 0;

	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_id) = 0;
	virtual ObjectID area_get_object_instance_id(RID p_area) const = 0;

	virtual void area_attach_canvas_instance_id(RID p_area, ObjectID p_id) = 0;
	virtual ObjectID area_get_canvas_instance_id(RID p_area) const = 0;

	virtual void area_set_param(RID p_area, PS2DE::AreaParameter p_param, const Variant &p_value) = 0;
	virtual void area_set_transform(RID p_area, const Transform2D &p_transform) = 0;

	virtual Variant area_get_param(RID p_parea, PS2DE::AreaParameter p_param) const = 0;
	virtual Transform2D area_get_transform(RID p_area) const = 0;

	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer) = 0;
	virtual uint32_t area_get_collision_layer(RID p_area) const = 0;

	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask) = 0;
	virtual uint32_t area_get_collision_mask(RID p_area) const = 0;

	virtual void area_set_monitorable(RID p_area, bool p_monitorable) = 0;
	virtual void area_set_pickable(RID p_area, bool p_pickable) = 0;

	virtual void area_set_monitor_callback(RID p_area, const Callable &p_callback) = 0;
	virtual void area_set_area_monitor_callback(RID p_area, const Callable &p_callback) = 0;
	virtual void area_set_gravity_target_callback(RID p_area, const Callable &p_callback) = 0;

	/* BODY API */

	//missing ccd?

	virtual RID body_create() = 0;

	virtual void body_set_space(RID p_body, RID p_space) = 0;
	virtual RID body_get_space(RID p_body) const = 0;

	virtual void body_set_mode(RID p_body, PS2DE::BodyMode p_mode) = 0;
	virtual PS2DE::BodyMode body_get_mode(RID p_body) const = 0;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false) = 0;
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) = 0;
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform2D &p_transform) = 0;

	virtual int body_get_shape_count(RID p_body) const = 0;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const = 0;
	virtual Transform2D body_get_shape_transform(RID p_body, int p_shape_idx) const = 0;

	virtual void body_set_shape_disabled(RID p_body, int p_shape, bool p_disabled) = 0;
	virtual void body_set_shape_as_one_way_collision(RID p_body, int p_shape, bool p_enabled, real_t p_margin = 0, const Vector2 &p_direction = Vector2(0, 1)) = 0;

	virtual void body_remove_shape(RID p_body, int p_shape_idx) = 0;
	virtual void body_clear_shapes(RID p_body) = 0;

	virtual void body_attach_object_instance_id(RID p_body, ObjectID p_id) = 0;
	virtual ObjectID body_get_object_instance_id(RID p_body) const = 0;

	virtual void body_attach_canvas_instance_id(RID p_body, ObjectID p_id) = 0;
	virtual ObjectID body_get_canvas_instance_id(RID p_body) const = 0;

	virtual void body_set_continuous_collision_detection_mode(RID p_body, PS2DE::CCDMode p_mode) = 0;
	virtual PS2DE::CCDMode body_get_continuous_collision_detection_mode(RID p_body) const = 0;

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer) = 0;
	virtual uint32_t body_get_collision_layer(RID p_body) const = 0;

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask) = 0;
	virtual uint32_t body_get_collision_mask(RID p_body) const = 0;

	virtual void body_set_collision_priority(RID p_body, real_t p_priority) = 0;
	virtual real_t body_get_collision_priority(RID p_body) const = 0;

	virtual void body_set_param(RID p_body, PS2DE::BodyParameter p_param, const Variant &p_value) = 0;
	virtual Variant body_get_param(RID p_body, PS2DE::BodyParameter p_param) const = 0;

	virtual void body_reset_mass_properties(RID p_body) = 0;

	//state

	virtual void body_set_state(RID p_body, PS2DE::BodyState p_state, const Variant &p_variant) = 0;
	virtual Variant body_get_state(RID p_body, PS2DE::BodyState p_state) const = 0;

	virtual void body_apply_central_impulse(RID p_body, const Vector2 &p_impulse) = 0;
	virtual void body_apply_torque_impulse(RID p_body, real_t p_torque) = 0;
	virtual void body_apply_impulse(RID p_body, const Vector2 &p_impulse, const Vector2 &p_position = Vector2()) = 0;

	virtual void body_apply_central_force(RID p_body, const Vector2 &p_force) = 0;
	virtual void body_apply_force(RID p_body, const Vector2 &p_force, const Vector2 &p_position = Vector2()) = 0;
	virtual void body_apply_torque(RID p_body, real_t p_torque) = 0;

	virtual void body_add_constant_central_force(RID p_body, const Vector2 &p_force) = 0;
	virtual void body_add_constant_force(RID p_body, const Vector2 &p_force, const Vector2 &p_position = Vector2()) = 0;
	virtual void body_add_constant_torque(RID p_body, real_t p_torque) = 0;

	virtual void body_set_constant_force(RID p_body, const Vector2 &p_force) = 0;
	virtual Vector2 body_get_constant_force(RID p_body) const = 0;

	virtual void body_set_constant_torque(RID p_body, real_t p_torque) = 0;
	virtual real_t body_get_constant_torque(RID p_body) const = 0;

	virtual void body_set_axis_velocity(RID p_body, const Vector2 &p_axis_velocity) = 0;

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

	virtual bool body_collide_shape(RID p_body, int p_body_shape, RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, Vector2 *r_results, int p_result_max, int &r_result_count) = 0;

	virtual void body_set_pickable(RID p_body, bool p_pickable) = 0;

	// this function only works on physics process, errors and returns null otherwise
	virtual PhysicsDirectBodyState2D *body_get_direct_state(RID p_body) = 0;

	virtual bool body_test_motion(RID p_body, const PS2DT::MotionParameters &p_parameters, PS2DT::MotionResult *r_result = nullptr) = 0;

	/* JOINT API */

	virtual RID joint_create() = 0;

	virtual void joint_clear(RID p_joint) = 0;

	virtual void joint_set_param(RID p_joint, PS2DE::JointParam p_param, real_t p_value) = 0;
	virtual real_t joint_get_param(RID p_joint, PS2DE::JointParam p_param) const = 0;

	virtual void joint_disable_collisions_between_bodies(RID p_joint, const bool p_disable) = 0;
	virtual bool joint_is_disabled_collisions_between_bodies(RID p_joint) const = 0;

	virtual void joint_make_pin(RID p_joint, const Vector2 &p_anchor, RID p_body_a, RID p_body_b = RID()) = 0;
	virtual void joint_make_groove(RID p_joint, const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, RID p_body_a, RID p_body_b) = 0;
	virtual void joint_make_damped_spring(RID p_joint, const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, RID p_body_a, RID p_body_b = RID()) = 0;

	virtual void pin_joint_set_param(RID p_joint, PS2DE::PinJointParam p_param, real_t p_value) = 0;
	virtual real_t pin_joint_get_param(RID p_joint, PS2DE::PinJointParam p_param) const = 0;

	virtual void pin_joint_set_flag(RID p_joint, PS2DE::PinJointFlag p_flag, bool p_enabled) = 0;
	virtual bool pin_joint_get_flag(RID p_joint, PS2DE::PinJointFlag p_flag) const = 0;

	virtual void damped_spring_joint_set_param(RID p_joint, PS2DE::DampedSpringParam p_param, real_t p_value) = 0;
	virtual real_t damped_spring_joint_get_param(RID p_joint, PS2DE::DampedSpringParam p_param) const = 0;

	virtual PS2DE::JointType joint_get_type(RID p_joint) const = 0;

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

	virtual int get_process_info(PS2DE::ProcessInfo p_info) = 0;

	PhysicsServer2D();
	~PhysicsServer2D();
};

VARIANT_ENUM_CAST_EXT(PS2DE::ShapeType, PhysicsServer2D::ShapeType);
VARIANT_ENUM_CAST_EXT(PS2DE::SpaceParameter, PhysicsServer2D::SpaceParameter);
VARIANT_ENUM_CAST_EXT(PS2DE::AreaGravityType, PhysicsServer2D::AreaGravityType);
VARIANT_ENUM_CAST_EXT(PS2DE::AreaParameter, PhysicsServer2D::AreaParameter);
VARIANT_ENUM_CAST_EXT(PS2DE::AreaSpaceOverrideMode, PhysicsServer2D::AreaSpaceOverrideMode);
VARIANT_ENUM_CAST_EXT(PS2DE::BodyMode, PhysicsServer2D::BodyMode);
VARIANT_ENUM_CAST_EXT(PS2DE::BodyParameter, PhysicsServer2D::BodyParameter);
VARIANT_ENUM_CAST_EXT(PS2DE::BodyDampMode, PhysicsServer2D::BodyDampMode);
VARIANT_ENUM_CAST_EXT(PS2DE::BodyState, PhysicsServer2D::BodyState);
VARIANT_ENUM_CAST_EXT(PS2DE::CCDMode, PhysicsServer2D::CCDMode);
VARIANT_ENUM_CAST_EXT(PS2DE::JointParam, PhysicsServer2D::JointParam);
VARIANT_ENUM_CAST_EXT(PS2DE::JointType, PhysicsServer2D::JointType);
VARIANT_ENUM_CAST_EXT(PS2DE::PinJointParam, PhysicsServer2D::PinJointParam);
VARIANT_ENUM_CAST_EXT(PS2DE::PinJointFlag, PhysicsServer2D::PinJointFlag);
VARIANT_ENUM_CAST_EXT(PS2DE::DampedSpringParam, PhysicsServer2D::DampedSpringParam);
VARIANT_ENUM_CAST_EXT(PS2DE::AreaBodyStatus, PhysicsServer2D::AreaBodyStatus);
VARIANT_ENUM_CAST_EXT(PS2DE::ProcessInfo, PhysicsServer2D::ProcessInfo);
