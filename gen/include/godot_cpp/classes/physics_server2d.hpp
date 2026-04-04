/**************************************************************************/
/*  physics_server2d.hpp                                                  */
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

#pragma once

#include <godot_cpp/classes/physics_test_motion_result2d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class PhysicsDirectBodyState2D;
class PhysicsDirectSpaceState2D;
class PhysicsTestMotionParameters2D;

class PhysicsServer2D : public Object {
	GDEXTENSION_CLASS(PhysicsServer2D, Object)

	static PhysicsServer2D *singleton;

public:
	enum SpaceParameter {
		SPACE_PARAM_CONTACT_RECYCLE_RADIUS = 0,
		SPACE_PARAM_CONTACT_MAX_SEPARATION = 1,
		SPACE_PARAM_CONTACT_MAX_ALLOWED_PENETRATION = 2,
		SPACE_PARAM_CONTACT_DEFAULT_BIAS = 3,
		SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD = 4,
		SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD = 5,
		SPACE_PARAM_BODY_TIME_TO_SLEEP = 6,
		SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS = 7,
		SPACE_PARAM_SOLVER_ITERATIONS = 8,
	};

	enum ShapeType {
		SHAPE_WORLD_BOUNDARY = 0,
		SHAPE_SEPARATION_RAY = 1,
		SHAPE_SEGMENT = 2,
		SHAPE_CIRCLE = 3,
		SHAPE_RECTANGLE = 4,
		SHAPE_CAPSULE = 5,
		SHAPE_CONVEX_POLYGON = 6,
		SHAPE_CONCAVE_POLYGON = 7,
		SHAPE_CUSTOM = 8,
	};

	enum AreaParameter {
		AREA_PARAM_GRAVITY_OVERRIDE_MODE = 0,
		AREA_PARAM_GRAVITY = 1,
		AREA_PARAM_GRAVITY_VECTOR = 2,
		AREA_PARAM_GRAVITY_IS_POINT = 3,
		AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE = 4,
		AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE = 5,
		AREA_PARAM_LINEAR_DAMP = 6,
		AREA_PARAM_ANGULAR_DAMP_OVERRIDE_MODE = 7,
		AREA_PARAM_ANGULAR_DAMP = 8,
		AREA_PARAM_PRIORITY = 9,
	};

	enum AreaSpaceOverrideMode {
		AREA_SPACE_OVERRIDE_DISABLED = 0,
		AREA_SPACE_OVERRIDE_COMBINE = 1,
		AREA_SPACE_OVERRIDE_COMBINE_REPLACE = 2,
		AREA_SPACE_OVERRIDE_REPLACE = 3,
		AREA_SPACE_OVERRIDE_REPLACE_COMBINE = 4,
	};

	enum BodyMode {
		BODY_MODE_STATIC = 0,
		BODY_MODE_KINEMATIC = 1,
		BODY_MODE_RIGID = 2,
		BODY_MODE_RIGID_LINEAR = 3,
	};

	enum BodyParameter {
		BODY_PARAM_BOUNCE = 0,
		BODY_PARAM_FRICTION = 1,
		BODY_PARAM_MASS = 2,
		BODY_PARAM_INERTIA = 3,
		BODY_PARAM_CENTER_OF_MASS = 4,
		BODY_PARAM_GRAVITY_SCALE = 5,
		BODY_PARAM_LINEAR_DAMP_MODE = 6,
		BODY_PARAM_ANGULAR_DAMP_MODE = 7,
		BODY_PARAM_LINEAR_DAMP = 8,
		BODY_PARAM_ANGULAR_DAMP = 9,
		BODY_PARAM_MAX = 10,
	};

	enum BodyDampMode {
		BODY_DAMP_MODE_COMBINE = 0,
		BODY_DAMP_MODE_REPLACE = 1,
	};

	enum BodyState {
		BODY_STATE_TRANSFORM = 0,
		BODY_STATE_LINEAR_VELOCITY = 1,
		BODY_STATE_ANGULAR_VELOCITY = 2,
		BODY_STATE_SLEEPING = 3,
		BODY_STATE_CAN_SLEEP = 4,
	};

	enum JointType {
		JOINT_TYPE_PIN = 0,
		JOINT_TYPE_GROOVE = 1,
		JOINT_TYPE_DAMPED_SPRING = 2,
		JOINT_TYPE_MAX = 3,
	};

	enum JointParam {
		JOINT_PARAM_BIAS = 0,
		JOINT_PARAM_MAX_BIAS = 1,
		JOINT_PARAM_MAX_FORCE = 2,
	};

	enum PinJointParam {
		PIN_JOINT_SOFTNESS = 0,
		PIN_JOINT_LIMIT_UPPER = 1,
		PIN_JOINT_LIMIT_LOWER = 2,
		PIN_JOINT_MOTOR_TARGET_VELOCITY = 3,
	};

	enum PinJointFlag {
		PIN_JOINT_FLAG_ANGULAR_LIMIT_ENABLED = 0,
		PIN_JOINT_FLAG_MOTOR_ENABLED = 1,
	};

	enum DampedSpringParam {
		DAMPED_SPRING_REST_LENGTH = 0,
		DAMPED_SPRING_STIFFNESS = 1,
		DAMPED_SPRING_DAMPING = 2,
	};

	enum CCDMode {
		CCD_MODE_DISABLED = 0,
		CCD_MODE_CAST_RAY = 1,
		CCD_MODE_CAST_SHAPE = 2,
	};

	enum AreaBodyStatus {
		AREA_BODY_ADDED = 0,
		AREA_BODY_REMOVED = 1,
	};

	enum ProcessInfo {
		INFO_ACTIVE_OBJECTS = 0,
		INFO_COLLISION_PAIRS = 1,
		INFO_ISLAND_COUNT = 2,
	};

	static PhysicsServer2D *get_singleton();

	RID world_boundary_shape_create();
	RID separation_ray_shape_create();
	RID segment_shape_create();
	RID circle_shape_create();
	RID rectangle_shape_create();
	RID capsule_shape_create();
	RID convex_polygon_shape_create();
	RID concave_polygon_shape_create();
	void shape_set_data(const RID &p_shape, const Variant &p_data);
	PhysicsServer2D::ShapeType shape_get_type(const RID &p_shape) const;
	Variant shape_get_data(const RID &p_shape) const;
	RID space_create();
	void space_set_active(const RID &p_space, bool p_active);
	bool space_is_active(const RID &p_space) const;
	void space_set_param(const RID &p_space, PhysicsServer2D::SpaceParameter p_param, float p_value);
	float space_get_param(const RID &p_space, PhysicsServer2D::SpaceParameter p_param) const;
	PhysicsDirectSpaceState2D *space_get_direct_state(const RID &p_space);
	RID area_create();
	void area_set_space(const RID &p_area, const RID &p_space);
	RID area_get_space(const RID &p_area) const;
	void area_add_shape(const RID &p_area, const RID &p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false);
	void area_set_shape(const RID &p_area, int32_t p_shape_idx, const RID &p_shape);
	void area_set_shape_transform(const RID &p_area, int32_t p_shape_idx, const Transform2D &p_transform);
	void area_set_shape_disabled(const RID &p_area, int32_t p_shape_idx, bool p_disabled);
	int32_t area_get_shape_count(const RID &p_area) const;
	RID area_get_shape(const RID &p_area, int32_t p_shape_idx) const;
	Transform2D area_get_shape_transform(const RID &p_area, int32_t p_shape_idx) const;
	void area_remove_shape(const RID &p_area, int32_t p_shape_idx);
	void area_clear_shapes(const RID &p_area);
	void area_set_collision_layer(const RID &p_area, uint32_t p_layer);
	uint32_t area_get_collision_layer(const RID &p_area) const;
	void area_set_collision_mask(const RID &p_area, uint32_t p_mask);
	uint32_t area_get_collision_mask(const RID &p_area) const;
	void area_set_param(const RID &p_area, PhysicsServer2D::AreaParameter p_param, const Variant &p_value);
	void area_set_transform(const RID &p_area, const Transform2D &p_transform);
	Variant area_get_param(const RID &p_area, PhysicsServer2D::AreaParameter p_param) const;
	Transform2D area_get_transform(const RID &p_area) const;
	void area_attach_object_instance_id(const RID &p_area, uint64_t p_id);
	uint64_t area_get_object_instance_id(const RID &p_area) const;
	void area_attach_canvas_instance_id(const RID &p_area, uint64_t p_id);
	uint64_t area_get_canvas_instance_id(const RID &p_area) const;
	void area_set_monitor_callback(const RID &p_area, const Callable &p_callback);
	void area_set_area_monitor_callback(const RID &p_area, const Callable &p_callback);
	void area_set_monitorable(const RID &p_area, bool p_monitorable);
	RID body_create();
	void body_set_space(const RID &p_body, const RID &p_space);
	RID body_get_space(const RID &p_body) const;
	void body_set_mode(const RID &p_body, PhysicsServer2D::BodyMode p_mode);
	PhysicsServer2D::BodyMode body_get_mode(const RID &p_body) const;
	void body_add_shape(const RID &p_body, const RID &p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false);
	void body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape);
	void body_set_shape_transform(const RID &p_body, int32_t p_shape_idx, const Transform2D &p_transform);
	int32_t body_get_shape_count(const RID &p_body) const;
	RID body_get_shape(const RID &p_body, int32_t p_shape_idx) const;
	Transform2D body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const;
	void body_remove_shape(const RID &p_body, int32_t p_shape_idx);
	void body_clear_shapes(const RID &p_body);
	void body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled);
	void body_set_shape_as_one_way_collision(const RID &p_body, int32_t p_shape_idx, bool p_enable, float p_margin);
	void body_attach_object_instance_id(const RID &p_body, uint64_t p_id);
	uint64_t body_get_object_instance_id(const RID &p_body) const;
	void body_attach_canvas_instance_id(const RID &p_body, uint64_t p_id);
	uint64_t body_get_canvas_instance_id(const RID &p_body) const;
	void body_set_continuous_collision_detection_mode(const RID &p_body, PhysicsServer2D::CCDMode p_mode);
	PhysicsServer2D::CCDMode body_get_continuous_collision_detection_mode(const RID &p_body) const;
	void body_set_collision_layer(const RID &p_body, uint32_t p_layer);
	uint32_t body_get_collision_layer(const RID &p_body) const;
	void body_set_collision_mask(const RID &p_body, uint32_t p_mask);
	uint32_t body_get_collision_mask(const RID &p_body) const;
	void body_set_collision_priority(const RID &p_body, float p_priority);
	float body_get_collision_priority(const RID &p_body) const;
	void body_set_param(const RID &p_body, PhysicsServer2D::BodyParameter p_param, const Variant &p_value);
	Variant body_get_param(const RID &p_body, PhysicsServer2D::BodyParameter p_param) const;
	void body_reset_mass_properties(const RID &p_body);
	void body_set_state(const RID &p_body, PhysicsServer2D::BodyState p_state, const Variant &p_value);
	Variant body_get_state(const RID &p_body, PhysicsServer2D::BodyState p_state) const;
	void body_apply_central_impulse(const RID &p_body, const Vector2 &p_impulse);
	void body_apply_torque_impulse(const RID &p_body, float p_impulse);
	void body_apply_impulse(const RID &p_body, const Vector2 &p_impulse, const Vector2 &p_position = Vector2(0, 0));
	void body_apply_central_force(const RID &p_body, const Vector2 &p_force);
	void body_apply_force(const RID &p_body, const Vector2 &p_force, const Vector2 &p_position = Vector2(0, 0));
	void body_apply_torque(const RID &p_body, float p_torque);
	void body_add_constant_central_force(const RID &p_body, const Vector2 &p_force);
	void body_add_constant_force(const RID &p_body, const Vector2 &p_force, const Vector2 &p_position = Vector2(0, 0));
	void body_add_constant_torque(const RID &p_body, float p_torque);
	void body_set_constant_force(const RID &p_body, const Vector2 &p_force);
	Vector2 body_get_constant_force(const RID &p_body) const;
	void body_set_constant_torque(const RID &p_body, float p_torque);
	float body_get_constant_torque(const RID &p_body) const;
	void body_set_axis_velocity(const RID &p_body, const Vector2 &p_axis_velocity);
	void body_add_collision_exception(const RID &p_body, const RID &p_excepted_body);
	void body_remove_collision_exception(const RID &p_body, const RID &p_excepted_body);
	void body_set_max_contacts_reported(const RID &p_body, int32_t p_amount);
	int32_t body_get_max_contacts_reported(const RID &p_body) const;
	void body_set_omit_force_integration(const RID &p_body, bool p_enable);
	bool body_is_omitting_force_integration(const RID &p_body) const;
	void body_set_state_sync_callback(const RID &p_body, const Callable &p_callable);
	void body_set_force_integration_callback(const RID &p_body, const Callable &p_callable, const Variant &p_userdata = nullptr);
	bool body_test_motion(const RID &p_body, const Ref<PhysicsTestMotionParameters2D> &p_parameters, const Ref<PhysicsTestMotionResult2D> &p_result = nullptr);
	PhysicsDirectBodyState2D *body_get_direct_state(const RID &p_body);
	RID joint_create();
	void joint_clear(const RID &p_joint);
	void joint_set_param(const RID &p_joint, PhysicsServer2D::JointParam p_param, float p_value);
	float joint_get_param(const RID &p_joint, PhysicsServer2D::JointParam p_param) const;
	void joint_disable_collisions_between_bodies(const RID &p_joint, bool p_disable);
	bool joint_is_disabled_collisions_between_bodies(const RID &p_joint) const;
	void joint_make_pin(const RID &p_joint, const Vector2 &p_anchor, const RID &p_body_a, const RID &p_body_b = RID());
	void joint_make_groove(const RID &p_joint, const Vector2 &p_groove1_a, const Vector2 &p_groove2_a, const Vector2 &p_anchor_b, const RID &p_body_a = RID(), const RID &p_body_b = RID());
	void joint_make_damped_spring(const RID &p_joint, const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, const RID &p_body_a, const RID &p_body_b = RID());
	void pin_joint_set_flag(const RID &p_joint, PhysicsServer2D::PinJointFlag p_flag, bool p_enabled);
	bool pin_joint_get_flag(const RID &p_joint, PhysicsServer2D::PinJointFlag p_flag) const;
	void pin_joint_set_param(const RID &p_joint, PhysicsServer2D::PinJointParam p_param, float p_value);
	float pin_joint_get_param(const RID &p_joint, PhysicsServer2D::PinJointParam p_param) const;
	void damped_spring_joint_set_param(const RID &p_joint, PhysicsServer2D::DampedSpringParam p_param, float p_value);
	float damped_spring_joint_get_param(const RID &p_joint, PhysicsServer2D::DampedSpringParam p_param) const;
	PhysicsServer2D::JointType joint_get_type(const RID &p_joint) const;
	void free_rid(const RID &p_rid);
	void set_active(bool p_active);
	int32_t get_process_info(PhysicsServer2D::ProcessInfo p_process_info);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~PhysicsServer2D();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(PhysicsServer2D::SpaceParameter);
VARIANT_ENUM_CAST(PhysicsServer2D::ShapeType);
VARIANT_ENUM_CAST(PhysicsServer2D::AreaParameter);
VARIANT_ENUM_CAST(PhysicsServer2D::AreaSpaceOverrideMode);
VARIANT_ENUM_CAST(PhysicsServer2D::BodyMode);
VARIANT_ENUM_CAST(PhysicsServer2D::BodyParameter);
VARIANT_ENUM_CAST(PhysicsServer2D::BodyDampMode);
VARIANT_ENUM_CAST(PhysicsServer2D::BodyState);
VARIANT_ENUM_CAST(PhysicsServer2D::JointType);
VARIANT_ENUM_CAST(PhysicsServer2D::JointParam);
VARIANT_ENUM_CAST(PhysicsServer2D::PinJointParam);
VARIANT_ENUM_CAST(PhysicsServer2D::PinJointFlag);
VARIANT_ENUM_CAST(PhysicsServer2D::DampedSpringParam);
VARIANT_ENUM_CAST(PhysicsServer2D::CCDMode);
VARIANT_ENUM_CAST(PhysicsServer2D::AreaBodyStatus);
VARIANT_ENUM_CAST(PhysicsServer2D::ProcessInfo);

