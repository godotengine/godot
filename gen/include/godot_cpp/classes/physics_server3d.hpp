/**************************************************************************/
/*  physics_server3d.hpp                                                  */
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

#include <godot_cpp/classes/physics_test_motion_result3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class PhysicsDirectBodyState3D;
class PhysicsDirectSpaceState3D;
class PhysicsServer3DRenderingServerHandler;
class PhysicsTestMotionParameters3D;

class PhysicsServer3D : public Object {
	GDEXTENSION_CLASS(PhysicsServer3D, Object)

	static PhysicsServer3D *singleton;

public:
	enum JointType {
		JOINT_TYPE_PIN = 0,
		JOINT_TYPE_HINGE = 1,
		JOINT_TYPE_SLIDER = 2,
		JOINT_TYPE_CONE_TWIST = 3,
		JOINT_TYPE_6DOF = 4,
		JOINT_TYPE_MAX = 5,
	};

	enum PinJointParam {
		PIN_JOINT_BIAS = 0,
		PIN_JOINT_DAMPING = 1,
		PIN_JOINT_IMPULSE_CLAMP = 2,
	};

	enum HingeJointParam {
		HINGE_JOINT_BIAS = 0,
		HINGE_JOINT_LIMIT_UPPER = 1,
		HINGE_JOINT_LIMIT_LOWER = 2,
		HINGE_JOINT_LIMIT_BIAS = 3,
		HINGE_JOINT_LIMIT_SOFTNESS = 4,
		HINGE_JOINT_LIMIT_RELAXATION = 5,
		HINGE_JOINT_MOTOR_TARGET_VELOCITY = 6,
		HINGE_JOINT_MOTOR_MAX_IMPULSE = 7,
	};

	enum HingeJointFlag {
		HINGE_JOINT_FLAG_USE_LIMIT = 0,
		HINGE_JOINT_FLAG_ENABLE_MOTOR = 1,
	};

	enum SliderJointParam {
		SLIDER_JOINT_LINEAR_LIMIT_UPPER = 0,
		SLIDER_JOINT_LINEAR_LIMIT_LOWER = 1,
		SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS = 2,
		SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION = 3,
		SLIDER_JOINT_LINEAR_LIMIT_DAMPING = 4,
		SLIDER_JOINT_LINEAR_MOTION_SOFTNESS = 5,
		SLIDER_JOINT_LINEAR_MOTION_RESTITUTION = 6,
		SLIDER_JOINT_LINEAR_MOTION_DAMPING = 7,
		SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS = 8,
		SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION = 9,
		SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING = 10,
		SLIDER_JOINT_ANGULAR_LIMIT_UPPER = 11,
		SLIDER_JOINT_ANGULAR_LIMIT_LOWER = 12,
		SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS = 13,
		SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION = 14,
		SLIDER_JOINT_ANGULAR_LIMIT_DAMPING = 15,
		SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS = 16,
		SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION = 17,
		SLIDER_JOINT_ANGULAR_MOTION_DAMPING = 18,
		SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS = 19,
		SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION = 20,
		SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING = 21,
		SLIDER_JOINT_MAX = 22,
	};

	enum ConeTwistJointParam {
		CONE_TWIST_JOINT_SWING_SPAN = 0,
		CONE_TWIST_JOINT_TWIST_SPAN = 1,
		CONE_TWIST_JOINT_BIAS = 2,
		CONE_TWIST_JOINT_SOFTNESS = 3,
		CONE_TWIST_JOINT_RELAXATION = 4,
	};

	enum G6DOFJointAxisParam {
		G6DOF_JOINT_LINEAR_LOWER_LIMIT = 0,
		G6DOF_JOINT_LINEAR_UPPER_LIMIT = 1,
		G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS = 2,
		G6DOF_JOINT_LINEAR_RESTITUTION = 3,
		G6DOF_JOINT_LINEAR_DAMPING = 4,
		G6DOF_JOINT_LINEAR_MOTOR_TARGET_VELOCITY = 5,
		G6DOF_JOINT_LINEAR_MOTOR_FORCE_LIMIT = 6,
		G6DOF_JOINT_LINEAR_SPRING_STIFFNESS = 7,
		G6DOF_JOINT_LINEAR_SPRING_DAMPING = 8,
		G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT = 9,
		G6DOF_JOINT_ANGULAR_LOWER_LIMIT = 10,
		G6DOF_JOINT_ANGULAR_UPPER_LIMIT = 11,
		G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS = 12,
		G6DOF_JOINT_ANGULAR_DAMPING = 13,
		G6DOF_JOINT_ANGULAR_RESTITUTION = 14,
		G6DOF_JOINT_ANGULAR_FORCE_LIMIT = 15,
		G6DOF_JOINT_ANGULAR_ERP = 16,
		G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY = 17,
		G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT = 18,
		G6DOF_JOINT_ANGULAR_SPRING_STIFFNESS = 19,
		G6DOF_JOINT_ANGULAR_SPRING_DAMPING = 20,
		G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT = 21,
		G6DOF_JOINT_MAX = 22,
	};

	enum G6DOFJointAxisFlag {
		G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT = 0,
		G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT = 1,
		G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING = 2,
		G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING = 3,
		G6DOF_JOINT_FLAG_ENABLE_MOTOR = 4,
		G6DOF_JOINT_FLAG_ENABLE_LINEAR_MOTOR = 5,
		G6DOF_JOINT_FLAG_MAX = 6,
	};

	enum ShapeType {
		SHAPE_WORLD_BOUNDARY = 0,
		SHAPE_SEPARATION_RAY = 1,
		SHAPE_SPHERE = 2,
		SHAPE_BOX = 3,
		SHAPE_CAPSULE = 4,
		SHAPE_CYLINDER = 5,
		SHAPE_CONVEX_POLYGON = 6,
		SHAPE_CONCAVE_POLYGON = 7,
		SHAPE_HEIGHTMAP = 8,
		SHAPE_SOFT_BODY = 9,
		SHAPE_CUSTOM = 10,
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
		AREA_PARAM_WIND_FORCE_MAGNITUDE = 10,
		AREA_PARAM_WIND_SOURCE = 11,
		AREA_PARAM_WIND_DIRECTION = 12,
		AREA_PARAM_WIND_ATTENUATION_FACTOR = 13,
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

	enum AreaBodyStatus {
		AREA_BODY_ADDED = 0,
		AREA_BODY_REMOVED = 1,
	};

	enum ProcessInfo {
		INFO_ACTIVE_OBJECTS = 0,
		INFO_COLLISION_PAIRS = 1,
		INFO_ISLAND_COUNT = 2,
	};

	enum SpaceParameter {
		SPACE_PARAM_CONTACT_RECYCLE_RADIUS = 0,
		SPACE_PARAM_CONTACT_MAX_SEPARATION = 1,
		SPACE_PARAM_CONTACT_MAX_ALLOWED_PENETRATION = 2,
		SPACE_PARAM_CONTACT_DEFAULT_BIAS = 3,
		SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD = 4,
		SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD = 5,
		SPACE_PARAM_BODY_TIME_TO_SLEEP = 6,
		SPACE_PARAM_SOLVER_ITERATIONS = 7,
	};

	enum BodyAxis {
		BODY_AXIS_LINEAR_X = 1,
		BODY_AXIS_LINEAR_Y = 2,
		BODY_AXIS_LINEAR_Z = 4,
		BODY_AXIS_ANGULAR_X = 8,
		BODY_AXIS_ANGULAR_Y = 16,
		BODY_AXIS_ANGULAR_Z = 32,
	};

	static PhysicsServer3D *get_singleton();

	RID world_boundary_shape_create();
	RID separation_ray_shape_create();
	RID sphere_shape_create();
	RID box_shape_create();
	RID capsule_shape_create();
	RID cylinder_shape_create();
	RID convex_polygon_shape_create();
	RID concave_polygon_shape_create();
	RID heightmap_shape_create();
	RID custom_shape_create();
	void shape_set_data(const RID &p_shape, const Variant &p_data);
	void shape_set_margin(const RID &p_shape, float p_margin);
	PhysicsServer3D::ShapeType shape_get_type(const RID &p_shape) const;
	Variant shape_get_data(const RID &p_shape) const;
	float shape_get_margin(const RID &p_shape) const;
	RID space_create();
	void space_set_active(const RID &p_space, bool p_active);
	bool space_is_active(const RID &p_space) const;
	void space_set_param(const RID &p_space, PhysicsServer3D::SpaceParameter p_param, float p_value);
	float space_get_param(const RID &p_space, PhysicsServer3D::SpaceParameter p_param) const;
	PhysicsDirectSpaceState3D *space_get_direct_state(const RID &p_space);
	RID area_create();
	void area_set_space(const RID &p_area, const RID &p_space);
	RID area_get_space(const RID &p_area) const;
	void area_add_shape(const RID &p_area, const RID &p_shape, const Transform3D &p_transform = Transform3D(), bool p_disabled = false);
	void area_set_shape(const RID &p_area, int32_t p_shape_idx, const RID &p_shape);
	void area_set_shape_transform(const RID &p_area, int32_t p_shape_idx, const Transform3D &p_transform);
	void area_set_shape_disabled(const RID &p_area, int32_t p_shape_idx, bool p_disabled);
	int32_t area_get_shape_count(const RID &p_area) const;
	RID area_get_shape(const RID &p_area, int32_t p_shape_idx) const;
	Transform3D area_get_shape_transform(const RID &p_area, int32_t p_shape_idx) const;
	void area_remove_shape(const RID &p_area, int32_t p_shape_idx);
	void area_clear_shapes(const RID &p_area);
	void area_set_collision_layer(const RID &p_area, uint32_t p_layer);
	uint32_t area_get_collision_layer(const RID &p_area) const;
	void area_set_collision_mask(const RID &p_area, uint32_t p_mask);
	uint32_t area_get_collision_mask(const RID &p_area) const;
	void area_set_param(const RID &p_area, PhysicsServer3D::AreaParameter p_param, const Variant &p_value);
	void area_set_transform(const RID &p_area, const Transform3D &p_transform);
	Variant area_get_param(const RID &p_area, PhysicsServer3D::AreaParameter p_param) const;
	Transform3D area_get_transform(const RID &p_area) const;
	void area_attach_object_instance_id(const RID &p_area, uint64_t p_id);
	uint64_t area_get_object_instance_id(const RID &p_area) const;
	void area_set_monitor_callback(const RID &p_area, const Callable &p_callback);
	void area_set_area_monitor_callback(const RID &p_area, const Callable &p_callback);
	void area_set_monitorable(const RID &p_area, bool p_monitorable);
	void area_set_ray_pickable(const RID &p_area, bool p_enable);
	RID body_create();
	void body_set_space(const RID &p_body, const RID &p_space);
	RID body_get_space(const RID &p_body) const;
	void body_set_mode(const RID &p_body, PhysicsServer3D::BodyMode p_mode);
	PhysicsServer3D::BodyMode body_get_mode(const RID &p_body) const;
	void body_set_collision_layer(const RID &p_body, uint32_t p_layer);
	uint32_t body_get_collision_layer(const RID &p_body) const;
	void body_set_collision_mask(const RID &p_body, uint32_t p_mask);
	uint32_t body_get_collision_mask(const RID &p_body) const;
	void body_set_collision_priority(const RID &p_body, float p_priority);
	float body_get_collision_priority(const RID &p_body) const;
	void body_add_shape(const RID &p_body, const RID &p_shape, const Transform3D &p_transform = Transform3D(), bool p_disabled = false);
	void body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape);
	void body_set_shape_transform(const RID &p_body, int32_t p_shape_idx, const Transform3D &p_transform);
	void body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled);
	int32_t body_get_shape_count(const RID &p_body) const;
	RID body_get_shape(const RID &p_body, int32_t p_shape_idx) const;
	Transform3D body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const;
	void body_remove_shape(const RID &p_body, int32_t p_shape_idx);
	void body_clear_shapes(const RID &p_body);
	void body_attach_object_instance_id(const RID &p_body, uint64_t p_id);
	uint64_t body_get_object_instance_id(const RID &p_body) const;
	void body_set_enable_continuous_collision_detection(const RID &p_body, bool p_enable);
	bool body_is_continuous_collision_detection_enabled(const RID &p_body) const;
	void body_set_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param, const Variant &p_value);
	Variant body_get_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param) const;
	void body_reset_mass_properties(const RID &p_body);
	void body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state, const Variant &p_value);
	Variant body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const;
	void body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse);
	void body_apply_impulse(const RID &p_body, const Vector3 &p_impulse, const Vector3 &p_position = Vector3(0, 0, 0));
	void body_apply_torque_impulse(const RID &p_body, const Vector3 &p_impulse);
	void body_apply_central_force(const RID &p_body, const Vector3 &p_force);
	void body_apply_force(const RID &p_body, const Vector3 &p_force, const Vector3 &p_position = Vector3(0, 0, 0));
	void body_apply_torque(const RID &p_body, const Vector3 &p_torque);
	void body_add_constant_central_force(const RID &p_body, const Vector3 &p_force);
	void body_add_constant_force(const RID &p_body, const Vector3 &p_force, const Vector3 &p_position = Vector3(0, 0, 0));
	void body_add_constant_torque(const RID &p_body, const Vector3 &p_torque);
	void body_set_constant_force(const RID &p_body, const Vector3 &p_force);
	Vector3 body_get_constant_force(const RID &p_body) const;
	void body_set_constant_torque(const RID &p_body, const Vector3 &p_torque);
	Vector3 body_get_constant_torque(const RID &p_body) const;
	void body_set_axis_velocity(const RID &p_body, const Vector3 &p_axis_velocity);
	void body_set_axis_lock(const RID &p_body, PhysicsServer3D::BodyAxis p_axis, bool p_lock);
	bool body_is_axis_locked(const RID &p_body, PhysicsServer3D::BodyAxis p_axis) const;
	void body_add_collision_exception(const RID &p_body, const RID &p_excepted_body);
	void body_remove_collision_exception(const RID &p_body, const RID &p_excepted_body);
	void body_set_max_contacts_reported(const RID &p_body, int32_t p_amount);
	int32_t body_get_max_contacts_reported(const RID &p_body) const;
	void body_set_omit_force_integration(const RID &p_body, bool p_enable);
	bool body_is_omitting_force_integration(const RID &p_body) const;
	void body_set_state_sync_callback(const RID &p_body, const Callable &p_callable);
	void body_set_force_integration_callback(const RID &p_body, const Callable &p_callable, const Variant &p_userdata = nullptr);
	void body_set_ray_pickable(const RID &p_body, bool p_enable);
	bool body_test_motion(const RID &p_body, const Ref<PhysicsTestMotionParameters3D> &p_parameters, const Ref<PhysicsTestMotionResult3D> &p_result = nullptr);
	PhysicsDirectBodyState3D *body_get_direct_state(const RID &p_body);
	RID soft_body_create();
	void soft_body_update_rendering_server(const RID &p_body, PhysicsServer3DRenderingServerHandler *p_rendering_server_handler);
	void soft_body_set_space(const RID &p_body, const RID &p_space);
	RID soft_body_get_space(const RID &p_body) const;
	void soft_body_set_mesh(const RID &p_body, const RID &p_mesh);
	AABB soft_body_get_bounds(const RID &p_body) const;
	void soft_body_set_collision_layer(const RID &p_body, uint32_t p_layer);
	uint32_t soft_body_get_collision_layer(const RID &p_body) const;
	void soft_body_set_collision_mask(const RID &p_body, uint32_t p_mask);
	uint32_t soft_body_get_collision_mask(const RID &p_body) const;
	void soft_body_add_collision_exception(const RID &p_body, const RID &p_body_b);
	void soft_body_remove_collision_exception(const RID &p_body, const RID &p_body_b);
	void soft_body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state, const Variant &p_variant);
	Variant soft_body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const;
	void soft_body_set_transform(const RID &p_body, const Transform3D &p_transform);
	void soft_body_set_ray_pickable(const RID &p_body, bool p_enable);
	void soft_body_set_simulation_precision(const RID &p_body, int32_t p_simulation_precision);
	int32_t soft_body_get_simulation_precision(const RID &p_body) const;
	void soft_body_set_total_mass(const RID &p_body, float p_total_mass);
	float soft_body_get_total_mass(const RID &p_body) const;
	void soft_body_set_linear_stiffness(const RID &p_body, float p_stiffness);
	float soft_body_get_linear_stiffness(const RID &p_body) const;
	void soft_body_set_shrinking_factor(const RID &p_body, float p_shrinking_factor);
	float soft_body_get_shrinking_factor(const RID &p_body) const;
	void soft_body_set_pressure_coefficient(const RID &p_body, float p_pressure_coefficient);
	float soft_body_get_pressure_coefficient(const RID &p_body) const;
	void soft_body_set_damping_coefficient(const RID &p_body, float p_damping_coefficient);
	float soft_body_get_damping_coefficient(const RID &p_body) const;
	void soft_body_set_drag_coefficient(const RID &p_body, float p_drag_coefficient);
	float soft_body_get_drag_coefficient(const RID &p_body) const;
	void soft_body_move_point(const RID &p_body, int32_t p_point_index, const Vector3 &p_global_position);
	Vector3 soft_body_get_point_global_position(const RID &p_body, int32_t p_point_index) const;
	void soft_body_remove_all_pinned_points(const RID &p_body);
	void soft_body_pin_point(const RID &p_body, int32_t p_point_index, bool p_pin);
	bool soft_body_is_point_pinned(const RID &p_body, int32_t p_point_index) const;
	void soft_body_apply_point_impulse(const RID &p_body, int32_t p_point_index, const Vector3 &p_impulse);
	void soft_body_apply_point_force(const RID &p_body, int32_t p_point_index, const Vector3 &p_force);
	void soft_body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse);
	void soft_body_apply_central_force(const RID &p_body, const Vector3 &p_force);
	RID joint_create();
	void joint_clear(const RID &p_joint);
	void joint_make_pin(const RID &p_joint, const RID &p_body_A, const Vector3 &p_local_A, const RID &p_body_B, const Vector3 &p_local_B);
	void pin_joint_set_param(const RID &p_joint, PhysicsServer3D::PinJointParam p_param, float p_value);
	float pin_joint_get_param(const RID &p_joint, PhysicsServer3D::PinJointParam p_param) const;
	void pin_joint_set_local_a(const RID &p_joint, const Vector3 &p_local_A);
	Vector3 pin_joint_get_local_a(const RID &p_joint) const;
	void pin_joint_set_local_b(const RID &p_joint, const Vector3 &p_local_B);
	Vector3 pin_joint_get_local_b(const RID &p_joint) const;
	void joint_make_hinge(const RID &p_joint, const RID &p_body_A, const Transform3D &p_hinge_A, const RID &p_body_B, const Transform3D &p_hinge_B);
	void hinge_joint_set_param(const RID &p_joint, PhysicsServer3D::HingeJointParam p_param, float p_value);
	float hinge_joint_get_param(const RID &p_joint, PhysicsServer3D::HingeJointParam p_param) const;
	void hinge_joint_set_flag(const RID &p_joint, PhysicsServer3D::HingeJointFlag p_flag, bool p_enabled);
	bool hinge_joint_get_flag(const RID &p_joint, PhysicsServer3D::HingeJointFlag p_flag) const;
	void joint_make_slider(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B);
	void slider_joint_set_param(const RID &p_joint, PhysicsServer3D::SliderJointParam p_param, float p_value);
	float slider_joint_get_param(const RID &p_joint, PhysicsServer3D::SliderJointParam p_param) const;
	void joint_make_cone_twist(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B);
	void cone_twist_joint_set_param(const RID &p_joint, PhysicsServer3D::ConeTwistJointParam p_param, float p_value);
	float cone_twist_joint_get_param(const RID &p_joint, PhysicsServer3D::ConeTwistJointParam p_param) const;
	PhysicsServer3D::JointType joint_get_type(const RID &p_joint) const;
	void joint_set_solver_priority(const RID &p_joint, int32_t p_priority);
	int32_t joint_get_solver_priority(const RID &p_joint) const;
	void joint_disable_collisions_between_bodies(const RID &p_joint, bool p_disable);
	bool joint_is_disabled_collisions_between_bodies(const RID &p_joint) const;
	void joint_make_generic_6dof(const RID &p_joint, const RID &p_body_A, const Transform3D &p_local_ref_A, const RID &p_body_B, const Transform3D &p_local_ref_B);
	void generic_6dof_joint_set_param(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param, float p_value);
	float generic_6dof_joint_get_param(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param) const;
	void generic_6dof_joint_set_flag(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag, bool p_enable);
	bool generic_6dof_joint_get_flag(const RID &p_joint, Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag) const;
	void free_rid(const RID &p_rid);
	void set_active(bool p_active);
	int32_t get_process_info(PhysicsServer3D::ProcessInfo p_process_info);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~PhysicsServer3D();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(PhysicsServer3D::JointType);
VARIANT_ENUM_CAST(PhysicsServer3D::PinJointParam);
VARIANT_ENUM_CAST(PhysicsServer3D::HingeJointParam);
VARIANT_ENUM_CAST(PhysicsServer3D::HingeJointFlag);
VARIANT_ENUM_CAST(PhysicsServer3D::SliderJointParam);
VARIANT_ENUM_CAST(PhysicsServer3D::ConeTwistJointParam);
VARIANT_ENUM_CAST(PhysicsServer3D::G6DOFJointAxisParam);
VARIANT_ENUM_CAST(PhysicsServer3D::G6DOFJointAxisFlag);
VARIANT_ENUM_CAST(PhysicsServer3D::ShapeType);
VARIANT_ENUM_CAST(PhysicsServer3D::AreaParameter);
VARIANT_ENUM_CAST(PhysicsServer3D::AreaSpaceOverrideMode);
VARIANT_ENUM_CAST(PhysicsServer3D::BodyMode);
VARIANT_ENUM_CAST(PhysicsServer3D::BodyParameter);
VARIANT_ENUM_CAST(PhysicsServer3D::BodyDampMode);
VARIANT_ENUM_CAST(PhysicsServer3D::BodyState);
VARIANT_ENUM_CAST(PhysicsServer3D::AreaBodyStatus);
VARIANT_ENUM_CAST(PhysicsServer3D::ProcessInfo);
VARIANT_ENUM_CAST(PhysicsServer3D::SpaceParameter);
VARIANT_ENUM_CAST(PhysicsServer3D::BodyAxis);

