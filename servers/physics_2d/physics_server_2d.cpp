/**************************************************************************/
/*  physics_server_2d.cpp                                                 */
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

#include "physics_server_2d.h"
#include "physics_server_2d.compat.inc"

#include "core/config/project_settings.h"
#include "core/object/class_db.h"

PhysicsServer2D *PhysicsServer2D::singleton = nullptr;

PhysicsServer2D *PhysicsServer2D::get_singleton() {
	return singleton;
}

bool PhysicsServer2D::_body_test_motion(RID p_body, RequiredParam<PhysicsTestMotionParameters2D> rp_parameters, const Ref<PhysicsTestMotionResult2D> &p_result) {
	EXTRACT_PARAM_OR_FAIL_V(p_parameters, rp_parameters, false);

	PS2DT::MotionResult *result_ptr = nullptr;
	if (p_result.is_valid()) {
		result_ptr = p_result->get_result_ptr();
	}

	return body_test_motion(p_body, p_parameters->get_parameters(), result_ptr);
}

void PhysicsServer2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("world_boundary_shape_create"), &PhysicsServer2D::world_boundary_shape_create);
	ClassDB::bind_method(D_METHOD("separation_ray_shape_create"), &PhysicsServer2D::separation_ray_shape_create);
	ClassDB::bind_method(D_METHOD("segment_shape_create"), &PhysicsServer2D::segment_shape_create);
	ClassDB::bind_method(D_METHOD("circle_shape_create"), &PhysicsServer2D::circle_shape_create);
	ClassDB::bind_method(D_METHOD("rectangle_shape_create"), &PhysicsServer2D::rectangle_shape_create);
	ClassDB::bind_method(D_METHOD("capsule_shape_create"), &PhysicsServer2D::capsule_shape_create);
	ClassDB::bind_method(D_METHOD("convex_polygon_shape_create"), &PhysicsServer2D::convex_polygon_shape_create);
	ClassDB::bind_method(D_METHOD("concave_polygon_shape_create"), &PhysicsServer2D::concave_polygon_shape_create);

	ClassDB::bind_method(D_METHOD("shape_set_data", "shape", "data"), &PhysicsServer2D::shape_set_data);

	ClassDB::bind_method(D_METHOD("shape_get_type", "shape"), &PhysicsServer2D::shape_get_type);
	ClassDB::bind_method(D_METHOD("shape_get_data", "shape"), &PhysicsServer2D::shape_get_data);

	ClassDB::bind_method(D_METHOD("space_create"), &PhysicsServer2D::space_create);
	ClassDB::bind_method(D_METHOD("space_set_active", "space", "active"), &PhysicsServer2D::space_set_active);
	ClassDB::bind_method(D_METHOD("space_is_active", "space"), &PhysicsServer2D::space_is_active);
	ClassDB::bind_method(D_METHOD("space_set_param", "space", "param", "value"), &PhysicsServer2D::space_set_param);
	ClassDB::bind_method(D_METHOD("space_get_param", "space", "param"), &PhysicsServer2D::space_get_param);
	ClassDB::bind_method(D_METHOD("space_get_direct_state", "space"), &PhysicsServer2D::space_get_direct_state);

	ClassDB::bind_method(D_METHOD("area_create"), &PhysicsServer2D::area_create);
	ClassDB::bind_method(D_METHOD("area_set_space", "area", "space"), &PhysicsServer2D::area_set_space);
	ClassDB::bind_method(D_METHOD("area_get_space", "area"), &PhysicsServer2D::area_get_space);

	ClassDB::bind_method(D_METHOD("area_add_shape", "area", "shape", "transform", "disabled"), &PhysicsServer2D::area_add_shape, DEFVAL(Transform2D()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("area_set_shape", "area", "shape_idx", "shape"), &PhysicsServer2D::area_set_shape);
	ClassDB::bind_method(D_METHOD("area_set_shape_transform", "area", "shape_idx", "transform"), &PhysicsServer2D::area_set_shape_transform);
	ClassDB::bind_method(D_METHOD("area_set_shape_disabled", "area", "shape_idx", "disabled"), &PhysicsServer2D::area_set_shape_disabled);

	ClassDB::bind_method(D_METHOD("area_get_shape_count", "area"), &PhysicsServer2D::area_get_shape_count);
	ClassDB::bind_method(D_METHOD("area_get_shape", "area", "shape_idx"), &PhysicsServer2D::area_get_shape);
	ClassDB::bind_method(D_METHOD("area_get_shape_transform", "area", "shape_idx"), &PhysicsServer2D::area_get_shape_transform);

	ClassDB::bind_method(D_METHOD("area_remove_shape", "area", "shape_idx"), &PhysicsServer2D::area_remove_shape);
	ClassDB::bind_method(D_METHOD("area_clear_shapes", "area"), &PhysicsServer2D::area_clear_shapes);

	ClassDB::bind_method(D_METHOD("area_set_collision_layer", "area", "layer"), &PhysicsServer2D::area_set_collision_layer);
	ClassDB::bind_method(D_METHOD("area_get_collision_layer", "area"), &PhysicsServer2D::area_get_collision_layer);

	ClassDB::bind_method(D_METHOD("area_set_collision_mask", "area", "mask"), &PhysicsServer2D::area_set_collision_mask);
	ClassDB::bind_method(D_METHOD("area_get_collision_mask", "area"), &PhysicsServer2D::area_get_collision_mask);

	ClassDB::bind_method(D_METHOD("area_set_param", "area", "param", "value"), &PhysicsServer2D::area_set_param);
	ClassDB::bind_method(D_METHOD("area_set_transform", "area", "transform"), &PhysicsServer2D::area_set_transform);

	ClassDB::bind_method(D_METHOD("area_get_param", "area", "param"), &PhysicsServer2D::area_get_param);
	ClassDB::bind_method(D_METHOD("area_get_transform", "area"), &PhysicsServer2D::area_get_transform);

	ClassDB::bind_method(D_METHOD("area_attach_object_instance_id", "area", "id"), &PhysicsServer2D::area_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("area_get_object_instance_id", "area"), &PhysicsServer2D::area_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("area_attach_canvas_instance_id", "area", "id"), &PhysicsServer2D::area_attach_canvas_instance_id);
	ClassDB::bind_method(D_METHOD("area_get_canvas_instance_id", "area"), &PhysicsServer2D::area_get_canvas_instance_id);

	ClassDB::bind_method(D_METHOD("area_set_monitor_callback", "area", "callback"), &PhysicsServer2D::area_set_monitor_callback);
	ClassDB::bind_method(D_METHOD("area_set_area_monitor_callback", "area", "callback"), &PhysicsServer2D::area_set_area_monitor_callback);
	ClassDB::bind_method(D_METHOD("area_set_monitorable", "area", "monitorable"), &PhysicsServer2D::area_set_monitorable);

	ClassDB::bind_method(D_METHOD("body_create"), &PhysicsServer2D::body_create);

	ClassDB::bind_method(D_METHOD("body_set_space", "body", "space"), &PhysicsServer2D::body_set_space);
	ClassDB::bind_method(D_METHOD("body_get_space", "body"), &PhysicsServer2D::body_get_space);

	ClassDB::bind_method(D_METHOD("body_set_mode", "body", "mode"), &PhysicsServer2D::body_set_mode);
	ClassDB::bind_method(D_METHOD("body_get_mode", "body"), &PhysicsServer2D::body_get_mode);

	ClassDB::bind_method(D_METHOD("body_add_shape", "body", "shape", "transform", "disabled"), &PhysicsServer2D::body_add_shape, DEFVAL(Transform2D()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("body_set_shape", "body", "shape_idx", "shape"), &PhysicsServer2D::body_set_shape);
	ClassDB::bind_method(D_METHOD("body_set_shape_transform", "body", "shape_idx", "transform"), &PhysicsServer2D::body_set_shape_transform);

	ClassDB::bind_method(D_METHOD("body_get_shape_count", "body"), &PhysicsServer2D::body_get_shape_count);
	ClassDB::bind_method(D_METHOD("body_get_shape", "body", "shape_idx"), &PhysicsServer2D::body_get_shape);
	ClassDB::bind_method(D_METHOD("body_get_shape_transform", "body", "shape_idx"), &PhysicsServer2D::body_get_shape_transform);

	ClassDB::bind_method(D_METHOD("body_remove_shape", "body", "shape_idx"), &PhysicsServer2D::body_remove_shape);
	ClassDB::bind_method(D_METHOD("body_clear_shapes", "body"), &PhysicsServer2D::body_clear_shapes);

	ClassDB::bind_method(D_METHOD("body_set_shape_disabled", "body", "shape_idx", "disabled"), &PhysicsServer2D::body_set_shape_disabled);
	ClassDB::bind_method(D_METHOD("body_set_shape_as_one_way_collision", "body", "shape_idx", "enable", "margin", "direction"), &PhysicsServer2D::body_set_shape_as_one_way_collision, Vector2(0, 1));

	ClassDB::bind_method(D_METHOD("body_attach_object_instance_id", "body", "id"), &PhysicsServer2D::body_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("body_get_object_instance_id", "body"), &PhysicsServer2D::body_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("body_attach_canvas_instance_id", "body", "id"), &PhysicsServer2D::body_attach_canvas_instance_id);
	ClassDB::bind_method(D_METHOD("body_get_canvas_instance_id", "body"), &PhysicsServer2D::body_get_canvas_instance_id);

	ClassDB::bind_method(D_METHOD("body_set_continuous_collision_detection_mode", "body", "mode"), &PhysicsServer2D::body_set_continuous_collision_detection_mode);
	ClassDB::bind_method(D_METHOD("body_get_continuous_collision_detection_mode", "body"), &PhysicsServer2D::body_get_continuous_collision_detection_mode);

	ClassDB::bind_method(D_METHOD("body_set_collision_layer", "body", "layer"), &PhysicsServer2D::body_set_collision_layer);
	ClassDB::bind_method(D_METHOD("body_get_collision_layer", "body"), &PhysicsServer2D::body_get_collision_layer);

	ClassDB::bind_method(D_METHOD("body_set_collision_mask", "body", "mask"), &PhysicsServer2D::body_set_collision_mask);
	ClassDB::bind_method(D_METHOD("body_get_collision_mask", "body"), &PhysicsServer2D::body_get_collision_mask);

	ClassDB::bind_method(D_METHOD("body_set_collision_priority", "body", "priority"), &PhysicsServer2D::body_set_collision_priority);
	ClassDB::bind_method(D_METHOD("body_get_collision_priority", "body"), &PhysicsServer2D::body_get_collision_priority);

	ClassDB::bind_method(D_METHOD("body_set_param", "body", "param", "value"), &PhysicsServer2D::body_set_param);
	ClassDB::bind_method(D_METHOD("body_get_param", "body", "param"), &PhysicsServer2D::body_get_param);

	ClassDB::bind_method(D_METHOD("body_reset_mass_properties", "body"), &PhysicsServer2D::body_reset_mass_properties);

	ClassDB::bind_method(D_METHOD("body_set_state", "body", "state", "value"), &PhysicsServer2D::body_set_state);
	ClassDB::bind_method(D_METHOD("body_get_state", "body", "state"), &PhysicsServer2D::body_get_state);

	ClassDB::bind_method(D_METHOD("body_apply_central_impulse", "body", "impulse"), &PhysicsServer2D::body_apply_central_impulse);
	ClassDB::bind_method(D_METHOD("body_apply_torque_impulse", "body", "impulse"), &PhysicsServer2D::body_apply_torque_impulse);
	ClassDB::bind_method(D_METHOD("body_apply_impulse", "body", "impulse", "position"), &PhysicsServer2D::body_apply_impulse, Vector2());

	ClassDB::bind_method(D_METHOD("body_apply_central_force", "body", "force"), &PhysicsServer2D::body_apply_central_force);
	ClassDB::bind_method(D_METHOD("body_apply_force", "body", "force", "position"), &PhysicsServer2D::body_apply_force, Vector2());
	ClassDB::bind_method(D_METHOD("body_apply_torque", "body", "torque"), &PhysicsServer2D::body_apply_torque);

	ClassDB::bind_method(D_METHOD("body_add_constant_central_force", "body", "force"), &PhysicsServer2D::body_add_constant_central_force);
	ClassDB::bind_method(D_METHOD("body_add_constant_force", "body", "force", "position"), &PhysicsServer2D::body_add_constant_force, Vector2());
	ClassDB::bind_method(D_METHOD("body_add_constant_torque", "body", "torque"), &PhysicsServer2D::body_add_constant_torque);

	ClassDB::bind_method(D_METHOD("body_set_constant_force", "body", "force"), &PhysicsServer2D::body_set_constant_force);
	ClassDB::bind_method(D_METHOD("body_get_constant_force", "body"), &PhysicsServer2D::body_get_constant_force);

	ClassDB::bind_method(D_METHOD("body_set_constant_torque", "body", "torque"), &PhysicsServer2D::body_set_constant_torque);
	ClassDB::bind_method(D_METHOD("body_get_constant_torque", "body"), &PhysicsServer2D::body_get_constant_torque);

	ClassDB::bind_method(D_METHOD("body_set_axis_velocity", "body", "axis_velocity"), &PhysicsServer2D::body_set_axis_velocity);

	ClassDB::bind_method(D_METHOD("body_add_collision_exception", "body", "excepted_body"), &PhysicsServer2D::body_add_collision_exception);
	ClassDB::bind_method(D_METHOD("body_remove_collision_exception", "body", "excepted_body"), &PhysicsServer2D::body_remove_collision_exception);

	ClassDB::bind_method(D_METHOD("body_set_max_contacts_reported", "body", "amount"), &PhysicsServer2D::body_set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("body_get_max_contacts_reported", "body"), &PhysicsServer2D::body_get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("body_set_omit_force_integration", "body", "enable"), &PhysicsServer2D::body_set_omit_force_integration);
	ClassDB::bind_method(D_METHOD("body_is_omitting_force_integration", "body"), &PhysicsServer2D::body_is_omitting_force_integration);

	ClassDB::bind_method(D_METHOD("body_set_state_sync_callback", "body", "callable"), &PhysicsServer2D::body_set_state_sync_callback);

	ClassDB::bind_method(D_METHOD("body_set_force_integration_callback", "body", "callable", "userdata"), &PhysicsServer2D::body_set_force_integration_callback, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("body_test_motion", "body", "parameters", "result"), &PhysicsServer2D::_body_test_motion, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("body_get_direct_state", "body"), &PhysicsServer2D::body_get_direct_state);

	/* JOINT API */

	ClassDB::bind_method(D_METHOD("joint_create"), &PhysicsServer2D::joint_create);

	ClassDB::bind_method(D_METHOD("joint_clear", "joint"), &PhysicsServer2D::joint_clear);

	ClassDB::bind_method(D_METHOD("joint_set_param", "joint", "param", "value"), &PhysicsServer2D::joint_set_param);
	ClassDB::bind_method(D_METHOD("joint_get_param", "joint", "param"), &PhysicsServer2D::joint_get_param);

	ClassDB::bind_method(D_METHOD("joint_disable_collisions_between_bodies", "joint", "disable"), &PhysicsServer2D::joint_disable_collisions_between_bodies);
	ClassDB::bind_method(D_METHOD("joint_is_disabled_collisions_between_bodies", "joint"), &PhysicsServer2D::joint_is_disabled_collisions_between_bodies);

	ClassDB::bind_method(D_METHOD("joint_make_pin", "joint", "anchor", "body_a", "body_b"), &PhysicsServer2D::joint_make_pin, DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("joint_make_groove", "joint", "groove1_a", "groove2_a", "anchor_b", "body_a", "body_b"), &PhysicsServer2D::joint_make_groove, DEFVAL(RID()), DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("joint_make_damped_spring", "joint", "anchor_a", "anchor_b", "body_a", "body_b"), &PhysicsServer2D::joint_make_damped_spring, DEFVAL(RID()));

	ClassDB::bind_method(D_METHOD("pin_joint_set_flag", "joint", "flag", "enabled"), &PhysicsServer2D::pin_joint_set_flag);
	ClassDB::bind_method(D_METHOD("pin_joint_get_flag", "joint", "flag"), &PhysicsServer2D::pin_joint_get_flag);

	ClassDB::bind_method(D_METHOD("pin_joint_set_param", "joint", "param", "value"), &PhysicsServer2D::pin_joint_set_param);
	ClassDB::bind_method(D_METHOD("pin_joint_get_param", "joint", "param"), &PhysicsServer2D::pin_joint_get_param);

	ClassDB::bind_method(D_METHOD("damped_spring_joint_set_param", "joint", "param", "value"), &PhysicsServer2D::damped_spring_joint_set_param);
	ClassDB::bind_method(D_METHOD("damped_spring_joint_get_param", "joint", "param"), &PhysicsServer2D::damped_spring_joint_get_param);

	ClassDB::bind_method(D_METHOD("joint_get_type", "joint"), &PhysicsServer2D::joint_get_type);

	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &PhysicsServer2D::free_rid);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &PhysicsServer2D::set_active);

	ClassDB::bind_method(D_METHOD("get_process_info", "process_info"), &PhysicsServer2D::get_process_info);

	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_CONTACT_RECYCLE_RADIUS);
	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_CONTACT_MAX_SEPARATION);
	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_CONTACT_MAX_ALLOWED_PENETRATION);
	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_CONTACT_DEFAULT_BIAS);
	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD);
	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD);
	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_BODY_TIME_TO_SLEEP);
	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS);
	BIND_ENUM_CONSTANT(PS2DE::SPACE_PARAM_SOLVER_ITERATIONS);

	BIND_ENUM_CONSTANT(PS2DE::SHAPE_WORLD_BOUNDARY);
	BIND_ENUM_CONSTANT(PS2DE::SHAPE_SEPARATION_RAY);
	BIND_ENUM_CONSTANT(PS2DE::SHAPE_SEGMENT);
	BIND_ENUM_CONSTANT(PS2DE::SHAPE_CIRCLE);
	BIND_ENUM_CONSTANT(PS2DE::SHAPE_RECTANGLE);
	BIND_ENUM_CONSTANT(PS2DE::SHAPE_CAPSULE);
	BIND_ENUM_CONSTANT(PS2DE::SHAPE_CONVEX_POLYGON);
	BIND_ENUM_CONSTANT(PS2DE::SHAPE_CONCAVE_POLYGON);
	BIND_ENUM_CONSTANT(PS2DE::SHAPE_CUSTOM);

	BIND_ENUM_CONSTANT(PS2DE::AREA_GRAVITY_TYPE_DIRECTIONAL);
	BIND_ENUM_CONSTANT(PS2DE::AREA_GRAVITY_TYPE_POINT);

	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_GRAVITY_OVERRIDE_MODE);
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_GRAVITY);
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_GRAVITY_VECTOR);
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_GRAVITY_TYPE);
#ifndef DISABLE_DEPRECATED
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_GRAVITY_IS_POINT);
#endif // DISABLE_DEPRECATED
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE);
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE);
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_LINEAR_DAMP);
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_ANGULAR_DAMP_OVERRIDE_MODE);
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_ANGULAR_DAMP);
	BIND_ENUM_CONSTANT(PS2DE::AREA_PARAM_PRIORITY);

	BIND_ENUM_CONSTANT(PS2DE::AREA_SPACE_OVERRIDE_DISABLED);
	BIND_ENUM_CONSTANT(PS2DE::AREA_SPACE_OVERRIDE_COMBINE);
	BIND_ENUM_CONSTANT(PS2DE::AREA_SPACE_OVERRIDE_COMBINE_REPLACE);
	BIND_ENUM_CONSTANT(PS2DE::AREA_SPACE_OVERRIDE_REPLACE);
	BIND_ENUM_CONSTANT(PS2DE::AREA_SPACE_OVERRIDE_REPLACE_COMBINE);

	BIND_ENUM_CONSTANT(PS2DE::BODY_MODE_STATIC);
	BIND_ENUM_CONSTANT(PS2DE::BODY_MODE_KINEMATIC);
	BIND_ENUM_CONSTANT(PS2DE::BODY_MODE_RIGID);
	BIND_ENUM_CONSTANT(PS2DE::BODY_MODE_RIGID_LINEAR);

	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_BOUNCE);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_FRICTION);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_MASS);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_INERTIA);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_CENTER_OF_MASS);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_GRAVITY_SCALE);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_LINEAR_DAMP_MODE);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_ANGULAR_DAMP_MODE);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_LINEAR_DAMP);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_ANGULAR_DAMP);
	BIND_ENUM_CONSTANT(PS2DE::BODY_PARAM_MAX);

	BIND_ENUM_CONSTANT(PS2DE::BODY_DAMP_MODE_COMBINE);
	BIND_ENUM_CONSTANT(PS2DE::BODY_DAMP_MODE_REPLACE);

	BIND_ENUM_CONSTANT(PS2DE::BODY_STATE_TRANSFORM);
	BIND_ENUM_CONSTANT(PS2DE::BODY_STATE_LINEAR_VELOCITY);
	BIND_ENUM_CONSTANT(PS2DE::BODY_STATE_ANGULAR_VELOCITY);
	BIND_ENUM_CONSTANT(PS2DE::BODY_STATE_SLEEPING);
	BIND_ENUM_CONSTANT(PS2DE::BODY_STATE_CAN_SLEEP);

	BIND_ENUM_CONSTANT(PS2DE::JOINT_TYPE_PIN);
	BIND_ENUM_CONSTANT(PS2DE::JOINT_TYPE_GROOVE);
	BIND_ENUM_CONSTANT(PS2DE::JOINT_TYPE_DAMPED_SPRING);
	BIND_ENUM_CONSTANT(PS2DE::JOINT_TYPE_MAX);

	BIND_ENUM_CONSTANT(PS2DE::JOINT_PARAM_BIAS);
	BIND_ENUM_CONSTANT(PS2DE::JOINT_PARAM_MAX_BIAS);
	BIND_ENUM_CONSTANT(PS2DE::JOINT_PARAM_MAX_FORCE);

	BIND_ENUM_CONSTANT(PS2DE::PIN_JOINT_SOFTNESS);
	BIND_ENUM_CONSTANT(PS2DE::PIN_JOINT_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(PS2DE::PIN_JOINT_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(PS2DE::PIN_JOINT_MOTOR_TARGET_VELOCITY);

	BIND_ENUM_CONSTANT(PS2DE::PIN_JOINT_FLAG_ANGULAR_LIMIT_ENABLED);
	BIND_ENUM_CONSTANT(PS2DE::PIN_JOINT_FLAG_MOTOR_ENABLED);

	BIND_ENUM_CONSTANT(PS2DE::DAMPED_SPRING_REST_LENGTH);
	BIND_ENUM_CONSTANT(PS2DE::DAMPED_SPRING_STIFFNESS);
	BIND_ENUM_CONSTANT(PS2DE::DAMPED_SPRING_DAMPING);

	BIND_ENUM_CONSTANT(PS2DE::CCD_MODE_DISABLED);
	BIND_ENUM_CONSTANT(PS2DE::CCD_MODE_CAST_RAY);
	BIND_ENUM_CONSTANT(PS2DE::CCD_MODE_CAST_SHAPE);

	BIND_ENUM_CONSTANT(PS2DE::AREA_BODY_ADDED);
	BIND_ENUM_CONSTANT(PS2DE::AREA_BODY_REMOVED);

	BIND_ENUM_CONSTANT(PS2DE::INFO_ACTIVE_OBJECTS);
	BIND_ENUM_CONSTANT(PS2DE::INFO_COLLISION_PAIRS);
	BIND_ENUM_CONSTANT(PS2DE::INFO_ISLAND_COUNT);
}

PhysicsServer2D::PhysicsServer2D() {
	singleton = this;

	// World2D physics space
	GLOBAL_DEF_BASIC(PropertyInfo(Variant::FLOAT, "physics/2d/default_gravity", PROPERTY_HINT_RANGE, U"-4096,4096,0.001,or_less,or_greater,suffix:px/s\u00B2"), 980.0);
	GLOBAL_DEF_BASIC(PropertyInfo(Variant::VECTOR2, "physics/2d/default_gravity_vector", PROPERTY_HINT_RANGE, "-10,10,0.001,or_less,or_greater"), Vector2(0, 1));
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/default_linear_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), 0.1);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/default_angular_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"), 1.0);

	// PhysicsServer2D
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/sleep_threshold_linear", PROPERTY_HINT_RANGE, "0,10,0.001,or_greater"), 2.0);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/sleep_threshold_angular", PROPERTY_HINT_RANGE, "0,90,0.1,radians_as_degrees"), Math::deg_to_rad(8.0));
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/time_before_sleep", PROPERTY_HINT_RANGE, "0,5,0.01,or_greater,suffix:s"), 0.5);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "physics/2d/solver/solver_iterations", PROPERTY_HINT_RANGE, "1,32,1,or_greater"), 16);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/solver/contact_recycle_radius", PROPERTY_HINT_RANGE, "0,10,0.01,or_greater"), 1.0);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/solver/contact_max_separation", PROPERTY_HINT_RANGE, "0,10,0.01,or_greater"), 1.5);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/solver/contact_max_allowed_penetration", PROPERTY_HINT_RANGE, "0.01,10,0.01,or_greater"), 0.3);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/solver/default_contact_bias", PROPERTY_HINT_RANGE, "0,1,0.01"), 0.8);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "physics/2d/solver/default_constraint_bias", PROPERTY_HINT_RANGE, "0,1,0.01"), 0.2);
}

PhysicsServer2D::~PhysicsServer2D() {
	singleton = nullptr;
}
