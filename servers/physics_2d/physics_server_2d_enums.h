/**************************************************************************/
/*  physics_server_2d_enums.h                                             */
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

namespace PhysicsServer2DEnums {

/* SHAPE API */

enum ShapeType {
	SHAPE_WORLD_BOUNDARY, ///< plane:"plane"
	SHAPE_SEPARATION_RAY, ///< float:"length"
	SHAPE_SEGMENT, ///< float:"length"
	SHAPE_CIRCLE, ///< float:"radius"
	SHAPE_RECTANGLE, ///< vec3:"extents"
	SHAPE_CAPSULE,
	SHAPE_CONVEX_POLYGON, ///< array of planes:"planes"
	SHAPE_CONCAVE_POLYGON, ///< Vector2 array:"triangles" , or Dictionary with "indices" (int array) and "triangles" (Vector2 array)
	SHAPE_CUSTOM, ///< Server-Implementation based custom shape, calling shape_create() with this value will result in an error
};

/* SPACE API */

enum SpaceParameter {
	SPACE_PARAM_CONTACT_RECYCLE_RADIUS,
	SPACE_PARAM_CONTACT_MAX_SEPARATION,
	SPACE_PARAM_CONTACT_MAX_ALLOWED_PENETRATION,
	SPACE_PARAM_CONTACT_DEFAULT_BIAS,
	SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD,
	SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD,
	SPACE_PARAM_BODY_TIME_TO_SLEEP,
	SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS,
	SPACE_PARAM_SOLVER_ITERATIONS,
};

/* AREA API */

enum AreaParameter {
	AREA_PARAM_GRAVITY_OVERRIDE_MODE,
	AREA_PARAM_GRAVITY,
	AREA_PARAM_GRAVITY_VECTOR,
	AREA_PARAM_GRAVITY_IS_POINT,
	AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE,
	AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE,
	AREA_PARAM_LINEAR_DAMP,
	AREA_PARAM_ANGULAR_DAMP_OVERRIDE_MODE,
	AREA_PARAM_ANGULAR_DAMP,
	AREA_PARAM_PRIORITY
};

enum AreaSpaceOverrideMode {
	AREA_SPACE_OVERRIDE_DISABLED,
	AREA_SPACE_OVERRIDE_COMBINE,
	AREA_SPACE_OVERRIDE_COMBINE_REPLACE, // Combines, then discards all subsequent calculations
	AREA_SPACE_OVERRIDE_REPLACE,
	AREA_SPACE_OVERRIDE_REPLACE_COMBINE // Discards all previous calculations, then keeps combining
};

/* BODY API */

enum BodyMode {
	BODY_MODE_STATIC,
	BODY_MODE_KINEMATIC,
	BODY_MODE_RIGID,
	BODY_MODE_RIGID_LINEAR,
};

enum BodyDampMode {
	BODY_DAMP_MODE_COMBINE,
	BODY_DAMP_MODE_REPLACE,
};

// common body variables
enum BodyParameter {
	BODY_PARAM_BOUNCE,
	BODY_PARAM_FRICTION,
	BODY_PARAM_MASS, ///< unused for static, always infinite
	BODY_PARAM_INERTIA,
	BODY_PARAM_CENTER_OF_MASS,
	BODY_PARAM_GRAVITY_SCALE,
	BODY_PARAM_LINEAR_DAMP_MODE,
	BODY_PARAM_ANGULAR_DAMP_MODE,
	BODY_PARAM_LINEAR_DAMP,
	BODY_PARAM_ANGULAR_DAMP,
	BODY_PARAM_MAX,
};

enum BodyState {
	BODY_STATE_TRANSFORM,
	BODY_STATE_LINEAR_VELOCITY,
	BODY_STATE_ANGULAR_VELOCITY,
	BODY_STATE_SLEEPING,
	BODY_STATE_CAN_SLEEP,
};

enum CCDMode {
	CCD_MODE_DISABLED,
	CCD_MODE_CAST_RAY,
	CCD_MODE_CAST_SHAPE,
};

/* JOINT API */

enum JointType {
	JOINT_TYPE_PIN,
	JOINT_TYPE_GROOVE,
	JOINT_TYPE_DAMPED_SPRING,
	JOINT_TYPE_MAX
};

enum JointParam {
	JOINT_PARAM_BIAS,
	JOINT_PARAM_MAX_BIAS,
	JOINT_PARAM_MAX_FORCE,
};

enum PinJointParam {
	PIN_JOINT_SOFTNESS,
	PIN_JOINT_LIMIT_UPPER,
	PIN_JOINT_LIMIT_LOWER,
	PIN_JOINT_MOTOR_TARGET_VELOCITY
};

enum PinJointFlag {
	PIN_JOINT_FLAG_ANGULAR_LIMIT_ENABLED,
	PIN_JOINT_FLAG_MOTOR_ENABLED
};

enum DampedSpringParam {
	DAMPED_SPRING_REST_LENGTH,
	DAMPED_SPRING_STIFFNESS,
	DAMPED_SPRING_DAMPING
};

/* QUERY API */

enum AreaBodyStatus {
	AREA_BODY_ADDED,
	AREA_BODY_REMOVED
};

enum ProcessInfo {
	INFO_ACTIVE_OBJECTS,
	INFO_COLLISION_PAIRS,
	INFO_ISLAND_COUNT
};

#ifndef DISABLE_DEPRECATED
// Graveyard.
#endif

} // namespace PhysicsServer2DEnums

// Alias to make it easier to use.
#define PS2DE PhysicsServer2DEnums
