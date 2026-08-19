/**************************************************************************/
/*  physics_server_3d_types.h                                             */
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

#include "core/math/vector3.h"
#include "core/object/object.h"
#include "core/templates/hash_set.h"
#include "core/templates/rid.h"

namespace PhysicsServer3DTypes {

struct RayParameters {
	Vector3 from;
	Vector3 to;
	HashSet<RID> exclude;
	uint32_t collision_mask = UINT32_MAX;

	bool collide_with_bodies = true;
	bool collide_with_areas = false;

	bool hit_from_inside = false;
	bool hit_back_faces = true;

	bool pick_ray = false;
};

struct RayResult {
	Vector3 position;
	Vector3 normal;
	RID rid;
	ObjectID collider_id;
	Object *collider = nullptr;
	int shape = 0;
	int face_index = -1;
};

struct ShapeResult {
	RID rid;
	ObjectID collider_id;
	Object *collider = nullptr;
	int shape = 0;
};

struct PointParameters {
	Vector3 position;
	HashSet<RID> exclude;
	uint32_t collision_mask = UINT32_MAX;

	bool collide_with_bodies = true;
	bool collide_with_areas = false;
};

struct ShapeParameters {
	RID shape_rid;
	Transform3D transform;
	Vector3 motion;
	real_t margin = 0.0;
	HashSet<RID> exclude;
	uint32_t collision_mask = UINT32_MAX;

	bool collide_with_bodies = true;
	bool collide_with_areas = false;
};

struct ShapeRestInfo {
	Vector3 point;
	Vector3 normal;
	RID rid;
	ObjectID collider_id;
	int shape = 0;
	Vector3 linear_velocity; // Velocity at contact point.
};

struct MotionParameters {
	Transform3D from;
	Vector3 motion;
	real_t margin = 0.001;
	int max_collisions = 1;
	bool collide_separation_ray = false;
	HashSet<RID> exclude_bodies;
	HashSet<ObjectID> exclude_objects;
	bool recovery_as_collision = false;

	MotionParameters() {}

	MotionParameters(const Transform3D &p_from, const Vector3 &p_motion, real_t p_margin = 0.001) :
			from(p_from),
			motion(p_motion),
			margin(p_margin) {}
};

struct MotionCollision {
	Vector3 position;
	Vector3 normal;
	Vector3 collider_velocity;
	Vector3 collider_angular_velocity;
	real_t depth = 0.0;
	int local_shape = 0;
	ObjectID collider_id;
	RID collider;
	int collider_shape = 0;

	real_t get_angle(Vector3 p_up_direction) const {
		return Math::acos(normal.dot(p_up_direction));
	}
};

struct MotionResult {
	Vector3 travel;
	Vector3 remainder;
	real_t collision_depth = 0.0;
	real_t collision_safe_fraction = 0.0;
	real_t collision_unsafe_fraction = 0.0;

	static const int MAX_COLLISIONS = 32;
	MotionCollision collisions[MAX_COLLISIONS];
	int collision_count = 0;
};

#ifndef DISABLE_DEPRECATED
// Graveyard.
#endif

} // namespace PhysicsServer3DTypes

// Alias to make it easier to use.
#define PS3DT PhysicsServer3DTypes
