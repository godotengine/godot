/**************************************************************************/
/*  physics_testmotion_query_result_3d.cpp                                */
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

#include "physics_testmotion_query_result_3d.h"

#include "core/object/class_db.h"

Vector3 PhysicsTestMotionResult3D::get_travel() const {
	return result.travel;
}

Vector3 PhysicsTestMotionResult3D::get_remainder() const {
	return result.remainder;
}

real_t PhysicsTestMotionResult3D::get_collision_safe_fraction() const {
	return result.collision_safe_fraction;
}

real_t PhysicsTestMotionResult3D::get_collision_unsafe_fraction() const {
	return result.collision_unsafe_fraction;
}

int PhysicsTestMotionResult3D::get_collision_count() const {
	return result.collision_count;
}

Vector3 PhysicsTestMotionResult3D::get_collision_point(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, Vector3());
	return result.collisions[p_collision_index].position;
}

Vector3 PhysicsTestMotionResult3D::get_collision_normal(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, Vector3());
	return result.collisions[p_collision_index].normal;
}

Vector3 PhysicsTestMotionResult3D::get_collider_velocity(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, Vector3());
	return result.collisions[p_collision_index].collider_velocity;
}

ObjectID PhysicsTestMotionResult3D::get_collider_id(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, ObjectID());
	return result.collisions[p_collision_index].collider_id;
}

RID PhysicsTestMotionResult3D::get_collider_rid(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, RID());
	return result.collisions[p_collision_index].collider;
}

Object *PhysicsTestMotionResult3D::get_collider(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, nullptr);
	return ObjectDB::get_instance(result.collisions[p_collision_index].collider_id);
}

int PhysicsTestMotionResult3D::get_collider_shape(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, 0);
	return result.collisions[p_collision_index].collider_shape;
}

int PhysicsTestMotionResult3D::get_collision_local_shape(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, 0);
	return result.collisions[p_collision_index].local_shape;
}

real_t PhysicsTestMotionResult3D::get_collision_depth(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, result.collision_count, 0.0);
	return result.collisions[p_collision_index].depth;
}

void PhysicsTestMotionResult3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_travel"), &PhysicsTestMotionResult3D::get_travel);
	ClassDB::bind_method(D_METHOD("get_remainder"), &PhysicsTestMotionResult3D::get_remainder);
	ClassDB::bind_method(D_METHOD("get_collision_safe_fraction"), &PhysicsTestMotionResult3D::get_collision_safe_fraction);
	ClassDB::bind_method(D_METHOD("get_collision_unsafe_fraction"), &PhysicsTestMotionResult3D::get_collision_unsafe_fraction);
	ClassDB::bind_method(D_METHOD("get_collision_count"), &PhysicsTestMotionResult3D::get_collision_count);
	ClassDB::bind_method(D_METHOD("get_collision_point", "collision_index"), &PhysicsTestMotionResult3D::get_collision_point, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collision_normal", "collision_index"), &PhysicsTestMotionResult3D::get_collision_normal, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_velocity", "collision_index"), &PhysicsTestMotionResult3D::get_collider_velocity, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_id", "collision_index"), &PhysicsTestMotionResult3D::get_collider_id, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_rid", "collision_index"), &PhysicsTestMotionResult3D::get_collider_rid, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider", "collision_index"), &PhysicsTestMotionResult3D::get_collider, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_shape", "collision_index"), &PhysicsTestMotionResult3D::get_collider_shape, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collision_local_shape", "collision_index"), &PhysicsTestMotionResult3D::get_collision_local_shape, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collision_depth", "collision_index"), &PhysicsTestMotionResult3D::get_collision_depth, DEFVAL(0));
}
