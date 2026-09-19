/**************************************************************************/
/*  physics_testmotion_query_result_2d.cpp                                */
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

#include "physics_testmotion_query_result_2d.h"

#include "core/object/class_db.h"

Vector2 PhysicsTestMotionResult2D::get_travel() const {
	return result.travel;
}

Vector2 PhysicsTestMotionResult2D::get_remainder() const {
	return result.remainder;
}

Vector2 PhysicsTestMotionResult2D::get_collision_point() const {
	return result.collision_point;
}

Vector2 PhysicsTestMotionResult2D::get_collision_normal() const {
	return result.collision_normal;
}

Vector2 PhysicsTestMotionResult2D::get_collider_velocity() const {
	return result.collider_velocity;
}

ObjectID PhysicsTestMotionResult2D::get_collider_id() const {
	return result.collider_id;
}

RID PhysicsTestMotionResult2D::get_collider_rid() const {
	return result.collider;
}

Object *PhysicsTestMotionResult2D::get_collider() const {
	return ObjectDB::get_instance(result.collider_id);
}

int PhysicsTestMotionResult2D::get_collider_shape() const {
	return result.collider_shape;
}

int PhysicsTestMotionResult2D::get_collision_local_shape() const {
	return result.collision_local_shape;
}

real_t PhysicsTestMotionResult2D::get_collision_depth() const {
	return result.collision_depth;
}

real_t PhysicsTestMotionResult2D::get_collision_safe_fraction() const {
	return result.collision_safe_fraction;
}

real_t PhysicsTestMotionResult2D::get_collision_unsafe_fraction() const {
	return result.collision_unsafe_fraction;
}

void PhysicsTestMotionResult2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_travel"), &PhysicsTestMotionResult2D::get_travel);
	ClassDB::bind_method(D_METHOD("get_remainder"), &PhysicsTestMotionResult2D::get_remainder);
	ClassDB::bind_method(D_METHOD("get_collision_point"), &PhysicsTestMotionResult2D::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &PhysicsTestMotionResult2D::get_collision_normal);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &PhysicsTestMotionResult2D::get_collider_velocity);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &PhysicsTestMotionResult2D::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_rid"), &PhysicsTestMotionResult2D::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider"), &PhysicsTestMotionResult2D::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &PhysicsTestMotionResult2D::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collision_local_shape"), &PhysicsTestMotionResult2D::get_collision_local_shape);
	ClassDB::bind_method(D_METHOD("get_collision_depth"), &PhysicsTestMotionResult2D::get_collision_depth);
	ClassDB::bind_method(D_METHOD("get_collision_safe_fraction"), &PhysicsTestMotionResult2D::get_collision_safe_fraction);
	ClassDB::bind_method(D_METHOD("get_collision_unsafe_fraction"), &PhysicsTestMotionResult2D::get_collision_unsafe_fraction);
}
