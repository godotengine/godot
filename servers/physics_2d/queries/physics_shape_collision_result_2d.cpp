/**************************************************************************/
/*  physics_shape_collision_result_2d.cpp                                 */
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

#include "physics_shape_collision_result_2d.h"

#include "core/object/class_db.h"

PhysicsShapeCollisionResult2D::PhysicsShapeCollisionResult2D(int p_max_collisions) {
	result.resize(2 * p_max_collisions);
	collision_count = 0;
}

int PhysicsShapeCollisionResult2D::get_max_collisions() const {
	return result.size() / 2;
}

void PhysicsShapeCollisionResult2D::set_max_collisions(int p_max_collisions) {
	result.resize(2 * p_max_collisions);
}

int PhysicsShapeCollisionResult2D::get_collision_count() const {
	return collision_count;
}

Vector2 PhysicsShapeCollisionResult2D::get_collision_point_on_queried_shape(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, collision_count, Vector2());
	return result[2 * p_collision_index];
}

Vector2 PhysicsShapeCollisionResult2D::get_collision_point_on_colliding_shape(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, collision_count, Vector2());
	return result[2 * p_collision_index + 1];
}

void PhysicsShapeCollisionResult2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_max_collisions"), &PhysicsShapeCollisionResult2D::get_max_collisions);
	ClassDB::bind_method(D_METHOD("set_max_collisions", "max_collisions"), &PhysicsShapeCollisionResult2D::set_max_collisions);
	ClassDB::bind_method(D_METHOD("get_collision_count"), &PhysicsShapeCollisionResult2D::get_collision_count);
	ClassDB::bind_method(D_METHOD("get_collision_point_on_queried_shape", "collision_index"), &PhysicsShapeCollisionResult2D::get_collision_point_on_queried_shape, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collision_point_on_colliding_shape", "collision_index"), &PhysicsShapeCollisionResult2D::get_collision_point_on_colliding_shape, DEFVAL(0));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_collisions"), "set_max_collisions", "get_max_collisions");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_count"), "", "get_collision_count");
}
