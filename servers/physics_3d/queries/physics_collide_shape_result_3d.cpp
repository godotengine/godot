/**************************************************************************/
/*  physics_collide_shape_result_3d.cpp                                   */
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

#include "physics_collide_shape_result_3d.h"

#include "core/object/class_db.h"

PhysicsCollideShapeResult3D::PhysicsCollideShapeResult3D(int p_max_collisions) {
	ERR_FAIL_COND(p_max_collisions < 0);
	result.resize(2 * p_max_collisions);
}

int PhysicsCollideShapeResult3D::get_max_collisions() const {
	return result.size() / 2;
}

void PhysicsCollideShapeResult3D::set_max_collisions(int p_max_collisions) {
	ERR_FAIL_COND(p_max_collisions < 0);
	result.resize(2 * p_max_collisions);
	if (collision_count > p_max_collisions) {
		collision_count = p_max_collisions;
	}
}

int PhysicsCollideShapeResult3D::get_collision_count() const {
	return collision_count;
}

Vector3 PhysicsCollideShapeResult3D::get_point_on_queried_shape(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, collision_count, Vector3());
	ERR_FAIL_INDEX_V(2 * p_collision_index, static_cast<int>(result.size()), Vector3());
	return result[2 * p_collision_index];
}

Vector3 PhysicsCollideShapeResult3D::get_point_on_colliding_shape(int p_collision_index) const {
	ERR_FAIL_INDEX_V(p_collision_index, collision_count, Vector3());
	ERR_FAIL_INDEX_V(2 * p_collision_index, static_cast<int>(result.size()), Vector3());
	return result[2 * p_collision_index + 1];
}

void PhysicsCollideShapeResult3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_max_collisions"), &PhysicsCollideShapeResult3D::get_max_collisions);
	ClassDB::bind_method(D_METHOD("set_max_collisions", "max_collisions"), &PhysicsCollideShapeResult3D::set_max_collisions);
	ClassDB::bind_method(D_METHOD("get_collision_count"), &PhysicsCollideShapeResult3D::get_collision_count);
	ClassDB::bind_method(D_METHOD("get_point_on_queried_shape", "collision_index"), &PhysicsCollideShapeResult3D::get_point_on_queried_shape);
	ClassDB::bind_method(D_METHOD("get_point_on_colliding_shape", "collision_index"), &PhysicsCollideShapeResult3D::get_point_on_colliding_shape);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_collisions"), "set_max_collisions", "get_max_collisions");
}
