/**************************************************************************/
/*  physics_shape_intersection_result_2d.cpp                              */
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

#include "physics_shape_intersection_result_2d.h"

#include "core/object/class_db.h"

PhysicsShapeIntersectionResult2D::PhysicsShapeIntersectionResult2D(int p_max_collisions) {
	result.resize(p_max_collisions);
	collision_count = 0;
}

int PhysicsShapeIntersectionResult2D::get_max_collisions() const {
	return result.size();
}

void PhysicsShapeIntersectionResult2D::set_max_collisions(int p_max_collisions) {
	result.resize(p_max_collisions);
}

int PhysicsShapeIntersectionResult2D::get_collision_count() const {
	return collision_count;
}

RID PhysicsShapeIntersectionResult2D::get_collider_rid(int p_collider_index) const {
	ERR_FAIL_INDEX_V(p_collider_index, collision_count, RID());
	return result[p_collider_index].rid;
}

ObjectID PhysicsShapeIntersectionResult2D::get_collider_id(int p_collider_index) const {
	ERR_FAIL_INDEX_V(p_collider_index, collision_count, ObjectID());
	return result[p_collider_index].collider_id;
}

Object *PhysicsShapeIntersectionResult2D::get_collider(int p_collider_index) const {
	ERR_FAIL_INDEX_V(p_collider_index, collision_count, nullptr);
	return result[p_collider_index].collider;
}

int PhysicsShapeIntersectionResult2D::get_collider_shape(int p_collider_index) const {
	ERR_FAIL_INDEX_V(p_collider_index, collision_count, 0);
	return result[p_collider_index].shape;
}

void PhysicsShapeIntersectionResult2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_max_collisions"), &PhysicsShapeIntersectionResult2D::get_max_collisions);
	ClassDB::bind_method(D_METHOD("set_max_collisions", "max_collisions"), &PhysicsShapeIntersectionResult2D::set_max_collisions);
	ClassDB::bind_method(D_METHOD("get_collision_count"), &PhysicsShapeIntersectionResult2D::get_collision_count);
	ClassDB::bind_method(D_METHOD("get_collider_id", "collider_index"), &PhysicsShapeIntersectionResult2D::get_collider_id, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_rid", "collider_index"), &PhysicsShapeIntersectionResult2D::get_collider_rid, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider", "collider_index"), &PhysicsShapeIntersectionResult2D::get_collider, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_collider_shape", "collider_index"), &PhysicsShapeIntersectionResult2D::get_collider_shape, DEFVAL(0));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_collisions"), "set_max_collisions", "get_max_collisions");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_count"), "", "get_collision_count");
}
