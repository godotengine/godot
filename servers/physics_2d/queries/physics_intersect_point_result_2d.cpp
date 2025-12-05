/**************************************************************************/
/*  physics_intersect_point_result_2d.cpp                                 */
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

#include "physics_intersect_point_result_2d.h"

#include "core/object/class_db.h"

PhysicsIntersectPointResult2D::PhysicsIntersectPointResult2D(int p_max_intersections) {
	ERR_FAIL_COND(p_max_intersections < 0);
	result.resize(p_max_intersections);
}

int PhysicsIntersectPointResult2D::get_max_intersections() const {
	return result.size();
}

void PhysicsIntersectPointResult2D::set_max_intersections(int p_max_intersections) {
	ERR_FAIL_COND(p_max_intersections < 0);
	result.resize(p_max_intersections);
	if (intersection_count > p_max_intersections) {
		intersection_count = p_max_intersections;
	}
}

int PhysicsIntersectPointResult2D::get_intersection_count() const {
	return intersection_count;
}

RID PhysicsIntersectPointResult2D::get_collider_rid(int p_intersection_index) const {
	ERR_FAIL_INDEX_V(p_intersection_index, intersection_count, RID());
	ERR_FAIL_INDEX_V(p_intersection_index, static_cast<int>(result.size()), RID());
	return result[p_intersection_index].rid;
}

ObjectID PhysicsIntersectPointResult2D::get_collider_id(int p_intersection_index) const {
	ERR_FAIL_INDEX_V(p_intersection_index, intersection_count, ObjectID());
	ERR_FAIL_INDEX_V(p_intersection_index, static_cast<int>(result.size()), ObjectID());
	return result[p_intersection_index].collider_id;
}

Object *PhysicsIntersectPointResult2D::get_collider(int p_intersection_index) const {
	ERR_FAIL_INDEX_V(p_intersection_index, intersection_count, nullptr);
	ERR_FAIL_INDEX_V(p_intersection_index, static_cast<int>(result.size()), nullptr);
	return result[p_intersection_index].collider;
}

int PhysicsIntersectPointResult2D::get_collider_shape(int p_intersection_index) const {
	ERR_FAIL_INDEX_V(p_intersection_index, intersection_count, 0);
	ERR_FAIL_INDEX_V(p_intersection_index, static_cast<int>(result.size()), 0);
	return result[p_intersection_index].shape;
}

void PhysicsIntersectPointResult2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_max_intersections"), &PhysicsIntersectPointResult2D::get_max_intersections);
	ClassDB::bind_method(D_METHOD("set_max_intersections", "max_intersections"), &PhysicsIntersectPointResult2D::set_max_intersections);
	ClassDB::bind_method(D_METHOD("get_intersection_count"), &PhysicsIntersectPointResult2D::get_intersection_count);
	ClassDB::bind_method(D_METHOD("get_collider_id", "intersection_index"), &PhysicsIntersectPointResult2D::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_rid", "intersection_index"), &PhysicsIntersectPointResult2D::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider", "intersection_index"), &PhysicsIntersectPointResult2D::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape", "intersection_index"), &PhysicsIntersectPointResult2D::get_collider_shape);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_intersections"), "set_max_intersections", "get_max_intersections");
}
