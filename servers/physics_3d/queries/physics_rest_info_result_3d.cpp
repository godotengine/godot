/**************************************************************************/
/*  physics_rest_info_result_3d.cpp                                       */
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

#include "physics_rest_info_result_3d.h"

#include "core/object/class_db.h"

Vector3 PhysicsRestInfoResult3D::get_collision_point() const {
	return result.point;
}

Vector3 PhysicsRestInfoResult3D::get_collision_normal() const {
	return result.normal;
}

RID PhysicsRestInfoResult3D::get_collider_rid() const {
	return result.rid;
}

ObjectID PhysicsRestInfoResult3D::get_collider_id() const {
	return result.collider_id;
}

int PhysicsRestInfoResult3D::get_collider_shape() const {
	return result.shape;
}

Vector3 PhysicsRestInfoResult3D::get_collider_velocity() const {
	return result.linear_velocity;
}

void PhysicsRestInfoResult3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_collision_point"), &PhysicsRestInfoResult3D::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &PhysicsRestInfoResult3D::get_collision_normal);
	ClassDB::bind_method(D_METHOD("get_collider_rid"), &PhysicsRestInfoResult3D::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &PhysicsRestInfoResult3D::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &PhysicsRestInfoResult3D::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &PhysicsRestInfoResult3D::get_collider_velocity);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "collision_point"), "", "get_collision_point");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "collision_normal"), "", "get_collision_normal");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "collider_rid"), "", "get_collider_rid");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_id"), "", "get_collider_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_shape"), "", "get_collider_shape");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "collider_velocity"), "", "get_collider_velocity");
}
