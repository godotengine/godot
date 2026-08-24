/**************************************************************************/
/*  physics_cast_motion_result_3d.cpp                                     */
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

#include "physics_cast_motion_result_3d.h"

#include "core/object/class_db.h"

void PhysicsCastMotionResult3D::_reset() {
	collision_safe_fraction = 1.0;
	collision_unsafe_fraction = 1.0;
	rest_info = PS3DT::ShapeRestInfo();
}

void PhysicsCastMotionResult3D::_set_fractions(real_t p_safe_fraction, real_t p_unsafe_fraction) {
	collision_safe_fraction = p_safe_fraction;
	collision_unsafe_fraction = p_unsafe_fraction;
}

real_t PhysicsCastMotionResult3D::get_collision_safe_fraction() const {
	return collision_safe_fraction;
}

real_t PhysicsCastMotionResult3D::get_collision_unsafe_fraction() const {
	return collision_unsafe_fraction;
}

Vector3 PhysicsCastMotionResult3D::get_collision_point() const {
	return rest_info.point;
}

Vector3 PhysicsCastMotionResult3D::get_collision_normal() const {
	return rest_info.normal;
}

RID PhysicsCastMotionResult3D::get_collider_rid() const {
	return rest_info.rid;
}

ObjectID PhysicsCastMotionResult3D::get_collider_id() const {
	return rest_info.collider_id;
}

int PhysicsCastMotionResult3D::get_collider_shape() const {
	return rest_info.shape;
}

Vector3 PhysicsCastMotionResult3D::get_collider_velocity() const {
	return rest_info.linear_velocity;
}

void PhysicsCastMotionResult3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_collision_safe_fraction"), &PhysicsCastMotionResult3D::get_collision_safe_fraction);
	ClassDB::bind_method(D_METHOD("get_collision_unsafe_fraction"), &PhysicsCastMotionResult3D::get_collision_unsafe_fraction);
	ClassDB::bind_method(D_METHOD("get_collision_point"), &PhysicsCastMotionResult3D::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &PhysicsCastMotionResult3D::get_collision_normal);
	ClassDB::bind_method(D_METHOD("get_collider_rid"), &PhysicsCastMotionResult3D::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &PhysicsCastMotionResult3D::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &PhysicsCastMotionResult3D::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &PhysicsCastMotionResult3D::get_collider_velocity);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_safe_fraction", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_collision_safe_fraction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_unsafe_fraction", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_collision_unsafe_fraction");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "collision_point", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_collision_point");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "collision_normal", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_collision_normal");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "collider_rid", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_collider_rid");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_id", PROPERTY_HINT_OBJECT_ID, "", PROPERTY_USAGE_NONE), "", "get_collider_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_shape", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_collider_shape");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "collider_velocity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_collider_velocity");
}
