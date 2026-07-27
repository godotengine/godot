/**************************************************************************/
/*  physics_testray_query_result_3d.cpp                                   */
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

#include "physics_testray_query_result_3d.h"

#include "core/object/class_db.h"

void PhysicsTestRayResult3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_position"), &PhysicsTestRayResult3D::get_position);
	ClassDB::bind_method(D_METHOD("get_normal"), &PhysicsTestRayResult3D::get_normal);
	ClassDB::bind_method(D_METHOD("get_rid"), &PhysicsTestRayResult3D::get_rid);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &PhysicsTestRayResult3D::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider"), &PhysicsTestRayResult3D::get_collider);
	ClassDB::bind_method(D_METHOD("get_shape"), &PhysicsTestRayResult3D::get_shape);
	ClassDB::bind_method(D_METHOD("get_face_index"), &PhysicsTestRayResult3D::get_face_index);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position"), "", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "normal"), "", "get_normal");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "rid"), "", "get_rid");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_id"), "", "get_collider_id");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider"), "", "get_collider");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shape"), "", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "face_index"), "", "get_face_index");
}
