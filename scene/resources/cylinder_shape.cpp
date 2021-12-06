/*************************************************************************/
/*  cylinder_shape.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "cylinder_shape.h"
#include "servers/physics_server.h"

Vector<Vector3> CylinderShape::get_debug_mesh_lines() {
	float radius = get_radius();
	float height = get_height();

	Vector<Vector3> points;

	Vector3 d(0, height * 0.5, 0);
	for (int i = 0; i < 360; i++) {
		float ra = Math::deg2rad((float)i);
		float rb = Math::deg2rad((float)i + 1);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * radius;

		points.push_back(Vector3(a.x, 0, a.y) + d);
		points.push_back(Vector3(b.x, 0, b.y) + d);

		points.push_back(Vector3(a.x, 0, a.y) - d);
		points.push_back(Vector3(b.x, 0, b.y) - d);

		if (i % 90 == 0) {
			points.push_back(Vector3(a.x, 0, a.y) + d);
			points.push_back(Vector3(a.x, 0, a.y) - d);
		}
	}

	return points;
}

void CylinderShape::_update_shape() {
	Dictionary d;
	d["radius"] = radius;
	d["height"] = height;
	PhysicsServer::get_singleton()->shape_set_data(get_shape(), d);
	Shape::_update_shape();
}

void CylinderShape::set_radius(float p_radius) {
	radius = p_radius;
	_update_shape();
	notify_change_to_owners();
	_change_notify("radius");
}

float CylinderShape::get_radius() const {
	return radius;
}

void CylinderShape::set_height(float p_height) {
	height = p_height;
	_update_shape();
	notify_change_to_owners();
	_change_notify("height");
}

float CylinderShape::get_height() const {
	return height;
}

void CylinderShape::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CylinderShape::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CylinderShape::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &CylinderShape::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CylinderShape::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0.01,4096,0.01"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height", PROPERTY_HINT_RANGE, "0.01,4096,0.01"), "set_height", "get_height");
}

CylinderShape::CylinderShape() :
		Shape(RID_PRIME(PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_CYLINDER))) {
	radius = 1.0;
	height = 2.0;
	_update_shape();
}
