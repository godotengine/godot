/*************************************************************************/
/*  sphere_shape.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "sphere_shape.h"
#include "servers/physics_server.h"

Vector<Vector3> SphereShape::get_debug_mesh_lines() {
	float r = get_radius();

	Vector<Vector3> points;

	for (int i = 0; i <= 360; i++) {
		float ra = Math::deg2rad((float)i);
		float rb = Math::deg2rad((float)i + 1);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

		points.push_back(Vector3(a.x, 0, a.y));
		points.push_back(Vector3(b.x, 0, b.y));
		points.push_back(Vector3(0, a.x, a.y));
		points.push_back(Vector3(0, b.x, b.y));
		points.push_back(Vector3(a.x, a.y, 0));
		points.push_back(Vector3(b.x, b.y, 0));
	}

	return points;
}

real_t SphereShape::get_enclosing_radius() const {
	return radius;
}

void SphereShape::_update_shape() {
	PhysicsServer::get_singleton()->shape_set_data(get_shape(), radius);
	Shape::_update_shape();
}

void SphereShape::set_radius(float p_radius) {
	radius = p_radius;
	_update_shape();
	notify_change_to_owners();
	_change_notify("radius");
}

float SphereShape::get_radius() const {
	return radius;
}

void SphereShape::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &SphereShape::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &SphereShape::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0,4096,0.01"), "set_radius", "get_radius");
}

SphereShape::SphereShape() :
		Shape(RID_PRIME(PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_SPHERE))) {
	set_radius(1.0);
}
