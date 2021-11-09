/*************************************************************************/
/*  plane_shape.cpp                                                      */
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

#include "plane_shape.h"

#include "servers/physics_server.h"

Vector<Vector3> PlaneShape::get_debug_mesh_lines() {
	Plane p = get_plane();
	Vector<Vector3> points;

	Vector3 n1 = p.get_any_perpendicular_normal();
	Vector3 n2 = p.normal.cross(n1).normalized();

	Vector3 pface[4] = {
		p.normal * p.d + n1 * 10.0 + n2 * 10.0,
		p.normal * p.d + n1 * 10.0 + n2 * -10.0,
		p.normal * p.d + n1 * -10.0 + n2 * -10.0,
		p.normal * p.d + n1 * -10.0 + n2 * 10.0,
	};

	points.push_back(pface[0]);
	points.push_back(pface[1]);
	points.push_back(pface[1]);
	points.push_back(pface[2]);
	points.push_back(pface[2]);
	points.push_back(pface[3]);
	points.push_back(pface[3]);
	points.push_back(pface[0]);
	points.push_back(p.normal * p.d);
	points.push_back(p.normal * p.d + p.normal * 3);

	return points;
}

void PlaneShape::_update_shape() {
	PhysicsServer::get_singleton()->shape_set_data(get_shape(), plane);
	Shape::_update_shape();
}

void PlaneShape::set_plane(Plane p_plane) {
	plane = p_plane;
	_update_shape();
	notify_change_to_owners();
	_change_notify("plane");
}

Plane PlaneShape::get_plane() const {
	return plane;
}

void PlaneShape::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_plane", "plane"), &PlaneShape::set_plane);
	ClassDB::bind_method(D_METHOD("get_plane"), &PlaneShape::get_plane);

	ADD_PROPERTY(PropertyInfo(Variant::PLANE, "plane"), "set_plane", "get_plane");
}

PlaneShape::PlaneShape() :
		Shape(RID_PRIME(PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_PLANE))) {
	set_plane(Plane(0, 1, 0, 0));
}
