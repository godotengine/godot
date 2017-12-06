/*************************************************************************/
/*  box_shape.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "box_shape.h"
#include "servers/physics_server.h"

Vector<Vector3> BoxShape::_gen_debug_mesh_lines() {

	Vector<Vector3> lines;
	AABB aabb;
	aabb.position = -get_extents();
	aabb.size = aabb.position * -2;

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	return lines;
}

void BoxShape::_update_shape() {

	PhysicsServer::get_singleton()->shape_set_data(get_shape(), extents);
}

void BoxShape::set_extents(const Vector3 &p_extents) {

	extents = p_extents;
	_update_shape();
	notify_change_to_owners();
	_change_notify("extents");
}

Vector3 BoxShape::get_extents() const {

	return extents;
}

void BoxShape::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &BoxShape::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &BoxShape::get_extents);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents"), "set_extents", "get_extents");
}

BoxShape::BoxShape() :
		Shape(PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_BOX)) {

	set_extents(Vector3(1, 1, 1));
}
