/**************************************************************************/
/*  ray_shape.cpp                                                         */
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

#include "ray_shape.h"

#include "servers/physics_server.h"

Vector<Vector3> RayShape::get_debug_mesh_lines() {
	Vector<Vector3> points;
	points.push_back(Vector3());
	points.push_back(Vector3(0, 0, get_length()));

	return points;
}

real_t RayShape::get_enclosing_radius() const {
	return length;
}

void RayShape::_update_shape() {
	Dictionary d;
	d["length"] = length;
	d["slips_on_slope"] = slips_on_slope;
	PhysicsServer::get_singleton()->shape_set_data(get_shape(), d);
	Shape::_update_shape();
}

void RayShape::set_length(float p_length) {
	length = p_length;
	_update_shape();
	notify_change_to_owners();
	_change_notify("length");
}

float RayShape::get_length() const {
	return length;
}

void RayShape::set_slips_on_slope(bool p_active) {
	slips_on_slope = p_active;
	_update_shape();
	notify_change_to_owners();
	_change_notify("slips_on_slope");
}

bool RayShape::get_slips_on_slope() const {
	return slips_on_slope;
}

void RayShape::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_length", "length"), &RayShape::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &RayShape::get_length);

	ClassDB::bind_method(D_METHOD("set_slips_on_slope", "active"), &RayShape::set_slips_on_slope);
	ClassDB::bind_method(D_METHOD("get_slips_on_slope"), &RayShape::get_slips_on_slope);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "length", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "slips_on_slope"), "set_slips_on_slope", "get_slips_on_slope");
}

RayShape::RayShape() :
		Shape(RID_PRIME(PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_RAY))) {
	length = 1.0;
	slips_on_slope = false;

	/* Code copied from setters to prevent the use of uninitialized variables */
	_update_shape();
	notify_change_to_owners();
	_change_notify("length");
	_change_notify("slips_on_slope");
}
