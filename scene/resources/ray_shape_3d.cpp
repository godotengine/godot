/*************************************************************************/
/*  ray_shape_3d.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "ray_shape_3d.h"

#include "servers/physics_server_3d.h"

Vector<Vector3> RayShape3D::get_debug_mesh_lines() const {
	Vector<Vector3> points;
	points.push_back(Vector3());
	points.push_back(Vector3(0, 0, get_length()));

	return points;
}

real_t RayShape3D::get_enclosing_radius() const {
	return length;
}

void RayShape3D::_update_shape() {
	Dictionary d;
	d["length"] = length;
	d["slips_on_slope"] = slips_on_slope;
	PhysicsServer3D::get_singleton()->shape_set_data(get_shape(), d);
	Shape3D::_update_shape();
}

void RayShape3D::set_length(float p_length) {
	length = p_length;
	_update_shape();
	notify_change_to_owners();
	_change_notify("length");
}

float RayShape3D::get_length() const {
	return length;
}

void RayShape3D::set_slips_on_slope(bool p_active) {
	slips_on_slope = p_active;
	_update_shape();
	notify_change_to_owners();
	_change_notify("slips_on_slope");
}

bool RayShape3D::get_slips_on_slope() const {
	return slips_on_slope;
}

void RayShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_length", "length"), &RayShape3D::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &RayShape3D::get_length);

	ClassDB::bind_method(D_METHOD("set_slips_on_slope", "active"), &RayShape3D::set_slips_on_slope);
	ClassDB::bind_method(D_METHOD("get_slips_on_slope"), &RayShape3D::get_slips_on_slope);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "0,4096,0.01"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "slips_on_slope"), "set_slips_on_slope", "get_slips_on_slope");
}

RayShape3D::RayShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_RAY)) {
	length = 1.0;
	slips_on_slope = false;

	/* Code copied from setters to prevent the use of uninitialized variables */
	_update_shape();
	notify_change_to_owners();
	_change_notify("length");
	_change_notify("slips_on_slope");
}
