/*************************************************************************/
/*  space_2d.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "space_2d.h"

RID Space2D::get_rid() const {

	return space;
}

void Space2D::set_active(bool p_active) {

	active = p_active;
	Physics2DServer::get_singleton()->space_set_active(space, active);
}

bool Space2D::is_active() const {

	return active;
}

void Space2D::set_gravity(real_t p_gravity) {

	gravity = p_gravity;
	Physics2DServer::get_singleton()->area_set_param(get_rid(), Physics2DServer::AREA_PARAM_GRAVITY, p_gravity);
}
real_t Space2D::get_gravity() const {

	return gravity;
}

void Space2D::set_gravity_vector(const Vector2 &p_vec) {

	gravity_vec = p_vec;
	Physics2DServer::get_singleton()->area_set_param(get_rid(), Physics2DServer::AREA_PARAM_GRAVITY_VECTOR, p_vec);
}
Vector2 Space2D::get_gravity_vector() const {

	return gravity_vec;
}

void Space2D::set_linear_damp(real_t p_linear_damp) {

	linear_damp = p_linear_damp;
	Physics2DServer::get_singleton()->area_set_param(get_rid(), Physics2DServer::AREA_PARAM_LINEAR_DAMP, p_linear_damp);
}
real_t Space2D::get_linear_damp() const {

	return linear_damp;
}

void Space2D::set_angular_damp(real_t p_angular_damp) {

	angular_damp = p_angular_damp;
	Physics2DServer::get_singleton()->area_set_param(get_rid(), Physics2DServer::AREA_PARAM_ANGULAR_DAMP, p_angular_damp);
}

real_t Space2D::get_angular_damp() const {

	return angular_damp;
}

void Space2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_active", "active"), &Space2D::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &Space2D::is_active);

	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &Space2D::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &Space2D::get_gravity);

	ClassDB::bind_method(D_METHOD("set_gravity_vector", "vector"), &Space2D::set_gravity_vector);
	ClassDB::bind_method(D_METHOD("get_gravity_vector"), &Space2D::get_gravity_vector);

	ClassDB::bind_method(D_METHOD("set_linear_damp", "linear_damp"), &Space2D::set_linear_damp);
	ClassDB::bind_method(D_METHOD("get_linear_damp"), &Space2D::get_linear_damp);

	ClassDB::bind_method(D_METHOD("set_angular_damp", "angular_damp"), &Space2D::set_angular_damp);
	ClassDB::bind_method(D_METHOD("get_angular_damp"), &Space2D::get_angular_damp);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gravity", PROPERTY_HINT_RANGE, "-1024,1024,0.001"), "set_gravity", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "gravity_vec"), "set_gravity_vector", "get_gravity_vector");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "linear_damp", PROPERTY_HINT_RANGE, "0,100,0.01,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_damp", PROPERTY_HINT_RANGE, "0,100,0.01,or_greater"), "set_angular_damp", "get_angular_damp");
}

Space2D::Space2D() {

	space = Physics2DServer::get_singleton()->space_create();

	set_active(true);
	set_gravity(98);
	set_gravity_vector(Vector2(0, 1));
	set_linear_damp(0.1);
	set_angular_damp(1);
}

Space2D::~Space2D() {

	Physics2DServer::get_singleton()->free(space);
}
