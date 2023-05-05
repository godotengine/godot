/**************************************************************************/
/*  navigation_raycast_hit_2d.cpp                                         */
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

#include "navigation_raycast_hit_2d.h"

void NavigationRaycastHit2D::set_did_hit(const bool &p_did_hit) {
	did_hit = p_did_hit;
}

const bool &NavigationRaycastHit2D::get_did_hit() const {
	return did_hit;
}

void NavigationRaycastHit2D::set_hit_position(const Vector2 &p_hit_position) {
	hit_position = p_hit_position;
}

const Vector2 &NavigationRaycastHit2D::get_hit_position() {
	return hit_position;
}

void NavigationRaycastHit2D::set_hit_normal(const Vector2 &p_hit_normal) {
	hit_normal = p_hit_normal;
}

const Vector2 &NavigationRaycastHit2D::get_hit_normal() {
	return hit_normal;
}

void NavigationRaycastHit2D::set_raycast_path(const Vector<Vector2> &p_raycast_path) {
	raycast_path = p_raycast_path;
}

const Vector<Vector2> &NavigationRaycastHit2D::get_raycast_path() {
	return raycast_path;
}

void NavigationRaycastHit2D::reset() {
	did_hit = false;
	hit_position = Vector2();
	hit_normal = Vector2();
	raycast_path.clear();
}

void NavigationRaycastHit2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_did_hit", "did_hit"), &NavigationRaycastHit2D::set_did_hit);
	ClassDB::bind_method(D_METHOD("get_did_hit"), &NavigationRaycastHit2D::get_did_hit);

	ClassDB::bind_method(D_METHOD("set_hit_position", "hit_position"), &NavigationRaycastHit2D::set_hit_position);
	ClassDB::bind_method(D_METHOD("get_hit_position"), &NavigationRaycastHit2D::get_hit_position);

	ClassDB::bind_method(D_METHOD("set_hit_normal", "hit_normal"), &NavigationRaycastHit2D::set_hit_normal);
	ClassDB::bind_method(D_METHOD("get_hit_normal"), &NavigationRaycastHit2D::get_hit_normal);

	ClassDB::bind_method(D_METHOD("set_raycast_path", "raycast_path"), &NavigationRaycastHit2D::set_raycast_path);
	ClassDB::bind_method(D_METHOD("get_raycast_path"), &NavigationRaycastHit2D::get_raycast_path);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "did_hit"), "set_did_hit", "get_did_hit");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "hit_position"), "set_hit_position", "get_hit_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "hit_normal"), "set_hit_normal", "get_hit_normal");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR3_ARRAY, "raycast_path"), "set_raycast_path", "get_raycast_path");
}
