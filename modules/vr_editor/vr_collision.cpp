/**************************************************************************/
/*  vr_collision.cpp                                                      */
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

#include "vr_collision.h"

Vector<VRCollision *> VRCollision::collisions;

void VRCollision::_bind_methods() {
	ADD_SIGNAL(MethodInfo("interact_enter", PropertyInfo(Variant::VECTOR3, "position")));
	ADD_SIGNAL(MethodInfo("interact_moved", PropertyInfo(Variant::VECTOR3, "position"), PropertyInfo(Variant::FLOAT, "pressure")));
	ADD_SIGNAL(MethodInfo("interact_leave", PropertyInfo(Variant::VECTOR3, "position")));
	ADD_SIGNAL(MethodInfo("interact_pressed", PropertyInfo(Variant::VECTOR3, "position"), PropertyInfo(Variant::INT, "button")));
	ADD_SIGNAL(MethodInfo("interact_released", PropertyInfo(Variant::VECTOR3, "position"), PropertyInfo(Variant::INT, "button")));
	ADD_SIGNAL(MethodInfo("interact_scrolled", PropertyInfo(Variant::VECTOR3, "position"), PropertyInfo(Variant::VECTOR2, "scroll_delta")));
}

Vector<VRCollision *> VRCollision::get_hit_tests(bool p_inc_can_interact, bool p_inc_can_grab) {
	Vector<VRCollision *> res;

	for (int i = 0; i < collisions.size(); i++) {
		VRCollision *collision = collisions[i];
		if (collision->is_enabled() && ((p_inc_can_interact && collision->get_can_interact()) || (p_inc_can_grab && collision->get_can_grab()))) {
			res.push_back(collision);
		}
	}

	return res;
}

void VRCollision::_on_interact_enter(const Vector3 &p_position) {
	emit_signal("interact_enter", p_position);
}

void VRCollision::_on_interact_moved(const Vector3 &p_position, float p_pressure) {
	emit_signal("interact_moved", p_position, p_pressure);
}

void VRCollision::_on_interact_leave(const Vector3 &p_position) {
	emit_signal("interact_leave", p_position);
}

void VRCollision::_on_interact_pressed(const Vector3 &p_position, MouseButton p_button) {
	emit_signal("interact_pressed", p_position, p_button);
}

void VRCollision::_on_interact_released(const Vector3 &p_position, MouseButton p_button) {
	emit_signal("interact_released", p_position, p_button);
}

void VRCollision::_on_interact_scrolled(const Vector3 &p_position, const Vector2 &p_scroll_delta) {
	emit_signal("interact_scrolled", p_position, p_scroll_delta);
}

VRCollision::VRCollision() {
	collisions.push_back(this);
}

VRCollision::~VRCollision() {
	collisions.erase(this);
}
