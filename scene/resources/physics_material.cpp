/**************************************************************************/
/*  physics_material.cpp                                                  */
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

#include "physics_material.h"

#if !defined(PHYSICS_2D_DISABLED) || !defined(PHYSICS_3D_DISABLED)
void PhysicsMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &PhysicsMaterial::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &PhysicsMaterial::get_friction);

	ClassDB::bind_method(D_METHOD("set_rough", "rough"), &PhysicsMaterial::set_rough);
	ClassDB::bind_method(D_METHOD("is_rough"), &PhysicsMaterial::is_rough);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &PhysicsMaterial::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &PhysicsMaterial::get_bounce);

	ClassDB::bind_method(D_METHOD("set_absorbent", "absorbent"), &PhysicsMaterial::set_absorbent);
	ClassDB::bind_method(D_METHOD("is_absorbent"), &PhysicsMaterial::is_absorbent);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rough"), "set_rough", "is_rough");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bounce", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_bounce", "get_bounce");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "absorbent"), "set_absorbent", "is_absorbent");
}

void PhysicsMaterial::set_friction(real_t p_val) {
	ERR_FAIL_COND(p_val > 1.0);
	friction = p_val;
	emit_changed();
}

void PhysicsMaterial::set_rough(bool p_val) {
	rough = p_val;
	emit_changed();
}

void PhysicsMaterial::set_bounce(real_t p_val) {
	ERR_FAIL_COND(p_val > 1.0);
	bounce = p_val;
	emit_changed();
}

void PhysicsMaterial::set_absorbent(bool p_val) {
	absorbent = p_val;
	emit_changed();
}
#endif // !defined(PHYSICS_2D_DISABLED) || !defined(PHYSICS_3D_DISABLED)
