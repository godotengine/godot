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
	ClassDB::bind_method(D_METHOD("set_preset", "preset"), &PhysicsMaterial::set_preset);
	ClassDB::bind_method(D_METHOD("get_preset"), &PhysicsMaterial::get_preset);

	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &PhysicsMaterial::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &PhysicsMaterial::get_friction);

	ClassDB::bind_method(D_METHOD("set_rough", "rough"), &PhysicsMaterial::set_rough);
	ClassDB::bind_method(D_METHOD("is_rough"), &PhysicsMaterial::is_rough);

	ClassDB::bind_method(D_METHOD("set_bounce", "bounce"), &PhysicsMaterial::set_bounce);
	ClassDB::bind_method(D_METHOD("get_bounce"), &PhysicsMaterial::get_bounce);

	ClassDB::bind_method(D_METHOD("set_absorbent", "absorbent"), &PhysicsMaterial::set_absorbent);
	ClassDB::bind_method(D_METHOD("is_absorbent"), &PhysicsMaterial::is_absorbent);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "preset", PROPERTY_HINT_ENUM, "Generic,Brick,Concrete,Ceramic,Gravel,Carpet,Glass,Plaster,Wood,Metal,Rock,Custom"), "set_preset", "get_preset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rough"), "set_rough", "is_rough");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bounce", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"), "set_bounce", "get_bounce");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "absorbent"), "set_absorbent", "is_absorbent");

	BIND_ENUM_CONSTANT(PRESET_GENERIC);
	BIND_ENUM_CONSTANT(PRESET_BRICK);
	BIND_ENUM_CONSTANT(PRESET_CONCRETE);
	BIND_ENUM_CONSTANT(PRESET_CERAMIC);
	BIND_ENUM_CONSTANT(PRESET_GRAVEL);
	BIND_ENUM_CONSTANT(PRESET_CARPET);
	BIND_ENUM_CONSTANT(PRESET_GLASS);
	BIND_ENUM_CONSTANT(PRESET_PLASTER);
	BIND_ENUM_CONSTANT(PRESET_WOOD);
	BIND_ENUM_CONSTANT(PRESET_METAL);
	BIND_ENUM_CONSTANT(PRESET_ROCK);
	BIND_ENUM_CONSTANT(PRESET_CUSTOM);
}

void PhysicsMaterial::set_preset(Preset p_preset) {
	if (preset == p_preset && p_preset != PRESET_CUSTOM) {
		return;
	}
	preset = static_cast<Preset>(p_preset);
	switch (p_preset) {
		case PRESET_GENERIC:
			friction = 1.0;
			bounce = 0.0;
			break;
		case PRESET_BRICK:
			friction = 0.6;
			bounce = 0.05;
			break;
		case PRESET_CONCRETE:
			friction = 0.65;
			bounce = 0.1;
			break;
		case PRESET_CERAMIC:
			friction = 0.15;
			bounce = 0.7;
			break;
		case PRESET_GRAVEL:
			friction = 0.7;
			bounce = 0.15;
			break;
		case PRESET_CARPET:
			friction = 0.8;
			bounce = 0.05;
			break;
		case PRESET_GLASS:
			friction = 0.2;
			bounce = 0.6;
			break;
		case PRESET_PLASTER:
			friction = 0.55;
			bounce = 0.25;
			break;
		case PRESET_WOOD:
			friction = 0.45;
			bounce = 0.15;
			break;
		case PRESET_METAL:
			friction = 0.6;
			bounce = 0.1;
			break;
		case PRESET_ROCK:
			friction = 0.7;
			bounce = 0.2;
			break;
		case PRESET_CUSTOM:
			break;
		default:
			set_preset(PRESET_CUSTOM);
			break;
	}
	emit_changed();
}

void PhysicsMaterial::set_friction(real_t p_val) {
	friction = p_val;
	preset = static_cast<Preset>(PRESET_CUSTOM);
	emit_changed();
}

void PhysicsMaterial::set_rough(bool p_val) {
	rough = p_val;
	preset = static_cast<Preset>(PRESET_CUSTOM);
	emit_changed();
}

void PhysicsMaterial::set_bounce(real_t p_val) {
	bounce = p_val;
	preset = static_cast<Preset>(PRESET_CUSTOM);
	emit_changed();
}

void PhysicsMaterial::set_absorbent(bool p_val) {
	absorbent = p_val;
	preset = static_cast<Preset>(PRESET_CUSTOM);
	emit_changed();
}

PhysicsMaterial::PhysicsMaterial() {
	preset = PRESET_GENERIC;
}
#endif // !defined(PHYSICS_2D_DISABLED) || !defined(PHYSICS_3D_DISABLED)
