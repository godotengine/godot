/*************************************************************************/
/*  physics_material.cpp                                                 */
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
#include "physics_material.h"

bool PhysicsMaterial::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "bounce") {
		set_bounce(p_value);
	} else if (p_name == "bounce_combine_mode") {
		set_bounce_combine_mode(static_cast<PhysicsServer::CombineMode>(int(p_value)));
	} else if (p_name == "friction") {
		set_friction(p_value);
	} else if (p_name == "friction_combine_mode") {
		set_friction_combine_mode(static_cast<PhysicsServer::CombineMode>(int(p_value)));
	} else {
		return false;
	}

	emit_changed();
	return true;
}

bool PhysicsMaterial::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "bounce") {
		r_ret = bounce;
	} else if (p_name == "bounce_combine_mode") {
		r_ret = int(bounce_combine_mode);
	} else if (p_name == "friction") {
		r_ret = friction;
	} else if (p_name == "friction_combine_mode") {
		r_ret = int(friction_combine_mode);
	} else {
		return false;
	}

	return true;
}

void PhysicsMaterial::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::REAL, "bounce"));
	p_list->push_back(PropertyInfo(Variant::INT, "bounce_combine_mode", PROPERTY_HINT_ENUM, "Max,Min,Multiply,Average"));
	p_list->push_back(PropertyInfo(Variant::REAL, "friction"));
	p_list->push_back(PropertyInfo(Variant::INT, "friction_combine_mode", PROPERTY_HINT_ENUM, "Max,Min,Multiply,Average"));
}

void PhysicsMaterial::_bind_methods() {}

void PhysicsMaterial::set_bounce(real_t p_val) {
	bounce = p_val;
}

void PhysicsMaterial::set_bounce_combine_mode(PhysicsServer::CombineMode p_val) {
	bounce_combine_mode = p_val;
}

void PhysicsMaterial::set_friction(real_t p_val) {
	friction = p_val;
}

void PhysicsMaterial::set_friction_combine_mode(PhysicsServer::CombineMode p_val) {
	friction_combine_mode = p_val;
}

PhysicsMaterial::PhysicsMaterial() :
		bounce(0),
		bounce_combine_mode(PhysicsServer::COMBINE_MODE_MAX),
		friction(0),
		friction_combine_mode(PhysicsServer::COMBINE_MODE_MULTIPLY) {}
