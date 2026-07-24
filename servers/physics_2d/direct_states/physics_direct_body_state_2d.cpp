/**************************************************************************/
/*  physics_direct_body_state_2d.cpp                                      */
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

#include "physics_direct_body_state_2d.h"

#include "core/object/class_db.h"

void PhysicsDirectBodyState2D::integrate_forces() {
	real_t step = get_step();
	Vector2 lv = get_linear_velocity();
	lv += get_total_gravity() * step;

	real_t av = get_angular_velocity();

	real_t damp = 1.0 - step * get_total_linear_damp();

	if (damp < 0) { // reached zero in the given time
		damp = 0;
	}

	lv *= damp;

	damp = 1.0 - step * get_total_angular_damp();

	if (damp < 0) { // reached zero in the given time
		damp = 0;
	}

	av *= damp;

	set_linear_velocity(lv);
	set_angular_velocity(av);
}

Object *PhysicsDirectBodyState2D::get_contact_collider_object(int p_contact_idx) const {
	ObjectID objid = get_contact_collider_id(p_contact_idx);
	Object *obj = ObjectDB::get_instance(objid);
	return obj;
}

void PhysicsDirectBodyState2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_total_gravity"), &PhysicsDirectBodyState2D::get_total_gravity);
	ClassDB::bind_method(D_METHOD("get_total_linear_damp"), &PhysicsDirectBodyState2D::get_total_linear_damp);
	ClassDB::bind_method(D_METHOD("get_total_angular_damp"), &PhysicsDirectBodyState2D::get_total_angular_damp);

	ClassDB::bind_method(D_METHOD("get_center_of_mass"), &PhysicsDirectBodyState2D::get_center_of_mass);
	ClassDB::bind_method(D_METHOD("get_center_of_mass_local"), &PhysicsDirectBodyState2D::get_center_of_mass_local);
	ClassDB::bind_method(D_METHOD("get_inverse_mass"), &PhysicsDirectBodyState2D::get_inverse_mass);
	ClassDB::bind_method(D_METHOD("get_inverse_inertia"), &PhysicsDirectBodyState2D::get_inverse_inertia);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &PhysicsDirectBodyState2D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &PhysicsDirectBodyState2D::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &PhysicsDirectBodyState2D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &PhysicsDirectBodyState2D::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsDirectBodyState2D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsDirectBodyState2D::get_transform);

	ClassDB::bind_method(D_METHOD("get_velocity_at_local_position", "local_position"), &PhysicsDirectBodyState2D::get_velocity_at_local_position);

	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &PhysicsDirectBodyState2D::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "impulse"), &PhysicsDirectBodyState2D::apply_torque_impulse);
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "position"), &PhysicsDirectBodyState2D::apply_impulse, Vector2());
	ClassDB::bind_method(D_METHOD("apply_impulse_at_position", "impulse", "global_position"), &PhysicsDirectBodyState2D::apply_impulse_at_position, Vector2());

	ClassDB::bind_method(D_METHOD("apply_central_force", "force"), &PhysicsDirectBodyState2D::apply_central_force, Vector2());
	ClassDB::bind_method(D_METHOD("apply_force", "force", "position"), &PhysicsDirectBodyState2D::apply_force, Vector2());
	ClassDB::bind_method(D_METHOD("apply_force_at_position", "force", "global_position"), &PhysicsDirectBodyState2D::apply_force_at_position, Vector2());
	ClassDB::bind_method(D_METHOD("apply_torque", "torque"), &PhysicsDirectBodyState2D::apply_torque);

	ClassDB::bind_method(D_METHOD("add_constant_central_force", "force"), &PhysicsDirectBodyState2D::add_constant_central_force, Vector2());
	ClassDB::bind_method(D_METHOD("add_constant_force", "force", "position"), &PhysicsDirectBodyState2D::add_constant_force, Vector2());
	ClassDB::bind_method(D_METHOD("add_constant_torque", "torque"), &PhysicsDirectBodyState2D::add_constant_torque);

	ClassDB::bind_method(D_METHOD("set_constant_force", "force"), &PhysicsDirectBodyState2D::set_constant_force);
	ClassDB::bind_method(D_METHOD("get_constant_force"), &PhysicsDirectBodyState2D::get_constant_force);

	ClassDB::bind_method(D_METHOD("set_constant_torque", "torque"), &PhysicsDirectBodyState2D::set_constant_torque);
	ClassDB::bind_method(D_METHOD("get_constant_torque"), &PhysicsDirectBodyState2D::get_constant_torque);

	ClassDB::bind_method(D_METHOD("set_sleep_state", "enabled"), &PhysicsDirectBodyState2D::set_sleep_state);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &PhysicsDirectBodyState2D::is_sleeping);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &PhysicsDirectBodyState2D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &PhysicsDirectBodyState2D::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &PhysicsDirectBodyState2D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &PhysicsDirectBodyState2D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("get_contact_count"), &PhysicsDirectBodyState2D::get_contact_count);

	ClassDB::bind_method(D_METHOD("get_contact_local_position", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_local_position);
	ClassDB::bind_method(D_METHOD("get_contact_local_normal", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_local_normal);
	ClassDB::bind_method(D_METHOD("get_contact_local_shape", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_local_shape);
	ClassDB::bind_method(D_METHOD("get_contact_local_velocity_at_position", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_local_velocity_at_position);
	ClassDB::bind_method(D_METHOD("get_contact_collider", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider);
	ClassDB::bind_method(D_METHOD("get_contact_collider_position", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_position);
	ClassDB::bind_method(D_METHOD("get_contact_collider_id", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_id);
	ClassDB::bind_method(D_METHOD("get_contact_collider_object", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_object);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider_velocity_at_position", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_velocity_at_position);
	ClassDB::bind_method(D_METHOD("get_contact_impulse", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_impulse);
	ClassDB::bind_method(D_METHOD("get_step"), &PhysicsDirectBodyState2D::get_step);
	ClassDB::bind_method(D_METHOD("integrate_forces"), &PhysicsDirectBodyState2D::integrate_forces);
	ClassDB::bind_method(D_METHOD("get_space_state"), &PhysicsDirectBodyState2D::get_space_state);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step"), "", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inverse_mass"), "", "get_inverse_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inverse_inertia"), "", "get_inverse_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_angular_damp"), "", "get_total_angular_damp");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_linear_damp"), "", "get_total_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "total_gravity"), "", "get_total_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "center_of_mass"), "", "get_center_of_mass");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "center_of_mass_local"), "", "get_center_of_mass_local");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleep_state", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer"), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask"), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "transform"), "set_transform", "get_transform");
}

PhysicsDirectBodyState2D::PhysicsDirectBodyState2D() {}
