/**************************************************************************/
/*  physics_direct_body_state_3d.cpp                                      */
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

#include "physics_direct_body_state_3d.h"

#include "core/object/class_db.h"

void PhysicsDirectBodyState3D::integrate_forces() {
	real_t step = get_step();
	Vector3 lv = get_linear_velocity();
	lv += get_total_gravity() * step;

	Vector3 av = get_angular_velocity();

	real_t linear_damp = 1.0 - step * get_total_linear_damp();

	if (linear_damp < 0) { // reached zero in the given time
		linear_damp = 0;
	}

	real_t angular_damp = 1.0 - step * get_total_angular_damp();

	if (angular_damp < 0) { // reached zero in the given time
		angular_damp = 0;
	}

	lv *= linear_damp;
	av *= angular_damp;

	set_linear_velocity(lv);
	set_angular_velocity(av);
}

Object *PhysicsDirectBodyState3D::get_contact_collider_object(int p_contact_idx) const {
	ObjectID objid = get_contact_collider_id(p_contact_idx);
	Object *obj = ObjectDB::get_instance(objid);
	return obj;
}

void PhysicsDirectBodyState3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_total_gravity"), &PhysicsDirectBodyState3D::get_total_gravity);
	ClassDB::bind_method(D_METHOD("get_total_linear_damp"), &PhysicsDirectBodyState3D::get_total_linear_damp);
	ClassDB::bind_method(D_METHOD("get_total_angular_damp"), &PhysicsDirectBodyState3D::get_total_angular_damp);

	ClassDB::bind_method(D_METHOD("get_center_of_mass"), &PhysicsDirectBodyState3D::get_center_of_mass);
	ClassDB::bind_method(D_METHOD("get_center_of_mass_local"), &PhysicsDirectBodyState3D::get_center_of_mass_local);
	ClassDB::bind_method(D_METHOD("get_principal_inertia_axes"), &PhysicsDirectBodyState3D::get_principal_inertia_axes);

	ClassDB::bind_method(D_METHOD("get_inverse_mass"), &PhysicsDirectBodyState3D::get_inverse_mass);
	ClassDB::bind_method(D_METHOD("get_inverse_inertia"), &PhysicsDirectBodyState3D::get_inverse_inertia);
	ClassDB::bind_method(D_METHOD("get_inverse_inertia_tensor"), &PhysicsDirectBodyState3D::get_inverse_inertia_tensor);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &PhysicsDirectBodyState3D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &PhysicsDirectBodyState3D::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &PhysicsDirectBodyState3D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &PhysicsDirectBodyState3D::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsDirectBodyState3D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsDirectBodyState3D::get_transform);

	ClassDB::bind_method(D_METHOD("get_velocity_at_local_position", "local_position"), &PhysicsDirectBodyState3D::get_velocity_at_local_position);

	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &PhysicsDirectBodyState3D::apply_central_impulse, Vector3());
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "position"), &PhysicsDirectBodyState3D::apply_impulse, Vector3());
	ClassDB::bind_method(D_METHOD("apply_impulse_at_position", "impulse", "global_position"), &PhysicsDirectBodyState3D::apply_impulse_at_position, Vector3());
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "impulse"), &PhysicsDirectBodyState3D::apply_torque_impulse);

	ClassDB::bind_method(D_METHOD("apply_central_force", "force"), &PhysicsDirectBodyState3D::apply_central_force, Vector3());
	ClassDB::bind_method(D_METHOD("apply_force", "force", "position"), &PhysicsDirectBodyState3D::apply_force, Vector3());
	ClassDB::bind_method(D_METHOD("apply_force_at_position", "force", "global_position"), &PhysicsDirectBodyState3D::apply_force_at_position, Vector3());
	ClassDB::bind_method(D_METHOD("apply_torque", "torque"), &PhysicsDirectBodyState3D::apply_torque);

	ClassDB::bind_method(D_METHOD("add_constant_central_force", "force"), &PhysicsDirectBodyState3D::add_constant_central_force, Vector3());
	ClassDB::bind_method(D_METHOD("add_constant_force", "force", "position"), &PhysicsDirectBodyState3D::add_constant_force, Vector3());
	ClassDB::bind_method(D_METHOD("add_constant_torque", "torque"), &PhysicsDirectBodyState3D::add_constant_torque);

	ClassDB::bind_method(D_METHOD("set_constant_force", "force"), &PhysicsDirectBodyState3D::set_constant_force);
	ClassDB::bind_method(D_METHOD("get_constant_force"), &PhysicsDirectBodyState3D::get_constant_force);

	ClassDB::bind_method(D_METHOD("set_constant_torque", "torque"), &PhysicsDirectBodyState3D::set_constant_torque);
	ClassDB::bind_method(D_METHOD("get_constant_torque"), &PhysicsDirectBodyState3D::get_constant_torque);

	ClassDB::bind_method(D_METHOD("set_sleep_state", "enabled"), &PhysicsDirectBodyState3D::set_sleep_state);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &PhysicsDirectBodyState3D::is_sleeping);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &PhysicsDirectBodyState3D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &PhysicsDirectBodyState3D::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &PhysicsDirectBodyState3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &PhysicsDirectBodyState3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("get_contact_count"), &PhysicsDirectBodyState3D::get_contact_count);

	ClassDB::bind_method(D_METHOD("get_contact_local_position", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_local_position);
	ClassDB::bind_method(D_METHOD("get_contact_local_normal", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_local_normal);
	ClassDB::bind_method(D_METHOD("get_contact_impulse", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_impulse);
	ClassDB::bind_method(D_METHOD("get_contact_local_shape", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_local_shape);
	ClassDB::bind_method(D_METHOD("get_contact_local_velocity_at_position", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_local_velocity_at_position);
	ClassDB::bind_method(D_METHOD("get_contact_collider", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider);
	ClassDB::bind_method(D_METHOD("get_contact_collider_position", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_position);
	ClassDB::bind_method(D_METHOD("get_contact_collider_id", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_id);
	ClassDB::bind_method(D_METHOD("get_contact_collider_object", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_object);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider_velocity_at_position", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_velocity_at_position);
	ClassDB::bind_method(D_METHOD("get_step"), &PhysicsDirectBodyState3D::get_step);
	ClassDB::bind_method(D_METHOD("integrate_forces"), &PhysicsDirectBodyState3D::integrate_forces);
	ClassDB::bind_method(D_METHOD("get_space_state"), &PhysicsDirectBodyState3D::get_space_state);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step"), "", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inverse_mass"), "", "get_inverse_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_angular_damp"), "", "get_total_angular_damp");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_linear_damp"), "", "get_total_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "inverse_inertia"), "", "get_inverse_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::BASIS, "inverse_inertia_tensor"), "", "get_inverse_inertia_tensor");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "total_gravity"), "", "get_total_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_of_mass"), "", "get_center_of_mass");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_of_mass_local"), "", "get_center_of_mass_local");
	ADD_PROPERTY(PropertyInfo(Variant::BASIS, "principal_inertia_axes"), "", "get_principal_inertia_axes");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleep_state", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer"), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask"), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "transform"), "set_transform", "get_transform");
}

PhysicsDirectBodyState3D::PhysicsDirectBodyState3D() {}
