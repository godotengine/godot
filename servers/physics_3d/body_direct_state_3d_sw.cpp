/*************************************************************************/
/*  body_direct_state_3d_sw.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "body_direct_state_3d_sw.h"

#include "body_3d_sw.h"
#include "space_3d_sw.h"

Vector3 PhysicsDirectBodyState3DSW::get_total_gravity() const {
	return body->gravity;
}

real_t PhysicsDirectBodyState3DSW::get_total_angular_damp() const {
	return body->area_angular_damp;
}

real_t PhysicsDirectBodyState3DSW::get_total_linear_damp() const {
	return body->area_linear_damp;
}

Vector3 PhysicsDirectBodyState3DSW::get_center_of_mass() const {
	return body->get_center_of_mass();
}

Basis PhysicsDirectBodyState3DSW::get_principal_inertia_axes() const {
	return body->get_principal_inertia_axes();
}

real_t PhysicsDirectBodyState3DSW::get_inverse_mass() const {
	return body->get_inv_mass();
}

Vector3 PhysicsDirectBodyState3DSW::get_inverse_inertia() const {
	return body->get_inv_inertia();
}

Basis PhysicsDirectBodyState3DSW::get_inverse_inertia_tensor() const {
	return body->get_inv_inertia_tensor();
}

void PhysicsDirectBodyState3DSW::set_linear_velocity(const Vector3 &p_velocity) {
	body->wakeup();
	body->set_linear_velocity(p_velocity);
}

Vector3 PhysicsDirectBodyState3DSW::get_linear_velocity() const {
	return body->get_linear_velocity();
}

void PhysicsDirectBodyState3DSW::set_angular_velocity(const Vector3 &p_velocity) {
	body->wakeup();
	body->set_angular_velocity(p_velocity);
}

Vector3 PhysicsDirectBodyState3DSW::get_angular_velocity() const {
	return body->get_angular_velocity();
}

void PhysicsDirectBodyState3DSW::set_transform(const Transform3D &p_transform) {
	body->set_state(PhysicsServer3D::BODY_STATE_TRANSFORM, p_transform);
}

Transform3D PhysicsDirectBodyState3DSW::get_transform() const {
	return body->get_transform();
}

Vector3 PhysicsDirectBodyState3DSW::get_velocity_at_local_position(const Vector3 &p_position) const {
	return body->get_velocity_in_local_point(p_position);
}

void PhysicsDirectBodyState3DSW::add_central_force(const Vector3 &p_force) {
	body->wakeup();
	body->add_central_force(p_force);
}

void PhysicsDirectBodyState3DSW::add_force(const Vector3 &p_force, const Vector3 &p_position) {
	body->wakeup();
	body->add_force(p_force, p_position);
}

void PhysicsDirectBodyState3DSW::add_torque(const Vector3 &p_torque) {
	body->wakeup();
	body->add_torque(p_torque);
}

void PhysicsDirectBodyState3DSW::apply_central_impulse(const Vector3 &p_impulse) {
	body->wakeup();
	body->apply_central_impulse(p_impulse);
}

void PhysicsDirectBodyState3DSW::apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position) {
	body->wakeup();
	body->apply_impulse(p_impulse, p_position);
}

void PhysicsDirectBodyState3DSW::apply_torque_impulse(const Vector3 &p_impulse) {
	body->wakeup();
	body->apply_torque_impulse(p_impulse);
}

void PhysicsDirectBodyState3DSW::set_sleep_state(bool p_sleep) {
	body->set_active(!p_sleep);
}

bool PhysicsDirectBodyState3DSW::is_sleeping() const {
	return !body->is_active();
}

int PhysicsDirectBodyState3DSW::get_contact_count() const {
	return body->contact_count;
}

Vector3 PhysicsDirectBodyState3DSW::get_contact_local_position(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, Vector3());
	return body->contacts[p_contact_idx].local_pos;
}

Vector3 PhysicsDirectBodyState3DSW::get_contact_local_normal(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, Vector3());
	return body->contacts[p_contact_idx].local_normal;
}

real_t PhysicsDirectBodyState3DSW::get_contact_impulse(int p_contact_idx) const {
	return 0.0f; // Only implemented for bullet
}

int PhysicsDirectBodyState3DSW::get_contact_local_shape(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, -1);
	return body->contacts[p_contact_idx].local_shape;
}

RID PhysicsDirectBodyState3DSW::get_contact_collider(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, RID());
	return body->contacts[p_contact_idx].collider;
}

Vector3 PhysicsDirectBodyState3DSW::get_contact_collider_position(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, Vector3());
	return body->contacts[p_contact_idx].collider_pos;
}

ObjectID PhysicsDirectBodyState3DSW::get_contact_collider_id(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, ObjectID());
	return body->contacts[p_contact_idx].collider_instance_id;
}

int PhysicsDirectBodyState3DSW::get_contact_collider_shape(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, 0);
	return body->contacts[p_contact_idx].collider_shape;
}

Vector3 PhysicsDirectBodyState3DSW::get_contact_collider_velocity_at_position(int p_contact_idx) const {
	ERR_FAIL_INDEX_V(p_contact_idx, body->contact_count, Vector3());
	return body->contacts[p_contact_idx].collider_velocity_at_pos;
}

PhysicsDirectSpaceState3D *PhysicsDirectBodyState3DSW::get_space_state() {
	return body->get_space()->get_direct_state();
}

real_t PhysicsDirectBodyState3DSW::get_step() const {
	return body->get_space()->get_last_step();
}
