/**************************************************************************/
/*  physics_direct_body_state2d_extension.cpp                             */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/physics_direct_body_state2d_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_direct_space_state2d.hpp>
#include <godot_cpp/core/object.hpp>

namespace godot {

Vector2 PhysicsDirectBodyState2DExtension::_get_total_gravity() const {
	return Vector2();
}

float PhysicsDirectBodyState2DExtension::_get_total_linear_damp() const {
	return 0.0;
}

float PhysicsDirectBodyState2DExtension::_get_total_angular_damp() const {
	return 0.0;
}

Vector2 PhysicsDirectBodyState2DExtension::_get_center_of_mass() const {
	return Vector2();
}

Vector2 PhysicsDirectBodyState2DExtension::_get_center_of_mass_local() const {
	return Vector2();
}

float PhysicsDirectBodyState2DExtension::_get_inverse_mass() const {
	return 0.0;
}

float PhysicsDirectBodyState2DExtension::_get_inverse_inertia() const {
	return 0.0;
}

void PhysicsDirectBodyState2DExtension::_set_linear_velocity(const Vector2 &p_velocity) {}

Vector2 PhysicsDirectBodyState2DExtension::_get_linear_velocity() const {
	return Vector2();
}

void PhysicsDirectBodyState2DExtension::_set_angular_velocity(float p_velocity) {}

float PhysicsDirectBodyState2DExtension::_get_angular_velocity() const {
	return 0.0;
}

void PhysicsDirectBodyState2DExtension::_set_transform(const Transform2D &p_transform) {}

Transform2D PhysicsDirectBodyState2DExtension::_get_transform() const {
	return Transform2D();
}

Vector2 PhysicsDirectBodyState2DExtension::_get_velocity_at_local_position(const Vector2 &p_local_position) const {
	return Vector2();
}

void PhysicsDirectBodyState2DExtension::_apply_central_impulse(const Vector2 &p_impulse) {}

void PhysicsDirectBodyState2DExtension::_apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position) {}

void PhysicsDirectBodyState2DExtension::_apply_torque_impulse(float p_impulse) {}

void PhysicsDirectBodyState2DExtension::_apply_central_force(const Vector2 &p_force) {}

void PhysicsDirectBodyState2DExtension::_apply_force(const Vector2 &p_force, const Vector2 &p_position) {}

void PhysicsDirectBodyState2DExtension::_apply_torque(float p_torque) {}

void PhysicsDirectBodyState2DExtension::_add_constant_central_force(const Vector2 &p_force) {}

void PhysicsDirectBodyState2DExtension::_add_constant_force(const Vector2 &p_force, const Vector2 &p_position) {}

void PhysicsDirectBodyState2DExtension::_add_constant_torque(float p_torque) {}

void PhysicsDirectBodyState2DExtension::_set_constant_force(const Vector2 &p_force) {}

Vector2 PhysicsDirectBodyState2DExtension::_get_constant_force() const {
	return Vector2();
}

void PhysicsDirectBodyState2DExtension::_set_constant_torque(float p_torque) {}

float PhysicsDirectBodyState2DExtension::_get_constant_torque() const {
	return 0.0;
}

void PhysicsDirectBodyState2DExtension::_set_sleep_state(bool p_enabled) {}

bool PhysicsDirectBodyState2DExtension::_is_sleeping() const {
	return false;
}

void PhysicsDirectBodyState2DExtension::_set_collision_layer(uint32_t p_layer) {}

uint32_t PhysicsDirectBodyState2DExtension::_get_collision_layer() const {
	return 0;
}

void PhysicsDirectBodyState2DExtension::_set_collision_mask(uint32_t p_mask) {}

uint32_t PhysicsDirectBodyState2DExtension::_get_collision_mask() const {
	return 0;
}

int32_t PhysicsDirectBodyState2DExtension::_get_contact_count() const {
	return 0;
}

Vector2 PhysicsDirectBodyState2DExtension::_get_contact_local_position(int32_t p_contact_idx) const {
	return Vector2();
}

Vector2 PhysicsDirectBodyState2DExtension::_get_contact_local_normal(int32_t p_contact_idx) const {
	return Vector2();
}

int32_t PhysicsDirectBodyState2DExtension::_get_contact_local_shape(int32_t p_contact_idx) const {
	return 0;
}

Vector2 PhysicsDirectBodyState2DExtension::_get_contact_local_velocity_at_position(int32_t p_contact_idx) const {
	return Vector2();
}

RID PhysicsDirectBodyState2DExtension::_get_contact_collider(int32_t p_contact_idx) const {
	return RID();
}

Vector2 PhysicsDirectBodyState2DExtension::_get_contact_collider_position(int32_t p_contact_idx) const {
	return Vector2();
}

uint64_t PhysicsDirectBodyState2DExtension::_get_contact_collider_id(int32_t p_contact_idx) const {
	return 0;
}

Object *PhysicsDirectBodyState2DExtension::_get_contact_collider_object(int32_t p_contact_idx) const {
	return nullptr;
}

int32_t PhysicsDirectBodyState2DExtension::_get_contact_collider_shape(int32_t p_contact_idx) const {
	return 0;
}

Vector2 PhysicsDirectBodyState2DExtension::_get_contact_collider_velocity_at_position(int32_t p_contact_idx) const {
	return Vector2();
}

Vector2 PhysicsDirectBodyState2DExtension::_get_contact_impulse(int32_t p_contact_idx) const {
	return Vector2();
}

float PhysicsDirectBodyState2DExtension::_get_step() const {
	return 0.0;
}

void PhysicsDirectBodyState2DExtension::_integrate_forces() {}

PhysicsDirectSpaceState2D *PhysicsDirectBodyState2DExtension::_get_space_state() {
	return nullptr;
}

} // namespace godot
