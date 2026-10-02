/**************************************************************************/
/*  physics_direct_body_state_3d.h                                        */
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

#pragma once

#include "core/variant/type_info.h"
#include "servers/physics_3d/direct_states/physics_direct_space_state_3d.h"

class PhysicsDirectBodyState3D : public Object {
	GDCLASS(PhysicsDirectBodyState3D, Object);

protected:
	static void _bind_methods();

public:
	virtual Vector3 get_total_gravity() const = 0;
	virtual real_t get_total_angular_damp() const = 0;
	virtual real_t get_total_linear_damp() const = 0;

	virtual Vector3 get_center_of_mass() const = 0;
	virtual Vector3 get_center_of_mass_local() const = 0;
	virtual Basis get_principal_inertia_axes() const = 0;
	virtual real_t get_inverse_mass() const = 0; // get the mass
	virtual Vector3 get_inverse_inertia() const = 0; // get density of this body space
	virtual Basis get_inverse_inertia_tensor() const = 0; // get density of this body space

	virtual void set_linear_velocity(const Vector3 &p_velocity) = 0;
	virtual Vector3 get_linear_velocity() const = 0;

	virtual void set_angular_velocity(const Vector3 &p_velocity) = 0;
	virtual Vector3 get_angular_velocity() const = 0;

	virtual void set_transform(const Transform3D &p_transform) = 0;
	virtual Transform3D get_transform() const = 0;

	virtual Vector3 get_velocity_at_local_position(const Vector3 &p_position) const = 0;

	virtual void apply_central_impulse(const Vector3 &p_impulse) = 0;
	virtual void apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position = Vector3()) = 0;
	virtual void apply_impulse_at_position(const Vector3 &p_impulse, const Vector3 &p_global_position = Vector3()) = 0;
	virtual void apply_torque_impulse(const Vector3 &p_impulse) = 0;

	virtual void apply_central_force(const Vector3 &p_force) = 0;
	virtual void apply_force(const Vector3 &p_force, const Vector3 &p_position = Vector3()) = 0;
	virtual void apply_force_at_position(const Vector3 &p_force, const Vector3 &p_global_position = Vector3()) = 0;
	virtual void apply_torque(const Vector3 &p_torque) = 0;

	virtual void add_constant_central_force(const Vector3 &p_force) = 0;
	virtual void add_constant_force(const Vector3 &p_force, const Vector3 &p_position = Vector3()) = 0;
	virtual void add_constant_torque(const Vector3 &p_torque) = 0;

	virtual void set_constant_force(const Vector3 &p_force) = 0;
	virtual Vector3 get_constant_force() const = 0;

	virtual void set_constant_torque(const Vector3 &p_torque) = 0;
	virtual Vector3 get_constant_torque() const = 0;

	virtual void set_sleep_state(bool p_sleep) = 0;
	virtual bool is_sleeping() const = 0;

	virtual void set_collision_layer(uint32_t p_layer) = 0;
	virtual uint32_t get_collision_layer() const = 0;

	virtual void set_collision_mask(uint32_t p_mask) = 0;
	virtual uint32_t get_collision_mask() const = 0;

	virtual int get_contact_count() const = 0;

	virtual Vector3 get_contact_local_position(int p_contact_idx) const = 0;
	virtual Vector3 get_contact_local_normal(int p_contact_idx) const = 0;
	virtual Vector3 get_contact_impulse(int p_contact_idx) const = 0;
	virtual int get_contact_local_shape(int p_contact_idx) const = 0;
	virtual Vector3 get_contact_local_velocity_at_position(int p_contact_idx) const = 0;

	virtual RID get_contact_collider(int p_contact_idx) const = 0;
	virtual Vector3 get_contact_collider_position(int p_contact_idx) const = 0;
	virtual ObjectID get_contact_collider_id(int p_contact_idx) const = 0;
	virtual Object *get_contact_collider_object(int p_contact_idx) const;
	virtual int get_contact_collider_shape(int p_contact_idx) const = 0;
	virtual Vector3 get_contact_collider_velocity_at_position(int p_contact_idx) const = 0;

	virtual real_t get_step() const = 0;
	virtual void integrate_forces();

	virtual RequiredResult<PhysicsDirectSpaceState3D> get_space_state() = 0;

	PhysicsDirectBodyState3D();
};
