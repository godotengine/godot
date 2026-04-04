/**************************************************************************/
/*  physics_direct_body_state2d.hpp                                       */
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

#pragma once

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PhysicsDirectSpaceState2D;

class PhysicsDirectBodyState2D : public Object {
	GDEXTENSION_CLASS(PhysicsDirectBodyState2D, Object)

public:
	Vector2 get_total_gravity() const;
	float get_total_linear_damp() const;
	float get_total_angular_damp() const;
	Vector2 get_center_of_mass() const;
	Vector2 get_center_of_mass_local() const;
	float get_inverse_mass() const;
	float get_inverse_inertia() const;
	void set_linear_velocity(const Vector2 &p_velocity);
	Vector2 get_linear_velocity() const;
	void set_angular_velocity(float p_velocity);
	float get_angular_velocity() const;
	void set_transform(const Transform2D &p_transform);
	Transform2D get_transform() const;
	Vector2 get_velocity_at_local_position(const Vector2 &p_local_position) const;
	void apply_central_impulse(const Vector2 &p_impulse);
	void apply_torque_impulse(float p_impulse);
	void apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position = Vector2(0, 0));
	void apply_central_force(const Vector2 &p_force = Vector2(0, 0));
	void apply_force(const Vector2 &p_force, const Vector2 &p_position = Vector2(0, 0));
	void apply_torque(float p_torque);
	void add_constant_central_force(const Vector2 &p_force = Vector2(0, 0));
	void add_constant_force(const Vector2 &p_force, const Vector2 &p_position = Vector2(0, 0));
	void add_constant_torque(float p_torque);
	void set_constant_force(const Vector2 &p_force);
	Vector2 get_constant_force() const;
	void set_constant_torque(float p_torque);
	float get_constant_torque() const;
	void set_sleep_state(bool p_enabled);
	bool is_sleeping() const;
	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;
	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;
	int32_t get_contact_count() const;
	Vector2 get_contact_local_position(int32_t p_contact_idx) const;
	Vector2 get_contact_local_normal(int32_t p_contact_idx) const;
	int32_t get_contact_local_shape(int32_t p_contact_idx) const;
	Vector2 get_contact_local_velocity_at_position(int32_t p_contact_idx) const;
	RID get_contact_collider(int32_t p_contact_idx) const;
	Vector2 get_contact_collider_position(int32_t p_contact_idx) const;
	uint64_t get_contact_collider_id(int32_t p_contact_idx) const;
	Object *get_contact_collider_object(int32_t p_contact_idx) const;
	int32_t get_contact_collider_shape(int32_t p_contact_idx) const;
	Vector2 get_contact_collider_velocity_at_position(int32_t p_contact_idx) const;
	Vector2 get_contact_impulse(int32_t p_contact_idx) const;
	float get_step() const;
	void integrate_forces();
	PhysicsDirectSpaceState2D *get_space_state();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

