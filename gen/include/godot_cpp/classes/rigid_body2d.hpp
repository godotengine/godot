/**************************************************************************/
/*  rigid_body2d.hpp                                                      */
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

#include <godot_cpp/classes/physics_body2d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node2D;
class PhysicsDirectBodyState2D;
class PhysicsMaterial;

class RigidBody2D : public PhysicsBody2D {
	GDEXTENSION_CLASS(RigidBody2D, PhysicsBody2D)

public:
	enum FreezeMode {
		FREEZE_MODE_STATIC = 0,
		FREEZE_MODE_KINEMATIC = 1,
	};

	enum CenterOfMassMode {
		CENTER_OF_MASS_MODE_AUTO = 0,
		CENTER_OF_MASS_MODE_CUSTOM = 1,
	};

	enum DampMode {
		DAMP_MODE_COMBINE = 0,
		DAMP_MODE_REPLACE = 1,
	};

	enum CCDMode {
		CCD_MODE_DISABLED = 0,
		CCD_MODE_CAST_RAY = 1,
		CCD_MODE_CAST_SHAPE = 2,
	};

	void set_mass(float p_mass);
	float get_mass() const;
	float get_inertia() const;
	void set_inertia(float p_inertia);
	void set_center_of_mass_mode(RigidBody2D::CenterOfMassMode p_mode);
	RigidBody2D::CenterOfMassMode get_center_of_mass_mode() const;
	void set_center_of_mass(const Vector2 &p_center_of_mass);
	Vector2 get_center_of_mass() const;
	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;
	void set_gravity_scale(float p_gravity_scale);
	float get_gravity_scale() const;
	void set_linear_damp_mode(RigidBody2D::DampMode p_linear_damp_mode);
	RigidBody2D::DampMode get_linear_damp_mode() const;
	void set_angular_damp_mode(RigidBody2D::DampMode p_angular_damp_mode);
	RigidBody2D::DampMode get_angular_damp_mode() const;
	void set_linear_damp(float p_linear_damp);
	float get_linear_damp() const;
	void set_angular_damp(float p_angular_damp);
	float get_angular_damp() const;
	void set_linear_velocity(const Vector2 &p_linear_velocity);
	Vector2 get_linear_velocity() const;
	void set_angular_velocity(float p_angular_velocity);
	float get_angular_velocity() const;
	void set_max_contacts_reported(int32_t p_amount);
	int32_t get_max_contacts_reported() const;
	int32_t get_contact_count() const;
	void set_use_custom_integrator(bool p_enable);
	bool is_using_custom_integrator();
	void set_contact_monitor(bool p_enabled);
	bool is_contact_monitor_enabled() const;
	void set_continuous_collision_detection_mode(RigidBody2D::CCDMode p_mode);
	RigidBody2D::CCDMode get_continuous_collision_detection_mode() const;
	void set_axis_velocity(const Vector2 &p_axis_velocity);
	void apply_central_impulse(const Vector2 &p_impulse = Vector2(0, 0));
	void apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position = Vector2(0, 0));
	void apply_torque_impulse(float p_torque);
	void apply_central_force(const Vector2 &p_force);
	void apply_force(const Vector2 &p_force, const Vector2 &p_position = Vector2(0, 0));
	void apply_torque(float p_torque);
	void add_constant_central_force(const Vector2 &p_force);
	void add_constant_force(const Vector2 &p_force, const Vector2 &p_position = Vector2(0, 0));
	void add_constant_torque(float p_torque);
	void set_constant_force(const Vector2 &p_force);
	Vector2 get_constant_force() const;
	void set_constant_torque(float p_torque);
	float get_constant_torque() const;
	void set_sleeping(bool p_sleeping);
	bool is_sleeping() const;
	void set_can_sleep(bool p_able_to_sleep);
	bool is_able_to_sleep() const;
	void set_lock_rotation_enabled(bool p_lock_rotation);
	bool is_lock_rotation_enabled() const;
	void set_freeze_enabled(bool p_freeze_mode);
	bool is_freeze_enabled() const;
	void set_freeze_mode(RigidBody2D::FreezeMode p_freeze_mode);
	RigidBody2D::FreezeMode get_freeze_mode() const;
	TypedArray<Node2D> get_colliding_bodies() const;
	virtual void _integrate_forces(PhysicsDirectBodyState2D *p_state);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PhysicsBody2D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_integrate_forces), decltype(&T::_integrate_forces)>) {
			BIND_VIRTUAL_METHOD(T, _integrate_forces, 370287496);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(RigidBody2D::FreezeMode);
VARIANT_ENUM_CAST(RigidBody2D::CenterOfMassMode);
VARIANT_ENUM_CAST(RigidBody2D::DampMode);
VARIANT_ENUM_CAST(RigidBody2D::CCDMode);

