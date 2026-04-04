/**************************************************************************/
/*  physics_direct_body_state3d_extension.hpp                             */
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

#include <godot_cpp/classes/physics_direct_body_state3d.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Object;
class PhysicsDirectSpaceState3D;

class PhysicsDirectBodyState3DExtension : public PhysicsDirectBodyState3D {
	GDEXTENSION_CLASS(PhysicsDirectBodyState3DExtension, PhysicsDirectBodyState3D)

public:
	virtual Vector3 _get_total_gravity() const;
	virtual float _get_total_linear_damp() const;
	virtual float _get_total_angular_damp() const;
	virtual Vector3 _get_center_of_mass() const;
	virtual Vector3 _get_center_of_mass_local() const;
	virtual Basis _get_principal_inertia_axes() const;
	virtual float _get_inverse_mass() const;
	virtual Vector3 _get_inverse_inertia() const;
	virtual Basis _get_inverse_inertia_tensor() const;
	virtual void _set_linear_velocity(const Vector3 &p_velocity);
	virtual Vector3 _get_linear_velocity() const;
	virtual void _set_angular_velocity(const Vector3 &p_velocity);
	virtual Vector3 _get_angular_velocity() const;
	virtual void _set_transform(const Transform3D &p_transform);
	virtual Transform3D _get_transform() const;
	virtual Vector3 _get_velocity_at_local_position(const Vector3 &p_local_position) const;
	virtual void _apply_central_impulse(const Vector3 &p_impulse);
	virtual void _apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position);
	virtual void _apply_torque_impulse(const Vector3 &p_impulse);
	virtual void _apply_central_force(const Vector3 &p_force);
	virtual void _apply_force(const Vector3 &p_force, const Vector3 &p_position);
	virtual void _apply_torque(const Vector3 &p_torque);
	virtual void _add_constant_central_force(const Vector3 &p_force);
	virtual void _add_constant_force(const Vector3 &p_force, const Vector3 &p_position);
	virtual void _add_constant_torque(const Vector3 &p_torque);
	virtual void _set_constant_force(const Vector3 &p_force);
	virtual Vector3 _get_constant_force() const;
	virtual void _set_constant_torque(const Vector3 &p_torque);
	virtual Vector3 _get_constant_torque() const;
	virtual void _set_sleep_state(bool p_enabled);
	virtual bool _is_sleeping() const;
	virtual void _set_collision_layer(uint32_t p_layer);
	virtual uint32_t _get_collision_layer() const;
	virtual void _set_collision_mask(uint32_t p_mask);
	virtual uint32_t _get_collision_mask() const;
	virtual int32_t _get_contact_count() const;
	virtual Vector3 _get_contact_local_position(int32_t p_contact_idx) const;
	virtual Vector3 _get_contact_local_normal(int32_t p_contact_idx) const;
	virtual Vector3 _get_contact_impulse(int32_t p_contact_idx) const;
	virtual int32_t _get_contact_local_shape(int32_t p_contact_idx) const;
	virtual Vector3 _get_contact_local_velocity_at_position(int32_t p_contact_idx) const;
	virtual RID _get_contact_collider(int32_t p_contact_idx) const;
	virtual Vector3 _get_contact_collider_position(int32_t p_contact_idx) const;
	virtual uint64_t _get_contact_collider_id(int32_t p_contact_idx) const;
	virtual Object *_get_contact_collider_object(int32_t p_contact_idx) const;
	virtual int32_t _get_contact_collider_shape(int32_t p_contact_idx) const;
	virtual Vector3 _get_contact_collider_velocity_at_position(int32_t p_contact_idx) const;
	virtual float _get_step() const;
	virtual void _integrate_forces();
	virtual PhysicsDirectSpaceState3D *_get_space_state();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PhysicsDirectBodyState3D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_total_gravity), decltype(&T::_get_total_gravity)>) {
			BIND_VIRTUAL_METHOD(T, _get_total_gravity, 3360562783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_total_linear_damp), decltype(&T::_get_total_linear_damp)>) {
			BIND_VIRTUAL_METHOD(T, _get_total_linear_damp, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_total_angular_damp), decltype(&T::_get_total_angular_damp)>) {
			BIND_VIRTUAL_METHOD(T, _get_total_angular_damp, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_center_of_mass), decltype(&T::_get_center_of_mass)>) {
			BIND_VIRTUAL_METHOD(T, _get_center_of_mass, 3360562783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_center_of_mass_local), decltype(&T::_get_center_of_mass_local)>) {
			BIND_VIRTUAL_METHOD(T, _get_center_of_mass_local, 3360562783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_principal_inertia_axes), decltype(&T::_get_principal_inertia_axes)>) {
			BIND_VIRTUAL_METHOD(T, _get_principal_inertia_axes, 2716978435);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_inverse_mass), decltype(&T::_get_inverse_mass)>) {
			BIND_VIRTUAL_METHOD(T, _get_inverse_mass, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_inverse_inertia), decltype(&T::_get_inverse_inertia)>) {
			BIND_VIRTUAL_METHOD(T, _get_inverse_inertia, 3360562783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_inverse_inertia_tensor), decltype(&T::_get_inverse_inertia_tensor)>) {
			BIND_VIRTUAL_METHOD(T, _get_inverse_inertia_tensor, 2716978435);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_linear_velocity), decltype(&T::_set_linear_velocity)>) {
			BIND_VIRTUAL_METHOD(T, _set_linear_velocity, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_linear_velocity), decltype(&T::_get_linear_velocity)>) {
			BIND_VIRTUAL_METHOD(T, _get_linear_velocity, 3360562783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_angular_velocity), decltype(&T::_set_angular_velocity)>) {
			BIND_VIRTUAL_METHOD(T, _set_angular_velocity, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_angular_velocity), decltype(&T::_get_angular_velocity)>) {
			BIND_VIRTUAL_METHOD(T, _get_angular_velocity, 3360562783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_transform), decltype(&T::_set_transform)>) {
			BIND_VIRTUAL_METHOD(T, _set_transform, 2952846383);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_transform), decltype(&T::_get_transform)>) {
			BIND_VIRTUAL_METHOD(T, _get_transform, 3229777777);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_velocity_at_local_position), decltype(&T::_get_velocity_at_local_position)>) {
			BIND_VIRTUAL_METHOD(T, _get_velocity_at_local_position, 192990374);
		}
		if constexpr (!std::is_same_v<decltype(&B::_apply_central_impulse), decltype(&T::_apply_central_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _apply_central_impulse, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_apply_impulse), decltype(&T::_apply_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _apply_impulse, 1714681797);
		}
		if constexpr (!std::is_same_v<decltype(&B::_apply_torque_impulse), decltype(&T::_apply_torque_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _apply_torque_impulse, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_apply_central_force), decltype(&T::_apply_central_force)>) {
			BIND_VIRTUAL_METHOD(T, _apply_central_force, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_apply_force), decltype(&T::_apply_force)>) {
			BIND_VIRTUAL_METHOD(T, _apply_force, 1714681797);
		}
		if constexpr (!std::is_same_v<decltype(&B::_apply_torque), decltype(&T::_apply_torque)>) {
			BIND_VIRTUAL_METHOD(T, _apply_torque, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_add_constant_central_force), decltype(&T::_add_constant_central_force)>) {
			BIND_VIRTUAL_METHOD(T, _add_constant_central_force, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_add_constant_force), decltype(&T::_add_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _add_constant_force, 1714681797);
		}
		if constexpr (!std::is_same_v<decltype(&B::_add_constant_torque), decltype(&T::_add_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _add_constant_torque, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_constant_force), decltype(&T::_set_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _set_constant_force, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_constant_force), decltype(&T::_get_constant_force)>) {
			BIND_VIRTUAL_METHOD(T, _get_constant_force, 3360562783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_constant_torque), decltype(&T::_set_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _set_constant_torque, 3460891852);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_constant_torque), decltype(&T::_get_constant_torque)>) {
			BIND_VIRTUAL_METHOD(T, _get_constant_torque, 3360562783);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_sleep_state), decltype(&T::_set_sleep_state)>) {
			BIND_VIRTUAL_METHOD(T, _set_sleep_state, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_sleeping), decltype(&T::_is_sleeping)>) {
			BIND_VIRTUAL_METHOD(T, _is_sleeping, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_collision_layer), decltype(&T::_set_collision_layer)>) {
			BIND_VIRTUAL_METHOD(T, _set_collision_layer, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_collision_layer), decltype(&T::_get_collision_layer)>) {
			BIND_VIRTUAL_METHOD(T, _get_collision_layer, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_collision_mask), decltype(&T::_set_collision_mask)>) {
			BIND_VIRTUAL_METHOD(T, _set_collision_mask, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_collision_mask), decltype(&T::_get_collision_mask)>) {
			BIND_VIRTUAL_METHOD(T, _get_collision_mask, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_count), decltype(&T::_get_contact_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_local_position), decltype(&T::_get_contact_local_position)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_local_position, 711720468);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_local_normal), decltype(&T::_get_contact_local_normal)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_local_normal, 711720468);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_impulse), decltype(&T::_get_contact_impulse)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_impulse, 711720468);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_local_shape), decltype(&T::_get_contact_local_shape)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_local_shape, 923996154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_local_velocity_at_position), decltype(&T::_get_contact_local_velocity_at_position)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_local_velocity_at_position, 711720468);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_collider), decltype(&T::_get_contact_collider)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_collider, 495598643);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_collider_position), decltype(&T::_get_contact_collider_position)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_collider_position, 711720468);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_collider_id), decltype(&T::_get_contact_collider_id)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_collider_id, 923996154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_collider_object), decltype(&T::_get_contact_collider_object)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_collider_object, 3332903315);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_collider_shape), decltype(&T::_get_contact_collider_shape)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_collider_shape, 923996154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_contact_collider_velocity_at_position), decltype(&T::_get_contact_collider_velocity_at_position)>) {
			BIND_VIRTUAL_METHOD(T, _get_contact_collider_velocity_at_position, 711720468);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_step), decltype(&T::_get_step)>) {
			BIND_VIRTUAL_METHOD(T, _get_step, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_integrate_forces), decltype(&T::_integrate_forces)>) {
			BIND_VIRTUAL_METHOD(T, _integrate_forces, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_space_state), decltype(&T::_get_space_state)>) {
			BIND_VIRTUAL_METHOD(T, _get_space_state, 2069328350);
		}
	}

public:
};

} // namespace godot

