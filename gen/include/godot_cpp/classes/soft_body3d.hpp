/**************************************************************************/
/*  soft_body3d.hpp                                                       */
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

#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node;
class PhysicsBody3D;

class SoftBody3D : public MeshInstance3D {
	GDEXTENSION_CLASS(SoftBody3D, MeshInstance3D)

public:
	enum DisableMode {
		DISABLE_MODE_REMOVE = 0,
		DISABLE_MODE_KEEP_ACTIVE = 1,
	};

	RID get_physics_rid() const;
	void set_collision_mask(uint32_t p_collision_mask);
	uint32_t get_collision_mask() const;
	void set_collision_layer(uint32_t p_collision_layer);
	uint32_t get_collision_layer() const;
	void set_collision_mask_value(int32_t p_layer_number, bool p_value);
	bool get_collision_mask_value(int32_t p_layer_number) const;
	void set_collision_layer_value(int32_t p_layer_number, bool p_value);
	bool get_collision_layer_value(int32_t p_layer_number) const;
	void set_parent_collision_ignore(const NodePath &p_parent_collision_ignore);
	NodePath get_parent_collision_ignore() const;
	void set_disable_mode(SoftBody3D::DisableMode p_mode);
	SoftBody3D::DisableMode get_disable_mode() const;
	TypedArray<PhysicsBody3D> get_collision_exceptions();
	void add_collision_exception_with(Node *p_body);
	void remove_collision_exception_with(Node *p_body);
	void set_simulation_precision(int32_t p_simulation_precision);
	int32_t get_simulation_precision();
	void set_total_mass(float p_mass);
	float get_total_mass();
	void set_linear_stiffness(float p_linear_stiffness);
	float get_linear_stiffness();
	void set_shrinking_factor(float p_shrinking_factor);
	float get_shrinking_factor();
	void set_pressure_coefficient(float p_pressure_coefficient);
	float get_pressure_coefficient();
	void set_damping_coefficient(float p_damping_coefficient);
	float get_damping_coefficient();
	void set_drag_coefficient(float p_drag_coefficient);
	float get_drag_coefficient();
	Vector3 get_point_transform(int32_t p_point_index);
	void apply_impulse(int32_t p_point_index, const Vector3 &p_impulse);
	void apply_force(int32_t p_point_index, const Vector3 &p_force);
	void apply_central_impulse(const Vector3 &p_impulse);
	void apply_central_force(const Vector3 &p_force);
	void set_point_pinned(int32_t p_point_index, bool p_pinned, const NodePath &p_attachment_path = NodePath(""), int32_t p_insert_at = -1);
	bool is_point_pinned(int32_t p_point_index) const;
	void set_ray_pickable(bool p_ray_pickable);
	bool is_ray_pickable() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		MeshInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SoftBody3D::DisableMode);

