/**************************************************************************/
/*  iterate_ik3d.hpp                                                      */
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

#include <godot_cpp/classes/chain_ik3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/skeleton_modifier3d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class JointLimitation3D;

class IterateIK3D : public ChainIK3D {
	GDEXTENSION_CLASS(IterateIK3D, ChainIK3D)

public:
	void set_max_iterations(int32_t p_max_iterations);
	int32_t get_max_iterations() const;
	void set_min_distance(double p_min_distance);
	double get_min_distance() const;
	void set_angular_delta_limit(double p_angular_delta_limit);
	double get_angular_delta_limit() const;
	void set_deterministic(bool p_deterministic);
	bool is_deterministic() const;
	void set_target_node(int32_t p_index, const NodePath &p_target_node);
	NodePath get_target_node(int32_t p_index) const;
	void set_joint_rotation_axis(int32_t p_index, int32_t p_joint, SkeletonModifier3D::RotationAxis p_axis);
	SkeletonModifier3D::RotationAxis get_joint_rotation_axis(int32_t p_index, int32_t p_joint) const;
	void set_joint_rotation_axis_vector(int32_t p_index, int32_t p_joint, const Vector3 &p_axis_vector);
	Vector3 get_joint_rotation_axis_vector(int32_t p_index, int32_t p_joint) const;
	void set_joint_limitation(int32_t p_index, int32_t p_joint, const Ref<JointLimitation3D> &p_limitation);
	Ref<JointLimitation3D> get_joint_limitation(int32_t p_index, int32_t p_joint) const;
	void set_joint_limitation_right_axis(int32_t p_index, int32_t p_joint, SkeletonModifier3D::SecondaryDirection p_direction);
	SkeletonModifier3D::SecondaryDirection get_joint_limitation_right_axis(int32_t p_index, int32_t p_joint) const;
	void set_joint_limitation_right_axis_vector(int32_t p_index, int32_t p_joint, const Vector3 &p_vector);
	Vector3 get_joint_limitation_right_axis_vector(int32_t p_index, int32_t p_joint) const;
	void set_joint_limitation_rotation_offset(int32_t p_index, int32_t p_joint, const Quaternion &p_offset);
	Quaternion get_joint_limitation_rotation_offset(int32_t p_index, int32_t p_joint) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		ChainIK3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

