/**************************************************************************/
/*  spring_bone_simulator3d.hpp                                           */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/skeleton_modifier3d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Curve;

class SpringBoneSimulator3D : public SkeletonModifier3D {
	GDEXTENSION_CLASS(SpringBoneSimulator3D, SkeletonModifier3D)

public:
	enum CenterFrom {
		CENTER_FROM_WORLD_ORIGIN = 0,
		CENTER_FROM_NODE = 1,
		CENTER_FROM_BONE = 2,
	};

	void set_root_bone_name(int32_t p_index, const String &p_bone_name);
	String get_root_bone_name(int32_t p_index) const;
	void set_root_bone(int32_t p_index, int32_t p_bone);
	int32_t get_root_bone(int32_t p_index) const;
	void set_end_bone_name(int32_t p_index, const String &p_bone_name);
	String get_end_bone_name(int32_t p_index) const;
	void set_end_bone(int32_t p_index, int32_t p_bone);
	int32_t get_end_bone(int32_t p_index) const;
	void set_extend_end_bone(int32_t p_index, bool p_enabled);
	bool is_end_bone_extended(int32_t p_index) const;
	void set_end_bone_direction(int32_t p_index, SkeletonModifier3D::BoneDirection p_bone_direction);
	SkeletonModifier3D::BoneDirection get_end_bone_direction(int32_t p_index) const;
	void set_end_bone_length(int32_t p_index, float p_length);
	float get_end_bone_length(int32_t p_index) const;
	void set_center_from(int32_t p_index, SpringBoneSimulator3D::CenterFrom p_center_from);
	SpringBoneSimulator3D::CenterFrom get_center_from(int32_t p_index) const;
	void set_center_node(int32_t p_index, const NodePath &p_node_path);
	NodePath get_center_node(int32_t p_index) const;
	void set_center_bone_name(int32_t p_index, const String &p_bone_name);
	String get_center_bone_name(int32_t p_index) const;
	void set_center_bone(int32_t p_index, int32_t p_bone);
	int32_t get_center_bone(int32_t p_index) const;
	void set_radius(int32_t p_index, float p_radius);
	float get_radius(int32_t p_index) const;
	void set_rotation_axis(int32_t p_index, SkeletonModifier3D::RotationAxis p_axis);
	SkeletonModifier3D::RotationAxis get_rotation_axis(int32_t p_index) const;
	void set_rotation_axis_vector(int32_t p_index, const Vector3 &p_vector);
	Vector3 get_rotation_axis_vector(int32_t p_index) const;
	void set_radius_damping_curve(int32_t p_index, const Ref<Curve> &p_curve);
	Ref<Curve> get_radius_damping_curve(int32_t p_index) const;
	void set_stiffness(int32_t p_index, float p_stiffness);
	float get_stiffness(int32_t p_index) const;
	void set_stiffness_damping_curve(int32_t p_index, const Ref<Curve> &p_curve);
	Ref<Curve> get_stiffness_damping_curve(int32_t p_index) const;
	void set_drag(int32_t p_index, float p_drag);
	float get_drag(int32_t p_index) const;
	void set_drag_damping_curve(int32_t p_index, const Ref<Curve> &p_curve);
	Ref<Curve> get_drag_damping_curve(int32_t p_index) const;
	void set_gravity(int32_t p_index, float p_gravity);
	float get_gravity(int32_t p_index) const;
	void set_gravity_damping_curve(int32_t p_index, const Ref<Curve> &p_curve);
	Ref<Curve> get_gravity_damping_curve(int32_t p_index) const;
	void set_gravity_direction(int32_t p_index, const Vector3 &p_gravity_direction);
	Vector3 get_gravity_direction(int32_t p_index) const;
	void set_setting_count(int32_t p_count);
	int32_t get_setting_count() const;
	void clear_settings();
	void set_individual_config(int32_t p_index, bool p_enabled);
	bool is_config_individual(int32_t p_index) const;
	String get_joint_bone_name(int32_t p_index, int32_t p_joint) const;
	int32_t get_joint_bone(int32_t p_index, int32_t p_joint) const;
	void set_joint_rotation_axis(int32_t p_index, int32_t p_joint, SkeletonModifier3D::RotationAxis p_axis);
	SkeletonModifier3D::RotationAxis get_joint_rotation_axis(int32_t p_index, int32_t p_joint) const;
	void set_joint_rotation_axis_vector(int32_t p_index, int32_t p_joint, const Vector3 &p_vector);
	Vector3 get_joint_rotation_axis_vector(int32_t p_index, int32_t p_joint) const;
	void set_joint_radius(int32_t p_index, int32_t p_joint, float p_radius);
	float get_joint_radius(int32_t p_index, int32_t p_joint) const;
	void set_joint_stiffness(int32_t p_index, int32_t p_joint, float p_stiffness);
	float get_joint_stiffness(int32_t p_index, int32_t p_joint) const;
	void set_joint_drag(int32_t p_index, int32_t p_joint, float p_drag);
	float get_joint_drag(int32_t p_index, int32_t p_joint) const;
	void set_joint_gravity(int32_t p_index, int32_t p_joint, float p_gravity);
	float get_joint_gravity(int32_t p_index, int32_t p_joint) const;
	void set_joint_gravity_direction(int32_t p_index, int32_t p_joint, const Vector3 &p_gravity_direction);
	Vector3 get_joint_gravity_direction(int32_t p_index, int32_t p_joint) const;
	int32_t get_joint_count(int32_t p_index) const;
	void set_enable_all_child_collisions(int32_t p_index, bool p_enabled);
	bool are_all_child_collisions_enabled(int32_t p_index) const;
	void set_exclude_collision_path(int32_t p_index, int32_t p_collision, const NodePath &p_node_path);
	NodePath get_exclude_collision_path(int32_t p_index, int32_t p_collision) const;
	void set_exclude_collision_count(int32_t p_index, int32_t p_count);
	int32_t get_exclude_collision_count(int32_t p_index) const;
	void clear_exclude_collisions(int32_t p_index);
	void set_collision_path(int32_t p_index, int32_t p_collision, const NodePath &p_node_path);
	NodePath get_collision_path(int32_t p_index, int32_t p_collision) const;
	void set_collision_count(int32_t p_index, int32_t p_count);
	int32_t get_collision_count(int32_t p_index) const;
	void clear_collisions(int32_t p_index);
	void set_external_force(const Vector3 &p_force);
	Vector3 get_external_force() const;
	void set_mutable_bone_axes(bool p_enabled);
	bool are_bone_axes_mutable() const;
	void reset();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SkeletonModifier3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SpringBoneSimulator3D::CenterFrom);

