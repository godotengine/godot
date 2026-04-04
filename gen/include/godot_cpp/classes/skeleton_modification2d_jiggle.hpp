/**************************************************************************/
/*  skeleton_modification2d_jiggle.hpp                                    */
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
#include <godot_cpp/classes/skeleton_modification2d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class SkeletonModification2DJiggle : public SkeletonModification2D {
	GDEXTENSION_CLASS(SkeletonModification2DJiggle, SkeletonModification2D)

public:
	void set_target_node(const NodePath &p_target_nodepath);
	NodePath get_target_node() const;
	void set_jiggle_data_chain_length(int32_t p_length);
	int32_t get_jiggle_data_chain_length();
	void set_stiffness(float p_stiffness);
	float get_stiffness() const;
	void set_mass(float p_mass);
	float get_mass() const;
	void set_damping(float p_damping);
	float get_damping() const;
	void set_use_gravity(bool p_use_gravity);
	bool get_use_gravity() const;
	void set_gravity(const Vector2 &p_gravity);
	Vector2 get_gravity() const;
	void set_use_colliders(bool p_use_colliders);
	bool get_use_colliders() const;
	void set_collision_mask(int32_t p_collision_mask);
	int32_t get_collision_mask() const;
	void set_jiggle_joint_bone2d_node(int32_t p_joint_idx, const NodePath &p_bone2d_node);
	NodePath get_jiggle_joint_bone2d_node(int32_t p_joint_idx) const;
	void set_jiggle_joint_bone_index(int32_t p_joint_idx, int32_t p_bone_idx);
	int32_t get_jiggle_joint_bone_index(int32_t p_joint_idx) const;
	void set_jiggle_joint_override(int32_t p_joint_idx, bool p_override);
	bool get_jiggle_joint_override(int32_t p_joint_idx) const;
	void set_jiggle_joint_stiffness(int32_t p_joint_idx, float p_stiffness);
	float get_jiggle_joint_stiffness(int32_t p_joint_idx) const;
	void set_jiggle_joint_mass(int32_t p_joint_idx, float p_mass);
	float get_jiggle_joint_mass(int32_t p_joint_idx) const;
	void set_jiggle_joint_damping(int32_t p_joint_idx, float p_damping);
	float get_jiggle_joint_damping(int32_t p_joint_idx) const;
	void set_jiggle_joint_use_gravity(int32_t p_joint_idx, bool p_use_gravity);
	bool get_jiggle_joint_use_gravity(int32_t p_joint_idx) const;
	void set_jiggle_joint_gravity(int32_t p_joint_idx, const Vector2 &p_gravity);
	Vector2 get_jiggle_joint_gravity(int32_t p_joint_idx) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SkeletonModification2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

