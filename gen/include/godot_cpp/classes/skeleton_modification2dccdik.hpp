/**************************************************************************/
/*  skeleton_modification2dccdik.hpp                                      */
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

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class SkeletonModification2DCCDIK : public SkeletonModification2D {
	GDEXTENSION_CLASS(SkeletonModification2DCCDIK, SkeletonModification2D)

public:
	void set_target_node(const NodePath &p_target_nodepath);
	NodePath get_target_node() const;
	void set_tip_node(const NodePath &p_tip_nodepath);
	NodePath get_tip_node() const;
	void set_ccdik_data_chain_length(int32_t p_length);
	int32_t get_ccdik_data_chain_length();
	void set_ccdik_joint_bone2d_node(int32_t p_joint_idx, const NodePath &p_bone2d_nodepath);
	NodePath get_ccdik_joint_bone2d_node(int32_t p_joint_idx) const;
	void set_ccdik_joint_bone_index(int32_t p_joint_idx, int32_t p_bone_idx);
	int32_t get_ccdik_joint_bone_index(int32_t p_joint_idx) const;
	void set_ccdik_joint_rotate_from_joint(int32_t p_joint_idx, bool p_rotate_from_joint);
	bool get_ccdik_joint_rotate_from_joint(int32_t p_joint_idx) const;
	void set_ccdik_joint_enable_constraint(int32_t p_joint_idx, bool p_enable_constraint);
	bool get_ccdik_joint_enable_constraint(int32_t p_joint_idx) const;
	void set_ccdik_joint_constraint_angle_min(int32_t p_joint_idx, float p_angle_min);
	float get_ccdik_joint_constraint_angle_min(int32_t p_joint_idx) const;
	void set_ccdik_joint_constraint_angle_max(int32_t p_joint_idx, float p_angle_max);
	float get_ccdik_joint_constraint_angle_max(int32_t p_joint_idx) const;
	void set_ccdik_joint_constraint_angle_invert(int32_t p_joint_idx, bool p_invert);
	bool get_ccdik_joint_constraint_angle_invert(int32_t p_joint_idx) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SkeletonModification2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

