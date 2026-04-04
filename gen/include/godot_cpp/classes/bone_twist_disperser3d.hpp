/**************************************************************************/
/*  bone_twist_disperser3d.hpp                                            */
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
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Curve;

class BoneTwistDisperser3D : public SkeletonModifier3D {
	GDEXTENSION_CLASS(BoneTwistDisperser3D, SkeletonModifier3D)

public:
	enum DisperseMode {
		DISPERSE_MODE_EVEN = 0,
		DISPERSE_MODE_WEIGHTED = 1,
		DISPERSE_MODE_CUSTOM = 2,
	};

	void set_setting_count(int32_t p_count);
	int32_t get_setting_count() const;
	void clear_settings();
	void set_mutable_bone_axes(bool p_enabled);
	bool are_bone_axes_mutable() const;
	void set_root_bone_name(int32_t p_index, const String &p_bone_name);
	String get_root_bone_name(int32_t p_index) const;
	void set_root_bone(int32_t p_index, int32_t p_bone);
	int32_t get_root_bone(int32_t p_index) const;
	void set_end_bone_name(int32_t p_index, const String &p_bone_name);
	String get_end_bone_name(int32_t p_index) const;
	void set_end_bone(int32_t p_index, int32_t p_bone);
	int32_t get_end_bone(int32_t p_index) const;
	String get_reference_bone_name(int32_t p_index) const;
	int32_t get_reference_bone(int32_t p_index) const;
	void set_extend_end_bone(int32_t p_index, bool p_enabled);
	bool is_end_bone_extended(int32_t p_index) const;
	void set_end_bone_direction(int32_t p_index, SkeletonModifier3D::BoneDirection p_bone_direction);
	SkeletonModifier3D::BoneDirection get_end_bone_direction(int32_t p_index) const;
	void set_twist_from_rest(int32_t p_index, bool p_enabled);
	bool is_twist_from_rest(int32_t p_index) const;
	void set_twist_from(int32_t p_index, const Quaternion &p_from);
	Quaternion get_twist_from(int32_t p_index) const;
	void set_disperse_mode(int32_t p_index, BoneTwistDisperser3D::DisperseMode p_disperse_mode);
	BoneTwistDisperser3D::DisperseMode get_disperse_mode(int32_t p_index) const;
	void set_weight_position(int32_t p_index, float p_weight_position);
	float get_weight_position(int32_t p_index) const;
	void set_damping_curve(int32_t p_index, const Ref<Curve> &p_curve);
	Ref<Curve> get_damping_curve(int32_t p_index) const;
	String get_joint_bone_name(int32_t p_index, int32_t p_joint) const;
	int32_t get_joint_bone(int32_t p_index, int32_t p_joint) const;
	float get_joint_twist_amount(int32_t p_index, int32_t p_joint) const;
	void set_joint_twist_amount(int32_t p_index, int32_t p_joint, float p_twist_amount);
	int32_t get_joint_count(int32_t p_index) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SkeletonModifier3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(BoneTwistDisperser3D::DisperseMode);

