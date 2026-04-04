/**************************************************************************/
/*  look_at_modifier3d.hpp                                                */
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

#include <godot_cpp/classes/skeleton_modifier3d.hpp>
#include <godot_cpp/classes/tween.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class LookAtModifier3D : public SkeletonModifier3D {
	GDEXTENSION_CLASS(LookAtModifier3D, SkeletonModifier3D)

public:
	enum OriginFrom {
		ORIGIN_FROM_SELF = 0,
		ORIGIN_FROM_SPECIFIC_BONE = 1,
		ORIGIN_FROM_EXTERNAL_NODE = 2,
	};

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;
	void set_bone_name(const String &p_bone_name);
	String get_bone_name() const;
	void set_bone(int32_t p_bone);
	int32_t get_bone() const;
	void set_forward_axis(SkeletonModifier3D::BoneAxis p_forward_axis);
	SkeletonModifier3D::BoneAxis get_forward_axis() const;
	void set_primary_rotation_axis(Vector3::Axis p_axis);
	Vector3::Axis get_primary_rotation_axis() const;
	void set_use_secondary_rotation(bool p_enabled);
	bool is_using_secondary_rotation() const;
	void set_relative(bool p_enabled);
	bool is_relative() const;
	void set_origin_safe_margin(float p_margin);
	float get_origin_safe_margin() const;
	void set_origin_from(LookAtModifier3D::OriginFrom p_origin_from);
	LookAtModifier3D::OriginFrom get_origin_from() const;
	void set_origin_bone_name(const String &p_bone_name);
	String get_origin_bone_name() const;
	void set_origin_bone(int32_t p_bone);
	int32_t get_origin_bone() const;
	void set_origin_external_node(const NodePath &p_external_node);
	NodePath get_origin_external_node() const;
	void set_origin_offset(const Vector3 &p_offset);
	Vector3 get_origin_offset() const;
	void set_duration(float p_duration);
	float get_duration() const;
	void set_transition_type(Tween::TransitionType p_transition_type);
	Tween::TransitionType get_transition_type() const;
	void set_ease_type(Tween::EaseType p_ease_type);
	Tween::EaseType get_ease_type() const;
	void set_use_angle_limitation(bool p_enabled);
	bool is_using_angle_limitation() const;
	void set_symmetry_limitation(bool p_enabled);
	bool is_limitation_symmetry() const;
	void set_primary_limit_angle(float p_angle);
	float get_primary_limit_angle() const;
	void set_primary_damp_threshold(float p_power);
	float get_primary_damp_threshold() const;
	void set_primary_positive_limit_angle(float p_angle);
	float get_primary_positive_limit_angle() const;
	void set_primary_positive_damp_threshold(float p_power);
	float get_primary_positive_damp_threshold() const;
	void set_primary_negative_limit_angle(float p_angle);
	float get_primary_negative_limit_angle() const;
	void set_primary_negative_damp_threshold(float p_power);
	float get_primary_negative_damp_threshold() const;
	void set_secondary_limit_angle(float p_angle);
	float get_secondary_limit_angle() const;
	void set_secondary_damp_threshold(float p_power);
	float get_secondary_damp_threshold() const;
	void set_secondary_positive_limit_angle(float p_angle);
	float get_secondary_positive_limit_angle() const;
	void set_secondary_positive_damp_threshold(float p_power);
	float get_secondary_positive_damp_threshold() const;
	void set_secondary_negative_limit_angle(float p_angle);
	float get_secondary_negative_limit_angle() const;
	void set_secondary_negative_damp_threshold(float p_power);
	float get_secondary_negative_damp_threshold() const;
	float get_interpolation_remaining() const;
	bool is_interpolating() const;
	bool is_target_within_limitation() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SkeletonModifier3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(LookAtModifier3D::OriginFrom);

