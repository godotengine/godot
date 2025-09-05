/**************************************************************************/
/*  skeleton_modifier_3d.h                                                */
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

#pragma once

#include "scene/3d/node_3d.h"

#include "scene/3d/skeleton_3d.h"

class SkeletonModifier3D : public Node3D {
	GDCLASS(SkeletonModifier3D, Node3D);

	void rebind();

public:
	// For the case to indicate bone axis on basis without custom vector.
	enum BoneAxis {
		BONE_AXIS_PLUS_X,
		BONE_AXIS_MINUS_X,
		BONE_AXIS_PLUS_Y,
		BONE_AXIS_MINUS_Y,
		BONE_AXIS_PLUS_Z,
		BONE_AXIS_MINUS_Z,
	};
	static String get_hint_bone_axis() { return "+X,-X,+Y,-Y,+Z,-Z"; }

	// For the case to indicate Head-Tail of the bone.
	enum BoneDirection {
		BONE_DIRECTION_PLUS_X,
		BONE_DIRECTION_MINUS_X,
		BONE_DIRECTION_PLUS_Y,
		BONE_DIRECTION_MINUS_Y,
		BONE_DIRECTION_PLUS_Z,
		BONE_DIRECTION_MINUS_Z,
		BONE_DIRECTION_FROM_PARENT,
	};
	static String get_hint_bone_direction() { return "+X,-X,+Y,-Y,+Z,-Z,FromParent"; }

	// For the case to define secondary axis of the bone local space.
	enum SecondaryDirection {
		SECONDARY_DIRECTION_NONE,
		SECONDARY_DIRECTION_PLUS_X,
		SECONDARY_DIRECTION_MINUS_X,
		SECONDARY_DIRECTION_PLUS_Y,
		SECONDARY_DIRECTION_MINUS_Y,
		SECONDARY_DIRECTION_PLUS_Z,
		SECONDARY_DIRECTION_MINUS_Z,
		SECONDARY_DIRECTION_CUSTOM,
	};
	static String get_hint_secondary_direction() { return "None,+X,-X,+Y,-Y,+Z,-Z,Custom"; }

	// For the case to define rotation direction without identification plus/minus.
	enum RotationAxis {
		ROTATION_AXIS_X,
		ROTATION_AXIS_Y,
		ROTATION_AXIS_Z,
		ROTATION_AXIS_ALL,
		ROTATION_AXIS_CUSTOM,
	};
	static String get_hint_rotation_axis() { return "X,Y,Z,All,Custom"; }

protected:
	bool active = true;
	real_t influence = 1.0;

	// Cache them for the performance reason since finding node with NodePath is slow.
	ObjectID skeleton_id;

	void _update_skeleton();
	void _update_skeleton_path();
	void _force_update_skeleton_skin();

	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new);
	virtual void _validate_bone_names();
	GDVIRTUAL2(_skeleton_changed, Skeleton3D *, Skeleton3D *);
	GDVIRTUAL0(_validate_bone_names);

	void _notification(int p_what);
	static void _bind_methods();

	virtual void _set_active(bool p_active);

	virtual void _process_modification(double p_delta);
	// TODO: In Godot 5, should obsolete old GDVIRTUAL0(_process_modification); and replace it with _process_modification_with_delta as GDVIRTUAL1(_process_modification, double).
	GDVIRTUAL1(_process_modification_with_delta, double);
#ifndef DISABLE_DEPRECATED
	GDVIRTUAL0(_process_modification);
#endif

public:
	virtual PackedStringArray get_configuration_warnings() const override;
	virtual bool has_process() const { return false; } // Return true if modifier needs to modify bone pose without external animation such as physics, jiggle and etc.

	void set_active(bool p_active);
	bool is_active() const;

	void set_influence(real_t p_influence);
	real_t get_influence() const;

	Skeleton3D *get_skeleton() const;

	void process_modification(double p_delta);

	// Utility APIs.
	static Vector3 get_vector_from_bone_axis(BoneAxis p_axis);
	static Vector3 get_vector_from_axis(Vector3::Axis p_axis);
	static Vector3::Axis get_axis_from_bone_axis(BoneAxis p_axis);

	// 3D math.
	static Vector3 limit_length(const Vector3 &p_origin, const Vector3 &p_destination, float p_length);
	static Quaternion get_local_pose_rotation(Skeleton3D *p_skeleton, int p_bone, const Quaternion &p_global_pose_rotation);
	static Quaternion get_from_to_rotation(const Vector3 &p_from, const Vector3 &p_to, const Quaternion &p_prev_rot);
	static Quaternion get_from_to_rotation_by_axis(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_axis);
	static Quaternion get_swing(const Quaternion &p_rotation, const Vector3 &p_axis);
	static Vector3 snap_vector_to_plane(const Vector3 &p_plane_normal, const Vector3 &p_vector);
	static double symmetrize_angle(double p_angle);
	static double get_roll_angle(const Quaternion &p_rotation, const Vector3 &p_roll_axis);
	static Vector3 get_projected_normal(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_point);

#ifdef TOOLS_ENABLED
	virtual bool is_processed_on_saving() const { return false; }
#endif

	SkeletonModifier3D();
};

VARIANT_ENUM_CAST(SkeletonModifier3D::BoneAxis);
VARIANT_ENUM_CAST(SkeletonModifier3D::BoneDirection);
VARIANT_ENUM_CAST(SkeletonModifier3D::SecondaryDirection);
VARIANT_ENUM_CAST(SkeletonModifier3D::RotationAxis);
