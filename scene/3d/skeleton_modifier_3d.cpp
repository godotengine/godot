/**************************************************************************/
/*  skeleton_modifier_3d.cpp                                              */
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

#include "skeleton_modifier_3d.h"

PackedStringArray SkeletonModifier3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();
	if (skeleton_id.is_null()) {
		warnings.push_back(RTR("Skeleton3D node not set! SkeletonModifier3D must be child of Skeleton3D."));
	}
	return warnings;
}

/* Skeleton3D */

Skeleton3D *SkeletonModifier3D::get_skeleton() const {
	return ObjectDB::get_instance<Skeleton3D>(skeleton_id);
}

void SkeletonModifier3D::_update_skeleton_path() {
	skeleton_id = ObjectID();

	// Make sure parent is a Skeleton3D.
	Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
	if (sk) {
		skeleton_id = sk->get_instance_id();
	}
}

void SkeletonModifier3D::_update_skeleton() {
	if (!is_inside_tree()) {
		return;
	}
	Skeleton3D *old_sk = get_skeleton();
	_update_skeleton_path();
	Skeleton3D *new_sk = get_skeleton();
	if (old_sk != new_sk) {
		_skeleton_changed(old_sk, new_sk);
	}
	if (new_sk) {
		_validate_bone_names();
	}
	update_configuration_warnings();
}

void SkeletonModifier3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	GDVIRTUAL_CALL(_skeleton_changed, p_old, p_new);
}

void SkeletonModifier3D::_validate_bone_names() {
	GDVIRTUAL_CALL(_validate_bone_names);
}

void SkeletonModifier3D::_force_update_skeleton_skin() {
	if (!is_inside_tree()) {
		return;
	}
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	skeleton->force_update_deferred();
}

/* Process */

void SkeletonModifier3D::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}
	active = p_active;
	_set_active(active);
	_force_update_skeleton_skin();
}

bool SkeletonModifier3D::is_active() const {
	return active;
}

void SkeletonModifier3D::_set_active(bool p_active) {
	//
}

void SkeletonModifier3D::set_influence(real_t p_influence) {
	influence = p_influence;
}

real_t SkeletonModifier3D::get_influence() const {
	return influence;
}

void SkeletonModifier3D::process_modification(double p_delta) {
	if (!is_inside_tree() || !active) {
		return;
	}
	_process_modification(p_delta);
	emit_signal(SNAME("modification_processed"));
}

void SkeletonModifier3D::_process_modification(double p_delta) {
	if (GDVIRTUAL_CALL(_process_modification_with_delta, p_delta)) {
		return;
	}
#ifndef DISABLE_DEPRECATED
	if (GDVIRTUAL_CALL(_process_modification)) {
		return;
	}
#endif // DISABLE_DEPRECATED
}

void SkeletonModifier3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_PARENTED: {
			_update_skeleton();
		} break;
		case NOTIFICATION_EXIT_TREE:
		case NOTIFICATION_UNPARENTED: {
			_force_update_skeleton_skin();
		} break;
	}
}

void SkeletonModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_skeleton"), &SkeletonModifier3D::get_skeleton);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &SkeletonModifier3D::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &SkeletonModifier3D::is_active);

	ClassDB::bind_method(D_METHOD("set_influence", "influence"), &SkeletonModifier3D::set_influence);
	ClassDB::bind_method(D_METHOD("get_influence"), &SkeletonModifier3D::get_influence);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "influence", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_influence", "get_influence");

	ADD_SIGNAL(MethodInfo("modification_processed"));
	GDVIRTUAL_BIND(_process_modification_with_delta, "delta");
#ifndef DISABLE_DEPRECATED
	GDVIRTUAL_BIND(_process_modification);
#endif

	GDVIRTUAL_BIND(_skeleton_changed, "old_skeleton", "new_skeleton");
	GDVIRTUAL_BIND(_validate_bone_names);

	BIND_ENUM_CONSTANT(BONE_AXIS_PLUS_X);
	BIND_ENUM_CONSTANT(BONE_AXIS_MINUS_X);
	BIND_ENUM_CONSTANT(BONE_AXIS_PLUS_Y);
	BIND_ENUM_CONSTANT(BONE_AXIS_MINUS_Y);
	BIND_ENUM_CONSTANT(BONE_AXIS_PLUS_Z);
	BIND_ENUM_CONSTANT(BONE_AXIS_MINUS_Z);

	BIND_ENUM_CONSTANT(BONE_DIRECTION_PLUS_X);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_MINUS_X);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_PLUS_Y);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_MINUS_Y);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_PLUS_Z);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_MINUS_Z);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_FROM_PARENT);

	BIND_ENUM_CONSTANT(SECONDARY_DIRECTION_NONE);
	BIND_ENUM_CONSTANT(SECONDARY_DIRECTION_PLUS_X);
	BIND_ENUM_CONSTANT(SECONDARY_DIRECTION_MINUS_X);
	BIND_ENUM_CONSTANT(SECONDARY_DIRECTION_PLUS_Y);
	BIND_ENUM_CONSTANT(SECONDARY_DIRECTION_MINUS_Y);
	BIND_ENUM_CONSTANT(SECONDARY_DIRECTION_PLUS_Z);
	BIND_ENUM_CONSTANT(SECONDARY_DIRECTION_MINUS_Z);
	BIND_ENUM_CONSTANT(SECONDARY_DIRECTION_CUSTOM);

	BIND_ENUM_CONSTANT(ROTATION_AXIS_X);
	BIND_ENUM_CONSTANT(ROTATION_AXIS_Y);
	BIND_ENUM_CONSTANT(ROTATION_AXIS_Z);
	BIND_ENUM_CONSTANT(ROTATION_AXIS_ALL);
	BIND_ENUM_CONSTANT(ROTATION_AXIS_CUSTOM);
}

Vector3 SkeletonModifier3D::get_vector_from_bone_axis(BoneAxis p_axis) {
	Vector3 ret;
	switch (p_axis) {
		case BONE_AXIS_PLUS_X: {
			ret = Vector3(1, 0, 0);
		} break;
		case BONE_AXIS_MINUS_X: {
			ret = Vector3(-1, 0, 0);
		} break;
		case BONE_AXIS_PLUS_Y: {
			ret = Vector3(0, 1, 0);
		} break;
		case BONE_AXIS_MINUS_Y: {
			ret = Vector3(0, -1, 0);
		} break;
		case BONE_AXIS_PLUS_Z: {
			ret = Vector3(0, 0, 1);
		} break;
		case BONE_AXIS_MINUS_Z: {
			ret = Vector3(0, 0, -1);
		} break;
	}
	return ret;
}

Vector3 SkeletonModifier3D::get_vector_from_axis(Vector3::Axis p_axis) {
	Vector3 ret;
	switch (p_axis) {
		case Vector3::AXIS_X: {
			ret = Vector3(1, 0, 0);
		} break;
		case Vector3::AXIS_Y: {
			ret = Vector3(0, 1, 0);
		} break;
		case Vector3::AXIS_Z: {
			ret = Vector3(0, 0, 1);
		} break;
	}
	return ret;
}

Vector3::Axis SkeletonModifier3D::get_axis_from_bone_axis(BoneAxis p_axis) {
	Vector3::Axis ret = Vector3::AXIS_X;
	switch (p_axis) {
		case BONE_AXIS_PLUS_X:
		case BONE_AXIS_MINUS_X: {
			ret = Vector3::AXIS_X;
		} break;
		case BONE_AXIS_PLUS_Y:
		case BONE_AXIS_MINUS_Y: {
			ret = Vector3::AXIS_Y;
		} break;
		case BONE_AXIS_PLUS_Z:
		case BONE_AXIS_MINUS_Z: {
			ret = Vector3::AXIS_Z;
		} break;
	}
	return ret;
}

Vector3 SkeletonModifier3D::limit_length(const Vector3 &p_origin, const Vector3 &p_destination, float p_length) {
	return p_origin + (p_destination - p_origin).normalized() * p_length;
}

Quaternion SkeletonModifier3D::get_local_pose_rotation(Skeleton3D *p_skeleton, int p_bone, const Quaternion &p_global_pose_rotation) {
	int parent = p_skeleton->get_bone_parent(p_bone);
	if (parent < 0) {
		return p_global_pose_rotation;
	}
	return (p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion().inverse() * p_global_pose_rotation).normalized();
}

Quaternion SkeletonModifier3D::get_from_to_rotation(const Vector3 &p_from, const Vector3 &p_to, const Quaternion &p_prev_rot) {
	if (Math::is_equal_approx((float)p_from.dot(p_to), -1.0f)) {
		return p_prev_rot; // For preventing to glitch, checking dot for detecting flip is more accurate than checking cross.
	}
	Vector3 axis = p_from.cross(p_to);
	if (axis.is_zero_approx()) {
		return p_prev_rot;
	}
	float angle = p_from.angle_to(p_to);
	if (Math::is_zero_approx(angle)) {
		angle = 0.0;
	}
	return Quaternion(axis.normalized(), angle);
}

Quaternion SkeletonModifier3D::get_from_to_rotation_by_axis(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_axis) {
	const double ALMOST_ONE = 1.0 - CMP_EPSILON;
	double dot = p_from.dot(p_to);
	if (dot > ALMOST_ONE) {
		return Quaternion();
	}
	if (dot < -ALMOST_ONE) {
		return Quaternion(p_axis, Math::PI);
	}
	double angle = p_from.angle_to(p_to);
	Vector3 cross = p_from.cross(p_to);
	if (std::signbit(cross.dot(p_axis))) {
		angle = -angle;
	}
	return Quaternion(p_axis, angle);
}

Quaternion SkeletonModifier3D::get_swing(const Quaternion &p_rotation, const Vector3 &p_axis) {
	if (p_axis.is_zero_approx()) {
		return p_rotation;
	}
	Quaternion rot = p_rotation;
	if (!rot.is_normalized()) {
		rot.normalize();
	}
	Vector3 axis = p_axis.normalized();
	const Vector3 v(rot.x, rot.y, rot.z);
	const real_t proj_len = v.dot(axis);
	const Vector3 twist_vec = axis * proj_len;
	Quaternion twist(twist_vec.x, twist_vec.y, twist_vec.z, rot.w);
	if (!twist.is_normalized()) {
		if (Math::is_zero_approx(twist.length_squared())) {
			return rot;
		}
		twist.normalize();
	}
	Quaternion swing = rot * twist.inverse();
	swing.normalize();
	return swing;
}

Vector3 SkeletonModifier3D::snap_vector_to_plane(const Vector3 &p_plane_normal, const Vector3 &p_vector) {
	if (Math::is_zero_approx(p_plane_normal.length_squared())) {
		return p_vector;
	}
	double length = p_vector.length();
	Vector3 normalized_vec = p_vector.normalized();
	Vector3 normal = p_plane_normal.normalized();
	return normalized_vec.slide(normal) * length;
}

double SkeletonModifier3D::symmetrize_angle(double p_angle) {
	double angle = Math::fposmod(p_angle, Math::TAU);
	return angle > Math::PI ? angle - Math::TAU : angle;
}

double SkeletonModifier3D::get_roll_angle(const Quaternion &p_rotation, const Vector3 &p_roll_axis) {
	// Ensure roll axis is normalized.
	Vector3 roll_axis = p_roll_axis.normalized();

	// Project the quaternion rotation onto the roll axis.
	// This gives us the component of rotation around that axis.
	double dot = p_rotation.x * roll_axis.x +
			p_rotation.y * roll_axis.y +
			p_rotation.z * roll_axis.z;

	// Create a quaternion representing just the roll component.
	Quaternion roll_component;
	roll_component.x = roll_axis.x * dot;
	roll_component.y = roll_axis.y * dot;
	roll_component.z = roll_axis.z * dot;
	roll_component.w = p_rotation.w;

	// Normalize this component.
	double length = roll_component.length();
	if (length > CMP_EPSILON) {
		roll_component = roll_component / length;
	} else {
		return 0.0;
	}

	// Extract the angle.
	double angle = 2.0 * Math::acos(CLAMP(roll_component.w, -1.0, 1.0));

	// Determine the sign.
	double direction = (roll_component.x * roll_axis.x + roll_component.y * roll_axis.y + roll_component.z * roll_axis.z > 0) ? 1.0 : -1.0;

	return symmetrize_angle(angle * direction);
}

Vector3 SkeletonModifier3D::get_projected_normal(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_point) {
	// Get nearest normal vector to p_point from the infinite line p_a to p_b.
	const Vector3 dir = p_b - p_a;
	const real_t denom = dir.length_squared();
	if (Math::is_zero_approx(denom)) {
		return Vector3();
	}
	const Vector3 w = p_point - p_a;
	const real_t t = w.dot(dir) / denom;
	const Vector3 h = p_a + dir * t;
	return (p_point - h).normalized();
}

SkeletonModifier3D::SkeletonModifier3D() {
}
