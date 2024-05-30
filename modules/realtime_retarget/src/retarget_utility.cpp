/*************************************************************************/
/*  retarget_utility.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "retarget_utility.h"

Transform3D RetargetUtility::extract_global_transform(const Skeleton3D *p_skeleton, int p_bone_idx, Transform3D p_transform) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Transform3D());
	Transform3D tr = Transform3D();
	Transform3D s_grest = Transform3D();
	int s_parent = p_skeleton->get_bone_parent(p_bone_idx);
	if (s_parent >= 0) {
		s_grest = p_skeleton->get_bone_global_rest(s_parent);
	}
	tr.basis = s_grest.basis * p_transform.basis * p_skeleton->get_bone_rest(p_bone_idx).basis.inverse() * s_grest.basis.inverse();
	tr.origin = s_grest.basis.xform(p_transform.origin - p_skeleton->get_bone_rest(p_bone_idx).origin);
	return tr;
}

Vector3 RetargetUtility::extract_global_transform_position(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_position) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Vector3());
	Quaternion s_grest = Quaternion();
	int s_parent = p_skeleton->get_bone_parent(p_bone_idx);
	if (s_parent >= 0) {
		s_grest = p_skeleton->get_bone_global_rest(s_parent).basis.get_rotation_quaternion();
	}
	return s_grest.xform(p_position - p_skeleton->get_bone_rest(p_bone_idx).origin);
}

Quaternion RetargetUtility::extract_global_transform_rotation(const Skeleton3D *p_skeleton, int p_bone_idx, Quaternion p_rotation) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Quaternion());
	Quaternion s_grest = Quaternion();
	int s_parent = p_skeleton->get_bone_parent(p_bone_idx);
	if (s_parent >= 0) {
		s_grest = p_skeleton->get_bone_global_rest(s_parent).basis.get_rotation_quaternion();
	}
	return s_grest * p_rotation * p_skeleton->get_bone_rest(p_bone_idx).basis.get_rotation_quaternion().inverse() * s_grest.inverse();
}

Vector3 RetargetUtility::extract_global_transform_scale(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_scale) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Vector3(1, 1, 1));
	Basis s_grest = Basis();
	int s_parent = p_skeleton->get_bone_parent(p_bone_idx);
	if (s_parent >= 0) {
		s_grest = p_skeleton->get_bone_global_rest(s_parent).basis;
	}
	return (s_grest * Basis().scaled(p_scale) * p_skeleton->get_bone_rest(p_bone_idx).basis.inverse() * s_grest.inverse()).get_scale();
}

Transform3D RetargetUtility::global_transform_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Transform3D p_transform) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Transform3D());
	Transform3D tr = Transform3D();
	Transform3D t_grest = Transform3D();
	int t_parent = p_skeleton->get_bone_parent(p_bone_idx);
	if (t_parent >= 0) {
		t_grest = p_skeleton->get_bone_global_rest(t_parent);
	}
	tr.basis = t_grest.basis.inverse() * p_transform.basis * t_grest.basis * p_skeleton->get_bone_rest(p_bone_idx).basis;
	tr.origin = t_grest.basis.xform_inv(p_transform.origin) + p_skeleton->get_bone_rest(p_bone_idx).origin;
	return tr;
}

Vector3 RetargetUtility::global_transform_position_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_position) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Vector3());
	Quaternion t_grest = Quaternion();
	int t_parent = p_skeleton->get_bone_parent(p_bone_idx);
	if (t_parent >= 0) {
		t_grest = p_skeleton->get_bone_global_rest(t_parent).basis.get_rotation_quaternion();
	}
	return t_grest.xform_inv(p_position) + p_skeleton->get_bone_rest(p_bone_idx).origin;
}

Quaternion RetargetUtility::global_transform_rotation_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Quaternion p_rotation) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Quaternion());
	Quaternion t_grest = Quaternion();
	int t_parent = p_skeleton->get_bone_parent(p_bone_idx);
	if (t_parent >= 0) {
		t_grest = p_skeleton->get_bone_global_rest(t_parent).basis.get_rotation_quaternion();
	}
	return t_grest.inverse() * p_rotation * t_grest * p_skeleton->get_bone_rest(p_bone_idx).basis.get_rotation_quaternion();
}

Vector3 RetargetUtility::global_transform_scale_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_scale) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Vector3(1, 1, 1));
	Basis t_grest = Basis();
	int t_parent = p_skeleton->get_bone_parent(p_bone_idx);
	if (t_parent >= 0) {
		t_grest = p_skeleton->get_bone_global_rest(t_parent).basis;
	}
	return (t_grest.inverse() * Basis().scaled(p_scale) * t_grest * p_skeleton->get_bone_rest(p_bone_idx).basis).get_scale();
}

Transform3D RetargetUtility::extract_local_transform(const Skeleton3D *p_skeleton, int p_bone_idx, Transform3D p_transform) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Transform3D());
	return p_skeleton->get_bone_rest(p_bone_idx).affine_inverse() * p_transform;
}

Vector3 RetargetUtility::extract_local_transform_position(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_position) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Vector3());
	return p_position - p_skeleton->get_bone_rest(p_bone_idx).origin;
}

Quaternion RetargetUtility::extract_local_transform_rotation(const Skeleton3D *p_skeleton, int p_bone_idx, Quaternion p_rotation) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Quaternion());
	return p_skeleton->get_bone_rest(p_bone_idx).basis.get_rotation_quaternion().inverse() * p_rotation;
}

Vector3 RetargetUtility::extract_local_transform_scale(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_scale) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Vector3(1, 1, 1));
	return (p_skeleton->get_bone_rest(p_bone_idx).basis.inverse() * Basis().scaled(p_scale)).get_scale();
}

Transform3D RetargetUtility::local_transform_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Transform3D p_transform) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Transform3D());
	return p_skeleton->get_bone_rest(p_bone_idx) * p_transform;
}

Vector3 RetargetUtility::local_transform_position_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_position) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Vector3());
	return p_skeleton->get_bone_rest(p_bone_idx).origin + p_position;
}

Quaternion RetargetUtility::local_transform_rotation_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Quaternion p_rotation) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Quaternion());
	return p_skeleton->get_bone_rest(p_bone_idx).basis.get_rotation_quaternion() * p_rotation;
}

Vector3 RetargetUtility::local_transform_scale_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_scale) {
	ERR_FAIL_INDEX_V(p_bone_idx, p_skeleton->get_bone_count(), Vector3(1, 1, 1));
	return (p_skeleton->get_bone_rest(p_bone_idx).basis * Basis().scaled(p_scale)).get_scale();
}

void RetargetUtility::_bind_methods() {
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("extract_global_transform", "skeleton", "bone_idx", "transform"), &RetargetUtility::extract_global_transform);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("extract_global_transform_position", "skeleton", "bone_idx", "position"), &RetargetUtility::extract_global_transform_position);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("extract_global_transform_rotation", "skeleton", "bone_idx", "rotation"), &RetargetUtility::extract_global_transform_rotation);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("extract_global_transform_scale", "skeleton", "bone_idx", "scale"), &RetargetUtility::extract_global_transform_scale);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("global_transform_to_bone_pose", "skeleton", "bone_idx", "transform"), &RetargetUtility::global_transform_to_bone_pose);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("global_transform_position_to_bone_pose", "skeleton", "bone_idx", "position"), &RetargetUtility::global_transform_position_to_bone_pose);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("global_transform_rotation_to_bone_pose", "skeleton", "bone_idx", "rotation"), &RetargetUtility::global_transform_rotation_to_bone_pose);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("global_transform_scale_to_bone_pose", "skeleton", "bone_idx", "scale"), &RetargetUtility::global_transform_scale_to_bone_pose);

	ClassDB::bind_static_method("RetargetUtility", D_METHOD("extract_local_transform", "skeleton", "bone_idx", "transform"), &RetargetUtility::extract_local_transform);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("extract_local_transform_position", "skeleton", "bone_idx", "position"), &RetargetUtility::extract_local_transform_position);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("extract_local_transform_rotation", "skeleton", "bone_idx", "rotation"), &RetargetUtility::extract_local_transform_rotation);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("extract_local_transform_scale", "skeleton", "bone_idx", "scale"), &RetargetUtility::extract_local_transform_scale);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("local_transform_to_bone_pose", "skeleton", "bone_idx", "transform"), &RetargetUtility::local_transform_to_bone_pose);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("local_transform_position_to_bone_pose", "skeleton", "bone_idx", "position"), &RetargetUtility::local_transform_position_to_bone_pose);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("local_transform_rotation_to_bone_pose", "skeleton", "bone_idx", "rotation"), &RetargetUtility::local_transform_rotation_to_bone_pose);
	ClassDB::bind_static_method("RetargetUtility", D_METHOD("local_transform_scale_to_bone_pose", "skeleton", "bone_idx", "scale"), &RetargetUtility::local_transform_scale_to_bone_pose);

	BIND_ENUM_CONSTANT(TYPE_ABSOLUTE);
	BIND_ENUM_CONSTANT(TYPE_LOCAL);
	BIND_ENUM_CONSTANT(TYPE_GLOBAL);
}

RetargetUtility::RetargetUtility() {
}

RetargetUtility::~RetargetUtility() {
}
