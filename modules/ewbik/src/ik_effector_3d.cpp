/*************************************************************************/
/*  ik_effector_3d.cpp                                                   */
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

#include "ik_effector_3d.h"
#include "ik_ewbik.h"
#include "math/ik_node_3d.h"

void IKEffector3D::set_target_node(Skeleton3D *p_skeleton, const NodePath &p_target_node_path) {
	ERR_FAIL_NULL(p_skeleton);
	target_node_path = p_target_node_path;
}

NodePath IKEffector3D::get_target_node() const {
	return target_node_path;
}

void IKEffector3D::set_target_node_rotation(bool p_use) {
	use_target_node_rotation = p_use;
}

bool IKEffector3D::get_target_node_rotation() const {
	return use_target_node_rotation;
}

Ref<IKBone3D> IKEffector3D::get_ik_node_3d_bone() const {
	return for_bone;
}

bool IKEffector3D::is_following_translation_only() const {
	return false;
}

void IKEffector3D::set_direction_priorities(Vector3 p_direction_priorities) { direction_priorities = p_direction_priorities; }

Vector3 IKEffector3D::get_direction_priorities() const {
	return direction_priorities;
}

void IKEffector3D::update_target_global_transform(Skeleton3D *p_skeleton, SkeletonModification3DNBoneIK *p_ewbik) {
	ERR_FAIL_NULL(p_skeleton);
	ERR_FAIL_NULL(for_bone);
	target_relative_to_skeleton_origin = p_skeleton->get_global_transform().affine_inverse() * for_bone->get_ik_transform()->get_global_transform();
	Node3D *current_target_node = cast_to<Node3D>(p_ewbik->get_node_or_null(target_node_path));
	if (!current_target_node) {
		return;
	}
	target_relative_to_skeleton_origin = p_skeleton->get_global_transform().affine_inverse() * current_target_node->get_global_transform();
}

Transform3D IKEffector3D::get_target_global_transform() const {
	return target_relative_to_skeleton_origin;
}

int32_t IKEffector3D::update_effector_target_headings(PackedVector3Array *p_headings, int32_t p_index, Ref<IKBone3D> p_for_bone, const Vector<real_t> *p_weights) const {
	ERR_FAIL_COND_V(p_index == -1, -1);
	ERR_FAIL_NULL_V(p_headings, -1);
	ERR_FAIL_NULL_V(p_for_bone, -1);

	int32_t index = p_index;
	Vector3 bone_origin_relative_to_skeleton_origin = p_for_bone->get_global_pose().origin;
	p_headings->write[index] = target_relative_to_skeleton_origin.origin - bone_origin_relative_to_skeleton_origin;
	index++;
	Vector3 priority = get_direction_priorities();
	if (priority.x > 0.0) {
		real_t w = (*p_weights)[index];
		w = MAX(w, 1.0f);
		p_headings->write[index] = (target_relative_to_skeleton_origin.basis.get_column(Vector3::AXIS_X) + target_relative_to_skeleton_origin.origin) - bone_origin_relative_to_skeleton_origin;
		p_headings->write[index] *= Vector3(w, w, w);
		index++;

		p_headings->write[index] = (target_relative_to_skeleton_origin.origin - target_relative_to_skeleton_origin.basis.get_column(Vector3::AXIS_X)) - bone_origin_relative_to_skeleton_origin;
		p_headings->write[index] *= Vector3(w, w, w);
		index++;
	}
	if (priority.y > 0.0) {
		real_t w = (*p_weights)[index];
		w = MAX(w, 1.0f);
		p_headings->write[index] = (target_relative_to_skeleton_origin.basis.get_column(Vector3::AXIS_Y) + target_relative_to_skeleton_origin.origin) - bone_origin_relative_to_skeleton_origin;
		p_headings->write[index] *= Vector3(w, w, w);
		index++;

		p_headings->write[index] = (target_relative_to_skeleton_origin.origin - target_relative_to_skeleton_origin.basis.get_column(Vector3::AXIS_Y)) - bone_origin_relative_to_skeleton_origin;
		p_headings->write[index] *= Vector3(w, w, w);
		index++;
	}
	if (priority.z > 0.0) {
		real_t w = (*p_weights)[index];
		w = MAX(w, 1.0f);
		p_headings->write[index] = (target_relative_to_skeleton_origin.basis.get_column(Vector3::AXIS_Z) + target_relative_to_skeleton_origin.origin) - bone_origin_relative_to_skeleton_origin;
		p_headings->write[index] *= Vector3(w, w, w);
		index++;

		p_headings->write[index] = (target_relative_to_skeleton_origin.origin - target_relative_to_skeleton_origin.basis.get_column(Vector3::AXIS_Z)) - bone_origin_relative_to_skeleton_origin;
		p_headings->write[index] *= Vector3(w, w, w);
		index++;
	}
	return index;
}

int32_t IKEffector3D::update_effector_tip_headings(PackedVector3Array *p_headings, int32_t p_index, Ref<IKBone3D> p_for_bone) const {
	ERR_FAIL_COND_V(p_index == -1, -1);
	ERR_FAIL_NULL_V(p_headings, -1);
	ERR_FAIL_NULL_V(for_bone, -1);
	ERR_FAIL_NULL_V(p_for_bone, -1);
	Transform3D tip_xform_relative_to_skeleton_origin = for_bone->get_global_pose();
	Basis tip_basis = tip_xform_relative_to_skeleton_origin.basis;
	Vector3 bone_origin_relative_to_skeleton_origin = p_for_bone->get_global_pose().origin;
	int32_t index = p_index;
	p_headings->write[index] = tip_xform_relative_to_skeleton_origin.origin - bone_origin_relative_to_skeleton_origin;
	index++;
	double distance = target_relative_to_skeleton_origin.origin.distance_to(bone_origin_relative_to_skeleton_origin);
	double scale_by = MAX(1.0, distance);
	const Vector3 priority = get_direction_priorities();
	if (priority.x > 0.0) {
		p_headings->write[index] = ((tip_basis.get_column(Vector3::AXIS_X) * scale_by) + tip_xform_relative_to_skeleton_origin.origin) - bone_origin_relative_to_skeleton_origin;
		index++;
		p_headings->write[index] = (tip_xform_relative_to_skeleton_origin.origin - (tip_basis.get_column(Vector3::AXIS_X) * scale_by)) - bone_origin_relative_to_skeleton_origin;
		index++;
	}
	if (priority.y > 0.0) {
		p_headings->write[index] = ((tip_basis.get_column(Vector3::AXIS_Y) * scale_by) + tip_xform_relative_to_skeleton_origin.origin) - bone_origin_relative_to_skeleton_origin;
		index++;
		p_headings->write[index] = (tip_xform_relative_to_skeleton_origin.origin - (tip_basis.get_column(Vector3::AXIS_Y) * scale_by)) - bone_origin_relative_to_skeleton_origin;
		index++;
	}
	if (priority.z > 0.0) {
		p_headings->write[index] = ((tip_basis.get_column(Vector3::AXIS_Z) * scale_by) + tip_xform_relative_to_skeleton_origin.origin) - bone_origin_relative_to_skeleton_origin;
		index++;
		p_headings->write[index] = (tip_xform_relative_to_skeleton_origin.origin - (tip_basis.get_column(Vector3::AXIS_Z) * scale_by)) - bone_origin_relative_to_skeleton_origin;
		index++;
	}
	return index;
}

void IKEffector3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "skeleton", "node"),
			&IKEffector3D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"),
			&IKEffector3D::get_target_node);
	ClassDB::bind_method(D_METHOD("set_depth_falloff", "amount"),
			&IKEffector3D::set_depth_falloff);
	ClassDB::bind_method(D_METHOD("get_depth_falloff"),
			&IKEffector3D::get_depth_falloff);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "depth_falloff"), "set_depth_falloff", "get_depth_falloff");
}

void IKEffector3D::set_weight(real_t p_weight) {
	weight = p_weight;
}

real_t IKEffector3D::get_weight() const {
	return weight;
}

IKEffector3D::IKEffector3D(const Ref<IKBone3D> &p_current_bone) {
	ERR_FAIL_NULL(p_current_bone);
	for_bone = p_current_bone;
}

void IKEffector3D::set_depth_falloff(float p_depth_falloff) {
	depth_falloff = CLAMP(p_depth_falloff, 0.0, 1.0);
}

float IKEffector3D::get_depth_falloff() const {
	return depth_falloff;
}
