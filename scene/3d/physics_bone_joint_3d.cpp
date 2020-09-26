/*************************************************************************/
/*  physics_bone_joint_3d.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/3d/physics_bone_joint_3d.h"

//// Shared by all derived
_FORCE_INLINE_ void compute_body_offsets_impl(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) {
	const PhysicsBone3D *bone_a = Object::cast_to<PhysicsBone3D>(&p_body_a);

	if (!bone_a) {
		ERR_FAIL_MSG("Incorrect Type");
		return;
	}

	const PhysicsBone3D *bone_b = nullptr;
	if (p_body_b) {
		bone_b = Object::cast_to<PhysicsBone3D>(p_body_b);
		if (!bone_b) {
			ERR_FAIL_MSG("Incorrect Type");
			return;
		}
	}

	const Skeleton3D *skeleton = bone_a->get_skeleton();
	// Not *technically* required for boneA, but we really should check...
	if (!skeleton) {
		ERR_FAIL_MSG("No Skeleton found");
	}

	const BoneId bone_a_id = bone_a->get_bone_id();
	ERR_FAIL_COND_MSG(bone_a_id == -1, "Invalid bone");

	if (!bone_b) {
		o_offset_a = Transform();
		o_offset_b = Transform();
		return;
	}

	ERR_FAIL_COND_MSG(bone_b->get_skeleton() != skeleton, "Skeletons are not the same");

	const BoneId bone_b_id = bone_b->get_bone_id();
	ERR_FAIL_COND_MSG(bone_b_id == -1, "Invalid bone");

	// the joint global position is at the oriign of bone_b in the skeleton
	const Transform joint_global_pos_a = skeleton->get_bone_global_rest(bone_a_id);
	const Transform joint_global_pos_b = skeleton->get_bone_global_rest(bone_b_id);

	Transform bone_a_body_rest = bone_a->get_p_body_global_rest();
	Transform bone_b_body_rest = bone_b->get_p_body_global_rest();

	// we now want to compute the joint local space binds at the point of bone_b's bone rest

	o_offset_a = (bone_a_body_rest.affine_inverse() * joint_global_pos_b);
	o_offset_a.basis = joint_global_pos_a.basis.orthonormalized();

	o_offset_b = (bone_b_body_rest.affine_inverse() * joint_global_pos_b);
	o_offset_b.basis = joint_global_pos_b.basis.orthonormalized().transposed();
}

/////////////////////////////////////////////////////////////////////

void BonePinJoint3D::compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) {
	compute_body_offsets_impl(o_offset_a, o_offset_b, p_body_a, p_body_b);
}

/////////////////////////////////////////////////////////////////////

void BoneHingeJoint3D::compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) {
	compute_body_offsets_impl(o_offset_a, o_offset_b, p_body_a, p_body_b);
}

/////////////////////////////////////////////////////////////////////

void BoneSliderJoint3D::compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) {
	compute_body_offsets_impl(o_offset_a, o_offset_b, p_body_a, p_body_b);
}

/////////////////////////////////////////////////////////////////////

void BoneConeTwistJoint3D::compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) {
	compute_body_offsets_impl(o_offset_a, o_offset_b, p_body_a, p_body_b);
}

/////////////////////////////////////////////////////////////////////

void BoneGeneric6DOFJoint3D::compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) {
	compute_body_offsets_impl(o_offset_a, o_offset_b, p_body_a, p_body_b);
}
