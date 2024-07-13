/**************************************************************************/
/*  renik_chain.cpp                                                       */
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

#include "renik_chain.h"

void RenIKChain::init(Vector3 p_chain_curve_direction, float p_root_influence,
		float p_leaf_influence, float p_twist_influence,
		float p_twist_start) {
	chain_curve_direction = p_chain_curve_direction;
	root_influence = p_root_influence;
	leaf_influence = p_leaf_influence;
	twist_influence = p_twist_influence;
	twist_start = p_twist_start;
}

void RenIKChain::init_chain(Skeleton3D *p_skeleton) {
	joints.clear();
	total_length = 0;
	if (p_skeleton && root_bone >= 0 && leaf_bone >= 0 &&
			root_bone < p_skeleton->get_bone_count() &&
			leaf_bone < p_skeleton->get_bone_count()) {
		BoneId bone = p_skeleton->get_bone_parent(leaf_bone);
		// generate the chain of bones
		Vector<BoneId> chain;
		float last_length = 0.0f;
		rest_leaf = p_skeleton->get_bone_rest(leaf_bone);
		while (bone != root_bone) {
			Transform3D rest_pose = p_skeleton->get_bone_rest(bone);
			rest_leaf = rest_pose * rest_leaf.orthonormalized();
			last_length = rest_pose.origin.length();
			total_length += last_length;
			if (bone < 0) { // invalid chain
				total_length = 0;
				first_bone = -1;
				rest_leaf = Transform3D();
				return;
			}
			chain.push_back(bone);
			first_bone = bone;
			bone = p_skeleton->get_bone_parent(bone);
		}
		total_length -= last_length;
		total_length += p_skeleton->get_bone_rest(leaf_bone).origin.length();

		if (total_length <= 0) { // invalid chain
			total_length = 0;
			first_bone = -1;
			rest_leaf = Transform3D();
			return;
		}

		Basis totalRotation;
		float progress = 0;
		// flip the order and figure out the relative distances of these joints
		for (int i = chain.size() - 1; i >= 0; i--) {
			RenIKChain::Joint j;
			j.id = chain[i];
			Transform3D boneTransform = p_skeleton->get_bone_rest(j.id);
			j.rotation = boneTransform.basis.get_rotation_quaternion();
			j.relative_prev = totalRotation.xform_inv(boneTransform.origin);
			j.prev_distance = j.relative_prev.length();

			// calc influences
			progress += j.prev_distance;
			float percentage = (progress / total_length);
			float effectiveRootInfluence =
					root_influence <= 0 || percentage >= root_influence
					? 0
					: (percentage - root_influence) / -root_influence;
			float effectiveLeafInfluence =
					leaf_influence <= 0 || percentage <= 1 - leaf_influence
					? 0
					: (percentage - (1 - leaf_influence)) / leaf_influence;
			float effectiveTwistInfluence =
					twist_start >= 1 || twist_influence <= 0 || percentage <= twist_start
					? 0
					: (percentage - twist_start) *
							(twist_influence / (1 - twist_start));
			j.root_influence =
					effectiveRootInfluence > 1 ? 1 : effectiveRootInfluence;
			j.leaf_influence =
					effectiveLeafInfluence > 1 ? 1 : effectiveLeafInfluence;
			j.twist_influence =
					effectiveTwistInfluence > 1 ? 1 : effectiveTwistInfluence;

			if (!joints.is_empty()) {
				RenIKChain::Joint oldJ = joints[joints.size() - 1];
				oldJ.relative_next = -j.relative_prev;
				oldJ.next_distance = j.prev_distance;
				joints.set(joints.size() - 1, oldJ);
			}
			joints.push_back(j);
			totalRotation = (totalRotation * boneTransform.basis).orthonormalized();
		}
		if (!joints.is_empty()) {
			RenIKChain::Joint oldJ = joints[joints.size() - 1];
			oldJ.relative_next = -p_skeleton->get_bone_rest(leaf_bone).origin;
			oldJ.next_distance = oldJ.relative_next.length();
			joints.set(joints.size() - 1, oldJ);
		}
	}
}

void RenIKChain::set_root_bone(Skeleton3D *skeleton, BoneId p_root_bone) {
	root_bone = p_root_bone;
	init_chain(skeleton);
}
void RenIKChain::set_leaf_bone(Skeleton3D *skeleton, BoneId p_leaf_bone) {
	leaf_bone = p_leaf_bone;
	init_chain(skeleton);
}

bool RenIKChain::is_valid() { return !joints.is_empty(); }

float RenIKChain::get_total_length() { return total_length; }

Vector<RenIKChain::Joint> RenIKChain::get_joints() { return joints; }

Transform3D RenIKChain::get_relative_rest_leaf() { return rest_leaf; }

BoneId RenIKChain::get_first_bone() { return first_bone; }

BoneId RenIKChain::get_root_bone() { return root_bone; }

BoneId RenIKChain::get_leaf_bone() { return leaf_bone; }

float RenIKChain::get_root_stiffness() { return root_influence; }

void RenIKChain::set_root_stiffness(Skeleton3D *p_skeleton, float p_stiffness) {
	root_influence = p_stiffness;
	init_chain(p_skeleton);
}

float RenIKChain::get_leaf_stiffness() { return leaf_influence; }

void RenIKChain::set_leaf_stiffness(Skeleton3D *p_skeleton, float p_stiffness) {
	leaf_influence = p_stiffness;
	init_chain(p_skeleton);
}

float RenIKChain::get_twist() { return twist_influence; }

void RenIKChain::set_twist(Skeleton3D *p_skeleton, float p_twist) {
	twist_influence = p_twist;
	init_chain(p_skeleton);
}

float RenIKChain::get_twist_start() { return twist_start; }

void RenIKChain::set_twist_start(Skeleton3D *p_skeleton, float p_twist_start) {
	twist_start = p_twist_start;
	init_chain(p_skeleton);
}

bool RenIKChain::contains_bone(Skeleton3D *p_skeleton, BoneId p_bone) {
	if (p_skeleton) {
		BoneId spineBone = leaf_bone;
		while (spineBone >= 0) {
			if (spineBone == p_bone) {
				return true;
			}
			spineBone = p_skeleton->get_bone_parent(spineBone);
		}
	}
	return false;
}
