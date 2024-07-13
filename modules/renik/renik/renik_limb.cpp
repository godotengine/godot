/**************************************************************************/
/*  renik_limb.cpp                                                        */
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

#include "renik_limb.h"

void RenIKLimb::init(float p_upper_twist_offset, float p_lower_twist_offset,
		float p_roll_offset, float p_upper_limb_twist,
		float p_lower_limb_twist,
		float p_twist_inflection_point_offset,
		float p_twist_overflow, float p_target_rotation_influence,
		Vector3 p_pole_offset,
		Vector3 p_target_position_influence) {
	upper_twist_offset = p_upper_twist_offset;
	lower_twist_offset = p_lower_twist_offset;
	roll_offset = p_roll_offset;
	upper_limb_twist = p_upper_limb_twist;
	lower_limb_twist = p_lower_limb_twist;
	twist_inflection_point_offset = p_twist_inflection_point_offset;
	twist_overflow = p_twist_overflow;
	target_rotation_influence = p_target_rotation_influence;
	pole_offset = p_pole_offset;
	target_position_influence = p_target_position_influence;
}

Transform3D RenIKLimb::get_extra_bones(Skeleton3D *p_skeleton,
		BoneId p_root_bone_id,
		BoneId p_tip_bone_id) {
	Transform3D cumulative_rest;
	BoneId current_bone_id = p_tip_bone_id;
	while (current_bone_id != -1 && current_bone_id != p_root_bone_id) {
		current_bone_id = p_skeleton->get_bone_parent(current_bone_id);
		if (current_bone_id == -1 || current_bone_id == p_root_bone_id) {
			break;
		}
		cumulative_rest =
				p_skeleton->get_bone_rest(current_bone_id) * cumulative_rest;
	}

	return cumulative_rest;
}
Vector<BoneId> RenIKLimb::get_extra_bone_ids(Skeleton3D *p_skeleton,
		BoneId p_root_bone_id,
		BoneId p_tip_bone_id) {
	Vector<BoneId> output;
	BoneId current_bone_id = p_tip_bone_id;
	while (current_bone_id != -1) {
		current_bone_id = p_skeleton->get_bone_parent(current_bone_id);
		if (current_bone_id == -1 || current_bone_id == p_root_bone_id) {
			break;
		}
		output.push_back(current_bone_id);
	}

	return output;
}

void RenIKLimb::update(Skeleton3D *skeleton) {
	if (skeleton != nullptr && leaf_id >= 0) {
		lower_id = lower_id >= 0 ? lower_id : skeleton->get_bone_parent(leaf_id);
		if (lower_id >= 0) {
			upper_id = upper_id >= 0 ? upper_id : skeleton->get_bone_parent(lower_id);
			if (upper_id >= 0) {
				// leaf = get_full_rest(skeleton, leaf_id, lower_id);
				// lower = get_full_rest(skeleton, lower_id, upper_id);
				// upper = skeleton->get_bone_rest(upper_id);

				lower_extra_bones = get_extra_bones(
						skeleton, lower_id,
						leaf_id); // lower bone + all bones after that except the leaf
				upper_extra_bones = get_extra_bones(
						skeleton, upper_id,
						lower_id); // upper bone + all bones between upper and lower
				lower_extra_bone_ids = get_extra_bone_ids(skeleton, lower_id, leaf_id);
				upper_extra_bone_ids = get_extra_bone_ids(skeleton, upper_id, lower_id);

				leaf = Transform3D(Basis(), skeleton->get_bone_rest(leaf_id).get_origin());
				lower = Transform3D(Basis(), skeleton->get_bone_rest(lower_id).get_origin());
				upper = Transform3D(Basis(), skeleton->get_bone_rest(upper_id).get_origin());
			}
		}
	}
}

void RenIKLimb::set_leaf(Skeleton3D *p_skeleton, BoneId p_leaf_id) {
	leaf_id = p_leaf_id;
	update(p_skeleton);
}

void RenIKLimb::set_upper(Skeleton3D *p_skeleton, BoneId p_upper_id) {
	upper_id = p_upper_id;
	update(p_skeleton);
}

void RenIKLimb::set_lower(Skeleton3D *p_skeleton, BoneId p_lower_id) {
	lower_id = p_lower_id;
	update(p_skeleton);
}

bool RenIKLimb::is_valid() {
	return upper_id >= 0 && lower_id >= 0 && leaf_id >= 0;
}

bool RenIKLimb::is_valid_in_skeleton(Skeleton3D *p_skeleton) {
	if (p_skeleton == nullptr || upper_id < 0 || lower_id < 0 || leaf_id < 0 ||
			upper_id >= p_skeleton->get_bone_count() ||
			lower_id >= p_skeleton->get_bone_count() ||
			leaf_id >= p_skeleton->get_bone_count()) {
		return false;
	}
	BoneId curr = p_skeleton->get_bone_parent(leaf_id);
	while (curr != -1 && curr != lower_id) {
		curr = p_skeleton->get_bone_parent(curr);
	}
	while (curr != -1 && curr != upper_id) {
		curr = p_skeleton->get_bone_parent(curr);
	}
	return curr != -1;
}

BoneId RenIKLimb::get_leaf_bone() { return leaf_id; }
BoneId RenIKLimb::get_lower_bone() { return lower_id; }
BoneId RenIKLimb::get_upper_bone() { return upper_id; }
Transform3D RenIKLimb::get_upper() { return upper; }
Transform3D RenIKLimb::get_lower() { return lower; }
Transform3D RenIKLimb::get_leaf() { return leaf; }
