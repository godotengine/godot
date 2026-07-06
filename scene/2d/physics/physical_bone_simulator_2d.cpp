/**************************************************************************/
/*  physical_bone_simulator_2d.cpp                                        */
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

#include "physical_bone_simulator_2d.h"

#include "core/object/class_db.h"
#include "scene/2d/physics/physical_bone_2d.h"
#include "scene/2d/skeleton_2d.h"

void PhysicalBoneSimulator2D::_collect_physical_bones(Node *p_node, LocalVector<PhysicalBone2D *> &r_physical_bones) const {
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		PhysicalBone2D *physical_bone = Object::cast_to<PhysicalBone2D>(child);
		if (physical_bone) {
			r_physical_bones.push_back(physical_bone);
		}
		_collect_physical_bones(child, r_physical_bones);
	}
}

PackedStringArray PhysicalBoneSimulator2D::get_configuration_warnings() const {
	PackedStringArray warnings = SkeletonModifier2D::get_configuration_warnings();
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return warnings;
	}
	LocalVector<PhysicalBone2D *> physical_bones;
	_collect_physical_bones(skeleton, physical_bones);
	if (physical_bones.is_empty()) {
		warnings.push_back(RTR("No PhysicalBone2D nodes were found under the parent Skeleton2D."));
	}
	return warnings;
}

void PhysicalBoneSimulator2D::_process_modification(double p_delta) {
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	LocalVector<PhysicalBone2D *> physical_bones;
	_collect_physical_bones(skeleton, physical_bones);
	if (physical_bones.is_empty()) {
		return;
	}
	const Transform2D skeleton_global_inverse = skeleton->get_global_transform().affine_inverse();
	const int bone_count = skeleton->get_bone_count();
	for (PhysicalBone2D *physical_bone : physical_bones) {
		if (!physical_bone || (apply_to_simulating_bones_only && !physical_bone->is_simulating_physics())) {
			continue;
		}
		const int bone_idx = physical_bone->get_bone2d_index();
		if (bone_idx < 0 || bone_idx >= bone_count) {
			continue;
		}
		skeleton->set_bone_global_pose(bone_idx, skeleton_global_inverse * physical_bone->get_global_transform());
	}
}

void PhysicalBoneSimulator2D::set_apply_to_simulating_bones_only(bool p_enabled) {
	apply_to_simulating_bones_only = p_enabled;
}

bool PhysicalBoneSimulator2D::is_applying_to_simulating_bones_only() const {
	return apply_to_simulating_bones_only;
}

void PhysicalBoneSimulator2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_apply_to_simulating_bones_only", "enabled"), &PhysicalBoneSimulator2D::set_apply_to_simulating_bones_only);
	ClassDB::bind_method(D_METHOD("is_applying_to_simulating_bones_only"), &PhysicalBoneSimulator2D::is_applying_to_simulating_bones_only);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "apply_to_simulating_bones_only"), "set_apply_to_simulating_bones_only", "is_applying_to_simulating_bones_only");
}
