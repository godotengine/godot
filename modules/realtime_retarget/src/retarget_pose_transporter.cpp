/*************************************************************************/
/*  retarget_pose_transporter.cpp                                        */
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

#include "retarget_pose_transporter.h"

#include "retarget_utility.h"

void RetargetPoseTransporter::set_source_skeleton(const NodePath &p_skeleton) {
	ERR_FAIL_COND_MSG(!p_skeleton.is_empty() && !target_skeleton_path.is_empty() && p_skeleton == target_skeleton_path, "The source and target skeletons are not allowed to be the same.");
	source_skeleton_path = p_skeleton;
}

NodePath RetargetPoseTransporter::get_source_skeleton() const {
	return source_skeleton_path;
}

void RetargetPoseTransporter::set_target_skeleton(const NodePath &p_skeleton) {
	ERR_FAIL_COND_MSG(!p_skeleton.is_empty() && !source_skeleton_path.is_empty() && p_skeleton == source_skeleton_path, "The source and target skeletons are not allowed to be the same.");
	target_skeleton_path = p_skeleton;
}

NodePath RetargetPoseTransporter::get_target_skeleton() const {
	return target_skeleton_path;
}

void RetargetPoseTransporter::set_profile(const Ref<RetargetProfile> &p_profile) {
	profile = p_profile;
}

Ref<RetargetProfile> RetargetPoseTransporter::get_profile() const {
	return profile;
}

void RetargetPoseTransporter::set_position_enabled(bool p_enabled) {
	position_enabled = p_enabled;
}

bool RetargetPoseTransporter::is_position_enabled() const {
	return position_enabled;
}

void RetargetPoseTransporter::set_rotation_enabled(bool p_enabled) {
	rotation_enabled = p_enabled;
}

bool RetargetPoseTransporter::is_rotation_enabled() const {
	return rotation_enabled;
}

void RetargetPoseTransporter::set_scale_enabled(bool p_enabled) {
	scale_enabled = p_enabled;
}

bool RetargetPoseTransporter::is_scale_enabled() const {
	return scale_enabled;
}

void RetargetPoseTransporter::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}

	active = p_active;

	if (process_callback == ANIMATION_PROCESS_IDLE) {
		set_process_internal(active);
	} else {
		set_physics_process_internal(active);
	}
}

bool RetargetPoseTransporter::is_active() const {
	return active;
}

void RetargetPoseTransporter::set_process_callback(AnimationProcessCallback p_mode) {
	if (process_callback == p_mode) {
		return;
	}

	bool was_active = is_active();
	if (was_active) {
		set_active(false);
	}

	process_callback = p_mode;

	if (was_active) {
		set_active(true);
	}
}

RetargetPoseTransporter::AnimationProcessCallback RetargetPoseTransporter::get_process_callback() const {
	return process_callback;
}

void RetargetPoseTransporter::_process() {
	Skeleton3D *src_skeleton = Object::cast_to<Skeleton3D>(get_node_or_null(source_skeleton_path));
	Skeleton3D *tgt_skeleton = Object::cast_to<Skeleton3D>(get_node_or_null(target_skeleton_path));
	if (!src_skeleton || !tgt_skeleton || !profile.is_valid()) {
		return;
	}

	// Global
	int len = profile->get_global_transform_target_size();
	for (int i = 0; i < len; i++) {
		String bn = profile->get_global_transform_target(i);
		int src_idx = src_skeleton->find_bone(bn);
		int tgt_idx = tgt_skeleton->find_bone(bn);
		if (src_idx < 0 || tgt_idx < 0) {
			continue;
		}
		if (position_enabled) {
			tgt_skeleton->set_bone_pose_position(tgt_idx,
					RetargetUtility::global_transform_position_to_bone_pose(tgt_skeleton, tgt_idx,
							RetargetUtility::extract_global_transform_position(src_skeleton, src_idx, src_skeleton->get_bone_pose_position(src_idx)) / src_skeleton->get_motion_scale() * tgt_skeleton->get_motion_scale()));
		}
		if (rotation_enabled) {
			tgt_skeleton->set_bone_pose_rotation(tgt_idx,
					RetargetUtility::global_transform_rotation_to_bone_pose(tgt_skeleton, tgt_idx,
							RetargetUtility::extract_global_transform_rotation(src_skeleton, src_idx, src_skeleton->get_bone_pose_rotation(src_idx))));
		}
		if (scale_enabled) {
			tgt_skeleton->set_bone_pose_scale(tgt_idx,
					RetargetUtility::global_transform_scale_to_bone_pose(tgt_skeleton, tgt_idx,
							RetargetUtility::extract_global_transform_scale(src_skeleton, src_idx, src_skeleton->get_bone_pose_scale(src_idx))));
		}
	}

	// Local
	len = profile->get_local_transform_target_size();
	for (int i = 0; i < len; i++) {
		String bn = profile->get_local_transform_target(i);
		int src_idx = src_skeleton->find_bone(bn);
		int tgt_idx = tgt_skeleton->find_bone(bn);
		if (src_idx < 0 || tgt_idx < 0) {
			continue;
		}
		if (position_enabled) {
			tgt_skeleton->set_bone_pose_position(tgt_idx,
					RetargetUtility::local_transform_position_to_bone_pose(tgt_skeleton, tgt_idx,
							RetargetUtility::extract_local_transform_position(src_skeleton, src_idx, src_skeleton->get_bone_pose_position(src_idx)) / src_skeleton->get_motion_scale() * tgt_skeleton->get_motion_scale()));
		}
		if (rotation_enabled) {
			tgt_skeleton->set_bone_pose_rotation(tgt_idx,
					RetargetUtility::local_transform_rotation_to_bone_pose(tgt_skeleton, tgt_idx,
							RetargetUtility::extract_local_transform_rotation(src_skeleton, src_idx, src_skeleton->get_bone_pose_rotation(src_idx))));
		}
		if (scale_enabled) {
			tgt_skeleton->set_bone_pose_scale(tgt_idx,
					RetargetUtility::local_transform_scale_to_bone_pose(tgt_skeleton, tgt_idx,
							RetargetUtility::extract_local_transform_scale(src_skeleton, src_idx, src_skeleton->get_bone_pose_scale(src_idx))));
		}
	}

	// Absolute
	len = profile->get_absolute_transform_target_size();
	for (int i = 0; i < len; i++) {
		String bn = profile->get_absolute_transform_target(i);
		int src_idx = src_skeleton->find_bone(bn);
		int tgt_idx = tgt_skeleton->find_bone(bn);
		if (src_idx < 0 || tgt_idx < 0) {
			continue;
		}
		if (position_enabled) {
			tgt_skeleton->set_bone_pose_position(tgt_idx, src_skeleton->get_bone_pose_position(src_idx) / src_skeleton->get_motion_scale() * tgt_skeleton->get_motion_scale());
		}
		if (rotation_enabled) {
			tgt_skeleton->set_bone_pose_rotation(tgt_idx, src_skeleton->get_bone_pose_rotation(src_idx));
		}
		if (scale_enabled) {
			tgt_skeleton->set_bone_pose_scale(tgt_idx, src_skeleton->get_bone_pose_scale(src_idx));
		}
	}
}

void RetargetPoseTransporter::advance() {
	_process();
}

void RetargetPoseTransporter::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (process_callback == ANIMATION_PROCESS_PHYSICS) {
				break;
			}
			_process();
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (process_callback == ANIMATION_PROCESS_IDLE) {
				break;
			}
			_process();
		} break;
	}
}

void RetargetPoseTransporter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source_skeleton", "skeleton"), &RetargetPoseTransporter::set_source_skeleton);
	ClassDB::bind_method(D_METHOD("get_source_skeleton"), &RetargetPoseTransporter::get_source_skeleton);

	ClassDB::bind_method(D_METHOD("set_target_skeleton", "skeleton"), &RetargetPoseTransporter::set_target_skeleton);
	ClassDB::bind_method(D_METHOD("get_target_skeleton"), &RetargetPoseTransporter::get_target_skeleton);

	ClassDB::bind_method(D_METHOD("set_profile", "profile"), &RetargetPoseTransporter::set_profile);
	ClassDB::bind_method(D_METHOD("get_profile"), &RetargetPoseTransporter::get_profile);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &RetargetPoseTransporter::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &RetargetPoseTransporter::is_active);

	ClassDB::bind_method(D_METHOD("set_position_enabled", "enable"), &RetargetPoseTransporter::set_position_enabled);
	ClassDB::bind_method(D_METHOD("is_position_enabled"), &RetargetPoseTransporter::is_position_enabled);

	ClassDB::bind_method(D_METHOD("set_rotation_enabled", "enable"), &RetargetPoseTransporter::set_rotation_enabled);
	ClassDB::bind_method(D_METHOD("is_rotation_enabled"), &RetargetPoseTransporter::is_rotation_enabled);

	ClassDB::bind_method(D_METHOD("set_scale_enabled", "enable"), &RetargetPoseTransporter::set_scale_enabled);
	ClassDB::bind_method(D_METHOD("is_scale_enabled"), &RetargetPoseTransporter::is_scale_enabled);

	ClassDB::bind_method(D_METHOD("set_process_callback", "mode"), &RetargetPoseTransporter::set_process_callback);
	ClassDB::bind_method(D_METHOD("get_process_callback"), &RetargetPoseTransporter::get_process_callback);

	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_IDLE);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_MANUAL);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "source_skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_source_skeleton", "get_source_skeleton");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_target_skeleton", "get_target_skeleton");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "profile", PROPERTY_HINT_RESOURCE_TYPE, "RetargetProfile"), "set_profile", "get_profile");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_position"), "set_position_enabled", "is_position_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_rotation"), "set_rotation_enabled", "is_rotation_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_scale"), "set_scale_enabled", "is_scale_enabled");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_callback", PROPERTY_HINT_ENUM, "Physics,Idle,Manual"), "set_process_callback", "get_process_callback");
}

RetargetPoseTransporter::RetargetPoseTransporter() {
}

RetargetPoseTransporter::~RetargetPoseTransporter() {
}
