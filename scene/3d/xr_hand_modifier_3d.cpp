/**************************************************************************/
/*  xr_hand_modifier_3d.cpp                                               */
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

#include "xr_hand_modifier_3d.h"

#include "scene/3d/skeleton_3d.h"
#include "servers/xr/xr_pose.h"
#include "servers/xr_server.h"

void XRHandModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_hand_tracker", "tracker_name"), &XRHandModifier3D::set_hand_tracker);
	ClassDB::bind_method(D_METHOD("get_hand_tracker"), &XRHandModifier3D::get_hand_tracker);

	ClassDB::bind_method(D_METHOD("set_target", "target"), &XRHandModifier3D::set_target);
	ClassDB::bind_method(D_METHOD("get_target"), &XRHandModifier3D::get_target);

	ClassDB::bind_method(D_METHOD("set_bone_update", "bone_update"), &XRHandModifier3D::set_bone_update);
	ClassDB::bind_method(D_METHOD("get_bone_update"), &XRHandModifier3D::get_bone_update);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "hand_tracker", PROPERTY_HINT_ENUM_SUGGESTION, "/user/left,/user/right"), "set_hand_tracker", "get_hand_tracker");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_target", "get_target");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_update", PROPERTY_HINT_ENUM, "Full,Rotation Only"), "set_bone_update", "get_bone_update");

	BIND_ENUM_CONSTANT(BONE_UPDATE_FULL);
	BIND_ENUM_CONSTANT(BONE_UPDATE_ROTATION_ONLY);
	BIND_ENUM_CONSTANT(BONE_UPDATE_MAX);
}

void XRHandModifier3D::set_hand_tracker(const StringName &p_tracker_name) {
	tracker_name = p_tracker_name;
}

StringName XRHandModifier3D::get_hand_tracker() const {
	return tracker_name;
}

void XRHandModifier3D::set_target(const NodePath &p_target) {
	target = p_target;

	if (is_inside_tree()) {
		_get_joint_data();
	}
}

NodePath XRHandModifier3D::get_target() const {
	return target;
}

void XRHandModifier3D::set_bone_update(BoneUpdate p_bone_update) {
	ERR_FAIL_INDEX(p_bone_update, BONE_UPDATE_MAX);
	bone_update = p_bone_update;
}

XRHandModifier3D::BoneUpdate XRHandModifier3D::get_bone_update() const {
	return bone_update;
}

Skeleton3D *XRHandModifier3D::get_skeleton() {
	if (!has_node(target)) {
		return nullptr;
	}

	Node *node = get_node(target);
	if (!node) {
		return nullptr;
	}

	Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(node);
	return skeleton;
}

void XRHandModifier3D::_get_joint_data() {
	// Table of bone names for different rig types.
	static const String bone_names[XRHandTracker::HAND_JOINT_MAX] = {
		"Palm",
		"Hand",
		"ThumbMetacarpal",
		"ThumbProximal",
		"ThumbDistal",
		"ThumbTip",
		"IndexMetacarpal",
		"IndexProximal",
		"IndexIntermediate",
		"IndexDistal",
		"IndexTip",
		"MiddleMetacarpal",
		"MiddleProximal",
		"MiddleIntermediate",
		"MiddleDistal",
		"MiddleTip",
		"RingMetacarpal",
		"RingProximal",
		"RingIntermediate",
		"RingDistal",
		"RingTip",
		"LittleMetacarpal",
		"LittleProximal",
		"LittleIntermediate",
		"LittleDistal",
		"LittleTip",
	};

	static const String bone_name_format[2] = {
		"Left<bone>",
		"Right<bone>",
	};

	// reset JIC
	for (int i = 0; i < XRHandTracker::HAND_JOINT_MAX; i++) {
		joints[i].bone = -1;
		joints[i].parent_joint = -1;
	}

	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (!xr_server) {
		return;
	}

	Ref<XRHandTracker> tracker = xr_server->get_hand_tracker(tracker_name);
	if (tracker.is_null()) {
		return;
	}

	XRHandTracker::Hand hand = tracker->get_hand();

	// Find the skeleton-bones associated with each joint.
	int bones[XRHandTracker::HAND_JOINT_MAX];
	for (int i = 0; i < XRHandTracker::HAND_JOINT_MAX; i++) {
		// Construct the expected bone name.
		String bone_name = bone_name_format[hand].replace("<bone>", bone_names[i]);

		// Find the skeleton bone.
		bones[i] = skeleton->find_bone(bone_name);
		if (bones[i] == -1) {
			WARN_PRINT(vformat("Couldn't obtain bone for %s", bone_name));
		}
	}

	// Assemble the joint relationship to the available skeleton bones.
	for (int i = 0; i < XRHandTracker::HAND_JOINT_MAX; i++) {
		// Get the skeleton bone (skip if not found).
		const int bone = bones[i];
		if (bone == -1) {
			continue;
		}

		// Find the parent skeleton-bone.
		const int parent_bone = skeleton->get_bone_parent(bone);
		if (parent_bone == -1) {
			// If no parent skeleton-bone exists then drive this relative to palm joint.
			joints[i].bone = bone;
			joints[i].parent_joint = XRHandTracker::HAND_JOINT_PALM;
			continue;
		}

		// Find the joint associated with the parent skeleton-bone.
		for (int j = 0; j < XRHandTracker::HAND_JOINT_MAX; ++j) {
			if (bones[j] == parent_bone) {
				// If a parent joint is found then drive this bone relative to it.
				joints[i].bone = bone;
				joints[i].parent_joint = j;
				break;
			}
		}
	}
}

void XRHandModifier3D::_update_skeleton() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (!xr_server) {
		return;
	}

	Ref<XRHandTracker> tracker = xr_server->get_hand_tracker(tracker_name);
	if (tracker.is_null()) {
		return;
	}

	// Get the world and skeleton scale.
	const float ws = xr_server->get_world_scale();
	const float ss = skeleton->get_motion_scale();

	// We cache our transforms so we can quickly calculate local transforms.
	bool has_valid_data[XRHandTracker::HAND_JOINT_MAX];
	Transform3D transforms[XRHandTracker::HAND_JOINT_MAX];
	Transform3D inv_transforms[XRHandTracker::HAND_JOINT_MAX];

	if (tracker->get_has_tracking_data()) {
		for (int joint = 0; joint < XRHandTracker::HAND_JOINT_MAX; joint++) {
			BitField<XRHandTracker::HandJointFlags> flags = tracker->get_hand_joint_flags((XRHandTracker::HandJoint)joint);
			has_valid_data[joint] = flags.has_flag(XRHandTracker::HAND_JOINT_FLAG_ORIENTATION_VALID);

			if (has_valid_data[joint]) {
				transforms[joint] = tracker->get_hand_joint_transform((XRHandTracker::HandJoint)joint);
				transforms[joint].origin *= ss;
				inv_transforms[joint] = transforms[joint].inverse();
			}
		}

		if (has_valid_data[XRHandTracker::HAND_JOINT_PALM]) {
			for (int joint = 0; joint < XRHandTracker::HAND_JOINT_MAX; joint++) {
				// Get the skeleton bone (skip if none).
				const int bone = joints[joint].bone;
				if (bone == -1) {
					continue;
				}

				// Calculate the relative relationship to the parent bone joint.
				const int parent_joint = joints[joint].parent_joint;
				const Transform3D relative_transform = inv_transforms[parent_joint] * transforms[joint];

				// Update the bone position if enabled by update mode.
				if (bone_update == BONE_UPDATE_FULL) {
					skeleton->set_bone_pose_position(joints[joint].bone, relative_transform.origin);
				}

				// Always update the bone rotation.
				skeleton->set_bone_pose_rotation(joints[joint].bone, Quaternion(relative_transform.basis));
			}

			// Transform to the skeleton pose. This uses the HAND_JOINT_PALM position without skeleton-scaling, as it
			// must be positioned to match the physical hand location. It is scaled with the world space to match
			// the scaling done to the camera and eyes.
			set_transform(
					tracker->get_hand_joint_transform(XRHandTracker::HAND_JOINT_PALM) * ws);

			set_visible(true);
		} else {
			set_visible(false);
		}
	} else {
		set_visible(false);
	}
}

void XRHandModifier3D::_tracker_changed(StringName p_tracker_name, const Ref<XRHandTracker> &p_tracker) {
	if (tracker_name == p_tracker_name) {
		_get_joint_data();
	}
}

void XRHandModifier3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			XRServer *xr_server = XRServer::get_singleton();
			if (xr_server) {
				xr_server->connect("hand_tracker_added", callable_mp(this, &XRHandModifier3D::_tracker_changed));
				xr_server->connect("hand_tracker_updated", callable_mp(this, &XRHandModifier3D::_tracker_changed));
				xr_server->connect("hand_tracker_removed", callable_mp(this, &XRHandModifier3D::_tracker_changed).bind(Ref<XRHandTracker>()));
			}

			_get_joint_data();

			set_process_internal(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			XRServer *xr_server = XRServer::get_singleton();
			if (xr_server) {
				xr_server->disconnect("hand_tracker_added", callable_mp(this, &XRHandModifier3D::_tracker_changed));
				xr_server->disconnect("hand_tracker_updated", callable_mp(this, &XRHandModifier3D::_tracker_changed));
				xr_server->disconnect("hand_tracker_removed", callable_mp(this, &XRHandModifier3D::_tracker_changed).bind(Ref<XRHandTracker>()));
			}

			set_process_internal(false);

			for (int i = 0; i < XRHandTracker::HAND_JOINT_MAX; i++) {
				joints[i].bone = -1;
				joints[i].parent_joint = -1;
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			_update_skeleton();
		} break;
		default: {
		} break;
	}
}
