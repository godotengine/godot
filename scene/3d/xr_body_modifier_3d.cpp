/**************************************************************************/
/*  xr_body_modifier_3d.cpp                                               */
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

#include "xr_body_modifier_3d.h"

#include "scene/3d/skeleton_3d.h"
#include "servers/xr/xr_pose.h"
#include "servers/xr_server.h"

void XRBodyModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_body_tracker", "tracker_name"), &XRBodyModifier3D::set_body_tracker);
	ClassDB::bind_method(D_METHOD("get_body_tracker"), &XRBodyModifier3D::get_body_tracker);

	ClassDB::bind_method(D_METHOD("set_target", "target"), &XRBodyModifier3D::set_target);
	ClassDB::bind_method(D_METHOD("get_target"), &XRBodyModifier3D::get_target);

	ClassDB::bind_method(D_METHOD("set_body_update", "body_update"), &XRBodyModifier3D::set_body_update);
	ClassDB::bind_method(D_METHOD("get_body_update"), &XRBodyModifier3D::get_body_update);

	ClassDB::bind_method(D_METHOD("set_bone_update", "bone_update"), &XRBodyModifier3D::set_bone_update);
	ClassDB::bind_method(D_METHOD("get_bone_update"), &XRBodyModifier3D::get_bone_update);

	ClassDB::bind_method(D_METHOD("set_show_when_tracked", "show"), &XRBodyModifier3D::set_show_when_tracked);
	ClassDB::bind_method(D_METHOD("get_show_when_tracked"), &XRBodyModifier3D::get_show_when_tracked);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "body_tracker", PROPERTY_HINT_ENUM_SUGGESTION, "/user/body"), "set_body_tracker", "get_body_tracker");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_target", "get_target");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "body_update", PROPERTY_HINT_FLAGS, "Upper Body,Lower Body,Hands"), "set_body_update", "get_body_update");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_update", PROPERTY_HINT_ENUM, "Full,Rotation Only"), "set_bone_update", "get_bone_update");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_when_tracked"), "set_show_when_tracked", "get_show_when_tracked");

	BIND_BITFIELD_FLAG(BODY_UPDATE_UPPER_BODY);
	BIND_BITFIELD_FLAG(BODY_UPDATE_LOWER_BODY);
	BIND_BITFIELD_FLAG(BODY_UPDATE_HANDS);

	BIND_ENUM_CONSTANT(BONE_UPDATE_FULL);
	BIND_ENUM_CONSTANT(BONE_UPDATE_ROTATION_ONLY);
	BIND_ENUM_CONSTANT(BONE_UPDATE_MAX);
}

void XRBodyModifier3D::set_body_tracker(const StringName &p_tracker_name) {
	tracker_name = p_tracker_name;
}

StringName XRBodyModifier3D::get_body_tracker() const {
	return tracker_name;
}

void XRBodyModifier3D::set_target(const NodePath &p_target) {
	target = p_target;

	if (is_inside_tree()) {
		_get_joint_data();
	}
}

NodePath XRBodyModifier3D::get_target() const {
	return target;
}

void XRBodyModifier3D::set_body_update(BitField<BodyUpdate> p_body_update) {
	body_update = p_body_update;
}

BitField<XRBodyModifier3D::BodyUpdate> XRBodyModifier3D::get_body_update() const {
	return body_update;
}

void XRBodyModifier3D::set_bone_update(BoneUpdate p_bone_update) {
	ERR_FAIL_INDEX(p_bone_update, BONE_UPDATE_MAX);
	bone_update = p_bone_update;
}

XRBodyModifier3D::BoneUpdate XRBodyModifier3D::get_bone_update() const {
	return bone_update;
}

void XRBodyModifier3D::set_show_when_tracked(bool p_show_when_tracked) {
	show_when_tracked = p_show_when_tracked;
}

bool XRBodyModifier3D::get_show_when_tracked() const {
	return show_when_tracked;
}

Skeleton3D *XRBodyModifier3D::get_skeleton() {
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

void XRBodyModifier3D::_get_joint_data() {
	// Table of Godot Humanoid bone names.
	static const String bone_names[XRBodyTracker::JOINT_MAX] = {
		"Root", // XRBodyTracker::JOINT_ROOT

		// Upper Body Joints.
		"Hips", // XRBodyTracker::JOINT_HIPS
		"Spine", // XRBodyTracker::JOINT_SPINE
		"Chest", // XRBodyTracker::JOINT_CHEST
		"UpperChest", // XRBodyTracker::JOINT_UPPER_CHEST
		"Neck", // XRBodyTracker::JOINT_NECK"
		"Head", // XRBodyTracker::JOINT_HEAD"
		"HeadTip", // XRBodyTracker::JOINT_HEAD_TIP"
		"LeftShoulder", // XRBodyTracker::JOINT_LEFT_SHOULDER"
		"LeftUpperArm", // XRBodyTracker::JOINT_LEFT_UPPER_ARM"
		"LeftLowerArm", // XRBodyTracker::JOINT_LEFT_LOWER_ARM"
		"RightShoulder", // XRBodyTracker::JOINT_RIGHT_SHOULDER"
		"RightUpperArm", // XRBodyTracker::JOINT_RIGHT_UPPER_ARM"
		"RightLowerArm", // XRBodyTracker::JOINT_RIGHT_LOWER_ARM"

		// Lower Body Joints.
		"LeftUpperLeg", // XRBodyTracker::JOINT_LEFT_UPPER_LEG
		"LeftLowerLeg", // XRBodyTracker::JOINT_LEFT_LOWER_LEG
		"LeftFoot", // XRBodyTracker::JOINT_LEFT_FOOT
		"LeftToes", // XRBodyTracker::JOINT_LEFT_TOES
		"RightUpperLeg", // XRBodyTracker::JOINT_RIGHT_UPPER_LEG
		"RightLowerLeg", // XRBodyTracker::JOINT_RIGHT_LOWER_LEG
		"RightFoot", // XRBodyTracker::JOINT_RIGHT_FOOT
		"RightToes", // XRBodyTracker::JOINT_RIGHT_TOES

		// Left Hand Joints.
		"LeftHand", // XRBodyTracker::JOINT_LEFT_HAND
		"LeftPalm", // XRBodyTracker::JOINT_LEFT_PALM
		"LeftWrist", // XRBodyTracker::JOINT_LEFT_WRIST
		"LeftThumbMetacarpal", // XRBodyTracker::JOINT_LEFT_THUMB_METACARPAL
		"LeftThumbProximal", // XRBodyTracker::JOINT_LEFT_THUMB_PHALANX_PROXIMAL
		"LeftThumbDistal", // XRBodyTracker::JOINT_LEFT_THUMB_PHALANX_DISTAL
		"LeftThumbTip", // XRBodyTracker::JOINT_LEFT_THUMB_TIP
		"LeftIndexMetacarpal", // XRBodyTracker::JOINT_LEFT_INDEX_FINGER_METACARPAL
		"LeftIndexProximal", // XRBodyTracker::JOINT_LEFT_INDEX_FINGER_PHALANX_PROXIMAL
		"LeftIndexIntermediate", // XRBodyTracker::JOINT_LEFT_INDEX_FINGER_PHALANX_INTERMEDIATE
		"LeftIndexDistal", // XRBodyTracker::JOINT_LEFT_INDEX_FINGER_PHALANX_DISTAL
		"LeftIndexTip", // XRBodyTracker::JOINT_LEFT_INDEX_FINGER_TIP
		"LeftMiddleMetacarpal", // XRBodyTracker::JOINT_LEFT_MIDDLE_FINGER_METACARPAL
		"LeftMiddleProximal", // XRBodyTracker::JOINT_LEFT_MIDDLE_FINGER_PHALANX_PROXIMAL
		"LeftMiddleIntermediate", // XRBodyTracker::JOINT_LEFT_MIDDLE_FINGER_PHALANX_INTERMEDIATE
		"LeftMiddleDistal", // XRBodyTracker::JOINT_LEFT_MIDDLE_FINGER_PHALANX_DISTAL
		"LeftMiddleTip", // XRBodyTracker::JOINT_LEFT_MIDDLE_FINGER_TIP
		"LeftRingMetacarpal", // XRBodyTracker::JOINT_LEFT_RING_FINGER_METACARPAL
		"LeftRingProximal", // XRBodyTracker::JOINT_LEFT_RING_FINGER_PHALANX_PROXIMAL
		"LeftRingIntermediate", // XRBodyTracker::JOINT_LEFT_RING_FINGER_PHALANX_INTERMEDIATE
		"LeftRingDistal", // XRBodyTracker::JOINT_LEFT_RING_FINGER_PHALANX_DISTAL
		"LeftRingTip", // XRBodyTracker::JOINT_LEFT_RING_FINGER_TIP
		"LeftLittleMetacarpal", // XRBodyTracker::JOINT_LEFT_PINKY_FINGER_METACARPAL
		"LeftLittleProximal", // XRBodyTracker::JOINT_LEFT_PINKY_FINGER_PHALANX_PROXIMAL
		"LeftLittleIntermediate", // XRBodyTracker::JOINT_LEFT_PINKY_FINGER_PHALANX_INTERMEDIATE
		"LeftLittleDistal", // XRBodyTracker::JOINT_LEFT_PINKY_FINGER_PHALANX_DISTAL
		"LeftLittleTip", // XRBodyTracker::JOINT_LEFT_PINKY_FINGER_TIP

		// Right Hand Joints.
		"RightHand", // XRBodyTracker::JOINT_RIGHT_HAND
		"RightPalm", // XRBodyTracker::JOINT_RIGHT_PALM
		"RightWrist", // XRBodyTracker::JOINT_RIGHT_WRIST
		"RightThumbMetacarpal", // XRBodyTracker::JOINT_RIGHT_THUMB_METACARPAL
		"RightThumbProximal", // XRBodyTracker::JOINT_RIGHT_THUMB_PHALANX_PROXIMAL
		"RightThumbDistal", // XRBodyTracker::JOINT_RIGHT_THUMB_PHALANX_DISTAL
		"RightThumbTip", // XRBodyTracker::JOINT_RIGHT_THUMB_TIP
		"RightIndexMetacarpal", // XRBodyTracker::JOINT_RIGHT_INDEX_FINGER_METACARPAL
		"RightIndexProximal", // XRBodyTracker::JOINT_RIGHT_INDEX_FINGER_PHALANX_PROXIMAL
		"RightIndexIntermediate", // XRBodyTracker::JOINT_RIGHT_INDEX_FINGER_PHALANX_INTERMEDIATE
		"RightIndexDistal", // XRBodyTracker::JOINT_RIGHT_INDEX_FINGER_PHALANX_DISTAL
		"RightIndexTip", // XRBodyTracker::JOINT_RIGHT_INDEX_FINGER_TIP
		"RightMiddleMetacarpal", // XRBodyTracker::JOINT_RIGHT_MIDDLE_FINGER_METACARPAL
		"RightMiddleProximal", // XRBodyTracker::JOINT_RIGHT_MIDDLE_FINGER_PHALANX_PROXIMAL
		"RightMiddleIntermediate", // XRBodyTracker::JOINT_RIGHT_MIDDLE_FINGER_PHALANX_INTERMEDIATE
		"RightMiddleDistal", // XRBodyTracker::JOINT_RIGHT_MIDDLE_FINGER_PHALANX_DISTAL
		"RightMiddleTip", // XRBodyTracker::JOINT_RIGHT_MIDDLE_FINGER_TIP
		"RightRingMetacarpal", // XRBodyTracker::JOINT_RIGHT_RING_FINGER_METACARPAL
		"RightRingProximal", // XRBodyTracker::JOINT_RIGHT_RING_FINGER_PHALANX_PROXIMAL
		"RightRingIntermediate", // XRBodyTracker::JOINT_RIGHT_RING_FINGER_PHALANX_INTERMEDIATE
		"RightRingDistal", // XRBodyTracker::JOINT_RIGHT_RING_FINGER_PHALANX_DISTAL
		"RightRingTip", // XRBodyTracker::JOINT_RIGHT_RING_FINGER_TIP
		"RightLittleMetacarpal", // XRBodyTracker::JOINT_RIGHT_PINKY_FINGER_METACARPAL
		"RightLittleProximal", // XRBodyTracker::JOINT_RIGHT_PINKY_FINGER_PHALANX_PROXIMAL
		"RightLittleIntermediate", // XRBodyTracker::JOINT_RIGHT_PINKY_FINGER_PHALANX_INTERMEDIATE
		"RightLittleDistal", // XRBodyTracker::JOINT_RIGHT_PINKY_FINGER_PHALANX_DISTAL
		"RightLittleTip", // XRBodyTracker::JOINT_RIGHT_PINKY_FINGER_TIP
	};

	// reset JIC.
	for (int i = 0; i < XRBodyTracker::JOINT_MAX; i++) {
		joints[i].bone = -1;
		joints[i].parent_joint = -1;
	}

	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	// Find the skeleton-bones associated with each joint.
	int bones[XRBodyTracker::JOINT_MAX];
	for (int i = 0; i < XRBodyTracker::JOINT_MAX; i++) {
		// Skip upper body joints if not enabled.
		if (!body_update.has_flag(BODY_UPDATE_UPPER_BODY) && i >= XRBodyTracker::JOINT_HIPS && i <= XRBodyTracker::JOINT_RIGHT_LOWER_ARM) {
			bones[i] = -1;
			continue;
		}

		// Skip lower body joints if not enabled.
		if (!body_update.has_flag(BODY_UPDATE_LOWER_BODY) && i >= XRBodyTracker::JOINT_LEFT_UPPER_LEG && i <= XRBodyTracker::JOINT_RIGHT_TOES) {
			bones[i] = -1;
			continue;
		}

		// Skip hand joints if not enabled.
		if (!body_update.has_flag(BODY_UPDATE_HANDS) && i >= XRBodyTracker::JOINT_LEFT_HAND && i <= XRBodyTracker::JOINT_RIGHT_PINKY_FINGER_TIP) {
			bones[i] = -1;
			continue;
		}

		// Find the skeleton bone.
		bones[i] = skeleton->find_bone(bone_names[i]);
		if (bones[i] == -1) {
			WARN_PRINT(vformat("Couldn't obtain bone for %s", bone_names[i]));
		}
	}

	// Assemble the joint relationship to the available skeleton bones.
	for (int i = 0; i < XRBodyTracker::JOINT_MAX; i++) {
		// Get the skeleton bone (skip if not found).
		const int bone = bones[i];
		if (bone == -1) {
			continue;
		}

		// Find the parent skeleton-bone.
		const int parent_bone = skeleton->get_bone_parent(bone);
		if (parent_bone == -1) {
			// If no parent skeleton-bone exists then drive this relative to the root joint.
			joints[i].bone = bone;
			joints[i].parent_joint = XRBodyTracker::JOINT_ROOT;
			continue;
		}

		// Find the joint associated with the parent skeleton-bone.
		for (int j = 0; j < XRBodyTracker::JOINT_MAX; ++j) {
			if (bones[j] == parent_bone) {
				// If a parent joint is found then drive this bone relative to it.
				joints[i].bone = bone;
				joints[i].parent_joint = j;
				break;
			}
		}
	}
}

void XRBodyModifier3D::_update_skeleton() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (!xr_server) {
		return;
	}

	Ref<XRBodyTracker> tracker = xr_server->get_body_tracker(tracker_name);
	if (tracker.is_null()) {
		return;
	}

	// Handle no tracking data.
	if (!tracker->get_has_tracking_data()) {
		// If tracking-state determines visibility then hide the node.
		if (show_when_tracked) {
			set_visible(false);
		}
		return;
	}

	// Get the world and skeleton scale.
	const float ws = xr_server->get_world_scale();
	const float ss = skeleton->get_motion_scale();

	// Read the relevant tracking data. This applies the skeleton motion scale to
	// the joint transforms, allowing the tracking data to be scaled to the skeleton.
	bool has_valid_data[XRBodyTracker::JOINT_MAX];
	Transform3D transforms[XRBodyTracker::JOINT_MAX];
	Transform3D inv_transforms[XRBodyTracker::JOINT_MAX];
	for (int joint = 0; joint < XRBodyTracker::JOINT_MAX; joint++) {
		BitField<XRBodyTracker::JointFlags> flags = tracker->get_joint_flags(static_cast<XRBodyTracker::Joint>(joint));
		has_valid_data[joint] = flags.has_flag(XRBodyTracker::JOINT_FLAG_ORIENTATION_VALID) && flags.has_flag(XRBodyTracker::JOINT_FLAG_POSITION_VALID);

		if (has_valid_data[joint]) {
			transforms[joint] = tracker->get_joint_transform(static_cast<XRBodyTracker::Joint>(joint));
			transforms[joint].origin *= ss;
			inv_transforms[joint] = transforms[joint].inverse();
		}
	}

	// Handle root joint not tracked.
	if (!has_valid_data[XRBodyTracker::JOINT_ROOT]) {
		// If tracking-state determines visibility then hide the node.
		if (show_when_tracked) {
			set_visible(false);
		}
		return;
	}

	// Apply the joint information to the skeleton.
	for (int joint = 0; joint < XRBodyTracker::JOINT_MAX; joint++) {
		// Skip if no valid joint data
		if (!has_valid_data[joint]) {
			continue;
		}

		// Get the skeleton bone (skip if none).
		const int bone = joints[joint].bone;
		if (bone == -1) {
			continue;
		}

		// Calculate the relative relationship to the parent bone joint.
		const int parent_joint = joints[joint].parent_joint;
		const Transform3D relative_transform = inv_transforms[parent_joint] * transforms[joint];

		// Update the bone position if enabled by update mode, or if the joint is the hips, to allow
		// for climbing or crouching.
		if (bone_update == BONE_UPDATE_FULL || joint == XRBodyTracker::JOINT_HIPS) {
			skeleton->set_bone_pose_position(joints[joint].bone, relative_transform.origin);
		}

		// Always update the bone rotation.
		skeleton->set_bone_pose_rotation(joints[joint].bone, Quaternion(relative_transform.basis));
	}

	// Transform to the tracking data root pose. This also applies the XR world-scale to allow
	// scaling the avatars mesh and skeleton appropriately (if they are child nodes).
	set_transform(
			transforms[XRBodyTracker::JOINT_ROOT] * ws);

	// If tracking-state determines visibility then show the node.
	if (show_when_tracked) {
		set_visible(true);
	}
}

void XRBodyModifier3D::_tracker_changed(const StringName &p_tracker_name, const Ref<XRBodyTracker> &p_tracker) {
	if (tracker_name == p_tracker_name) {
		_get_joint_data();
	}
}

void XRBodyModifier3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			XRServer *xr_server = XRServer::get_singleton();
			if (xr_server) {
				xr_server->connect("body_tracker_added", callable_mp(this, &XRBodyModifier3D::_tracker_changed));
				xr_server->connect("body_tracker_updated", callable_mp(this, &XRBodyModifier3D::_tracker_changed));
				xr_server->connect("body_tracker_removed", callable_mp(this, &XRBodyModifier3D::_tracker_changed).bind(Ref<XRBodyTracker>()));
			}

			_get_joint_data();

			set_process_internal(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			XRServer *xr_server = XRServer::get_singleton();
			if (xr_server) {
				xr_server->disconnect("body_tracker_added", callable_mp(this, &XRBodyModifier3D::_tracker_changed));
				xr_server->disconnect("body_tracker_updated", callable_mp(this, &XRBodyModifier3D::_tracker_changed));
				xr_server->disconnect("body_tracker_removed", callable_mp(this, &XRBodyModifier3D::_tracker_changed).bind(Ref<XRBodyTracker>()));
			}

			set_process_internal(false);

			for (int i = 0; i < XRBodyTracker::JOINT_MAX; i++) {
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
