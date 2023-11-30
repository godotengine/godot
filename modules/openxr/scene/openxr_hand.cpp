/**************************************************************************/
/*  openxr_hand.cpp                                                       */
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

#include "openxr_hand.h"

#include "../extensions/openxr_hand_tracking_extension.h"
#include "../openxr_api.h"

#include "scene/3d/skeleton_3d.h"
#include "servers/xr_server.h"

void OpenXRHand::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_hand", "hand"), &OpenXRHand::set_hand);
	ClassDB::bind_method(D_METHOD("get_hand"), &OpenXRHand::get_hand);

	ClassDB::bind_method(D_METHOD("set_hand_skeleton", "hand_skeleton"), &OpenXRHand::set_hand_skeleton);
	ClassDB::bind_method(D_METHOD("get_hand_skeleton"), &OpenXRHand::get_hand_skeleton);

	ClassDB::bind_method(D_METHOD("set_motion_range", "motion_range"), &OpenXRHand::set_motion_range);
	ClassDB::bind_method(D_METHOD("get_motion_range"), &OpenXRHand::get_motion_range);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "hand", PROPERTY_HINT_ENUM, "Left,Right"), "set_hand", "get_hand");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "motion_range", PROPERTY_HINT_ENUM, "Unobstructed,Conform to controller"), "set_motion_range", "get_motion_range");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "hand_skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_hand_skeleton", "get_hand_skeleton");

	BIND_ENUM_CONSTANT(HAND_LEFT);
	BIND_ENUM_CONSTANT(HAND_RIGHT);
	BIND_ENUM_CONSTANT(HAND_MAX);

	BIND_ENUM_CONSTANT(MOTION_RANGE_UNOBSTRUCTED);
	BIND_ENUM_CONSTANT(MOTION_RANGE_CONFORM_TO_CONTROLLER);
	BIND_ENUM_CONSTANT(MOTION_RANGE_MAX);
}

OpenXRHand::OpenXRHand() {
	openxr_api = OpenXRAPI::get_singleton();
	hand_tracking_ext = OpenXRHandTrackingExtension::get_singleton();
}

void OpenXRHand::set_hand(const Hands p_hand) {
	ERR_FAIL_INDEX(p_hand, HAND_MAX);

	hand = p_hand;
}

OpenXRHand::Hands OpenXRHand::get_hand() const {
	return hand;
}

void OpenXRHand::set_hand_skeleton(const NodePath &p_hand_skeleton) {
	hand_skeleton = p_hand_skeleton;

	// TODO if inside tree call _get_bones()
}

void OpenXRHand::set_motion_range(const MotionRange p_motion_range) {
	ERR_FAIL_INDEX(p_motion_range, MOTION_RANGE_MAX);
	motion_range = p_motion_range;

	_set_motion_range();
}

OpenXRHand::MotionRange OpenXRHand::get_motion_range() const {
	return motion_range;
}

NodePath OpenXRHand::get_hand_skeleton() const {
	return hand_skeleton;
}

void OpenXRHand::_set_motion_range() {
	if (!hand_tracking_ext) {
		return;
	}

	XrHandJointsMotionRangeEXT xr_motion_range;
	switch (motion_range) {
		case MOTION_RANGE_UNOBSTRUCTED:
			xr_motion_range = XR_HAND_JOINTS_MOTION_RANGE_UNOBSTRUCTED_EXT;
			break;
		case MOTION_RANGE_CONFORM_TO_CONTROLLER:
			xr_motion_range = XR_HAND_JOINTS_MOTION_RANGE_CONFORMING_TO_CONTROLLER_EXT;
			break;
		default:
			xr_motion_range = XR_HAND_JOINTS_MOTION_RANGE_CONFORMING_TO_CONTROLLER_EXT;
			break;
	}

	hand_tracking_ext->set_motion_range(OpenXRHandTrackingExtension::HandTrackedHands(hand), xr_motion_range);
}

Skeleton3D *OpenXRHand::get_skeleton() {
	if (!has_node(hand_skeleton)) {
		return nullptr;
	}

	Node *node = get_node(hand_skeleton);
	if (!node) {
		return nullptr;
	}

	Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(node);
	return skeleton;
}

void OpenXRHand::_get_bones() {
	const char *bone_names[XR_HAND_JOINT_COUNT_EXT] = {
		"Palm",
		"Wrist",
		"Thumb_Metacarpal",
		"Thumb_Proximal",
		"Thumb_Distal",
		"Thumb_Tip",
		"Index_Metacarpal",
		"Index_Proximal",
		"Index_Intermediate",
		"Index_Distal",
		"Index_Tip",
		"Middle_Metacarpal",
		"Middle_Proximal",
		"Middle_Intermediate",
		"Middle_Distal",
		"Middle_Tip",
		"Ring_Metacarpal",
		"Ring_Proximal",
		"Ring_Intermediate",
		"Ring_Distal",
		"Ring_Tip",
		"Little_Metacarpal",
		"Little_Proximal",
		"Little_Intermediate",
		"Little_Distal",
		"Little_Tip",
	};

	// reset JIC
	for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; i++) {
		bones[i] = -1;
	}

	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	// We cast to spatials which should allow us to use any subclass of that.
	for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; i++) {
		String bone_name = bone_names[i];
		if (hand == 0) {
			bone_name += String("_L");
		} else {
			bone_name += String("_R");
		}

		bones[i] = skeleton->find_bone(bone_name);
		if (bones[i] == -1) {
			print_line("Couldn't obtain bone for", bone_name);
		}
	}
}

void OpenXRHand::_update_skeleton() {
	if (openxr_api == nullptr || !openxr_api->is_initialized()) {
		return;
	} else if (hand_tracking_ext == nullptr || !hand_tracking_ext->get_active()) {
		return;
	}

	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	// we cache our transforms so we can quickly calculate local transforms
	XRPose::TrackingConfidence confidences[XR_HAND_JOINT_COUNT_EXT];
	Quaternion quaternions[XR_HAND_JOINT_COUNT_EXT];
	Quaternion inv_quaternions[XR_HAND_JOINT_COUNT_EXT];
	Vector3 positions[XR_HAND_JOINT_COUNT_EXT];

	const OpenXRHandTrackingExtension::HandTracker *hand_tracker = hand_tracking_ext->get_hand_tracker(OpenXRHandTrackingExtension::HandTrackedHands(hand));
	const float ws = XRServer::get_singleton()->get_world_scale();

	if (hand_tracker->is_initialized && hand_tracker->locations.isActive) {
		for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; i++) {
			confidences[i] = XRPose::XR_TRACKING_CONFIDENCE_NONE;
			quaternions[i] = Quaternion();
			positions[i] = Vector3();

			const XrHandJointLocationEXT &location = hand_tracker->joint_locations[i];
			const XrPosef &pose = location.pose;

			if (location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) {
				if (pose.orientation.x != 0 || pose.orientation.y != 0 || pose.orientation.z != 0 || pose.orientation.w != 0) {
					quaternions[i] = Quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
					inv_quaternions[i] = quaternions[i].inverse();

					if (location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) {
						confidences[i] = XRPose::XR_TRACKING_CONFIDENCE_HIGH;
						positions[i] = Vector3(pose.position.x * ws, pose.position.y * ws, pose.position.z * ws);

						// TODO get inverse of position, we'll do this later. For now we're ignoring bone positions which generally works better anyway
					} else {
						confidences[i] = XRPose::XR_TRACKING_CONFIDENCE_LOW;
					}
				}
			}
		}

		if (confidences[XR_HAND_JOINT_PALM_EXT] != XRPose::XR_TRACKING_CONFIDENCE_NONE) {
			// now update our skeleton
			for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; i++) {
				if (bones[i] != -1) {
					int bone = bones[i];
					int parent = skeleton->get_bone_parent(bone);

					// Get our target quaternion
					Quaternion q = quaternions[i];

					// Get our target position
					Vector3 p = positions[i];

					// get local translation, parent should already be processed
					if (parent == -1) {
						// use our palm location here, that is what we are tracking
						q = inv_quaternions[XR_HAND_JOINT_PALM_EXT] * q;
						p = inv_quaternions[XR_HAND_JOINT_PALM_EXT].xform(p - positions[XR_HAND_JOINT_PALM_EXT]);
					} else {
						int found = false;
						for (int b = 0; b < XR_HAND_JOINT_COUNT_EXT && !found; b++) {
							if (bones[b] == parent) {
								q = inv_quaternions[b] * q;
								p = inv_quaternions[b].xform(p - positions[b]);
								found = true;
							}
						}
					}

					// and set our pose
					skeleton->set_bone_pose_position(bones[i], p);
					skeleton->set_bone_pose_rotation(bones[i], q);
				}
			}

			Transform3D t;
			t.basis = Basis(quaternions[XR_HAND_JOINT_PALM_EXT]);
			t.origin = positions[XR_HAND_JOINT_PALM_EXT];
			set_transform(t);

			// show it
			set_visible(true);
		} else {
			// hide it
			set_visible(false);
		}
	} else {
		// hide it
		set_visible(false);
	}
}

void OpenXRHand::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_get_bones();

			set_process_internal(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			set_process_internal(false);

			// reset
			for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; i++) {
				bones[i] = -1;
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			_update_skeleton();
		} break;
		default: {
		} break;
	}
}
