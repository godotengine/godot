/**************************************************************************/
/*  visionos_hand_tracking.mm                                             */
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

#ifdef VISIONOS_ENABLED

#include "visionos_hand_tracking.h"

#include "visionos_simd_helpers.h"

namespace {

_FORCE_INLINE_ XRHandTracker::HandJoint joint_from_arkit(ar_hand_skeleton_joint_name_t p_joint_name) {
	switch (p_joint_name) {
		case ar_hand_skeleton_joint_name_wrist:
			return XRHandTracker::HAND_JOINT_WRIST;
		case ar_hand_skeleton_joint_name_thumb_knuckle:
			return XRHandTracker::HAND_JOINT_THUMB_METACARPAL;
		case ar_hand_skeleton_joint_name_thumb_intermediate_base:
			return XRHandTracker::HAND_JOINT_THUMB_PHALANX_PROXIMAL;
		case ar_hand_skeleton_joint_name_thumb_intermediate_tip:
			return XRHandTracker::HAND_JOINT_THUMB_PHALANX_DISTAL;
		case ar_hand_skeleton_joint_name_thumb_tip:
			return XRHandTracker::HAND_JOINT_THUMB_TIP;
		case ar_hand_skeleton_joint_name_index_finger_metacarpal:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_METACARPAL;
		case ar_hand_skeleton_joint_name_index_finger_knuckle:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_PROXIMAL;
		case ar_hand_skeleton_joint_name_index_finger_intermediate_base:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_INTERMEDIATE;
		case ar_hand_skeleton_joint_name_index_finger_intermediate_tip:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_DISTAL;
		case ar_hand_skeleton_joint_name_index_finger_tip:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_TIP;
		case ar_hand_skeleton_joint_name_middle_finger_metacarpal:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_METACARPAL;
		case ar_hand_skeleton_joint_name_middle_finger_knuckle:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_PROXIMAL;
		case ar_hand_skeleton_joint_name_middle_finger_intermediate_base:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_INTERMEDIATE;
		case ar_hand_skeleton_joint_name_middle_finger_intermediate_tip:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_DISTAL;
		case ar_hand_skeleton_joint_name_middle_finger_tip:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_TIP;
		case ar_hand_skeleton_joint_name_ring_finger_metacarpal:
			return XRHandTracker::HAND_JOINT_RING_FINGER_METACARPAL;
		case ar_hand_skeleton_joint_name_ring_finger_knuckle:
			return XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_PROXIMAL;
		case ar_hand_skeleton_joint_name_ring_finger_intermediate_base:
			return XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_INTERMEDIATE;
		case ar_hand_skeleton_joint_name_ring_finger_intermediate_tip:
			return XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_DISTAL;
		case ar_hand_skeleton_joint_name_ring_finger_tip:
			return XRHandTracker::HAND_JOINT_RING_FINGER_TIP;
		case ar_hand_skeleton_joint_name_little_finger_metacarpal:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_METACARPAL;
		case ar_hand_skeleton_joint_name_little_finger_knuckle:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_PROXIMAL;
		case ar_hand_skeleton_joint_name_little_finger_intermediate_base:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_INTERMEDIATE;
		case ar_hand_skeleton_joint_name_little_finger_intermediate_tip:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_DISTAL;
		case ar_hand_skeleton_joint_name_little_finger_tip:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_TIP;
		case ar_hand_skeleton_joint_name_forearm_wrist:
		case ar_hand_skeleton_joint_name_forearm_arm:
		default:
			// These don't have direct equivalents or are invalid
			return XRHandTracker::HAND_JOINT_MAX;
	}
}
} // namespace

void VisionOSHandTracking::initialize(XRServer *p_xr_server) {
	// Hand tracking provider (registered with the shared ARKit session)
	ar_hand_tracking_configuration_t hand_tracking_configuration = ar_hand_tracking_configuration_create();
	hand_tracking_provider = ar_hand_tracking_provider_create(hand_tracking_configuration);

	// Hand tracker initialization
	left_hand_tracker.instantiate();
	left_hand_tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_LEFT);
	left_hand_tracker->set_tracker_name("/user/hand_tracker/left");
	p_xr_server->add_tracker(left_hand_tracker);

	right_hand_tracker.instantiate();
	right_hand_tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_RIGHT);
	right_hand_tracker->set_tracker_name("/user/hand_tracker/right");
	p_xr_server->add_tracker(right_hand_tracker);

	left_hand_anchor = ar_hand_anchor_create();
	right_hand_anchor = ar_hand_anchor_create();
}

void VisionOSHandTracking::update_hand_trackers_from_arkit(CFTimeInterval p_trackable_anchor_time) {
	if (p_trackable_anchor_time != 0) {
		ar_hand_anchor_query_status_t query_anchor_result =
				ar_hand_tracking_provider_query_anchors_at_timestamp(hand_tracking_provider,
						p_trackable_anchor_time,
						left_hand_anchor,
						right_hand_anchor);

		if (query_anchor_result != ar_hand_anchor_query_status_success) {
			reset_hand_tracker_data(left_hand_tracker);
			reset_hand_tracker_data(right_hand_tracker);
			ERR_FAIL_MSG("Cannot query hand anchors, result: " + itos(query_anchor_result) + ".");
		}
	} else {
		// If we failed to get a trackable_anchor_time, we just get the latest anchors.
		// Tracking will be less precise in this case
		bool result = ar_hand_tracking_provider_get_latest_anchors(hand_tracking_provider, left_hand_anchor, right_hand_anchor);
		if (!result) {
			reset_hand_tracker_data(left_hand_tracker);
			reset_hand_tracker_data(right_hand_tracker);
			ERR_FAIL_MSG("Cannot query latest anchors, the ARKit session is probably not running.");
		}
	}

	if (ar_hand_anchor_is_tracked(left_hand_anchor)) {
		set_hand_tracker_data_from_arkit(left_hand_tracker, left_hand_anchor);
	} else {
		reset_hand_tracker_data(left_hand_tracker);
	}

	if (ar_hand_anchor_is_tracked(right_hand_anchor)) {
		set_hand_tracker_data_from_arkit(right_hand_tracker, right_hand_anchor);
	} else {
		reset_hand_tracker_data(right_hand_tracker);
	}
}

void VisionOSHandTracking::reset_hand_tracker_data(Ref<XRHandTracker> p_hand_tracker) {
	p_hand_tracker->set_hand_tracking_source(XRHandTracker::HAND_TRACKING_SOURCE_UNKNOWN);
	p_hand_tracker->set_has_tracking_data(false);
	p_hand_tracker->invalidate_pose("default");
}

void VisionOSHandTracking::set_hand_tracker_data_from_arkit(Ref<XRHandTracker> p_hand_tracker, ar_hand_anchor_t p_hand_anchor) {
	simd_float4x4 origin_from_hand_anchor_simd = ar_hand_anchor_get_origin_from_anchor_transform(p_hand_anchor);

	ar_hand_skeleton_t hand_skeleton = ar_hand_anchor_get_hand_skeleton(p_hand_anchor);
	Transform3D origin_from_hand_anchor = MTL::simd_to_transform3D(origin_from_hand_anchor_simd);

	// Rotate from ARKit coordinates to Godot Humanoid coordinates
	ar_hand_chirality_t chirality = ar_hand_anchor_get_chirality(p_hand_anchor);
	bool is_left_hand = (chirality == ar_hand_chirality_left);
	real_t rotation_angle = (is_left_hand ? -1 : 1) * Math::PI * 0.5;
	const Quaternion rotationX(Vector3(1, 0, 0), rotation_angle);

	BitField<XRHandTracker::HandJointFlags> flags = {};
	flags.set_flag(XRHandTracker::HAND_JOINT_FLAG_ORIENTATION_VALID);
	flags.set_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_VALID);
	flags.set_flag(XRHandTracker::HAND_JOINT_FLAG_ORIENTATION_TRACKED);
	flags.set_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_TRACKED);

	// Updating all the hand joints
	const Quaternion rotationZ(Vector3(0, 0, 1), rotation_angle);
	const Quaternion joint_axis_adjustment = rotationX * rotationZ;
	ar_hand_skeleton_enumerate_joints(hand_skeleton, ^bool(ar_skeleton_joint_t joint) {
		uint64_t joint_index = ar_skeleton_joint_get_index(joint);
		XRHandTracker::HandJoint hand_joint = joint_from_arkit((ar_hand_skeleton_joint_name_t)joint_index);
		if (hand_joint == XRHandTracker::HAND_JOINT_MAX) {
			return true;
		}
		simd_float4x4 hand_anchor_from_joint_simd = ar_skeleton_joint_get_anchor_from_joint_transform(joint);
		Transform3D hand_anchor_from_joint = MTL::simd_to_transform3D(hand_anchor_from_joint_simd);
		Transform3D origin_from_joint = origin_from_hand_anchor * hand_anchor_from_joint;
		origin_from_joint.basis = origin_from_joint.basis * joint_axis_adjustment;
		p_hand_tracker->set_hand_joint_transform(hand_joint, origin_from_joint);
		p_hand_tracker->set_hand_joint_flags(hand_joint, flags);
		return true;
	});

	// ARKit hands don't have a palm joint, so computing it the same way WebXR (webxr_interface_js.cpp) does it:
	// finding the middle of the middle-finger's metacarpal bone
	{
		// Start by getting the middle finger metacarpal joint.
		Transform3D palm_transform = p_hand_tracker->get_hand_joint_transform(XRHandTracker::HAND_JOINT_MIDDLE_FINGER_METACARPAL);

		// Get the middle finger phalanx position.
		Vector3 phalanx = p_hand_tracker->get_hand_joint_transform(XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_PROXIMAL).origin;

		// Offset the palm half-way towards the phalanx joint.
		palm_transform.origin = (palm_transform.origin + phalanx) / 2.0;

		// Set the palm joint and the pose.
		p_hand_tracker->set_hand_joint_transform(XRHandTracker::HAND_JOINT_PALM, palm_transform);
		p_hand_tracker->set_hand_joint_flags(XRHandTracker::HAND_JOINT_PALM, flags);
		// Note: ARKit does not have API for linear/angular velocity; so leaving it at 0
		p_hand_tracker->set_pose("default", palm_transform, Vector3(), Vector3());
	}

	p_hand_tracker->set_hand_tracking_source(XRHandTracker::HAND_TRACKING_SOURCE_UNOBSTRUCTED);
	p_hand_tracker->set_has_tracking_data(true);
}

#endif // VISIONOS_ENABLED
