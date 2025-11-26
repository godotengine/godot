/**************************************************************************/
/*  hand_tracking_server.cpp                                              */
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

#include "hand_tracking_server.h"

#include "hand_tracking_bridge.h"
#include "servers/xr/xr_server.h"

HandTrackingServer *HandTrackingServer::singleton = nullptr;

// Map ARKit hand joint IDs to Godot's XRHandTracker::HandJoint enum
static XRHandTracker::HandJoint map_joint_id_to_xr_joint(int joint_id) {
	switch (joint_id) {
		case GODOT_HAND_JOINT_WRIST:
			return XRHandTracker::HAND_JOINT_WRIST;

		// Thumb
		case GODOT_HAND_JOINT_THUMB_KNUCKLE:
			return XRHandTracker::HAND_JOINT_THUMB_METACARPAL;
		case GODOT_HAND_JOINT_THUMB_INTERMEDIATE:
			return XRHandTracker::HAND_JOINT_THUMB_PHALANX_PROXIMAL;
		case GODOT_HAND_JOINT_THUMB_TIP:
			return XRHandTracker::HAND_JOINT_THUMB_TIP;

		// Index finger
		case GODOT_HAND_JOINT_INDEX_KNUCKLE:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_METACARPAL;
		case GODOT_HAND_JOINT_INDEX_INTERMEDIATE:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_PROXIMAL;
		case GODOT_HAND_JOINT_INDEX_DISTAL:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_INTERMEDIATE;
		case GODOT_HAND_JOINT_INDEX_TIP:
			return XRHandTracker::HAND_JOINT_INDEX_FINGER_TIP;

		// Middle finger
		case GODOT_HAND_JOINT_MIDDLE_KNUCKLE:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_METACARPAL;
		case GODOT_HAND_JOINT_MIDDLE_INTERMEDIATE:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_PROXIMAL;
		case GODOT_HAND_JOINT_MIDDLE_DISTAL:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_INTERMEDIATE;
		case GODOT_HAND_JOINT_MIDDLE_TIP:
			return XRHandTracker::HAND_JOINT_MIDDLE_FINGER_TIP;

		// Ring finger
		case GODOT_HAND_JOINT_RING_KNUCKLE:
			return XRHandTracker::HAND_JOINT_RING_FINGER_METACARPAL;
		case GODOT_HAND_JOINT_RING_INTERMEDIATE:
			return XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_PROXIMAL;
		case GODOT_HAND_JOINT_RING_DISTAL:
			return XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_INTERMEDIATE;
		case GODOT_HAND_JOINT_RING_TIP:
			return XRHandTracker::HAND_JOINT_RING_FINGER_TIP;

		// Little finger
		case GODOT_HAND_JOINT_LITTLE_KNUCKLE:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_METACARPAL;
		case GODOT_HAND_JOINT_LITTLE_INTERMEDIATE:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_PROXIMAL;
		case GODOT_HAND_JOINT_LITTLE_DISTAL:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_INTERMEDIATE;
		case GODOT_HAND_JOINT_LITTLE_TIP:
			return XRHandTracker::HAND_JOINT_PINKY_FINGER_TIP;

		default:
			return XRHandTracker::HAND_JOINT_PALM; // Fallback
	}
}

void HandTrackingServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_hand_tracking_available"), &HandTrackingServer::is_hand_tracking_available);
	ClassDB::bind_method(D_METHOD("get_left_hand_tracker"), &HandTrackingServer::get_left_hand_tracker);
	ClassDB::bind_method(D_METHOD("get_right_hand_tracker"), &HandTrackingServer::get_right_hand_tracker);
	ClassDB::bind_method(D_METHOD("update_hand_tracking"), &HandTrackingServer::update_hand_tracking);
}

HandTrackingServer *HandTrackingServer::get_singleton() {
	return singleton;
}

void HandTrackingServer::_ensure_trackers() {
	if (initialized) {
		return;
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (!xr_server) {
		return;
	}

	// Create left hand tracker
	left_hand_tracker.instantiate();
	left_hand_tracker->set_tracker_type(XRServer::TRACKER_HAND);
	left_hand_tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_LEFT);
	left_hand_tracker->set_tracker_name("left_hand");
	left_hand_tracker->set_tracker_desc("VisionOS Left Hand");
	xr_server->add_tracker(left_hand_tracker);

	// Create right hand tracker
	right_hand_tracker.instantiate();
	right_hand_tracker->set_tracker_type(XRServer::TRACKER_HAND);
	right_hand_tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_RIGHT);
	right_hand_tracker->set_tracker_name("right_hand");
	right_hand_tracker->set_tracker_desc("VisionOS Right Hand");
	xr_server->add_tracker(right_hand_tracker);

	initialized = true;
}

void HandTrackingServer::_update_hand_tracker(Ref<XRHandTracker> tracker, const godot_hand_joint *joints, int joint_count) {
	if (tracker.is_null()) {
		return;
	}

	bool has_data = joint_count > 0;
	tracker->set_has_tracking_data(has_data);

	if (!has_data) {
		tracker->set_hand_tracking_source(XRHandTracker::HAND_TRACKING_SOURCE_NOT_TRACKED);
		return;
	}

	tracker->set_hand_tracking_source(XRHandTracker::HAND_TRACKING_SOURCE_UNOBSTRUCTED);

	// Update each joint
	for (int i = 0; i < joint_count; i++) {
		const godot_hand_joint &joint = joints[i];

		if (!joint.valid) {
			continue;
		}

		XRHandTracker::HandJoint xr_joint = map_joint_id_to_xr_joint(joint.joint_id);

		// Create transform from position and quaternion
		Vector3 position(joint.position[0], joint.position[1], joint.position[2]);
		Quaternion rotation(joint.orientation[0], joint.orientation[1], joint.orientation[2], joint.orientation[3]);

		Transform3D transform;
		transform.basis = Basis(rotation);
		transform.origin = position;

		// Set joint transform
		tracker->set_hand_joint_transform(xr_joint, transform);

		// Set joint flags to indicate position and orientation are valid and tracked
		BitField<XRHandTracker::HandJointFlags> flags;
		flags.set_flag(XRHandTracker::HAND_JOINT_FLAG_ORIENTATION_VALID);
		flags.set_flag(XRHandTracker::HAND_JOINT_FLAG_ORIENTATION_TRACKED);
		flags.set_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_VALID);
		flags.set_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_TRACKED);
		tracker->set_hand_joint_flags(xr_joint, flags);
	}
}

bool HandTrackingServer::is_hand_tracking_available() const {
	return hand_tracking_is_available();
}

Ref<XRHandTracker> HandTrackingServer::get_left_hand_tracker() const {
	return left_hand_tracker;
}

Ref<XRHandTracker> HandTrackingServer::get_right_hand_tracker() const {
	return right_hand_tracker;
}

void HandTrackingServer::update_hand_tracking() {
	godot_hand_frame frame;
	if (!hand_tracking_get_latest_frame(frame)) {
		return;
	}

	_ensure_trackers();

	// Update left hand
	_update_hand_tracker(left_hand_tracker, frame.left_joints, frame.left_joint_count);

	// Update right hand
	_update_hand_tracker(right_hand_tracker, frame.right_joints, frame.right_joint_count);
}

HandTrackingServer::HandTrackingServer() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

HandTrackingServer::~HandTrackingServer() {
	if (initialized) {
		XRServer *xr_server = XRServer::get_singleton();
		if (xr_server) {
			if (left_hand_tracker.is_valid()) {
				xr_server->remove_tracker(left_hand_tracker);
			}
			if (right_hand_tracker.is_valid()) {
				xr_server->remove_tracker(right_hand_tracker);
			}
		}
	}

	singleton = nullptr;
}
