/**************************************************************************/
/*  hand_tracker_3d.cpp                                                   */
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

#include "hand_tracker_3d.h"

#include "../hand_tracking_server.h"

void HandTracker3D::_bind_methods() {
	// Hand selection
	ClassDB::bind_method(D_METHOD("set_hand", "hand"), &HandTracker3D::set_hand);
	ClassDB::bind_method(D_METHOD("get_hand"), &HandTracker3D::get_hand);

	// Joint selection
	ClassDB::bind_method(D_METHOD("set_joint", "joint"), &HandTracker3D::set_joint);
	ClassDB::bind_method(D_METHOD("get_joint"), &HandTracker3D::get_joint);

	// Tracking options
	ClassDB::bind_method(D_METHOD("set_track_position", "enable"), &HandTracker3D::set_track_position);
	ClassDB::bind_method(D_METHOD("get_track_position"), &HandTracker3D::get_track_position);

	ClassDB::bind_method(D_METHOD("set_track_rotation", "enable"), &HandTracker3D::set_track_rotation);
	ClassDB::bind_method(D_METHOD("get_track_rotation"), &HandTracker3D::get_track_rotation);

	// Visibility
	ClassDB::bind_method(D_METHOD("set_hide_when_invalid", "enable"), &HandTracker3D::set_hide_when_invalid);
	ClassDB::bind_method(D_METHOD("get_hide_when_invalid"), &HandTracker3D::get_hide_when_invalid);

	// Smoothing
	ClassDB::bind_method(D_METHOD("set_smoothing", "smoothing"), &HandTracker3D::set_smoothing);
	ClassDB::bind_method(D_METHOD("get_smoothing"), &HandTracker3D::get_smoothing);

	// Status
	ClassDB::bind_method(D_METHOD("is_tracking_valid"), &HandTracker3D::is_tracking_valid);
	ClassDB::bind_method(D_METHOD("get_tracker"), &HandTracker3D::get_tracker);

	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hand", PROPERTY_HINT_ENUM, "Left,Right"), "set_hand", "get_hand");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "joint", PROPERTY_HINT_ENUM, "Palm,Wrist,Thumb Metacarpal,Thumb Proximal,Thumb Distal,Thumb Tip,Index Metacarpal,Index Proximal,Index Intermediate,Index Distal,Index Tip,Middle Metacarpal,Middle Proximal,Middle Intermediate,Middle Distal,Middle Tip,Ring Metacarpal,Ring Proximal,Ring Intermediate,Ring Distal,Ring Tip,Pinky Metacarpal,Pinky Proximal,Pinky Intermediate,Pinky Distal,Pinky Tip"), "set_joint", "get_joint");

	ADD_GROUP("Tracking", "track_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "track_position"), "set_track_position", "get_track_position");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "track_rotation"), "set_track_rotation", "get_track_rotation");

	ADD_GROUP("Behavior", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hide_when_invalid"), "set_hide_when_invalid", "get_hide_when_invalid");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_smoothing", "get_smoothing");

	// Enums
	BIND_ENUM_CONSTANT(HAND_LEFT);
	BIND_ENUM_CONSTANT(HAND_RIGHT);
	BIND_ENUM_CONSTANT(HAND_MAX);
}

void HandTracker3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_tracker_reference();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			tracker.unref();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			_update_tracking();
		} break;
	}
}

void HandTracker3D::_update_tracker_reference() {
	HandTrackingServer *server = HandTrackingServer::get_singleton();
	if (!server) {
		return;
	}

	// Get the appropriate hand tracker
	if (hand == HAND_LEFT) {
		tracker = server->get_left_hand_tracker();
	} else {
		tracker = server->get_right_hand_tracker();
	}
}

void HandTracker3D::_update_tracking() {
	if (tracker.is_null()) {
		_update_tracker_reference();
		if (tracker.is_null()) {
			return;
		}
	}

	// Check if we have tracking data
	if (!tracker->get_has_tracking_data()) {
		if (hide_when_invalid && is_visible_in_tree()) {
			was_visible = true;
			set_visible(false);
		}
		return;
	}

	// Restore visibility if we had hidden the node
	if (hide_when_invalid && !is_visible_in_tree() && was_visible) {
		set_visible(true);
	}

	// Get joint flags
	BitField<XRHandTracker::HandJointFlags> flags = tracker->get_hand_joint_flags(joint);

	// Check if position/orientation are valid
	bool position_valid = flags.has_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_VALID);
	bool orientation_valid = flags.has_flag(XRHandTracker::HAND_JOINT_FLAG_ORIENTATION_VALID);

	if (!position_valid && !orientation_valid) {
		if (hide_when_invalid && is_visible_in_tree()) {
			was_visible = true;
			set_visible(false);
		}
		return;
	}

	// Restore visibility
	if (hide_when_invalid && !is_visible_in_tree() && was_visible) {
		set_visible(true);
	}

	// Get joint transform
	Transform3D joint_transform = tracker->get_hand_joint_transform(joint);

	// Build target transform
	Transform3D new_transform = get_transform();

	if (track_position && position_valid) {
		target_transform.origin = joint_transform.origin;
	}

	if (track_rotation && orientation_valid) {
		target_transform.basis = joint_transform.basis;
	}

	// Apply smoothing if enabled
	if (smoothing > 0.0f) {
		// Interpolate position
		if (track_position) {
			new_transform.origin = new_transform.origin.lerp(target_transform.origin, 1.0f - smoothing);
		}

		// Interpolate rotation
		if (track_rotation) {
			Quaternion current_rot = Quaternion(new_transform.basis);
			Quaternion target_rot = Quaternion(target_transform.basis);
			Quaternion smoothed_rot = current_rot.slerp(target_rot, 1.0f - smoothing);
			new_transform.basis = Basis(smoothed_rot);
		}
	} else {
		// No smoothing - direct assignment
		if (track_position) {
			new_transform.origin = target_transform.origin;
		}
		if (track_rotation) {
			new_transform.basis = target_transform.basis;
		}
	}

	set_transform(new_transform);
}

void HandTracker3D::set_hand(Hand p_hand) {
	if (hand != p_hand) {
		hand = p_hand;
		_update_tracker_reference();
	}
}

HandTracker3D::Hand HandTracker3D::get_hand() const {
	return hand;
}

void HandTracker3D::set_joint(XRHandTracker::HandJoint p_joint) {
	joint = p_joint;
}

XRHandTracker::HandJoint HandTracker3D::get_joint() const {
	return joint;
}

void HandTracker3D::set_track_position(bool p_enable) {
	track_position = p_enable;
}

bool HandTracker3D::get_track_position() const {
	return track_position;
}

void HandTracker3D::set_track_rotation(bool p_enable) {
	track_rotation = p_enable;
}

bool HandTracker3D::get_track_rotation() const {
	return track_rotation;
}

void HandTracker3D::set_hide_when_invalid(bool p_enable) {
	hide_when_invalid = p_enable;
}

bool HandTracker3D::get_hide_when_invalid() const {
	return hide_when_invalid;
}

void HandTracker3D::set_smoothing(float p_smoothing) {
	smoothing = CLAMP(p_smoothing, 0.0f, 1.0f);
}

float HandTracker3D::get_smoothing() const {
	return smoothing;
}

bool HandTracker3D::is_tracking_valid() const {
	if (tracker.is_null()) {
		return false;
	}

	if (!tracker->get_has_tracking_data()) {
		return false;
	}

	BitField<XRHandTracker::HandJointFlags> flags = tracker->get_hand_joint_flags(joint);
	return flags.has_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_VALID) ||
		   flags.has_flag(XRHandTracker::HAND_JOINT_FLAG_ORIENTATION_VALID);
}

Ref<XRHandTracker> HandTracker3D::get_tracker() const {
	return tracker;
}

HandTracker3D::HandTracker3D() {
	set_process_internal(true);
}

HandTracker3D::~HandTracker3D() {
}
