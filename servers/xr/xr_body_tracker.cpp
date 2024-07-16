/**************************************************************************/
/*  xr_body_tracker.cpp                                                   */
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

#include "xr_body_tracker.h"

void XRBodyTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_has_tracking_data", "has_data"), &XRBodyTracker::set_has_tracking_data);
	ClassDB::bind_method(D_METHOD("get_has_tracking_data"), &XRBodyTracker::get_has_tracking_data);

	ClassDB::bind_method(D_METHOD("set_body_flags", "flags"), &XRBodyTracker::set_body_flags);
	ClassDB::bind_method(D_METHOD("get_body_flags"), &XRBodyTracker::get_body_flags);

	ClassDB::bind_method(D_METHOD("set_joint_flags", "joint", "flags"), &XRBodyTracker::set_joint_flags);
	ClassDB::bind_method(D_METHOD("get_joint_flags", "joint"), &XRBodyTracker::get_joint_flags);

	ClassDB::bind_method(D_METHOD("set_joint_transform", "joint", "transform"), &XRBodyTracker::set_joint_transform);
	ClassDB::bind_method(D_METHOD("get_joint_transform", "joint"), &XRBodyTracker::get_joint_transform);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "has_tracking_data", PROPERTY_HINT_NONE), "set_has_tracking_data", "get_has_tracking_data");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "body_flags", PROPERTY_HINT_FLAGS, "Upper Body,Lower Body,Hands"), "set_body_flags", "get_body_flags");

	BIND_BITFIELD_FLAG(BODY_FLAG_UPPER_BODY_SUPPORTED);
	BIND_BITFIELD_FLAG(BODY_FLAG_LOWER_BODY_SUPPORTED);
	BIND_BITFIELD_FLAG(BODY_FLAG_HANDS_SUPPORTED);

	BIND_ENUM_CONSTANT(JOINT_ROOT);
	BIND_ENUM_CONSTANT(JOINT_HIPS);
	BIND_ENUM_CONSTANT(JOINT_SPINE);
	BIND_ENUM_CONSTANT(JOINT_CHEST);
	BIND_ENUM_CONSTANT(JOINT_UPPER_CHEST);
	BIND_ENUM_CONSTANT(JOINT_NECK);
	BIND_ENUM_CONSTANT(JOINT_HEAD);
	BIND_ENUM_CONSTANT(JOINT_HEAD_TIP);
	BIND_ENUM_CONSTANT(JOINT_LEFT_SHOULDER);
	BIND_ENUM_CONSTANT(JOINT_LEFT_UPPER_ARM);
	BIND_ENUM_CONSTANT(JOINT_LEFT_LOWER_ARM);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_SHOULDER);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_UPPER_ARM);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_LOWER_ARM);
	BIND_ENUM_CONSTANT(JOINT_LEFT_UPPER_LEG);
	BIND_ENUM_CONSTANT(JOINT_LEFT_LOWER_LEG);
	BIND_ENUM_CONSTANT(JOINT_LEFT_FOOT);
	BIND_ENUM_CONSTANT(JOINT_LEFT_TOES);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_UPPER_LEG);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_LOWER_LEG);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_FOOT);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_TOES);
	BIND_ENUM_CONSTANT(JOINT_LEFT_HAND);
	BIND_ENUM_CONSTANT(JOINT_LEFT_PALM);
	BIND_ENUM_CONSTANT(JOINT_LEFT_WRIST);
	BIND_ENUM_CONSTANT(JOINT_LEFT_THUMB_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_THUMB_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_THUMB_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_THUMB_TIP);
	BIND_ENUM_CONSTANT(JOINT_LEFT_INDEX_FINGER_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_INDEX_FINGER_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_INDEX_FINGER_PHALANX_INTERMEDIATE);
	BIND_ENUM_CONSTANT(JOINT_LEFT_INDEX_FINGER_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_INDEX_FINGER_TIP);
	BIND_ENUM_CONSTANT(JOINT_LEFT_MIDDLE_FINGER_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_MIDDLE_FINGER_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_MIDDLE_FINGER_PHALANX_INTERMEDIATE);
	BIND_ENUM_CONSTANT(JOINT_LEFT_MIDDLE_FINGER_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_MIDDLE_FINGER_TIP);
	BIND_ENUM_CONSTANT(JOINT_LEFT_RING_FINGER_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_RING_FINGER_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_RING_FINGER_PHALANX_INTERMEDIATE);
	BIND_ENUM_CONSTANT(JOINT_LEFT_RING_FINGER_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_RING_FINGER_TIP);
	BIND_ENUM_CONSTANT(JOINT_LEFT_PINKY_FINGER_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_PINKY_FINGER_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_PINKY_FINGER_PHALANX_INTERMEDIATE);
	BIND_ENUM_CONSTANT(JOINT_LEFT_PINKY_FINGER_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_LEFT_PINKY_FINGER_TIP);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_HAND);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_PALM);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_WRIST);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_THUMB_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_THUMB_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_THUMB_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_THUMB_TIP);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_INDEX_FINGER_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_INDEX_FINGER_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_INDEX_FINGER_PHALANX_INTERMEDIATE);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_INDEX_FINGER_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_INDEX_FINGER_TIP);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_MIDDLE_FINGER_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_MIDDLE_FINGER_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_MIDDLE_FINGER_PHALANX_INTERMEDIATE);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_MIDDLE_FINGER_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_MIDDLE_FINGER_TIP);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_RING_FINGER_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_RING_FINGER_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_RING_FINGER_PHALANX_INTERMEDIATE);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_RING_FINGER_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_RING_FINGER_TIP);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_PINKY_FINGER_METACARPAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_PINKY_FINGER_PHALANX_PROXIMAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_PINKY_FINGER_PHALANX_INTERMEDIATE);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_PINKY_FINGER_PHALANX_DISTAL);
	BIND_ENUM_CONSTANT(JOINT_RIGHT_PINKY_FINGER_TIP);
	BIND_ENUM_CONSTANT(JOINT_MAX);

	BIND_BITFIELD_FLAG(JOINT_FLAG_ORIENTATION_VALID);
	BIND_BITFIELD_FLAG(JOINT_FLAG_ORIENTATION_TRACKED);
	BIND_BITFIELD_FLAG(JOINT_FLAG_POSITION_VALID);
	BIND_BITFIELD_FLAG(JOINT_FLAG_POSITION_TRACKED);
}

void XRBodyTracker::set_tracker_type(XRServer::TrackerType p_type) {
	ERR_FAIL_COND_MSG(p_type != XRServer::TRACKER_BODY, "XRBodyTracker must be of type TRACKER_BODY.");
}

void XRBodyTracker::set_tracker_hand(const XRPositionalTracker::TrackerHand p_hand) {
	ERR_FAIL_COND_MSG(p_hand != XRPositionalTracker::TRACKER_HAND_UNKNOWN, "XRBodyTracker cannot specify hand.");
}

void XRBodyTracker::set_has_tracking_data(bool p_has_tracking_data) {
	has_tracking_data = p_has_tracking_data;
}

bool XRBodyTracker::get_has_tracking_data() const {
	return has_tracking_data;
}

void XRBodyTracker::set_body_flags(BitField<BodyFlags> p_body_flags) {
	body_flags = p_body_flags;
}

BitField<XRBodyTracker::BodyFlags> XRBodyTracker::get_body_flags() const {
	return body_flags;
}

void XRBodyTracker::set_joint_flags(Joint p_joint, BitField<JointFlags> p_flags) {
	ERR_FAIL_INDEX(p_joint, JOINT_MAX);
	joint_flags[p_joint] = p_flags;
}

BitField<XRBodyTracker::JointFlags> XRBodyTracker::get_joint_flags(Joint p_joint) const {
	ERR_FAIL_INDEX_V(p_joint, JOINT_MAX, BitField<JointFlags>());
	return joint_flags[p_joint];
}

void XRBodyTracker::set_joint_transform(Joint p_joint, const Transform3D &p_transform) {
	ERR_FAIL_INDEX(p_joint, JOINT_MAX);
	joint_transforms[p_joint] = p_transform;
}

Transform3D XRBodyTracker::get_joint_transform(Joint p_joint) const {
	ERR_FAIL_INDEX_V(p_joint, JOINT_MAX, Transform3D());
	return joint_transforms[p_joint];
}

XRBodyTracker::XRBodyTracker() {
	type = XRServer::TRACKER_BODY;
}
