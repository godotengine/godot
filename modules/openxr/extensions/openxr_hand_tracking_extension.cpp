/**************************************************************************/
/*  openxr_hand_tracking_extension.cpp                                    */
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

#include "openxr_hand_tracking_extension.h"

#include "../openxr_api.h"

#include "core/config/project_settings.h"
#include "core/string/print_string.h"
#include "servers/xr_server.h"

#include <openxr/openxr.h>

OpenXRHandTrackingExtension *OpenXRHandTrackingExtension::singleton = nullptr;

OpenXRHandTrackingExtension *OpenXRHandTrackingExtension::get_singleton() {
	return singleton;
}

OpenXRHandTrackingExtension::OpenXRHandTrackingExtension() {
	singleton = this;

	// Make sure this is cleared until we actually request it
	handTrackingSystemProperties.supportsHandTracking = false;
}

OpenXRHandTrackingExtension::~OpenXRHandTrackingExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRHandTrackingExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_EXT_HAND_TRACKING_EXTENSION_NAME] = &hand_tracking_ext;
	request_extensions[XR_EXT_HAND_JOINTS_MOTION_RANGE_EXTENSION_NAME] = &hand_motion_range_ext;

	return request_extensions;
}

void OpenXRHandTrackingExtension::on_instance_created(const XrInstance p_instance) {
	if (hand_tracking_ext) {
		EXT_INIT_XR_FUNC(xrCreateHandTrackerEXT);
		EXT_INIT_XR_FUNC(xrDestroyHandTrackerEXT);
		EXT_INIT_XR_FUNC(xrLocateHandJointsEXT);

		hand_tracking_ext = xrCreateHandTrackerEXT_ptr && xrDestroyHandTrackerEXT_ptr && xrLocateHandJointsEXT_ptr;
	}
}

void OpenXRHandTrackingExtension::on_session_destroyed() {
	cleanup_hand_tracking();
}

void OpenXRHandTrackingExtension::on_instance_destroyed() {
	xrCreateHandTrackerEXT_ptr = nullptr;
	xrDestroyHandTrackerEXT_ptr = nullptr;
	xrLocateHandJointsEXT_ptr = nullptr;
}

void *OpenXRHandTrackingExtension::set_system_properties_and_get_next_pointer(void *p_next_pointer) {
	if (!hand_tracking_ext) {
		// not supported...
		return p_next_pointer;
	}

	handTrackingSystemProperties = {
		XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT, // type
		p_next_pointer, // next
		false, // supportsHandTracking
	};

	return &handTrackingSystemProperties;
}

void OpenXRHandTrackingExtension::on_state_ready() {
	if (!handTrackingSystemProperties.supportsHandTracking) {
		// not supported...
		return;
	}

	// Setup our hands and reset data
	for (int i = 0; i < OPENXR_MAX_TRACKED_HANDS; i++) {
		// we'll do this later
		hand_trackers[i].is_initialized = false;
		hand_trackers[i].hand_tracker = XR_NULL_HANDLE;

		hand_trackers[i].locations.isActive = false;

		for (int j = 0; j < XR_HAND_JOINT_COUNT_EXT; j++) {
			hand_trackers[i].joint_locations[j] = { 0, { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } }, 0.0 };
			hand_trackers[i].joint_velocities[j] = { 0, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };
		}
	}
}

void OpenXRHandTrackingExtension::on_process() {
	if (!handTrackingSystemProperties.supportsHandTracking) {
		// not supported...
		return;
	}

	// process our hands
	const XrTime time = OpenXRAPI::get_singleton()->get_next_frame_time(); // This data will be used for the next frame we render
	if (time == 0) {
		// we don't have timing info yet, or we're skipping a frame...
		return;
	}

	XrResult result;

	for (int i = 0; i < OPENXR_MAX_TRACKED_HANDS; i++) {
		if (hand_trackers[i].hand_tracker == XR_NULL_HANDLE) {
			XrHandTrackerCreateInfoEXT createInfo = {
				XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT, // type
				nullptr, // next
				i == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT, // hand
				XR_HAND_JOINT_SET_DEFAULT_EXT, // handJointSet
			};

			result = xrCreateHandTrackerEXT(OpenXRAPI::get_singleton()->get_session(), &createInfo, &hand_trackers[i].hand_tracker);
			if (XR_FAILED(result)) {
				// not successful? then we do nothing.
				print_line("OpenXR: Failed to obtain hand tracking information [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
				hand_trackers[i].is_initialized = false;
			} else {
				void *next_pointer = nullptr;

				hand_trackers[i].velocities.type = XR_TYPE_HAND_JOINT_VELOCITIES_EXT;
				hand_trackers[i].velocities.next = next_pointer;
				hand_trackers[i].velocities.jointCount = XR_HAND_JOINT_COUNT_EXT;
				hand_trackers[i].velocities.jointVelocities = hand_trackers[i].joint_velocities;
				next_pointer = &hand_trackers[i].velocities;

				hand_trackers[i].locations.type = XR_TYPE_HAND_JOINT_LOCATIONS_EXT;
				hand_trackers[i].locations.next = next_pointer;
				hand_trackers[i].locations.isActive = false;
				hand_trackers[i].locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
				hand_trackers[i].locations.jointLocations = hand_trackers[i].joint_locations;

				hand_trackers[i].is_initialized = true;
			}
		}

		if (hand_trackers[i].is_initialized) {
			void *next_pointer = nullptr;

			XrHandJointsMotionRangeInfoEXT motionRangeInfo;

			if (hand_motion_range_ext) {
				motionRangeInfo.type = XR_TYPE_HAND_JOINTS_MOTION_RANGE_INFO_EXT;
				motionRangeInfo.next = next_pointer;
				motionRangeInfo.handJointsMotionRange = hand_trackers[i].motion_range;

				next_pointer = &motionRangeInfo;
			}

			XrHandJointsLocateInfoEXT locateInfo = {
				XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT, // type
				next_pointer, // next
				OpenXRAPI::get_singleton()->get_play_space(), // baseSpace
				time, // time
			};

			result = xrLocateHandJointsEXT(hand_trackers[i].hand_tracker, &locateInfo, &hand_trackers[i].locations);
			if (XR_FAILED(result)) {
				// not successful? then we do nothing.
				print_line("OpenXR: Failed to get tracking for hand", i, "[", OpenXRAPI::get_singleton()->get_error_string(result), "]");
				continue;
			}

			// For some reason an inactive controller isn't coming back as inactive but has coordinates either as NAN or very large
			const XrPosef &palm = hand_trackers[i].joint_locations[XR_HAND_JOINT_PALM_EXT].pose;
			if (
					!hand_trackers[i].locations.isActive || isnan(palm.position.x) || palm.position.x < -1000000.00 || palm.position.x > 1000000.00) {
				hand_trackers[i].locations.isActive = false; // workaround, make sure its inactive
			}
		}
	}
}

void OpenXRHandTrackingExtension::on_state_stopping() {
	// cleanup
	cleanup_hand_tracking();
}

void OpenXRHandTrackingExtension::cleanup_hand_tracking() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	for (int i = 0; i < OPENXR_MAX_TRACKED_HANDS; i++) {
		if (hand_trackers[i].hand_tracker != XR_NULL_HANDLE) {
			xrDestroyHandTrackerEXT(hand_trackers[i].hand_tracker);

			hand_trackers[i].is_initialized = false;
			hand_trackers[i].hand_tracker = XR_NULL_HANDLE;
		}
	}
}

bool OpenXRHandTrackingExtension::get_active() {
	return handTrackingSystemProperties.supportsHandTracking;
}

const OpenXRHandTrackingExtension::HandTracker *OpenXRHandTrackingExtension::get_hand_tracker(HandTrackedHands p_hand) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, nullptr);

	return &hand_trackers[p_hand];
}

XrHandJointsMotionRangeEXT OpenXRHandTrackingExtension::get_motion_range(HandTrackedHands p_hand) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, XR_HAND_JOINTS_MOTION_RANGE_MAX_ENUM_EXT);

	return hand_trackers[p_hand].motion_range;
}

void OpenXRHandTrackingExtension::set_motion_range(HandTrackedHands p_hand, XrHandJointsMotionRangeEXT p_motion_range) {
	ERR_FAIL_UNSIGNED_INDEX(p_hand, OPENXR_MAX_TRACKED_HANDS);
	hand_trackers[p_hand].motion_range = p_motion_range;
}

XrSpaceLocationFlags OpenXRHandTrackingExtension::get_hand_joint_location_flags(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, XrSpaceLocationFlags(0));
	ERR_FAIL_UNSIGNED_INDEX_V(p_joint, XR_HAND_JOINT_COUNT_EXT, XrSpaceLocationFlags(0));

	if (!hand_trackers[p_hand].is_initialized) {
		return XrSpaceLocationFlags(0);
	}

	const XrHandJointLocationEXT &location = hand_trackers[p_hand].joint_locations[p_joint];
	return location.locationFlags;
}

Quaternion OpenXRHandTrackingExtension::get_hand_joint_rotation(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, Quaternion());
	ERR_FAIL_UNSIGNED_INDEX_V(p_joint, XR_HAND_JOINT_COUNT_EXT, Quaternion());

	if (!hand_trackers[p_hand].is_initialized) {
		return Quaternion();
	}

	const XrHandJointLocationEXT &location = hand_trackers[p_hand].joint_locations[p_joint];
	return Quaternion(location.pose.orientation.x, location.pose.orientation.y, location.pose.orientation.z, location.pose.orientation.w);
}

Vector3 OpenXRHandTrackingExtension::get_hand_joint_position(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, Vector3());
	ERR_FAIL_UNSIGNED_INDEX_V(p_joint, XR_HAND_JOINT_COUNT_EXT, Vector3());

	if (!hand_trackers[p_hand].is_initialized) {
		return Vector3();
	}

	const XrHandJointLocationEXT &location = hand_trackers[p_hand].joint_locations[p_joint];
	return Vector3(location.pose.position.x, location.pose.position.y, location.pose.position.z);
}

float OpenXRHandTrackingExtension::get_hand_joint_radius(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_joint, XR_HAND_JOINT_COUNT_EXT, 0.0);

	if (!hand_trackers[p_hand].is_initialized) {
		return 0.0;
	}

	return hand_trackers[p_hand].joint_locations[p_joint].radius;
}

XrSpaceVelocityFlags OpenXRHandTrackingExtension::get_hand_joint_velocity_flags(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, XrSpaceVelocityFlags(0));
	ERR_FAIL_UNSIGNED_INDEX_V(p_joint, XR_HAND_JOINT_COUNT_EXT, XrSpaceVelocityFlags(0));

	if (!hand_trackers[p_hand].is_initialized) {
		return XrSpaceVelocityFlags(0);
	}

	const XrHandJointVelocityEXT &velocity = hand_trackers[p_hand].joint_velocities[p_joint];
	return velocity.velocityFlags;
}

Vector3 OpenXRHandTrackingExtension::get_hand_joint_linear_velocity(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, Vector3());
	ERR_FAIL_UNSIGNED_INDEX_V(p_joint, XR_HAND_JOINT_COUNT_EXT, Vector3());

	if (!hand_trackers[p_hand].is_initialized) {
		return Vector3();
	}

	const XrHandJointVelocityEXT &velocity = hand_trackers[p_hand].joint_velocities[p_joint];
	return Vector3(velocity.linearVelocity.x, velocity.linearVelocity.y, velocity.linearVelocity.z);
}

Vector3 OpenXRHandTrackingExtension::get_hand_joint_angular_velocity(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, Vector3());
	ERR_FAIL_UNSIGNED_INDEX_V(p_joint, XR_HAND_JOINT_COUNT_EXT, Vector3());

	if (!hand_trackers[p_hand].is_initialized) {
		return Vector3();
	}

	const XrHandJointVelocityEXT &velocity = hand_trackers[p_hand].joint_velocities[p_joint];
	return Vector3(velocity.angularVelocity.x, velocity.angularVelocity.y, velocity.angularVelocity.z);
}
