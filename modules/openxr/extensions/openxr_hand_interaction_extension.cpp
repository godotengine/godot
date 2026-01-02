/**************************************************************************/
/*  openxr_hand_interaction_extension.cpp                                 */
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

#include "openxr_hand_interaction_extension.h"

#include "../action_map/openxr_interaction_profile_metadata.h"
#include "../openxr_api.h"
#include "core/config/project_settings.h"

#include <openxr/openxr.h>

OpenXRHandInteractionExtension *OpenXRHandInteractionExtension::singleton = nullptr;

OpenXRHandInteractionExtension *OpenXRHandInteractionExtension::get_singleton() {
	return singleton;
}

OpenXRHandInteractionExtension::OpenXRHandInteractionExtension() {
	singleton = this;
}

OpenXRHandInteractionExtension::~OpenXRHandInteractionExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRHandInteractionExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	// Only enable this extension when requested.
	// We still register our meta data or the action map editor will fail.
	if (GLOBAL_GET("xr/openxr/extensions/hand_interaction_profile")) {
		request_extensions[XR_EXT_HAND_INTERACTION_EXTENSION_NAME] = &available;
		request_extensions[XR_META_HAND_TRACKING_MICROGESTURES_EXTENSION_NAME] = &available;
	}

	return request_extensions;
}

bool OpenXRHandInteractionExtension::is_available() {
	return available;
}

void OpenXRHandInteractionExtension::on_register_metadata() {
	OpenXRInteractionProfileMetadata *openxr_metadata = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(openxr_metadata);

	// Hand interaction profile.
	const String profile_path = "/interaction_profiles/ext/hand_interaction_ext";
	openxr_metadata->register_interaction_profile("Hand interaction", profile_path, XR_EXT_HAND_INTERACTION_EXTENSION_NAME);
	for (const String user_path : { "/user/hand/left", "/user/hand/right" }) {
		openxr_metadata->register_io_path(profile_path, "Grip pose", user_path, user_path + "/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
		openxr_metadata->register_io_path(profile_path, "Aim pose", user_path, user_path + "/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
		openxr_metadata->register_io_path(profile_path, "Pinch pose", user_path, user_path + "/input/pinch_ext/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
		openxr_metadata->register_io_path(profile_path, "Poke pose", user_path, user_path + "/input/poke_ext/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
		openxr_metadata->register_io_path(profile_path, "Grip surface pose", user_path, user_path + "/input/grip_surface/pose", XR_EXT_PALM_POSE_EXTENSION_NAME "," XR_KHR_MAINTENANCE1_EXTENSION_NAME "," XR_OPENXR_1_1_NAME, OpenXRAction::OPENXR_ACTION_POSE);

		openxr_metadata->register_io_path(profile_path, "Pinch", user_path, user_path + "/input/pinch_ext/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		openxr_metadata->register_io_path(profile_path, "Pinch ready", user_path, user_path + "/input/pinch_ext/ready_ext", "", OpenXRAction::OPENXR_ACTION_BOOL);

		openxr_metadata->register_io_path(profile_path, "Aim activate", user_path, user_path + "/input/aim_activate_ext/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		openxr_metadata->register_io_path(profile_path, "Aim activate ready", user_path, user_path + "/input/aim_activate_ext/ready_ext", "", OpenXRAction::OPENXR_ACTION_BOOL);

		openxr_metadata->register_io_path(profile_path, "Grasp", user_path, user_path + "/input/grasp_ext/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
		openxr_metadata->register_io_path(profile_path, "Grasp ready", user_path, user_path + "/input/grasp_ext/ready_ext", "", OpenXRAction::OPENXR_ACTION_BOOL);

		// Hand tracking microgestures.
		openxr_metadata->register_io_path(profile_path, "Swipe left", user_path, user_path + "/input/swipe_left_meta/click", "XR_META_hand_tracking_microgestures", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "Swipe right", user_path, user_path + "/input/swipe_right_meta/click", "XR_META_hand_tracking_microgestures", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "Swipe forward", user_path, user_path + "/input/swipe_forward_meta/click", "XR_META_hand_tracking_microgestures", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "Swipe backward", user_path, user_path + "/input/swipe_backward_meta/click", "XR_META_hand_tracking_microgestures", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "Tap thumb", user_path, user_path + "/input/tap_thumb_meta/click", "XR_META_hand_tracking_microgestures", OpenXRAction::OPENXR_ACTION_BOOL);
	}
}
