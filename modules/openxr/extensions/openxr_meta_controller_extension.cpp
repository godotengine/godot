/**************************************************************************/
/*  openxr_meta_controller_extension.cpp                                  */
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

#include "openxr_meta_controller_extension.h"

#include "../action_map/openxr_interaction_profile_metadata.h"
#include "../openxr_api.h"

const char *touch_controller = "/interaction_profiles/oculus/touch_controller";
const char *touch_pro_controller = "/interaction_profiles/meta/touch_pro_controller";
const char *touch_plus_controller = "/interaction_profiles/meta/touch_plus_controller";
const char *rift_cv1_controller = "/interaction_profiles/meta/touch_controller_rift_cv1";
const char *quest1_rift_s_controller = "/interaction_profiles/meta/touch_controller_quest_1_rift_s";
const char *quest2_controller = "/interaction_profiles/meta/touch_controller_quest_2";

// Vendor extension paths (promoted)
const char *touch_controller_pro_fb = "/interaction_profiles/facebook/touch_controller_pro";
const char *touch_controller_plus_fb = "/interaction_profiles/facebook/touch_controller_plus";
const char *touch_controller_plus_meta = "/interaction_profiles/meta/touch_controller_plus";

HashMap<String, bool *> OpenXRMetaControllerExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	if (p_version < XR_API_VERSION_1_1_0) {
		// Extensions where promoted in OpenXR 1.1, only include it in OpenXR 1.0.
		request_extensions[XR_FB_TOUCH_CONTROLLER_PRO_EXTENSION_NAME] = &available[META_TOUCH_PRO];
		request_extensions[XR_META_TOUCH_CONTROLLER_PLUS_EXTENSION_NAME] = &available[META_TOUCH_PLUS];
	}

	request_extensions[XR_FB_TOUCH_CONTROLLER_PROXIMITY_EXTENSION_NAME] = &available[META_TOUCH_PROXIMITY];

	return HashMap<String, bool *>(request_extensions);
}

bool OpenXRMetaControllerExtension::is_available(MetaControllers p_type) {
	return available[p_type];
}

void OpenXRMetaControllerExtension::on_register_metadata() {
	OpenXRInteractionProfileMetadata *openxr_metadata = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(openxr_metadata);

	// Note, we register controllers regardless if they are supported on the current hardware,
	// this allows the editor to configure controllers for other platforms.
	// Unsupported entries will be filtered out of the action map if applicable.

	// OpenXR 1.1 uses various new names for profile and input/output paths.
	// We use these new names in our action map and translate back IF OpenXR 1.1 is not supported.
	openxr_metadata->register_profile_rename(touch_controller_pro_fb, touch_pro_controller);
	openxr_metadata->register_profile_rename(touch_controller_plus_fb, touch_controller_plus_meta); // This was an incorrect profile path that we need to correct.
	openxr_metadata->register_profile_rename(touch_controller_plus_meta, touch_plus_controller);

	openxr_metadata->register_path_rename("/user/hand/left/input/stylus_fb/force", "/user/hand/left/input/stylus/force");
	openxr_metadata->register_path_rename("/user/hand/right/input/stylus_fb/force", "/user/hand/right/input/stylus/force");
	openxr_metadata->register_path_rename("/user/hand/left/input/trigger/proximity_fb", "/user/hand/left/input/trigger/proximity");
	openxr_metadata->register_path_rename("/user/hand/right/input/trigger/proximity_fb", "/user/hand/right/input/trigger/proximity");
	openxr_metadata->register_path_rename("/user/hand/left/output/haptic_trigger_fb", "/user/hand/left/output/haptic_trigger");
	openxr_metadata->register_path_rename("/user/hand/right/output/haptic_trigger_fb", "/user/hand/right/output/haptic_trigger");
	openxr_metadata->register_path_rename("/user/hand/left/output/haptic_thumb_fb", "/user/hand/left/output/haptic_thumb");
	openxr_metadata->register_path_rename("/user/hand/right/output/haptic_thumb_fb", "/user/hand/right/output/haptic_thumb");
	openxr_metadata->register_path_rename("/user/hand/left/input/thumb_fb/proximity_fb", "/user/hand/left/input/thumb_resting_surfaces/proximity");
	openxr_metadata->register_path_rename("/user/hand/right/input/thumb_fb/proximity_fb", "/user/hand/right/input/thumb_resting_surfaces/proximity");
	openxr_metadata->register_path_rename("/user/hand/left/input/trigger/curl_fb", "/user/hand/left/input/trigger_curl/value");
	openxr_metadata->register_path_rename("/user/hand/right/input/trigger/curl_fb", "/user/hand/right/input/trigger_curl/value");
	openxr_metadata->register_path_rename("/user/hand/left/input/trigger/slide_fb", "/user/hand/left/input/trigger_slide/value");
	openxr_metadata->register_path_rename("/user/hand/right/input/trigger/slide_fb", "/user/hand/right/input/trigger_slide/value");

	openxr_metadata->register_path_rename("/user/hand/left/input/trigger/proximity_meta", "/user/hand/left/input/trigger/proximity");
	openxr_metadata->register_path_rename("/user/hand/right/input/trigger/proximity_meta", "/user/hand/right/input/trigger/proximity");
	openxr_metadata->register_path_rename("/user/hand/left/input/thumb_meta/proximity_meta", "/user/hand/left/input/thumb_resting_surfaces/proximity");
	openxr_metadata->register_path_rename("/user/hand/right/input/thumb_meta/proximity_meta", "/user/hand/right/input/thumb_resting_surfaces/proximity");
	openxr_metadata->register_path_rename("/user/hand/left/input/trigger/curl_meta", "/user/hand/left/input/trigger_curl/value");
	openxr_metadata->register_path_rename("/user/hand/right/input/trigger/curl_meta", "/user/hand/right/input/trigger_curl/value");
	openxr_metadata->register_path_rename("/user/hand/left/input/trigger/slide_meta", "/user/hand/left/input/trigger_slide/value");
	openxr_metadata->register_path_rename("/user/hand/right/input/trigger/slide_meta", "/user/hand/right/input/trigger_slide/value");

	{ // Normal touch controller is part of the core spec and still relies on the XR_FB_touch_controller_proximity extension for these two new paths.
		const String profile_path = touch_controller;
		for (const String user_path : { "/user/hand/left", "/user/hand/right" }) {
			openxr_metadata->register_io_path(profile_path, "Trigger proximity", user_path, user_path + "/input/trigger/proximity", XR_FB_TOUCH_CONTROLLER_PROXIMITY_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumb proximity", user_path, user_path + "/input/thumb_resting_surfaces/proximity", XR_FB_TOUCH_CONTROLLER_PROXIMITY_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
		}
	}

	{ // Touch controller pro (Quest Pro)
		const String profile_path = touch_pro_controller;
		openxr_metadata->register_interaction_profile("Touch controller pro", profile_path, XR_FB_TOUCH_CONTROLLER_PRO_EXTENSION_NAME "," XR_OPENXR_1_1_NAME);
		for (const String user_path : { "/user/hand/left", "/user/hand/right" }) {
			openxr_metadata->register_io_path(profile_path, "Grip pose", user_path, user_path + "/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
			openxr_metadata->register_io_path(profile_path, "Aim pose", user_path, user_path + "/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
			openxr_metadata->register_io_path(profile_path, "Grip surface pose", user_path, user_path + "/input/grip_surface/pose", XR_EXT_PALM_POSE_EXTENSION_NAME "," XR_KHR_MAINTENANCE1_EXTENSION_NAME "," XR_OPENXR_1_1_NAME, OpenXRAction::OPENXR_ACTION_POSE);

			openxr_metadata->register_io_path(profile_path, "Pinch pose", user_path, user_path + "/input/pinch_ext/pose", XR_EXT_HAND_INTERACTION_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
			openxr_metadata->register_io_path(profile_path, "Poke pose", user_path, user_path + "/input/poke_ext/pose", XR_EXT_HAND_INTERACTION_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

			openxr_metadata->register_io_path(profile_path, "Trigger", user_path, user_path + "/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Trigger touch", user_path, user_path + "/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Trigger proximity", user_path, user_path + "/input/trigger/proximity", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Trigger curl", user_path, user_path + "/input/trigger_curl/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Trigger slide", user_path, user_path + "/input/trigger_slide/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Trigger force", user_path, user_path + "/input/trigger/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);

			openxr_metadata->register_io_path(profile_path, "Squeeze", user_path, user_path + "/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

			openxr_metadata->register_io_path(profile_path, "Thumb proximity", user_path, user_path + "/input/thumb_resting_surfaces/proximity", "", OpenXRAction::OPENXR_ACTION_BOOL);

			openxr_metadata->register_io_path(profile_path, "Thumbstick", user_path, user_path + "/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
			openxr_metadata->register_io_path(profile_path, "Thumbstick X", user_path, user_path + "/input/thumbstick/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Y", user_path, user_path + "/input/thumbstick/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Thumbstick click", user_path, user_path + "/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick touch", user_path, user_path + "/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Up", user_path, user_path + "/input/thumbstick/dpad_up", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Down", user_path, user_path + "/input/thumbstick/dpad_down", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Left", user_path, user_path + "/input/thumbstick/dpad_left", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Right", user_path, user_path + "/input/thumbstick/dpad_right", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);

			openxr_metadata->register_io_path(profile_path, "Thumbrest touch", user_path, user_path + "/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbrest force", user_path, user_path + "/input/thumbrest/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);

			openxr_metadata->register_io_path(profile_path, "Stylus force", user_path, user_path + "/input/stylus/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);

			openxr_metadata->register_io_path(profile_path, "Haptic output", user_path, user_path + "/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
			openxr_metadata->register_io_path(profile_path, "Haptic trigger output", user_path, user_path + "/output/haptic_trigger", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
			openxr_metadata->register_io_path(profile_path, "Haptic thumb output", user_path, user_path + "/output/haptic_thumb", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
		}

		openxr_metadata->register_io_path(profile_path, "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

		openxr_metadata->register_io_path(profile_path, "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "X touch", "/user/hand/left", "/user/hand/left/input/x/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "Y touch", "/user/hand/left", "/user/hand/left/input/y/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	}

	{ // Touch controller plus (Quest 3 + 3S)
		const String profile_path = touch_plus_controller;
		openxr_metadata->register_interaction_profile("Touch controller plus", profile_path, XR_META_TOUCH_CONTROLLER_PLUS_EXTENSION_NAME "," XR_OPENXR_1_1_NAME);
		for (const String user_path : { "/user/hand/left", "/user/hand/right" }) {
			openxr_metadata->register_io_path(profile_path, "Grip pose", user_path, user_path + "/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
			openxr_metadata->register_io_path(profile_path, "Aim pose", user_path, user_path + "/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
			openxr_metadata->register_io_path(profile_path, "Grip surface pose", user_path, user_path + "/input/grip_surface/pose", XR_EXT_PALM_POSE_EXTENSION_NAME "," XR_KHR_MAINTENANCE1_EXTENSION_NAME "," XR_OPENXR_1_1_NAME, OpenXRAction::OPENXR_ACTION_POSE);

			openxr_metadata->register_io_path(profile_path, "Pinch pose", user_path, user_path + "/input/pinch_ext/pose", XR_EXT_HAND_INTERACTION_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
			openxr_metadata->register_io_path(profile_path, "Poke pose", user_path, user_path + "/input/poke_ext/pose", XR_EXT_HAND_INTERACTION_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

			openxr_metadata->register_io_path(profile_path, "Trigger", user_path, user_path + "/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Trigger touch", user_path, user_path + "/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Trigger proximity", user_path, user_path + "/input/trigger/proximity", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Trigger curl", user_path, user_path + "/input/trigger_curl/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Trigger slide", user_path, user_path + "/input/trigger_slide/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Trigger force", user_path, user_path + "/input/trigger/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);

			openxr_metadata->register_io_path(profile_path, "Squeeze", user_path, user_path + "/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

			openxr_metadata->register_io_path(profile_path, "Thumb proximity", user_path, user_path + "/input/thumb_resting_surfaces/proximity", "", OpenXRAction::OPENXR_ACTION_BOOL);

			openxr_metadata->register_io_path(profile_path, "Thumbstick", user_path, user_path + "/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
			openxr_metadata->register_io_path(profile_path, "Thumbstick X", user_path, user_path + "/input/thumbstick/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Y", user_path, user_path + "/input/thumbstick/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
			openxr_metadata->register_io_path(profile_path, "Thumbstick click", user_path, user_path + "/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick touch", user_path, user_path + "/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Up", user_path, user_path + "/input/thumbstick/dpad_up", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Down", user_path, user_path + "/input/thumbstick/dpad_down", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Left", user_path, user_path + "/input/thumbstick/dpad_left", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Right", user_path, user_path + "/input/thumbstick/dpad_right", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);

			openxr_metadata->register_io_path(profile_path, "Thumbrest touch", user_path, user_path + "/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

			openxr_metadata->register_io_path(profile_path, "Haptic output", user_path, user_path + "/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
		}

		openxr_metadata->register_io_path(profile_path, "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

		openxr_metadata->register_io_path(profile_path, "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "X touch", "/user/hand/left", "/user/hand/left/input/x/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "Y touch", "/user/hand/left", "/user/hand/left/input/y/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
		openxr_metadata->register_io_path(profile_path, "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	}

	{
		// In OpenXR 1.1 we have controller profiles for Meta legacy controllers. They are all the same as the normal touch controller profile.
		openxr_metadata->register_interaction_profile("Touch controller Rift CV1", rift_cv1_controller, XR_OPENXR_1_1_NAME);
		openxr_metadata->register_interaction_profile("Touch controller Quest 1 or Rift S", quest1_rift_s_controller, XR_OPENXR_1_1_NAME);
		openxr_metadata->register_interaction_profile("Touch controller Quest 2", quest2_controller, XR_OPENXR_1_1_NAME);

		for (const String profile_path : { rift_cv1_controller, quest1_rift_s_controller, quest2_controller }) {
			for (const String user_path : { "/user/hand/left", "/user/hand/right" }) {
				openxr_metadata->register_io_path(profile_path, "Grip pose", user_path, user_path + "/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
				openxr_metadata->register_io_path(profile_path, "Aim pose", user_path, user_path + "/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
				openxr_metadata->register_io_path(profile_path, "Grip surface pose", user_path, user_path + "/input/grip_surface/pose", "", OpenXRAction::OPENXR_ACTION_POSE);

				openxr_metadata->register_io_path(profile_path, "Pinch pose", user_path, user_path + "/input/pinch_ext/pose", XR_EXT_HAND_INTERACTION_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
				openxr_metadata->register_io_path(profile_path, "Poke pose", user_path, user_path + "/input/poke_ext/pose", XR_EXT_HAND_INTERACTION_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

				openxr_metadata->register_io_path(profile_path, "Trigger", user_path, user_path + "/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
				openxr_metadata->register_io_path(profile_path, "Trigger touch", user_path, user_path + "/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

				openxr_metadata->register_io_path(profile_path, "Squeeze", user_path, user_path + "/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

				// Note, unlike the default touch controller interaction profile, these are not dependent on the XR_FB_touch_controller_proximity extension!
				openxr_metadata->register_io_path(profile_path, "Trigger proximity", user_path, user_path + "/input/trigger/proximity", "", OpenXRAction::OPENXR_ACTION_BOOL);
				openxr_metadata->register_io_path(profile_path, "Thumb proximity", user_path, user_path + "/input/thumb_resting_surfaces/proximity", "", OpenXRAction::OPENXR_ACTION_BOOL);

				openxr_metadata->register_io_path(profile_path, "Thumbstick", user_path, user_path + "/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
				openxr_metadata->register_io_path(profile_path, "Thumbstick X", user_path, user_path + "/input/thumbstick/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
				openxr_metadata->register_io_path(profile_path, "Thumbstick Y", user_path, user_path + "/input/thumbstick/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
				openxr_metadata->register_io_path(profile_path, "Thumbstick click", user_path, user_path + "/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
				openxr_metadata->register_io_path(profile_path, "Thumbstick touch", user_path, user_path + "/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
				openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Up", user_path, user_path + "/input/thumbstick/dpad_up", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
				openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Down", user_path, user_path + "/input/thumbstick/dpad_down", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
				openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Left", user_path, user_path + "/input/thumbstick/dpad_left", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);
				openxr_metadata->register_io_path(profile_path, "Thumbstick Dpad Right", user_path, user_path + "/input/thumbstick/dpad_right", XR_EXT_DPAD_BINDING_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_BOOL);

				openxr_metadata->register_io_path(profile_path, "Thumbrest touch", user_path, user_path + "/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

				openxr_metadata->register_io_path(profile_path, "Haptic output", user_path, user_path + "/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
			}

			openxr_metadata->register_io_path(profile_path, "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

			openxr_metadata->register_io_path(profile_path, "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "X touch", "/user/hand/left", "/user/hand/left/input/x/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "Y touch", "/user/hand/left", "/user/hand/left/input/y/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
			openxr_metadata->register_io_path(profile_path, "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
		}
	}
}
