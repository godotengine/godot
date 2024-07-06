/**************************************************************************/
/*  openxr_htc_vive_tracker_extension.cpp                                 */
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

#include "openxr_htc_vive_tracker_extension.h"

#include "../action_map/openxr_interaction_profile_metadata.h"

#include "core/string/print_string.h"

HashMap<String, bool *> OpenXRHTCViveTrackerExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME] = &available;

	return request_extensions;
}

PackedStringArray OpenXRHTCViveTrackerExtension::get_suggested_tracker_names() {
	PackedStringArray arr = {
		"/user/vive_tracker_htcx/role/handheld_object",
		"/user/vive_tracker_htcx/role/left_foot",
		"/user/vive_tracker_htcx/role/right_foot",
		"/user/vive_tracker_htcx/role/left_shoulder",
		"/user/vive_tracker_htcx/role/right_shoulder",
		"/user/vive_tracker_htcx/role/left_elbow",
		"/user/vive_tracker_htcx/role/right_elbow",
		"/user/vive_tracker_htcx/role/left_knee",
		"/user/vive_tracker_htcx/role/right_knee",
		"/user/vive_tracker_htcx/role/waist",
		"/user/vive_tracker_htcx/role/chest",
		"/user/vive_tracker_htcx/role/camera",
		"/user/vive_tracker_htcx/role/keyboard",
	};
	return arr;
}

bool OpenXRHTCViveTrackerExtension::is_available() {
	return available;
}

void OpenXRHTCViveTrackerExtension::on_register_metadata() {
	OpenXRInteractionProfileMetadata *metadata = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(metadata);

	// register_top_level_path("Handheld object tracker", "/user/vive_tracker_htcx/role/handheld_object", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Left foot tracker", "/user/vive_tracker_htcx/role/left_foot", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Right foot tracker", "/user/vive_tracker_htcx/role/right_foot", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Left shoulder tracker", "/user/vive_tracker_htcx/role/left_shoulder", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Right shoulder tracker", "/user/vive_tracker_htcx/role/right_shoulder", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Left elbow tracker", "/user/vive_tracker_htcx/role/left_elbow", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Right elbow tracker", "/user/vive_tracker_htcx/role/right_elbow", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Left knee tracker", "/user/vive_tracker_htcx/role/left_knee", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Right knee tracker", "/user/vive_tracker_htcx/role/right_knee", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Waist tracker", "/user/vive_tracker_htcx/role/waist", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Chest tracker", "/user/vive_tracker_htcx/role/chest", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Camera tracker", "/user/vive_tracker_htcx/role/camera", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_top_level_path("Keyboard tracker", "/user/vive_tracker_htcx/role/keyboard", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);

	// HTC Vive tracker
	// Interestingly enough trackers don't have buttons or inputs, yet these are defined in the spec.
	// I think this can be supported through attachments on the trackers.
	metadata->register_interaction_profile("HTC Vive tracker", "/interaction_profiles/htc/vive_tracker_htcx", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	// metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);

	// metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);

	// metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
}

bool OpenXRHTCViveTrackerExtension::on_event_polled(const XrEventDataBuffer &event) {
	switch (event.type) {
		case XR_TYPE_EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX: {
			// Investigate if we need to do more here
			print_verbose("OpenXR EVENT: VIVE tracker connected");

			return true;
		} break;
		default: {
			return false;
		} break;
	}
}
