/**************************************************************************/
/*  openxr_htc_controller_extension.cpp                                   */
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

#include "openxr_htc_controller_extension.h"

#include "../action_map/openxr_interaction_profile_metadata.h"

HashMap<String, bool *> OpenXRHTCControllerExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_HTC_VIVE_COSMOS_CONTROLLER_INTERACTION_EXTENSION_NAME] = &available[HTC_VIVE_COSMOS];
	request_extensions[XR_HTC_VIVE_FOCUS3_CONTROLLER_INTERACTION_EXTENSION_NAME] = &available[HTC_VIVE_FOCUS3];

	return request_extensions;
}

bool OpenXRHTCControllerExtension::is_available(HTCControllers p_type) {
	return available[p_type];
}

void OpenXRHTCControllerExtension::on_register_metadata() {
	OpenXRInteractionProfileMetadata *metadata = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(metadata);

	// HTC Vive Cosmos controller
	metadata->register_interaction_profile("Vive Cosmos controller", "/interaction_profiles/htc/vive_cosmos_controller", XR_HTC_VIVE_COSMOS_CONTROLLER_INTERACTION_EXTENSION_NAME);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Shoulder click", "/user/hand/left", "/user/hand/left/input/shoulder/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Shoulder click", "/user/hand/right", "/user/hand/right/input/shoulder/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Squeeze click", "/user/hand/left", "/user/hand/left/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Squeeze click", "/user/hand/right", "/user/hand/right/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// HTC Vive Focus 3 controller
	metadata->register_interaction_profile("Vive Focus 3 controller", "/interaction_profiles/htc/vive_focus3_controller", XR_HTC_VIVE_FOCUS3_CONTROLLER_INTERACTION_EXTENSION_NAME);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger touch", "/user/hand/left", "/user/hand/left/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger touch", "/user/hand/right", "/user/hand/right/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Squeeze click", "/user/hand/left", "/user/hand/left/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Squeeze touch", "/user/hand/left", "/user/hand/left/input/squeeze/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Squeeze click", "/user/hand/right", "/user/hand/right/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Squeeze touch", "/user/hand/right", "/user/hand/right/input/squeeze/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbrest touch", "/user/hand/left", "/user/hand/left/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbrest touch", "/user/hand/right", "/user/hand/right/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
}
