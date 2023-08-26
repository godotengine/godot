/**************************************************************************/
/*  openxr_huawei_controller_extension.cpp                                */
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

#include "openxr_huawei_controller_extension.h"

#include "../action_map/openxr_interaction_profile_metadata.h"

HashMap<String, bool *> OpenXRHuaweiControllerExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_HUAWEI_CONTROLLER_INTERACTION_EXTENSION_NAME] = &available;

	return request_extensions;
}

bool OpenXRHuaweiControllerExtension::is_available() {
	return available;
}

void OpenXRHuaweiControllerExtension::on_register_metadata() {
	OpenXRInteractionProfileMetadata *metadata = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(metadata);

	// Huawei controller
	metadata->register_interaction_profile("Huawei controller", "/interaction_profiles/huawei/controller", XR_HUAWEI_CONTROLLER_INTERACTION_EXTENSION_NAME);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", "", OpenXRAction::OPENXR_ACTION_POSE);

	metadata->register_io_path("/interaction_profiles/huawei/controller", "Home click", "/user/hand/left", "/user/hand/left/input/home/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Home click", "/user/hand/right", "/user/hand/right/input/home/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Back click", "/user/hand/left", "/user/hand/left/input/back/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Back click", "/user/hand/right", "/user/hand/right/input/back/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/huawei/controller", "Volume up click", "/user/hand/left", "/user/hand/left/input/volume_up/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Volume up click", "/user/hand/right", "/user/hand/right/input/volume_up/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Volume down click", "/user/hand/left", "/user/hand/left/input/volume_down/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Volume down click", "/user/hand/right", "/user/hand/right/input/volume_down/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trackpad", "/user/hand/left", "/user/hand/left/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trackpad click", "/user/hand/left", "/user/hand/left/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trackpad touch", "/user/hand/left", "/user/hand/left/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trackpad", "/user/hand/right", "/user/hand/right/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trackpad click", "/user/hand/right", "/user/hand/right/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Trackpad touch", "/user/hand/right", "/user/hand/right/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/huawei/controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/huawei/controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
}
