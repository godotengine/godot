/**************************************************************************/
/*  openxr_pico_controller_extension.cpp                                  */
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

#include "openxr_pico_controller_extension.h"

#include "../action_map/openxr_interaction_profile_metadata.h"

HashMap<String, bool *> OpenXRPicoControllerExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	// Note, this used to be XR_PICO_controller_interaction but that has since been retired
	// and was never part of the OpenXX specification.
	// All PICO devices should be updated to an OS supporting the official extension.

	request_extensions[XR_BD_CONTROLLER_INTERACTION_EXTENSION_NAME] = &available;

	return request_extensions;
}

bool OpenXRPicoControllerExtension::is_available() {
	return available;
}

void OpenXRPicoControllerExtension::on_register_metadata() {
	OpenXRInteractionProfileMetadata *metadata = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(metadata);

	// Make sure we switch to our new name.
	metadata->register_profile_rename("/interaction_profiles/pico/neo3_controller", "/interaction_profiles/bytedance/pico_neo3_controller");

	// Pico neo 3 controller.
	metadata->register_interaction_profile("Pico Neo3 controller", "/interaction_profiles/bytedance/pico_neo3_controller", XR_BD_CONTROLLER_INTERACTION_EXTENSION_NAME);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Menu click", "/user/hand/right", "/user/hand/right/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "System click", "/user/hand/left", "/user/hand/left/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "X touch", "/user/hand/left", "/user/hand/left/input/x/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Y touch", "/user/hand/left", "/user/hand/left/input/y/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Trigger touch", "/user/hand/left", "/user/hand/left/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Trigger touch", "/user/hand/right", "/user/hand/right/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Squeeze", "/user/hand/left", "/user/hand/left/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Squeeze", "/user/hand/right", "/user/hand/right/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/bytedance/pico_neo3_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// Pico 4 controller.
	metadata->register_interaction_profile("Pico 4 controller", "/interaction_profiles/bytedance/pico4_controller", XR_BD_CONTROLLER_INTERACTION_EXTENSION_NAME);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	// Note, no menu on right controller!
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "System click", "/user/hand/left", "/user/hand/left/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "X touch", "/user/hand/left", "/user/hand/left/input/x/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Y touch", "/user/hand/left", "/user/hand/left/input/y/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Trigger touch", "/user/hand/left", "/user/hand/left/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Trigger touch", "/user/hand/right", "/user/hand/right/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Squeeze", "/user/hand/left", "/user/hand/left/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Squeeze", "/user/hand/right", "/user/hand/right/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/bytedance/pico4_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
}
