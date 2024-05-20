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

HashMap<String, bool *> OpenXRMetaControllerExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_FB_TOUCH_CONTROLLER_PROXIMITY_EXTENSION_NAME] = &available[META_TOUCH_PROXIMITY];
	request_extensions[XR_FB_TOUCH_CONTROLLER_PRO_EXTENSION_NAME] = &available[META_TOUCH_PRO];
	request_extensions[XR_META_TOUCH_CONTROLLER_PLUS_EXTENSION_NAME] = &available[META_TOUCH_PLUS];

	return request_extensions;
}

bool OpenXRMetaControllerExtension::is_available(MetaControllers p_type) {
	return available[p_type];
}

void OpenXRMetaControllerExtension::on_register_metadata() {
	OpenXRInteractionProfileMetadata *metadata = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(metadata);

	// Note, we register controllers regardless if they are supported on the current hardware.

	// Normal touch controller is part of the core spec, but we do have some extensions.
	metadata->register_io_path("/interaction_profiles/oculus/touch_controller", "Trigger proximity", "/user/hand/left", "/user/hand/left/input/trigger/proximity_fb ", "XR_FB_touch_controller_proximity", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/oculus/touch_controller", "Trigger proximity", "/user/hand/right", "/user/hand/right/input/trigger/proximity_fb ", "XR_FB_touch_controller_proximity", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/oculus/touch_controller", "Thumb proximity", "/user/hand/left", "/user/hand/left/input/thumb_fb/proximity_fb ", "XR_FB_touch_controller_proximity", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/oculus/touch_controller", "Thumb proximity", "/user/hand/right", "/user/hand/right/input/thumb_fb/proximity_fb ", "XR_FB_touch_controller_proximity", OpenXRAction::OPENXR_ACTION_BOOL);

	// Touch controller pro (Quest Pro)
	metadata->register_interaction_profile("Touch controller pro", "/interaction_profiles/facebook/touch_controller_pro", "XR_FB_touch_controller_pro");
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "X touch", "/user/hand/left", "/user/hand/left/input/x/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Y touch", "/user/hand/left", "/user/hand/left/input/y/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger touch", "/user/hand/left", "/user/hand/left/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger proximity", "/user/hand/left", "/user/hand/left/input/trigger/proximity_fb", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger curl", "/user/hand/left", "/user/hand/left/input/trigger/curl_fb", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger slide", "/user/hand/left", "/user/hand/left/input/trigger/slide_fb", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger force", "/user/hand/left", "/user/hand/left/input/trigger/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger touch", "/user/hand/right", "/user/hand/right/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger proximity", "/user/hand/right", "/user/hand/right/input/trigger/proximity_fb", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger curl", "/user/hand/right", "/user/hand/right/input/trigger/curl_fb", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger slide", "/user/hand/right", "/user/hand/right/input/trigger/slide_fb", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Trigger force", "/user/hand/right", "/user/hand/right/input/trigger/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Squeeze", "/user/hand/left", "/user/hand/left/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Squeeze", "/user/hand/right", "/user/hand/right/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumb proximity", "/user/hand/left", "/user/hand/left/input/thumb_fb/proximity_fb", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumb proximity", "/user/hand/right", "/user/hand/right/input/thumb_fb/proximity_fb", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick X", "/user/hand/left", "/user/hand/left/input/thumbstick/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick Y", "/user/hand/left", "/user/hand/left/input/thumbstick/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick X", "/user/hand/right", "/user/hand/right/input/thumbstick/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick Y", "/user/hand/right", "/user/hand/right/input/thumbstick/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbrest touch", "/user/hand/left", "/user/hand/left/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbrest force", "/user/hand/left", "/user/hand/left/input/thumbrest/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbrest touch", "/user/hand/right", "/user/hand/right/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Thumbrest force", "/user/hand/right", "/user/hand/right/input/thumbrest/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Stylus force", "/user/hand/left", "/user/hand/left/input/stylus_fb/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Stylus force", "/user/hand/right", "/user/hand/right/input/stylus_fb/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Haptic trigger output", "/user/hand/left", "/user/hand/left/output/haptic_trigger_fb", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Haptic thumb output", "/user/hand/left", "/user/hand/left/output/haptic_thumb_fb", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Haptic trigger output", "/user/hand/right", "/user/hand/right/output/haptic_trigger_fb", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/facebook/touch_controller_pro", "Haptic thumb output", "/user/hand/right", "/user/hand/right/output/haptic_thumb_fb", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// Touch controller plus (Quest 3)
	metadata->register_interaction_profile("Touch controller plus", "/interaction_profiles/meta/touch_controller_plus", "XR_META_touch_controller_plus");
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "X touch", "/user/hand/left", "/user/hand/left/input/x/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Y touch", "/user/hand/left", "/user/hand/left/input/y/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger touch", "/user/hand/left", "/user/hand/left/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger proximity", "/user/hand/left", "/user/hand/left/input/trigger/proximity_meta", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger curl", "/user/hand/left", "/user/hand/left/input/trigger/curl_meta", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger slide", "/user/hand/left", "/user/hand/left/input/trigger/slide_meta", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger force", "/user/hand/left", "/user/hand/left/input/trigger/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger touch", "/user/hand/right", "/user/hand/right/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger proximity", "/user/hand/right", "/user/hand/right/input/trigger/proximity_meta", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger curl", "/user/hand/right", "/user/hand/right/input/trigger/curl_meta", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger slide", "/user/hand/right", "/user/hand/right/input/trigger/slide_meta", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Trigger force", "/user/hand/right", "/user/hand/right/input/trigger/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Squeeze", "/user/hand/left", "/user/hand/left/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Squeeze", "/user/hand/right", "/user/hand/right/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumb proximity", "/user/hand/left", "/user/hand/left/input/thumb_meta/proximity_meta", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumb proximity", "/user/hand/right", "/user/hand/right/input/thumb_meta/proximity_meta", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick X", "/user/hand/left", "/user/hand/left/input/thumbstick/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick Y", "/user/hand/left", "/user/hand/left/input/thumbstick/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick X", "/user/hand/right", "/user/hand/right/input/thumbstick/x", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick Y", "/user/hand/right", "/user/hand/right/input/thumbstick/y", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbrest touch", "/user/hand/left", "/user/hand/left/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Thumbrest touch", "/user/hand/right", "/user/hand/right/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	metadata->register_io_path("/interaction_profiles/meta/touch_controller_plus", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
}
