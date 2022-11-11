/*************************************************************************/
/*  openxr_interaction_profile_meta_data.cpp                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "openxr_interaction_profile_meta_data.h"

#include <openxr/openxr.h>

OpenXRInteractionProfileMetaData *OpenXRInteractionProfileMetaData::singleton = nullptr;

OpenXRInteractionProfileMetaData::OpenXRInteractionProfileMetaData() {
	singleton = this;

	_register_core_metadata();
}

OpenXRInteractionProfileMetaData::~OpenXRInteractionProfileMetaData() {
	singleton = nullptr;
}

void OpenXRInteractionProfileMetaData::_bind_methods() {
	ClassDB::bind_method(D_METHOD("register_top_level_path", "display_name", "openxr_path", "openxr_extension_name"), &OpenXRInteractionProfileMetaData::register_top_level_path);
	ClassDB::bind_method(D_METHOD("register_interaction_profile", "display_name", "openxr_path", "openxr_extension_name"), &OpenXRInteractionProfileMetaData::register_interaction_profile);
	ClassDB::bind_method(D_METHOD("register_io_path", "interaction_profile", "display_name", "toplevel_path", "openxr_path", "openxr_extension_name", "action_type"), &OpenXRInteractionProfileMetaData::register_io_path);
}

void OpenXRInteractionProfileMetaData::register_top_level_path(const String &p_display_name, const String &p_openxr_path, const String &p_openxr_extension_name) {
	ERR_FAIL_COND_MSG(has_top_level_path(p_openxr_path), p_openxr_path + " had already been registered");

	TopLevelPath new_toplevel_path = {
		p_display_name,
		p_openxr_path,
		p_openxr_extension_name
	};

	top_level_paths.push_back(new_toplevel_path);
}

void OpenXRInteractionProfileMetaData::register_interaction_profile(const String &p_display_name, const String &p_openxr_path, const String &p_openxr_extension_name) {
	ERR_FAIL_COND_MSG(has_interaction_profile(p_openxr_path), p_openxr_path + " has already been registered");

	InteractionProfile new_profile;
	new_profile.display_name = p_display_name;
	new_profile.openxr_path = p_openxr_path;
	new_profile.openxr_extension_name = p_openxr_extension_name;

	interaction_profiles.push_back(new_profile);
}

void OpenXRInteractionProfileMetaData::register_io_path(const String &p_interaction_profile, const String &p_display_name, const String &p_toplevel_path, const String &p_openxr_path, const String &p_openxr_extension_name, OpenXRAction::ActionType p_action_type) {
	ERR_FAIL_COND_MSG(!has_interaction_profile(p_interaction_profile), "Unknown interaction profile " + p_interaction_profile);
	ERR_FAIL_COND_MSG(!has_top_level_path(p_toplevel_path), "Unknown top level path " + p_toplevel_path);

	for (int i = 0; i < interaction_profiles.size(); i++) {
		if (interaction_profiles[i].openxr_path == p_interaction_profile) {
			ERR_FAIL_COND_MSG(interaction_profiles[i].has_io_path(p_openxr_path), p_interaction_profile + " already has io path " + p_openxr_path + " registered!");

			IOPath new_io_path = {
				p_display_name,
				p_toplevel_path,
				p_openxr_path,
				p_openxr_extension_name,
				p_action_type
			};

			interaction_profiles.ptrw()[i].io_paths.push_back(new_io_path);
		}
	}
}

bool OpenXRInteractionProfileMetaData::has_top_level_path(const String p_openxr_path) const {
	for (int i = 0; i < top_level_paths.size(); i++) {
		if (top_level_paths[i].openxr_path == p_openxr_path) {
			return true;
		}
	}

	return false;
}

String OpenXRInteractionProfileMetaData::get_top_level_name(const String p_openxr_path) const {
	for (int i = 0; i < top_level_paths.size(); i++) {
		if (top_level_paths[i].openxr_path == p_openxr_path) {
			return top_level_paths[i].display_name;
		}
	}

	return String();
}

String OpenXRInteractionProfileMetaData::get_top_level_extension(const String p_openxr_path) const {
	for (int i = 0; i < top_level_paths.size(); i++) {
		if (top_level_paths[i].openxr_path == p_openxr_path) {
			return top_level_paths[i].openxr_extension_name;
		}
	}

	return XR_PATH_UNSUPPORTED_NAME;
}

bool OpenXRInteractionProfileMetaData::has_interaction_profile(const String p_openxr_path) const {
	for (int i = 0; i < interaction_profiles.size(); i++) {
		if (interaction_profiles[i].openxr_path == p_openxr_path) {
			return true;
		}
	}

	return false;
}

String OpenXRInteractionProfileMetaData::get_interaction_profile_extension(const String p_openxr_path) const {
	for (int i = 0; i < interaction_profiles.size(); i++) {
		if (interaction_profiles[i].openxr_path == p_openxr_path) {
			return interaction_profiles[i].openxr_extension_name;
		}
	}

	return XR_PATH_UNSUPPORTED_NAME;
}

const OpenXRInteractionProfileMetaData::InteractionProfile *OpenXRInteractionProfileMetaData::get_profile(const String p_openxr_path) const {
	for (int i = 0; i < interaction_profiles.size(); i++) {
		if (interaction_profiles[i].openxr_path == p_openxr_path) {
			return &interaction_profiles[i];
		}
	}

	return nullptr;
}

bool OpenXRInteractionProfileMetaData::InteractionProfile::has_io_path(const String p_io_path) const {
	for (int i = 0; i < io_paths.size(); i++) {
		if (io_paths[i].openxr_path == p_io_path) {
			return true;
		}
	}

	return false;
}

const OpenXRInteractionProfileMetaData::IOPath *OpenXRInteractionProfileMetaData::InteractionProfile::get_io_path(const String p_io_path) const {
	for (int i = 0; i < io_paths.size(); i++) {
		if (io_paths[i].openxr_path == p_io_path) {
			return &io_paths[i];
		}
	}

	return nullptr;
}

const OpenXRInteractionProfileMetaData::IOPath *OpenXRInteractionProfileMetaData::get_io_path(const String p_interaction_profile, const String p_io_path) const {
	const OpenXRInteractionProfileMetaData::InteractionProfile *profile = get_profile(p_interaction_profile);
	if (profile != nullptr) {
		return profile->get_io_path(p_io_path);
	}

	return nullptr;
}

PackedStringArray OpenXRInteractionProfileMetaData::get_interaction_profile_paths() const {
	PackedStringArray arr;

	for (int i = 0; i < interaction_profiles.size(); i++) {
		arr.push_back(interaction_profiles[i].openxr_path);
	}

	return arr;
}

void OpenXRInteractionProfileMetaData::_register_core_metadata() {
	// Note, currently we add definitions that belong in extensions.
	// Extensions are registered when our OpenXRAPI is instantiated
	// however this does not happen in the editor.
	// We are changing this in another PR, once that is accepted we
	// can make the changes to move code into extensions where needed.

	// Note that we'll make an exception for XR_EXT_palm_pose, which is used everywhere

	// Our core toplevel paths
	register_top_level_path("Left hand controller", "/user/hand/left", "");
	register_top_level_path("Right hand controller", "/user/hand/right", "");
	register_top_level_path("Head", "/user/head", "");
	register_top_level_path("Gamepad", "/user/gamepad", "");
	register_top_level_path("Treadmill", "/user/treadmill", "");

	// TODO move this into OpenXRHTCViveTrackerExtension once this is supported.
	// register_top_level_path("Handheld object tracker", "/user/vive_tracker_htcx/role/handheld_object", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Left foot tracker", "/user/vive_tracker_htcx/role/left_foot", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Right foot tracker", "/user/vive_tracker_htcx/role/right_foot", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Left shoulder tracker", "/user/vive_tracker_htcx/role/left_shoulder", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Right shoulder tracker", "/user/vive_tracker_htcx/role/right_shoulder", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Left elbow tracker", "/user/vive_tracker_htcx/role/left_elbow", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Right elbow tracker", "/user/vive_tracker_htcx/role/right_elbow", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Left knee tracker", "/user/vive_tracker_htcx/role/left_knee", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Right knee tracker", "/user/vive_tracker_htcx/role/right_knee", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Waist tracker", "/user/vive_tracker_htcx/role/waist", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Chest tracker", "/user/vive_tracker_htcx/role/chest", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Camera tracker", "/user/vive_tracker_htcx/role/camera", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_top_level_path("Keyboard tracker", "/user/vive_tracker_htcx/role/keyboard", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);

	// Fallback Khronos simple controller
	register_interaction_profile("Simple controller", "/interaction_profiles/khr/simple_controller", "");
	register_io_path("/interaction_profiles/khr/simple_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/khr/simple_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/khr/simple_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/khr/simple_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/khr/simple_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/khr/simple_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/khr/simple_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/khr/simple_controller", "Menu click", "/user/hand/right", "/user/hand/right/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/khr/simple_controller", "Select click", "/user/hand/left", "/user/hand/left/input/select/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/khr/simple_controller", "Select click", "/user/hand/right", "/user/hand/right/input/select/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/khr/simple_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/khr/simple_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// Original HTC Vive wands
	register_interaction_profile("HTC Vive wand", "/interaction_profiles/htc/vive_controller", "");
	register_io_path("/interaction_profiles/htc/vive_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/htc/vive_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_controller", "Menu click", "/user/hand/right", "/user/hand/right/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_controller", "System click", "/user/hand/left", "/user/hand/left/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_controller", "Squeeze click", "/user/hand/left", "/user/hand/left/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_controller", "Squeeze click", "/user/hand/right", "/user/hand/right/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_controller", "Trackpad", "/user/hand/left", "/user/hand/left/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_controller", "Trackpad click", "/user/hand/left", "/user/hand/left/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_controller", "Trackpad touch", "/user/hand/left", "/user/hand/left/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_controller", "Trackpad", "/user/hand/right", "/user/hand/right/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_controller", "Trackpad click", "/user/hand/right", "/user/hand/right/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_controller", "Trackpad touch", "/user/hand/right", "/user/hand/right/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// Microsoft motion controller (original WMR controllers)
	register_interaction_profile("MS Motion controller", "/interaction_profiles/microsoft/motion_controller", "");
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/microsoft/motion_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Menu click", "/user/hand/right", "/user/hand/right/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/microsoft/motion_controller", "Squeeze click", "/user/hand/left", "/user/hand/left/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Squeeze click", "/user/hand/right", "/user/hand/right/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/microsoft/motion_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trackpad", "/user/hand/left", "/user/hand/left/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trackpad click", "/user/hand/left", "/user/hand/left/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trackpad touch", "/user/hand/left", "/user/hand/left/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trackpad", "/user/hand/right", "/user/hand/right/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trackpad click", "/user/hand/right", "/user/hand/right/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Trackpad touch", "/user/hand/right", "/user/hand/right/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/microsoft/motion_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/microsoft/motion_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// Meta touch controller (original touch controllers, Quest 1 and Quest 2 controllers)
	register_interaction_profile("Touch controller", "/interaction_profiles/oculus/touch_controller", "");
	register_io_path("/interaction_profiles/oculus/touch_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/oculus/touch_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/oculus/touch_controller", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "X touch", "/user/hand/left", "/user/hand/left/input/x/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Y touch", "/user/hand/left", "/user/hand/left/input/y/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/oculus/touch_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Trigger touch", "/user/hand/left", "/user/hand/left/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Trigger touch", "/user/hand/right", "/user/hand/right/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/oculus/touch_controller", "Squeeze", "/user/hand/left", "/user/hand/left/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Squeeze", "/user/hand/right", "/user/hand/right/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	register_io_path("/interaction_profiles/oculus/touch_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/oculus/touch_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/oculus/touch_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// Valve Index controller
	register_interaction_profile("Index controller", "/interaction_profiles/valve/index_controller", "");
	register_io_path("/interaction_profiles/valve/index_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/valve/index_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/valve/index_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/valve/index_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/valve/index_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/valve/index_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/valve/index_controller", "System click", "/user/hand/left", "/user/hand/left/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/valve/index_controller", "A click", "/user/hand/left", "/user/hand/left/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "A touch", "/user/hand/left", "/user/hand/left/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "A touch", "/user/hand/right", "/user/hand/right/input/a/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "B click", "/user/hand/left", "/user/hand/left/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "B touch", "/user/hand/left", "/user/hand/left/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "B touch", "/user/hand/right", "/user/hand/right/input/b/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/valve/index_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/valve/index_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "Trigger touch", "/user/hand/left", "/user/hand/left/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/valve/index_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "Trigger touch", "/user/hand/right", "/user/hand/right/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/valve/index_controller", "Squeeze", "/user/hand/left", "/user/hand/left/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/valve/index_controller", "Squeeze", "/user/hand/right", "/user/hand/right/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	register_io_path("/interaction_profiles/valve/index_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/valve/index_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/valve/index_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/valve/index_controller", "Trackpad", "/user/hand/left", "/user/hand/left/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/valve/index_controller", "Trackpad force", "/user/hand/left", "/user/hand/left/input/trackpad/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/valve/index_controller", "Trackpad touch", "/user/hand/left", "/user/hand/left/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/valve/index_controller", "Trackpad", "/user/hand/right", "/user/hand/right/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/valve/index_controller", "Trackpad force", "/user/hand/right", "/user/hand/right/input/trackpad/force", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/valve/index_controller", "Trackpad touch", "/user/hand/right", "/user/hand/right/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/valve/index_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/valve/index_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// HP MR controller (newer G2 controllers)
	// TODO move this into an extension once this is supported.
	register_interaction_profile("HPMR controller", "/interaction_profiles/hp/mixed_reality_controller", XR_EXT_HP_MIXED_REALITY_CONTROLLER_EXTENSION_NAME);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Menu click", "/user/hand/right", "/user/hand/right/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Squeeze", "/user/hand/left", "/user/hand/left/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Squeeze", "/user/hand/right", "/user/hand/right/input/squeeze/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/hp/mixed_reality_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// Samsung Odyssey controller
	// TODO move this into an extension once this is supported.
	register_interaction_profile("Samsung Odyssey controller", "/interaction_profiles/samsung/odyssey_controller", XR_EXT_SAMSUNG_ODYSSEY_CONTROLLER_EXTENSION_NAME);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Menu click", "/user/hand/right", "/user/hand/right/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Squeeze click", "/user/hand/left", "/user/hand/left/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Squeeze click", "/user/hand/right", "/user/hand/right/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trackpad", "/user/hand/left", "/user/hand/left/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trackpad click", "/user/hand/left", "/user/hand/left/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trackpad touch", "/user/hand/left", "/user/hand/left/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trackpad", "/user/hand/right", "/user/hand/right/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trackpad click", "/user/hand/right", "/user/hand/right/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Trackpad touch", "/user/hand/right", "/user/hand/right/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/samsung/odyssey_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// HTC Vive Cosmos controller
	// TODO move this into an extension once this is supported.
	register_interaction_profile("Vive Cosmos controller", "/interaction_profiles/htc/vive_cosmos_controller", XR_HTC_VIVE_COSMOS_CONTROLLER_INTERACTION_EXTENSION_NAME);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Shoulder click", "/user/hand/left", "/user/hand/left/input/shoulder/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Shoulder click", "/user/hand/right", "/user/hand/right/input/shoulder/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Squeeze click", "/user/hand/left", "/user/hand/left/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Squeeze click", "/user/hand/right", "/user/hand/right/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_cosmos_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// HTC Vive Focus 3 controller
	// TODO move this into an extension once this is supported.
	register_interaction_profile("Vive Focus 3 controller", "/interaction_profiles/htc/vive_focus3_controller", XR_HTC_VIVE_FOCUS3_CONTROLLER_INTERACTION_EXTENSION_NAME);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", XR_EXT_PALM_POSE_EXTENSION_NAME, OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Menu click", "/user/hand/left", "/user/hand/left/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "System click", "/user/hand/right", "/user/hand/right/input/system/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "X click", "/user/hand/left", "/user/hand/left/input/x/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Y click", "/user/hand/left", "/user/hand/left/input/y/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "B click", "/user/hand/right", "/user/hand/right/input/b/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger touch", "/user/hand/left", "/user/hand/left/input/trigger/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Trigger touch", "/user/hand/right", "/user/hand/right/input/trigger/touch	", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Squeeze click", "/user/hand/left", "/user/hand/left/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Squeeze touch", "/user/hand/left", "/user/hand/left/input/squeeze/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Squeeze click", "/user/hand/right", "/user/hand/right/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Squeeze touch", "/user/hand/right", "/user/hand/right/input/squeeze/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick", "/user/hand/left", "/user/hand/left/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick click", "/user/hand/left", "/user/hand/left/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick touch", "/user/hand/left", "/user/hand/left/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick", "/user/hand/right", "/user/hand/right/input/thumbstick", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick click", "/user/hand/right", "/user/hand/right/input/thumbstick/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbstick touch", "/user/hand/right", "/user/hand/right/input/thumbstick/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Thumbrest touch", "/user/hand/right", "/user/hand/right/input/thumbrest/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_focus3_controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// Huawei controller
	// TODO move this into an extension once this is supported.
	register_interaction_profile("Huawei controller", "/interaction_profiles/huawei/controller", XR_HUAWEI_CONTROLLER_INTERACTION_EXTENSION_NAME);
	register_io_path("/interaction_profiles/huawei/controller", "Grip pose", "/user/hand/left", "/user/hand/left/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/huawei/controller", "Grip pose", "/user/hand/right", "/user/hand/right/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/huawei/controller", "Aim pose", "/user/hand/left", "/user/hand/left/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/huawei/controller", "Aim pose", "/user/hand/right", "/user/hand/right/input/aim/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/huawei/controller", "Palm pose", "/user/hand/left", "/user/hand/left/input/palm_ext/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/huawei/controller", "Palm pose", "/user/hand/right", "/user/hand/right/input/palm_ext/pose", "", OpenXRAction::OPENXR_ACTION_POSE);

	register_io_path("/interaction_profiles/huawei/controller", "Home click", "/user/hand/left", "/user/hand/left/input/home/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Home click", "/user/hand/right", "/user/hand/right/input/home/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Back click", "/user/hand/left", "/user/hand/left/input/back/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Back click", "/user/hand/right", "/user/hand/right/input/back/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/huawei/controller", "Volume up click", "/user/hand/left", "/user/hand/left/input/volume_up/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Volume up click", "/user/hand/right", "/user/hand/right/input/volume_up/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Volume down click", "/user/hand/left", "/user/hand/left/input/volume_down/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Volume down click", "/user/hand/right", "/user/hand/right/input/volume_down/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/huawei/controller", "Trigger", "/user/hand/left", "/user/hand/left/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/huawei/controller", "Trigger click", "/user/hand/left", "/user/hand/left/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Trigger", "/user/hand/right", "/user/hand/right/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/huawei/controller", "Trigger click", "/user/hand/right", "/user/hand/right/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/huawei/controller", "Trackpad", "/user/hand/left", "/user/hand/left/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/huawei/controller", "Trackpad click", "/user/hand/left", "/user/hand/left/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Trackpad touch", "/user/hand/left", "/user/hand/left/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Trackpad", "/user/hand/right", "/user/hand/right/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/huawei/controller", "Trackpad click", "/user/hand/right", "/user/hand/right/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/huawei/controller", "Trackpad touch", "/user/hand/right", "/user/hand/right/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	register_io_path("/interaction_profiles/huawei/controller", "Haptic output", "/user/hand/left", "/user/hand/left/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/huawei/controller", "Haptic output", "/user/hand/right", "/user/hand/right/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);

	// HTC Vive tracker
	// Interestingly enough trackers don't have buttons or inputs, yet these are defined in the spec.
	// I think this can be supported through attachments on the trackers.
	// TODO move this into an extension once this is supported.
	register_interaction_profile("HTC Vive tracker", "/interaction_profiles/htc/vive_tracker_htcx", XR_HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Menu click", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/menu/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trigger/value", "", OpenXRAction::OPENXR_ACTION_FLOAT);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trigger click", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trigger/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Squeeze click", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/squeeze/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trackpad", "", OpenXRAction::OPENXR_ACTION_VECTOR2);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad click", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trackpad/click", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Trackpad touch", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/trackpad/touch", "", OpenXRAction::OPENXR_ACTION_BOOL);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Grip pose", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/input/grip/pose", "", OpenXRAction::OPENXR_ACTION_POSE);

	// register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/handheld_object", "/user/vive_tracker_htcx/role/handheld_object/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/left_foot", "/user/vive_tracker_htcx/role/left_foot/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/right_foot", "/user/vive_tracker_htcx/role/right_foot/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/left_shoulder", "/user/vive_tracker_htcx/role/left_shoulder/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/right_shoulder", "/user/vive_tracker_htcx/role/right_shoulder/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/left_elbow", "/user/vive_tracker_htcx/role/left_elbow/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/right_elbow", "/user/vive_tracker_htcx/role/right_elbow/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/left_knee", "/user/vive_tracker_htcx/role/left_knee/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/right_knee", "/user/vive_tracker_htcx/role/right_knee/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/waist", "/user/vive_tracker_htcx/role/waist/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/chest", "/user/vive_tracker_htcx/role/chest/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/camera", "/user/vive_tracker_htcx/role/camera/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
	register_io_path("/interaction_profiles/htc/vive_tracker_htcx", "Haptic output", "/user/vive_tracker_htcx/role/keyboard", "/user/vive_tracker_htcx/role/keyboard/output/haptic", "", OpenXRAction::OPENXR_ACTION_HAPTIC);
}
