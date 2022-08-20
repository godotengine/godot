/*************************************************************************/
/*  openxr_defs.h                                                        */
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

#ifndef OPENXR_DEFS_H
#define OPENXR_DEFS_H

#include "openxr_action.h"

///////////////////////////////////////////////////////////////////////////
// Stores available interaction profiles
//
// OpenXR defines and hardcodes all the supported input devices and their
// paths as part of the OpenXR spec. When support for new devices is
// introduced this often starts life as extensions that need to be enabled
// until they are adopted into the core. As there is no interface to
// enumerate the possibly paths, and that any OpenXR runtime would likely
// limit such enumeration to those input devices supported by that runtime
// there is no other option than to hardcode this.
//
// Note on action type that automatic conversions between boolean and float
// are supported but otherwise action types should match between action and
// input/output paths.

class OpenXRDefs {
public:
	enum TOP_LEVEL_PATH {
		// Core OpenXR toplevel paths
		OPENXR_LEFT_HAND,
		OPENXR_RIGHT_HAND,
		OPENXR_HEAD,
		OPENXR_GAMEPAD,
		OPENXR_TREADMILL,

		// HTC tracker extension toplevel paths
		// OPENXR_HTC_HANDHELD_TRACKER,
		OPENXR_HTC_LEFT_FOOT_TRACKER,
		OPENXR_HTC_RIGHT_FOOT_TRACKER,
		OPENXR_HTC_LEFT_SHOULDER_TRACKER,
		OPENXR_HTC_RIGHT_SHOULDER_TRACKER,
		OPENXR_HTC_LEFT_ELBOW_TRACKER,
		OPENXR_HTC_RIGHT_ELBOW_TRACKER,
		OPENXR_HTC_LEFT_KNEE_TRACKER,
		OPENXR_HTC_RIGHT_KNEE_TRACKER,
		OPENXR_HTC_WAIST_TRACKER,
		OPENXR_HTC_CHEST_TRACKER,
		OPENXR_HTC_CAMERA_TRACKER,
		OPENXR_HTC_KEYBOARD_TRACKER,

		OPENXR_TOP_LEVEL_PATH_MAX
	};

	struct TopLevelPath {
		const char *display_name; // User friendly display name (i.e. Left controller)
		const char *openxr_path; // Path in OpenXR (i.e. /user/hand/left)
	};

	struct IOPath {
		const char *display_name; // User friendly display name (i.e. Grip pose (left controller))
		const TopLevelPath *top_level_path; // Top level path identifying the usage of the device in relation to this input/output
		const char *openxr_path; // Path in OpenXR (i.e. /user/hand/left/input/grip/pose)
		const OpenXRAction::ActionType action_type; // Type of input/output
	};

	struct InteractionProfile {
		const char *display_name; // User friendly display name (i.e. Simple controller)
		const char *openxr_path; // Path in OpenXR (i.e. /interaction_profiles/khr/simple_controller)
		const IOPath *io_paths; // Inputs and outputs for this device
		const int io_path_count; // Number of inputs and outputs for this device

		const IOPath *get_io_path(const String p_io_path) const;
	};

private:
	static TopLevelPath available_top_level_paths[OPENXR_TOP_LEVEL_PATH_MAX];
	static IOPath simple_io_paths[];
	static IOPath vive_io_paths[];
	static IOPath motion_io_paths[];
	static IOPath hpmr_io_paths[];
	static IOPath touch_io_paths[];
	static IOPath index_io_paths[];
	static IOPath odyssey_io_paths[];
	static IOPath vive_cosmos_paths[];
	static IOPath vive_focus3_paths[];
	static IOPath huawei_controller_paths[];
	static IOPath vive_tracker_controller_paths[];
	static InteractionProfile available_interaction_profiles[];
	static int available_interaction_profile_count;

public:
	static const TopLevelPath *get_top_level_path(const String p_top_level_path);
	static const InteractionProfile *get_profile(const String p_interaction_profile_path);
	static const IOPath *get_io_path(const String p_interaction_profile_path, const String p_io_path);

	static PackedStringArray get_interaction_profile_paths();
};

#endif // OPENXR_DEFS_H
