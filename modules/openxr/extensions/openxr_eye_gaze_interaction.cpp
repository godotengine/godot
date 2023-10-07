/**************************************************************************/
/*  openxr_eye_gaze_interaction.cpp                                       */
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

#include "openxr_eye_gaze_interaction.h"

#include "core/os/os.h"

#include "../action_map/openxr_interaction_profile_metadata.h"

OpenXREyeGazeInteractionExtension *OpenXREyeGazeInteractionExtension::singleton = nullptr;

OpenXREyeGazeInteractionExtension *OpenXREyeGazeInteractionExtension::get_singleton() {
	ERR_FAIL_NULL_V(singleton, nullptr);
	return singleton;
}

OpenXREyeGazeInteractionExtension::OpenXREyeGazeInteractionExtension() {
	singleton = this;
}

OpenXREyeGazeInteractionExtension::~OpenXREyeGazeInteractionExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXREyeGazeInteractionExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME] = &available;

	return request_extensions;
}

void *OpenXREyeGazeInteractionExtension::set_system_properties_and_get_next_pointer(void *p_next_pointer) {
	if (!available) {
		return p_next_pointer;
	}

	properties.type = XR_TYPE_SYSTEM_EYE_GAZE_INTERACTION_PROPERTIES_EXT;
	properties.next = p_next_pointer;
	properties.supportsEyeGazeInteraction = false;

	return &properties;
}

bool OpenXREyeGazeInteractionExtension::is_available() {
	return available;
}

bool OpenXREyeGazeInteractionExtension::supports_eye_gaze_interaction() {
	// The extension being available only means that the OpenXR Runtime supports the extension.
	// The `supportsEyeGazeInteraction` is set to true if the device also supports this.
	// Thus both need to be true.
	// In addition, on mobile runtimes, the proper permission needs to be granted.
	if (available && properties.supportsEyeGazeInteraction) {
		return !OS::get_singleton()->has_feature("mobile") || OS::get_singleton()->has_feature("PERMISSION_XR_EXT_eye_gaze_interaction");
	}

	return false;
}

void OpenXREyeGazeInteractionExtension::on_register_metadata() {
	OpenXRInteractionProfileMetadata *metadata = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(metadata);

	// Eyes top path
	metadata->register_top_level_path("Eye gaze tracker", "/user/eyes_ext", XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME);

	// Eye gaze interaction
	metadata->register_interaction_profile("Eye gaze", "/interaction_profiles/ext/eye_gaze_interaction", XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME);
	metadata->register_io_path("/interaction_profiles/ext/eye_gaze_interaction", "Gaze pose", "/user/eyes_ext", "/user/eyes_ext/input/gaze_ext/pose", "", OpenXRAction::OPENXR_ACTION_POSE);
}
