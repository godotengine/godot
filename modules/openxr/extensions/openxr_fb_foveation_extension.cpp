/**************************************************************************/
/*  openxr_fb_foveation_extension.cpp                                     */
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

#include "openxr_fb_foveation_extension.h"
#include "core/config/project_settings.h"

OpenXRFBFoveationExtension *OpenXRFBFoveationExtension::singleton = nullptr;

OpenXRFBFoveationExtension *OpenXRFBFoveationExtension::get_singleton() {
	return singleton;
}

OpenXRFBFoveationExtension::OpenXRFBFoveationExtension(const String &p_rendering_driver) {
	singleton = this;
	rendering_driver = p_rendering_driver;
	swapchain_update_state_ext = OpenXRFBUpdateSwapchainExtension::get_singleton();
	int fov_level = GLOBAL_GET("xr/openxr/foveation_level");
	if (fov_level >= 0 && fov_level < 4) {
		foveation_level = XrFoveationLevelFB(fov_level);
	}
	bool fov_dyn = GLOBAL_GET("xr/openxr/foveation_dynamic");
	foveation_dynamic = fov_dyn ? XR_FOVEATION_DYNAMIC_LEVEL_ENABLED_FB : XR_FOVEATION_DYNAMIC_DISABLED_FB;

	swapchain_create_info_foveation_fb.type = XR_TYPE_SWAPCHAIN_CREATE_INFO_FOVEATION_FB;
	swapchain_create_info_foveation_fb.next = nullptr;
	swapchain_create_info_foveation_fb.flags = 0;
}

OpenXRFBFoveationExtension::~OpenXRFBFoveationExtension() {
	singleton = nullptr;
	swapchain_update_state_ext = nullptr;
}

HashMap<String, bool *> OpenXRFBFoveationExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	if (rendering_driver == "vulkan") {
		// This is currently only supported on OpenGL, but we may add Vulkan support in the future...

	} else if (rendering_driver == "opengl3") {
		request_extensions[XR_FB_FOVEATION_EXTENSION_NAME] = &fb_foveation_ext;
		request_extensions[XR_FB_FOVEATION_CONFIGURATION_EXTENSION_NAME] = &fb_foveation_configuration_ext;
	}

	return request_extensions;
}

void OpenXRFBFoveationExtension::on_instance_created(const XrInstance p_instance) {
	if (fb_foveation_ext) {
		EXT_INIT_XR_FUNC(xrCreateFoveationProfileFB);
		EXT_INIT_XR_FUNC(xrDestroyFoveationProfileFB);
	}

	if (fb_foveation_configuration_ext) {
		// nothing to register here...
	}
}

void OpenXRFBFoveationExtension::on_instance_destroyed() {
	fb_foveation_ext = false;
	fb_foveation_configuration_ext = false;
}

bool OpenXRFBFoveationExtension::is_enabled() const {
	return swapchain_update_state_ext != nullptr && swapchain_update_state_ext->is_enabled() && fb_foveation_ext && fb_foveation_configuration_ext;
}

void *OpenXRFBFoveationExtension::set_swapchain_create_info_and_get_next_pointer(void *p_next_pointer) {
	if (is_enabled()) {
		swapchain_create_info_foveation_fb.next = p_next_pointer;
		return &swapchain_create_info_foveation_fb;
	} else {
		return p_next_pointer;
	}
}

void OpenXRFBFoveationExtension::on_state_ready() {
	update_profile();
}

XrFoveationLevelFB OpenXRFBFoveationExtension::get_foveation_level() const {
	return foveation_level;
}

void OpenXRFBFoveationExtension::set_foveation_level(XrFoveationLevelFB p_foveation_level) {
	foveation_level = p_foveation_level;

	// Update profile will do nothing if we're not yet initialized.
	update_profile();
}

XrFoveationDynamicFB OpenXRFBFoveationExtension::get_foveation_dynamic() const {
	return foveation_dynamic;
}

void OpenXRFBFoveationExtension::set_foveation_dynamic(XrFoveationDynamicFB p_foveation_dynamic) {
	foveation_dynamic = p_foveation_dynamic;

	// Update profile will do nothing if we're not yet initialized.
	update_profile();
}

void OpenXRFBFoveationExtension::update_profile() {
	if (!is_enabled()) {
		return;
	}

	XrFoveationLevelProfileCreateInfoFB level_profile_create_info;
	level_profile_create_info.type = XR_TYPE_FOVEATION_LEVEL_PROFILE_CREATE_INFO_FB;
	level_profile_create_info.next = nullptr;
	level_profile_create_info.level = foveation_level;
	level_profile_create_info.verticalOffset = 0.0f;
	level_profile_create_info.dynamic = foveation_dynamic;

	XrFoveationProfileCreateInfoFB profile_create_info;
	profile_create_info.type = XR_TYPE_FOVEATION_PROFILE_CREATE_INFO_FB;
	profile_create_info.next = &level_profile_create_info;

	XrFoveationProfileFB foveation_profile;
	XrResult result = xrCreateFoveationProfileFB(OpenXRAPI::get_singleton()->get_session(), &profile_create_info, &foveation_profile);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Unable to create the foveation profile [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
		return;
	}

	XrSwapchainStateFoveationFB foveation_update_state;
	foveation_update_state.type = XR_TYPE_SWAPCHAIN_STATE_FOVEATION_FB;
	foveation_update_state.profile = foveation_profile;

	result = swapchain_update_state_ext->xrUpdateSwapchainFB(OpenXRAPI::get_singleton()->get_color_swapchain(), (XrSwapchainStateBaseHeaderFB *)&foveation_update_state);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Unable to update the swapchain [", OpenXRAPI::get_singleton()->get_error_string(result), "]");

		// We still want to destroy our profile so keep going...
	}

	result = xrDestroyFoveationProfileFB(foveation_profile);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Unable to destroy the foveation profile [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
	}
}
