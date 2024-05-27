/**************************************************************************/
/*  openxr_fb_update_swapchain_extension.cpp                              */
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

#include "openxr_fb_update_swapchain_extension.h"

// Always include this as late as possible.
#include "../openxr_platform_inc.h"

OpenXRFBUpdateSwapchainExtension *OpenXRFBUpdateSwapchainExtension::singleton = nullptr;

OpenXRFBUpdateSwapchainExtension *OpenXRFBUpdateSwapchainExtension::get_singleton() {
	return singleton;
}

OpenXRFBUpdateSwapchainExtension::OpenXRFBUpdateSwapchainExtension(const String &p_rendering_driver) {
	singleton = this;
	rendering_driver = p_rendering_driver;
}

OpenXRFBUpdateSwapchainExtension::~OpenXRFBUpdateSwapchainExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRFBUpdateSwapchainExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_FB_SWAPCHAIN_UPDATE_STATE_EXTENSION_NAME] = &fb_swapchain_update_state_ext;

	if (rendering_driver == "vulkan") {
#ifdef XR_USE_GRAPHICS_API_VULKAN
		request_extensions[XR_FB_SWAPCHAIN_UPDATE_STATE_VULKAN_EXTENSION_NAME] = &fb_swapchain_update_state_vulkan_ext;
#endif
	} else if (rendering_driver == "opengl3") {
#ifdef XR_USE_GRAPHICS_API_OPENGL_ES
		request_extensions[XR_FB_SWAPCHAIN_UPDATE_STATE_OPENGL_ES_EXTENSION_NAME] = &fb_swapchain_update_state_opengles_ext;
#endif
	}

	return request_extensions;
}

void OpenXRFBUpdateSwapchainExtension::on_instance_created(const XrInstance p_instance) {
	if (fb_swapchain_update_state_ext) {
		EXT_INIT_XR_FUNC(xrUpdateSwapchainFB);
		EXT_INIT_XR_FUNC(xrGetSwapchainStateFB);
	}

	if (fb_swapchain_update_state_vulkan_ext) {
		// nothing to register here...
	}

	if (fb_swapchain_update_state_opengles_ext) {
		// nothing to register here...
	}
}

void OpenXRFBUpdateSwapchainExtension::on_instance_destroyed() {
	fb_swapchain_update_state_ext = false;
	fb_swapchain_update_state_vulkan_ext = false;
	fb_swapchain_update_state_opengles_ext = false;
}

bool OpenXRFBUpdateSwapchainExtension::is_enabled() const {
	if (rendering_driver == "vulkan") {
		return fb_swapchain_update_state_ext && fb_swapchain_update_state_vulkan_ext;
	} else if (rendering_driver == "opengl3") {
#ifdef XR_USE_GRAPHICS_API_OPENGL_ES
		return fb_swapchain_update_state_ext && fb_swapchain_update_state_opengles_ext;
#else
		return fb_swapchain_update_state_ext;
#endif
	}

	return false;
}
