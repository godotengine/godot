/**************************************************************************/
/*  openxr_fb_foveation_extension.h                                       */
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

#pragma once

// This extension implements the FB Foveation extension.
// This is an extension Meta added due to VRS being unavailable on Android.
// Other Android based devices are implementing this as well, see:
// https://github.khronos.org/OpenXR-Inventory/extension_support.html#XR_FB_foveation

#include "../openxr_api.h"
#include "../util.h"
#include "openxr_extension_wrapper.h"
#include "openxr_fb_update_swapchain_extension.h"

#include "../openxr_platform_inc.h"

class OpenXRFBFoveationExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRFBFoveationExtension, OpenXRExtensionWrapper);

protected:
	static void _bind_methods() {}

public:
	static OpenXRFBFoveationExtension *get_singleton();

	OpenXRFBFoveationExtension(const String &p_rendering_driver);
	virtual ~OpenXRFBFoveationExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_instance_destroyed() override;

	virtual void *set_system_properties_and_get_next_pointer(void *p_next_pointer) override;
	virtual void *set_swapchain_create_info_and_get_next_pointer(void *p_next_pointer) override;

	virtual void on_main_swapchains_created() override;

	bool is_enabled() const;

	XrFoveationLevelFB get_foveation_level() const;
	void set_foveation_level(XrFoveationLevelFB p_foveation_level);

	XrFoveationDynamicFB get_foveation_dynamic() const;
	void set_foveation_dynamic(XrFoveationDynamicFB p_foveation_dynamic);

	bool is_foveation_eye_tracked_enabled() const;
	void get_fragment_density_offsets(LocalVector<Vector2i> &r_offsets);

private:
	static OpenXRFBFoveationExtension *singleton;

	// Setup
	String rendering_driver;
	bool fb_foveation_ext = false;
	bool fb_foveation_configuration_ext = false;
	bool fb_foveation_vulkan_ext = false;
	bool meta_foveation_eye_tracked_ext = false;
	bool meta_vulkan_swapchain_create_info_ext = false;

	// Configuration
	XrFoveationLevelFB foveation_level = XR_FOVEATION_LEVEL_NONE_FB;
	XrFoveationDynamicFB foveation_dynamic = XR_FOVEATION_DYNAMIC_DISABLED_FB;

	void _update_profile_rt();

	void update_profile() {
		// If we're rendering on a separate thread, we may still be processing the last frame, don't communicate this till we're ready...
		RenderingServer *rendering_server = RenderingServer::get_singleton();
		ERR_FAIL_NULL(rendering_server);

		rendering_server->call_on_render_thread(callable_mp(this, &OpenXRFBFoveationExtension::_update_profile_rt));
	}

	// Enable foveation on this swapchain
	XrSwapchainCreateInfoFoveationFB swapchain_create_info_foveation_fb;
	OpenXRFBUpdateSwapchainExtension *swapchain_update_state_ext = nullptr;

	// Enable eye tracked foveation
	XrSystemFoveationEyeTrackedPropertiesMETA meta_foveation_eye_tracked_properties;
	XrFoveationEyeTrackedProfileCreateInfoMETA meta_foveation_eye_tracked_create_info;
#ifdef VULKAN_ENABLED
	XrVulkanSwapchainCreateInfoMETA meta_vulkan_swapchain_create_info;
#endif

	// OpenXR API call wrappers
	EXT_PROTO_XRRESULT_FUNC3(xrCreateFoveationProfileFB, (XrSession), session, (const XrFoveationProfileCreateInfoFB *), create_info, (XrFoveationProfileFB *), profile);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroyFoveationProfileFB, (XrFoveationProfileFB), profile);
	EXT_PROTO_XRRESULT_FUNC2(xrGetFoveationEyeTrackedStateMETA, (XrSession), session, (XrFoveationEyeTrackedStateMETA *), foveationState);
};
