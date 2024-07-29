/**************************************************************************/
/*  openxr_fb_space_warp_extension_wrapper.h                              */
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

#ifndef OPENXR_FB_SPACE_WARP_EXTENSION_WRAPPER_H
#define OPENXR_FB_SPACE_WARP_EXTENSION_WRAPPER_H

#include "openxr_extension_wrapper.h"

#include "../openxr_api.h"

class OpenXRFbSpaceWarpExtensionWrapper : public Object, public OpenXRExtensionWrapper {
	GDCLASS(OpenXRFbSpaceWarpExtensionWrapper, Object);

public:
	static OpenXRFbSpaceWarpExtensionWrapper *get_singleton();

	OpenXRFbSpaceWarpExtensionWrapper();
	virtual ~OpenXRFbSpaceWarpExtensionWrapper() override;

	virtual HashMap<String, bool *> get_requested_extensions() override;

	void *set_system_properties_and_get_next_pointer(void *p_next_pointer) override;
	void *set_projection_views_and_get_next_pointer(int p_view_index, void *p_next_pointer) override;

	void on_session_created(const XrSession p_instance) override;
	void on_session_destroyed() override;
	void on_state_ready() override;
	void on_main_swapchains_created() override;
	void on_pre_render() override;
	void on_post_draw_viewport(RID p_render_target) override;

	bool is_available();

	void enable_space_warp(bool p_enable);

protected:
	static void _bind_methods();

private:
	static OpenXRFbSpaceWarpExtensionWrapper *singleton;

	bool available = false;
	bool enabled = true;

	XrSystemSpaceWarpPropertiesFB system_space_warp_properties = {
		XR_TYPE_SYSTEM_SPACE_WARP_PROPERTIES_FB, // type
		nullptr, // next
		0, // recommendedMotionVectorImageRectWidth
		0, // recommendedMotionVectorImageRectHeight
	};

	OpenXRAPI::OpenXRSwapChainInfo motion_vector_swapchain_info;
	OpenXRAPI::OpenXRSwapChainInfo motion_vector_depth_swapchain_info;

	XrCompositionLayerSpaceWarpInfoFB *space_warp_info;
};

#endif // OPENXR_FB_SPACE_WARP_EXTENSION_WRAPPER_H
