/**************************************************************************/
/*  openxr_fb_space_warp_extension_wrapper.cpp                            */
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

#include "openxr_fb_space_warp_extension_wrapper.h"

#include "../openxr_interface.h"
#include "platform_gl.h"

OpenXRFbSpaceWarpExtensionWrapper *OpenXRFbSpaceWarpExtensionWrapper::singleton = nullptr;

OpenXRFbSpaceWarpExtensionWrapper *OpenXRFbSpaceWarpExtensionWrapper::get_singleton() {
	return singleton;
}

OpenXRFbSpaceWarpExtensionWrapper::OpenXRFbSpaceWarpExtensionWrapper() {
	singleton = this;
}

OpenXRFbSpaceWarpExtensionWrapper::~OpenXRFbSpaceWarpExtensionWrapper() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRFbSpaceWarpExtensionWrapper::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_FB_SPACE_WARP_EXTENSION_NAME] = &available;

	return request_extensions;
}

void *OpenXRFbSpaceWarpExtensionWrapper::set_system_properties_and_get_next_pointer(void *p_next_pointer) {
	return &system_space_warp_properties;
}

void *OpenXRFbSpaceWarpExtensionWrapper::set_projection_views_and_get_next_pointer(int p_view_index, void *p_next_pointer) {
	if (enabled) {
		return &space_warp_info[p_view_index];
	} else {
		return nullptr;
	}
}

void OpenXRFbSpaceWarpExtensionWrapper::on_session_created(const XrSession p_instance) {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	if (!openxr_api) {
		return;
	}

	openxr_api->register_projection_views_extension(this);
}

void OpenXRFbSpaceWarpExtensionWrapper::on_session_destroyed() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	if (!openxr_api) {
		return;
	}

	openxr_api->unregister_projection_views_extension(this);
}

void OpenXRFbSpaceWarpExtensionWrapper::on_state_ready() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();

	if (!openxr_api) {
		return;
	}

	int width = system_space_warp_properties.recommendedMotionVectorImageRectWidth;
	int height = system_space_warp_properties.recommendedMotionVectorImageRectHeight;
	int view_count = openxr_api->get_xr_interface()->get_view_count();

	motion_vector_swapchain_info.create(0, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT, GL_RGBA16F, width, height, 1, view_count);
	motion_vector_depth_swapchain_info.create(0, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, GL_DEPTH24_STENCIL8, width, height, 1, view_count);
}

void OpenXRFbSpaceWarpExtensionWrapper::on_main_swapchains_created() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	int view_count = openxr_api->get_xr_interface()->get_view_count();

	XrCompositionLayerProjectionView *projection_views = openxr_api->get_projection_views();
	XrPosef identity_pose = { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 } };

	space_warp_info = memnew_arr(XrCompositionLayerSpaceWarpInfoFB, 2);
	for (int i = 0; i < view_count; i++) {
		space_warp_info[i].type = XR_TYPE_COMPOSITION_LAYER_SPACE_WARP_INFO_FB;

		space_warp_info[i].next = nullptr;

		space_warp_info[i].layerFlags = 0;

		space_warp_info[i].motionVectorSubImage.swapchain = motion_vector_swapchain_info.get_swapchain();
		space_warp_info[i].motionVectorSubImage.imageRect.offset.x = 0;
		space_warp_info[i].motionVectorSubImage.imageRect.offset.y = 0;
		space_warp_info[i].motionVectorSubImage.imageRect.extent.width = system_space_warp_properties.recommendedMotionVectorImageRectWidth;
		space_warp_info[i].motionVectorSubImage.imageRect.extent.height = system_space_warp_properties.recommendedMotionVectorImageRectHeight;
		space_warp_info[i].motionVectorSubImage.imageArrayIndex = i;

		space_warp_info[i].appSpaceDeltaPose = identity_pose;

		space_warp_info[i].depthSubImage.swapchain = motion_vector_depth_swapchain_info.get_swapchain();
		space_warp_info[i].depthSubImage.imageRect.offset.x = 0;
		space_warp_info[i].depthSubImage.imageRect.offset.y = 0;
		space_warp_info[i].depthSubImage.imageRect.extent.width = system_space_warp_properties.recommendedMotionVectorImageRectWidth;
		space_warp_info[i].depthSubImage.imageRect.extent.height = system_space_warp_properties.recommendedMotionVectorImageRectHeight;
		space_warp_info[i].depthSubImage.imageArrayIndex = i;

		space_warp_info[i].minDepth = 0.0;
		space_warp_info[i].maxDepth = 1.0;

		space_warp_info[i].farZ = openxr_api->get_render_state_z_near();
		space_warp_info[i].nearZ = openxr_api->get_render_state_z_far();

		projection_views[i].next = &space_warp_info[i];
	}
}

void OpenXRFbSpaceWarpExtensionWrapper::on_pre_render() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();

	if (!openxr_api) {
		return;
	}

	bool should_render = true; // Can ignore should_render.
	motion_vector_swapchain_info.acquire(should_render);
	motion_vector_depth_swapchain_info.acquire(should_render);

	openxr_api->set_motion_vector_texture(motion_vector_swapchain_info.get_image());
	openxr_api->set_motion_vector_depth_texture(motion_vector_depth_swapchain_info.get_image());

	int target_width = system_space_warp_properties.recommendedMotionVectorImageRectWidth;
	int target_height = system_space_warp_properties.recommendedMotionVectorImageRectHeight;
	Size2i render_target_size = { target_width, target_height };

	openxr_api->set_motion_vector_target_size(render_target_size);

	int view_count = openxr_api->get_xr_interface()->get_view_count();
	for (int i = 0; i < view_count; i++) {
		space_warp_info[i].farZ = openxr_api->get_render_state_z_near();
		space_warp_info[i].nearZ = openxr_api->get_render_state_z_far();
	}
}

void OpenXRFbSpaceWarpExtensionWrapper::on_post_draw_viewport(RID p_render_target) {
	motion_vector_swapchain_info.release();
	motion_vector_depth_swapchain_info.release();
}

bool OpenXRFbSpaceWarpExtensionWrapper::is_available() {
	return available;
}

void OpenXRFbSpaceWarpExtensionWrapper::enable_space_warp(bool p_enable) {
	if (enabled == p_enable) {
		return;
	}

	enabled = p_enable;
}

void OpenXRFbSpaceWarpExtensionWrapper::_bind_methods() {
	ClassDB::bind_method(D_METHOD("enable_space_warp"), &OpenXRFbSpaceWarpExtensionWrapper::enable_space_warp);
}
