/**************************************************************************/
/*  openxr_varjo_quad_view_extension.cpp                                  */
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

#include "openxr_varjo_quad_view_extension.h"
#include "openxr_composition_layer_depth_extension.h"
#include "servers/rendering_server.h"

HashMap<String, bool *> OpenXRVarjoQuadViewExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_VARJO_QUAD_VIEWS_EXTENSION_NAME] = &available;

	return request_extensions;
}

void OpenXRVarjoQuadViewExtension::on_session_created(const XrSession p_instance) {
	if (!available) {
		// Not available? no need to do anything more...
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Note: after instance is created, our view configuration should NOT change
	XrViewConfigurationType view_configuration = openxr_api->get_view_configuration();
	if (view_configuration != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_QUAD_VARJO) {
		// Didn't select our view configuration? no need to do anything more...
		return;
	}

	// Get our rendering server
	RS *rendering_server = RS::get_singleton();
	ERR_FAIL_NULL(rendering_server);

	// We need to find a way to copy over settings from our primary XR viewport to our secondary viewport (and camera)
	// but we never get access to that. Or maybe we should make this configurable some other way?
	//
	// For now we'll accept this as a limitation.

	// Create a camera for our viewport (most of the state will be ignored)
	secondary_camera = rendering_server->camera_create();
	rendering_server->camera_set_perspective(secondary_camera, 60, 0.01f, 4096.0f); // only near and far of this will be used

	// Create the viewport we render too
	secondary_viewport = rendering_server->viewport_create();
	rendering_server->viewport_set_disable_2d(secondary_viewport, true);
	rendering_server->viewport_set_use_xr(secondary_viewport, true);
	rendering_server->viewport_set_update_mode(secondary_viewport, RS::VIEWPORT_UPDATE_ALWAYS);
	rendering_server->viewport_attach_camera(secondary_viewport, secondary_camera);
	rendering_server->viewport_set_active(secondary_viewport, true);

	enabled = true;
}

void OpenXRVarjoQuadViewExtension::on_session_destroyed() {
	if (enabled) {
		// We're disabling...
		enabled = false;

		// Make sure our swapchains are freed
		free_swapchains();
		OpenXRAPI::OpenXRSwapChainInfo::free_queued();

		RS *rendering_server = RS::get_singleton();
		ERR_FAIL_NULL(rendering_server);

		if (secondary_viewport.is_valid()) {
			rendering_server->viewport_set_active(secondary_viewport, false);
			rendering_server->free(secondary_viewport);
			secondary_viewport = RID();
		}

		if (secondary_camera.is_valid()) {
			rendering_server->free(secondary_camera);
			secondary_camera = RID();
		}
	}
}

bool OpenXRVarjoQuadViewExtension::owns_viewport(RID p_render_target) {
	if (!enabled) {
		return false;
	}

	RS *rendering_server = RS::get_singleton();
	ERR_FAIL_NULL_V(rendering_server, false);

	return rendering_server->viewport_get_render_target(secondary_viewport) == p_render_target;
}

bool OpenXRVarjoQuadViewExtension::on_pre_draw_viewport(RID p_render_target) {
	// This method is called if we own this viewport (it's our secondary viewport) or if it's the main XR viewport.
	if (!enabled) {
		// We're not enabled, so no need to check on this, this must be for the main XR viewport.
		return true;
	}

	RS *rendering_server = RS::get_singleton();
	ERR_FAIL_NULL_V(rendering_server, false);

	if (rendering_server->viewport_get_render_target(secondary_viewport) == p_render_target) {
		// This is our secondary viewport.
		if (!have_primary_viewport) {
			// this is ok if it happens once...
			WARN_PRINT("Attempting to render our secondary XR viewport before rendering our primary XR viewport.");
			return false;
		}

		// check our swapchain
		Size2i new_swapchain_size = get_viewport_size();

		if (swapchain_size != new_swapchain_size) {
			// out with the old
			free_swapchains();

			// in with the new
			create_swapchains(new_swapchain_size);
		}

		// Acquire our images
		for (int i = 0; i < OpenXRAPI::OPENXR_SWAPCHAIN_MAX; i++) {
			if (!swapchains[i].is_image_acquired() && swapchains[i].get_swapchain() != XR_NULL_HANDLE) {
				bool should_render = true;
				if (!swapchains[i].acquire(should_render)) {
					return false;
				}
			}
		}

		return true;
	} else {
		// We've obtained our primary viewport, make sure we use its scenario
		have_primary_viewport = true;
		RID main_xr_viewport = rendering_server->viewport_get_for_render_target(p_render_target);
		RID scenario = rendering_server->viewport_get_scenario(main_xr_viewport);
		rendering_server->viewport_set_scenario(secondary_viewport, scenario);
		return true;
	}
}

uint32_t OpenXRVarjoQuadViewExtension::get_viewport_view_count() {
	// This should only be called if we returned true on owns_viewport

	return 2; // always 2
}

Size2i OpenXRVarjoQuadViewExtension::get_viewport_size() {
	// This should only be called if we returned true on owns_viewport

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Size2i());

	// Views 2 and 3 are for our secondary view but should have the same size.
	XrViewConfigurationView view_configuration_view = openxr_api->get_view_configuration_view(2);

	// Do we apply render_target_multiplier here?

	// And return result.
	return Size2i(view_configuration_view.recommendedImageRectWidth, view_configuration_view.recommendedImageRectHeight);
}

bool OpenXRVarjoQuadViewExtension::get_view_transform(uint32_t p_view, XrTime p_display_time, Transform3D &r_transform) {
	// This should only be called if we returned true on owns_viewport

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, false);

	// Views 2 and 3 are for our secondary view
	XrView view = openxr_api->get_view(2 + p_view);

	r_transform = openxr_api->transform_from_pose(view.pose);

	return true;
}

bool OpenXRVarjoQuadViewExtension::get_view_projection(uint32_t p_view, double p_z_near, double p_z_far, XrTime p_display_time, Projection &r_projection) {
	// This should only be called if we returned true on owns_viewport

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, false);
	ERR_FAIL_NULL_V(openxr_api->get_graphics_extension(), false);

	// Views 2 and 3 are for our secondary view
	XrView view = openxr_api->get_view(2 + p_view);

	// Update near and far
	openxr_api->set_view_depth(2 + p_view, p_z_near, p_z_far);

	// And set our projection
	return openxr_api->get_graphics_extension()->create_projection_fov(view.fov, p_z_near, p_z_far, r_projection);
}

XrSwapchain OpenXRVarjoQuadViewExtension::get_color_swapchain() {
	// This should only be called if we returned true on owns_viewport

	return swapchains[OpenXRAPI::OPENXR_SWAPCHAIN_COLOR].get_swapchain();
}

RID OpenXRVarjoQuadViewExtension::get_color_texture() {
	// This should only be called if we returned true on owns_viewport

	return swapchains[OpenXRAPI::OPENXR_SWAPCHAIN_COLOR].get_image();
}

RID OpenXRVarjoQuadViewExtension::get_depth_texture() {
	// This should only be called if we returned true on owns_viewport

	return swapchains[OpenXRAPI::OPENXR_SWAPCHAIN_DEPTH].get_image();
}

void OpenXRVarjoQuadViewExtension::on_post_draw_viewport(RID p_render_target) {
	if (!enabled) {
		return;
	}

	RS *rendering_server = RS::get_singleton();
	ERR_FAIL_NULL(rendering_server);

	// For backwards compatibility reasons, double check if this is our viewport
	if (rendering_server->viewport_get_render_target(secondary_viewport) == p_render_target) {
		// nothing to do here atm
	}
}

void OpenXRVarjoQuadViewExtension::on_end_frame() {
	for (int i = 0; i < OpenXRAPI::OPENXR_SWAPCHAIN_MAX; i++) {
		if (swapchains[i].is_image_acquired()) {
			swapchains[i].release();
		}
	}
}

bool OpenXRVarjoQuadViewExtension::is_available() {
	return available;
}

bool OpenXRVarjoQuadViewExtension::is_enabled() {
	return enabled;
}

void OpenXRVarjoQuadViewExtension::create_swapchains(Size2i p_size) {
	swapchain_size = p_size;
	uint32_t sample_count = 1;
	uint32_t view_count = get_viewport_view_count();

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// We start with our color swapchain...
	int64_t color_swapchain_format = openxr_api->get_color_swapchain_format();
	if (color_swapchain_format != 0) {
		if (!swapchains[OpenXRAPI::OPENXR_SWAPCHAIN_COLOR].create(0, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT, color_swapchain_format, swapchain_size.width, swapchain_size.height, sample_count, view_count)) {
			return;
		}
	}

	// Then our depth swapchain
	int64_t depth_swapchain_format = openxr_api->get_depth_swapchain_format();
	bool submit_depth_buffer = openxr_api->get_submit_depth_buffer();
	if (depth_swapchain_format != 0 && submit_depth_buffer && OpenXRCompositionLayerDepthExtension::get_singleton()->is_available()) {
		if (!swapchains[OpenXRAPI::OPENXR_SWAPCHAIN_DEPTH].create(0, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depth_swapchain_format, swapchain_size.width, swapchain_size.height, sample_count, view_count)) {
			return;
		}
	}

	for (uint32_t i = 0; i < view_count; i++) {
		openxr_api->set_projection_image(2 + i, i, swapchain_size, swapchains[OpenXRAPI::OPENXR_SWAPCHAIN_COLOR].get_swapchain(), submit_depth_buffer ? swapchains[OpenXRAPI::OPENXR_SWAPCHAIN_DEPTH].get_swapchain() : XR_NULL_HANDLE);
	};
}

void OpenXRVarjoQuadViewExtension::free_swapchains() {
	for (int i = 0; i < OpenXRAPI::OPENXR_SWAPCHAIN_MAX; i++) {
		swapchains[i].queue_free();
	}
}
