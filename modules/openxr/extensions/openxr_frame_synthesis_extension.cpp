/**************************************************************************/
/*  openxr_frame_synthesis_extension.cpp                                  */
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

#include "openxr_frame_synthesis_extension.h"

#include "core/config/project_settings.h"
#include "servers/rendering/rendering_server.h"
#include "servers/xr/xr_server.h"

#define GL_RGBA16F 0x881A
#define GL_DEPTH24_STENCIL8 0x88F0

#define VK_FORMAT_R16G16B16A16_SFLOAT 97
#define VK_FORMAT_D24_UNORM_S8_UINT 129

OpenXRFrameSynthesisExtension *OpenXRFrameSynthesisExtension::singleton = nullptr;

OpenXRFrameSynthesisExtension *OpenXRFrameSynthesisExtension::get_singleton() {
	return singleton;
}

void OpenXRFrameSynthesisExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_available"), &OpenXRFrameSynthesisExtension::is_available);

	ClassDB::bind_method(D_METHOD("is_enabled"), &OpenXRFrameSynthesisExtension::is_enabled);
	ClassDB::bind_method(D_METHOD("set_enabled", "enable"), &OpenXRFrameSynthesisExtension::set_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");

	ClassDB::bind_method(D_METHOD("get_relax_frame_interval"), &OpenXRFrameSynthesisExtension::get_relax_frame_interval);
	ClassDB::bind_method(D_METHOD("set_relax_frame_interval", "relax_frame_interval"), &OpenXRFrameSynthesisExtension::set_relax_frame_interval);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "relax_frame_interval"), "set_relax_frame_interval", "get_relax_frame_interval");

	ClassDB::bind_method(D_METHOD("skip_next_frame"), &OpenXRFrameSynthesisExtension::skip_next_frame);
}

OpenXRFrameSynthesisExtension::OpenXRFrameSynthesisExtension() {
	singleton = this;
}

OpenXRFrameSynthesisExtension::~OpenXRFrameSynthesisExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRFrameSynthesisExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	if (GLOBAL_GET("xr/openxr/extensions/frame_synthesis")) {
		request_extensions[XR_EXT_FRAME_SYNTHESIS_EXTENSION_NAME] = &frame_synthesis_ext;
	}

	return request_extensions;
}

void OpenXRFrameSynthesisExtension::on_instance_created(const XrInstance p_instance) {
	// Enable this if our extension was successfully enabled
	enabled = frame_synthesis_ext;
	render_state.enabled = frame_synthesis_ext;

	// Register this as a projection view extension
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	openxr_api->register_projection_views_extension(this);
}

void OpenXRFrameSynthesisExtension::on_instance_destroyed() {
	frame_synthesis_ext = false;
	enabled = false;
	render_state.enabled = false;

	// Unregister this as a projection view extension.
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	openxr_api->unregister_projection_views_extension(this);
}

void OpenXRFrameSynthesisExtension::prepare_view_configuration(uint32_t p_view_count) {
	if (!frame_synthesis_ext) {
		return;
	}

	// Called during initialization, we can safely change this.
	render_state.config_views.resize(p_view_count);

	for (XrFrameSynthesisConfigViewEXT &config_view : render_state.config_views) {
		config_view.type = XR_TYPE_FRAME_SYNTHESIS_CONFIG_VIEW_EXT;
		config_view.next = nullptr;

		// These will be set by xrEnumerateViewConfigurationViews.
		config_view.recommendedMotionVectorImageRectWidth = 0;
		config_view.recommendedMotionVectorImageRectHeight = 0;
	}
}

void *OpenXRFrameSynthesisExtension::set_view_configuration_and_get_next_pointer(uint32_t p_view, void *p_next_pointer) {
	if (!frame_synthesis_ext) {
		return nullptr;
	}

	// Called during initialization, we can safely access this.
	ERR_FAIL_UNSIGNED_INDEX_V(p_view, render_state.config_views.size(), nullptr);

	XrFrameSynthesisConfigViewEXT &config_view = render_state.config_views[p_view];
	config_view.next = p_next_pointer;

	return &config_view;
}

void OpenXRFrameSynthesisExtension::print_view_configuration_info(uint32_t p_view) const {
	if (!frame_synthesis_ext) {
		return;
	}

	// Called during initialization, we can safely access this.
	if (p_view < render_state.config_views.size()) {
		const XrFrameSynthesisConfigViewEXT &config_view = render_state.config_views[p_view];

		print_line(" - motion vector width: ", itos(config_view.recommendedMotionVectorImageRectWidth));
		print_line(" - motion vector height: ", itos(config_view.recommendedMotionVectorImageRectHeight));
	}
}

void OpenXRFrameSynthesisExtension::on_session_destroyed() {
	if (!frame_synthesis_ext) {
		return;
	}

	// Free our swapchains.
	free_swapchains();
}

void OpenXRFrameSynthesisExtension::on_main_swapchains_created() {
	if (!frame_synthesis_ext) {
		return;
	}

	// It is possible that our swapchain information gets resized,
	// and that our motion vector and depth resolution changes with this.
	// So (re)create our swapchains here as well.
	// Note that we do this even if motion vectors aren't enabled yet.

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);

	// Out with the old.
	free_swapchains();

	// We only support stereo.
	size_t view_count = render_state.config_views.size();
	ERR_FAIL_COND(view_count != 2);

	// Determine specific values for each renderer.
	int swapchain_format = 0;
	int depth_swapchain_format = 0;
	String rendering_driver_name = rendering_server->get_current_rendering_driver_name();
	if (rendering_driver_name.contains("opengl")) {
		swapchain_format = GL_RGBA16F;
		depth_swapchain_format = GL_DEPTH24_STENCIL8;
	} else if (rendering_driver_name == "vulkan") {
		String rendering_method = rendering_server->get_current_rendering_method();
		if (rendering_method == "mobile") {
			swapchain_format = VK_FORMAT_R16G16B16A16_SFLOAT;
			depth_swapchain_format = VK_FORMAT_D24_UNORM_S8_UINT;
		} else {
			WARN_PRINT("OpenXR: Frame synthesis not supported for this rendering method!");
			frame_synthesis_ext = false;
			return;
		}
	} else {
		WARN_PRINT("OpenXR: Frame synthesis not supported for this rendering driver!");
		frame_synthesis_ext = false;
		return;
	}

	// We assume the size for each eye is the same, it should be.
	uint32_t width = render_state.config_views[0].recommendedMotionVectorImageRectWidth;
	uint32_t height = render_state.config_views[0].recommendedMotionVectorImageRectHeight;

	// Create swapchains for motion vectors and depth.
	render_state.swapchains[SWAPCHAIN_MOTION_VECTOR].create(0, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT, swapchain_format, width, height, 1, view_count);
	render_state.swapchains[SWAPCHAIN_DEPTH].create(0, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT, depth_swapchain_format, width, height, 1, view_count);

	// Set up our frame synthesis info.
	render_state.frame_synthesis_info.resize(view_count);

	uint32_t index = 0;
	for (XrFrameSynthesisInfoEXT &frame_synthesis_info : render_state.frame_synthesis_info) {
		frame_synthesis_info.type = XR_TYPE_FRAME_SYNTHESIS_INFO_EXT;
		frame_synthesis_info.next = nullptr;
		frame_synthesis_info.layerFlags = 0;

		// Set up motion vector.
		frame_synthesis_info.motionVectorSubImage.swapchain = render_state.swapchains[SWAPCHAIN_MOTION_VECTOR].get_swapchain();
		frame_synthesis_info.motionVectorSubImage.imageArrayIndex = index;
		frame_synthesis_info.motionVectorSubImage.imageRect.offset.x = 0;
		frame_synthesis_info.motionVectorSubImage.imageRect.offset.y = 0;
		frame_synthesis_info.motionVectorSubImage.imageRect.extent.width = width;
		frame_synthesis_info.motionVectorSubImage.imageRect.extent.height = height;

		// Q: this should be 1.0, -1.0, 1.0. We output OpenGL NDC, frame synthesis expects Vulkan NDC, but might be a problem on runtime I'm testing.
		frame_synthesis_info.motionVectorScale = { 1.0, 1.0, 1.0, 0.0 };
		frame_synthesis_info.motionVectorOffset = { 0.0, 0.0, 0.0, 0.0 };
		frame_synthesis_info.appSpaceDeltaPose = { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 } };

		// Set up depth image.
		frame_synthesis_info.depthSubImage.swapchain = render_state.swapchains[SWAPCHAIN_DEPTH].get_swapchain();
		frame_synthesis_info.depthSubImage.imageArrayIndex = index;
		frame_synthesis_info.depthSubImage.imageRect.offset.x = 0;
		frame_synthesis_info.depthSubImage.imageRect.offset.y = 0;
		frame_synthesis_info.depthSubImage.imageRect.extent.width = width;
		frame_synthesis_info.depthSubImage.imageRect.extent.height = height;

		frame_synthesis_info.minDepth = 0.0;
		frame_synthesis_info.maxDepth = 1.0;

		// Note: reverse-Z, these are just defaults for now.
		frame_synthesis_info.nearZ = 100.0;
		frame_synthesis_info.farZ = 0.01;

		index++;
	}
}

void OpenXRFrameSynthesisExtension::on_pre_render() {
	if (!frame_synthesis_ext) {
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	size_t view_count = render_state.config_views.size();
	if (!enabled || view_count != 2 || render_state.skip_next_frame) {
		// Unset these just in case.
		openxr_api->set_velocity_texture(RID());
		openxr_api->set_velocity_depth_texture(RID());

		// Remember our transform just in case we (re)start frame synthesis later on.
		render_state.previous_transform = XRServer::get_singleton()->get_world_origin();

		return;
	}

	// Acquire our swapchains.
	for (int i = 0; i < SWAPCHAIN_MAX; i++) {
		bool should_render = true;
		render_state.swapchains[i].acquire(should_render);
	}

	// Set our images.
	openxr_api->set_velocity_texture(render_state.swapchains[SWAPCHAIN_MOTION_VECTOR].get_image());
	openxr_api->set_velocity_depth_texture(render_state.swapchains[SWAPCHAIN_DEPTH].get_image());

	// Set our size.
	uint32_t width = render_state.config_views[0].recommendedMotionVectorImageRectWidth;
	uint32_t height = render_state.config_views[0].recommendedMotionVectorImageRectHeight;
	openxr_api->set_velocity_target_size(Size2i(width, height));

	// Get our head motion
	Transform3D world_transform = XRServer::get_singleton()->get_world_origin();
	Transform3D delta_transform = render_state.previous_transform.affine_inverse() * world_transform;
	Quaternion delta_quat = delta_transform.basis.get_quaternion();
	Vector3 delta_origin = delta_transform.origin;

	// Z near/far can change per frame, so make sure we update this.
	for (XrFrameSynthesisInfoEXT &frame_synthesis_info : render_state.frame_synthesis_info) {
		frame_synthesis_info.layerFlags = render_state.relax_frame_interval ? XR_FRAME_SYNTHESIS_INFO_REQUEST_RELAXED_FRAME_INTERVAL_BIT_EXT : 0;

		frame_synthesis_info.appSpaceDeltaPose = {
			{ (float)delta_quat.x, (float)delta_quat.y, (float)delta_quat.z, (float)delta_quat.w },
			{ (float)delta_origin.x, (float)delta_origin.y, (float)delta_origin.z }
		};

		// Note: reverse-Z.
		frame_synthesis_info.nearZ = openxr_api->get_render_state_z_far();
		frame_synthesis_info.farZ = openxr_api->get_render_state_z_near();
	}

	// Remember our transform.
	render_state.previous_transform = world_transform;
}

void OpenXRFrameSynthesisExtension::on_post_draw_viewport(RID p_render_target) {
	// Check if our extension is supported and enabled.
	if (!frame_synthesis_ext || !enabled || render_state.config_views.size() != 2 || render_state.skip_next_frame) {
		return;
	}

	// Release our swapchains.
	for (int i = 0; i < SWAPCHAIN_MAX; i++) {
		render_state.swapchains[i].release();
	}
}

void *OpenXRFrameSynthesisExtension::set_projection_views_and_get_next_pointer(int p_view_index, void *p_next_pointer) {
	// Check if our extension is supported and enabled.
	if (!frame_synthesis_ext || !enabled || render_state.config_views.size() != 2) {
		return nullptr;
	}

	// Did we skip this frame?
	if (render_state.skip_next_frame) {
		// Only unset when we've handled both eyes.
		if (p_view_index == 1) {
			render_state.skip_next_frame = false;
		}
		return nullptr;
	}

	// Check if we can run frame synthesis.
	size_t view_count = render_state.config_views.size();
	if (enabled && view_count == 2) {
		render_state.frame_synthesis_info[p_view_index].next = p_next_pointer;
		return &render_state.frame_synthesis_info[p_view_index];
	}

	return nullptr;
}

bool OpenXRFrameSynthesisExtension::is_available() const {
	return frame_synthesis_ext;
}

bool OpenXRFrameSynthesisExtension::is_enabled() const {
	return frame_synthesis_ext && enabled;
}

void OpenXRFrameSynthesisExtension::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	ERR_FAIL_COND(!frame_synthesis_ext && p_enabled);

	enabled = p_enabled;

	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);
	rendering_server->call_on_render_thread(callable_mp(this, &OpenXRFrameSynthesisExtension::_set_render_state_enabled_rt).bind(enabled));
}

bool OpenXRFrameSynthesisExtension::get_relax_frame_interval() const {
	return relax_frame_interval;
}

void OpenXRFrameSynthesisExtension::set_relax_frame_interval(bool p_relax_frame_interval) {
	if (relax_frame_interval == p_relax_frame_interval) {
		return;
	}
	relax_frame_interval = p_relax_frame_interval;

	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);
	rendering_server->call_on_render_thread(callable_mp(this, &OpenXRFrameSynthesisExtension::_set_relax_frame_interval_rt).bind(relax_frame_interval));
}

void OpenXRFrameSynthesisExtension::_set_render_state_enabled_rt(bool p_enabled) {
	render_state.enabled = p_enabled;
}

void OpenXRFrameSynthesisExtension::_set_relax_frame_interval_rt(bool p_relax_frame_interval) {
	render_state.relax_frame_interval = p_relax_frame_interval;
}

void OpenXRFrameSynthesisExtension::free_swapchains() {
	for (int i = 0; i < SWAPCHAIN_MAX; i++) {
		render_state.swapchains[i].queue_free();
	}
}

void OpenXRFrameSynthesisExtension::skip_next_frame() {
	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);
	rendering_server->call_on_render_thread(callable_mp(this, &OpenXRFrameSynthesisExtension::_set_skip_next_frame_rt));
}

void OpenXRFrameSynthesisExtension::_set_skip_next_frame_rt() {
	render_state.skip_next_frame = true;
}
