/**************************************************************************/
/*  openxr_foveated_inset_extension.cpp                                   */
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

#include "openxr_foveated_inset_extension.h"

#include "../scene/openxr_foveated_inset_viewport.h"

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h" // IWYU pragma: keep. Used via `Node::get_window()`.
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/xr/xr_server.h"

OpenXRFoveatedInsetExtension *OpenXRFoveatedInsetExtension::singleton = nullptr;

OpenXRFoveatedInsetExtension *OpenXRFoveatedInsetExtension::get_singleton() {
	return singleton;
}

OpenXRFoveatedInsetExtension::OpenXRFoveatedInsetExtension() {
	singleton = this;
}

OpenXRFoveatedInsetExtension::~OpenXRFoveatedInsetExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRFoveatedInsetExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	// Only request our extensions if we're attempting to use foveated inset.
	int view_configuration_setting = GLOBAL_GET("xr/openxr/view_configuration");
	if (view_configuration_setting == 2) {
		// Extension was replaced in OpenXR 1.1, use `XR_VARJO_quad_views` in OpenXR 1.0.
		// Note: we currently always include this as there is a dependency with `XR_VARJO_foveated_rendering`.
		request_extensions[XR_VARJO_QUAD_VIEWS_EXTENSION_NAME] = &varjo_ext_available;

		// This enables eye tracked foveated inset support.
		request_extensions[XR_VARJO_FOVEATED_RENDERING_EXTENSION_NAME] = &varjo_foveated_rendering_ext_available;
	}

	return request_extensions;
}

void OpenXRFoveatedInsetExtension::on_session_created(const XrSession p_session) {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// We must create a tracker for our foveated inset if we use the foveated inset rendering.
	if (openxr_api->get_view_configuration() == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET) {
		XRServer *xr_server = XRServer::get_singleton();
		ERR_FAIL_NULL(xr_server);

		foveated_inset.instantiate();
		foveated_inset->set_tracker_type(XRServer::TRACKER_CAMERA);
		foveated_inset->set_tracker_name(XR_TRACKER_INSET);
		foveated_inset->set_tracker_desc("Foveated Inset View");
		xr_server->add_tracker(foveated_inset);
	}
}

void OpenXRFoveatedInsetExtension::on_state_ready() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// It's too early in `on_session_created` to create our viewport.
	// It would be nicer if we could do this earlier...
	bool create_inset_viewport = GLOBAL_GET("xr/openxr/create_default_foveated_inset_viewport");
	if (openxr_api->get_view_configuration() == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET && create_inset_viewport) {
		OS *os = OS::get_singleton();
		ERR_FAIL_NULL(os);

		MainLoop *main_loop = os->get_main_loop();
		ERR_FAIL_NULL(main_loop);

		SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);
		ERR_FAIL_NULL(scene_tree);

		Node *root = scene_tree->get_root();
		ERR_FAIL_NULL(root);

		inset_viewport = memnew(OpenXRFoveatedInsetViewport);
		inset_viewport->set_name("OpenXRFoveatedInsetViewport");
		root->add_child(inset_viewport);
	}
}

void OpenXRFoveatedInsetExtension::on_session_destroyed() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	if (inset_viewport) {
		Node *parent = inset_viewport->get_parent();
		if (parent) {
			parent->remove_child(inset_viewport);
		}
		inset_viewport->queue_free();
		inset_viewport = nullptr;
	}

	_free_swapchains();

	if (foveated_inset.is_valid()) {
		xr_server->remove_tracker(foveated_inset);
		foveated_inset.unref();
	}
}

void OpenXRFoveatedInsetExtension::on_process() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	if (foveated_inset.is_valid()) {
		// We use the same center transform here as for our head tracker.
		Transform3D t;
		Vector3 linear_velocity;
		Vector3 angular_velocity;

		head_confidence = openxr_api->get_head_center(t, linear_velocity, angular_velocity);
		if (head_confidence != XRPose::XR_TRACKING_CONFIDENCE_NONE) {
			// Only update our transform if we have one to update it with
			// note that poses are stored without world scale and reference frame applied!
			head_transform = t;
			head_linear_velocity = linear_velocity;
			head_angular_velocity = angular_velocity;
		}

		foveated_inset->set_pose("default", head_transform, head_linear_velocity, head_angular_velocity, head_confidence);
	}
}

TypedArray<Projection> OpenXRFoveatedInsetExtension::get_camera_projections(const StringName &p_tracker_name, double p_aspect, double p_z_near, double p_z_far) {
	TypedArray<Projection> camera_projections;

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, camera_projections);

	if (p_tracker_name == XR_TRACKER_INSET) {
		if (openxr_api->get_view_configuration() != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET) {
			// It's possible our application was configured to support foveated inset,
			// but the current hardware used does not support it.
			return camera_projections;
		}

		ERR_FAIL_COND_V(openxr_api->get_view_count() != 4, camera_projections);

		for (uint32_t v = 2; v < 4; v++) {
			Projection cm;
			openxr_api->get_view_projection(v, p_z_near, p_z_far, cm);
			camera_projections.push_back(cm);
		}
	}

	return camera_projections;
}

TypedArray<Transform3D> OpenXRFoveatedInsetExtension::get_camera_offsets(const StringName &p_tracker_name) {
	TypedArray<Transform3D> camera_offsets;

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, camera_offsets);

	if (p_tracker_name == XR_TRACKER_INSET) {
		if (openxr_api->get_view_configuration() != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET) {
			// It's possible our application was configured to support foveated inset,
			// but the current hardware used does not support it.
			return camera_offsets;
		}

		ERR_FAIL_COND_V(openxr_api->get_view_count() != 4, camera_offsets);

		XRServer *xr_server = XRServer::get_singleton();
		ERR_FAIL_NULL_V(xr_server, camera_offsets);

		double world_scale = xr_server->get_world_scale();

		for (uint32_t v = 2; v < 4; v++) {
			Transform3D t;
			openxr_api->get_view_offset(v, t);

			// Apply our world scale
			t.origin *= world_scale;

			camera_offsets.push_back(t);
		}
	}

	return camera_offsets;
}

Size2i OpenXRFoveatedInsetExtension::get_render_size() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Size2i());

#ifdef DEBUG_ENABLED
	// Views 2 and 3 should have the same size!
	ERR_FAIL_COND_V(openxr_api->get_recommended_target_size(2) != openxr_api->get_recommended_target_size(3), Size2i());
#endif

	return openxr_api->get_recommended_target_size(2);
}

void OpenXRFoveatedInsetExtension::register_viewport(RID p_viewport) {
	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);

	rendering_server->call_on_render_thread(callable_mp(this, &OpenXRFoveatedInsetExtension::_register_viewport_rt).bind(p_viewport));
}

void OpenXRFoveatedInsetExtension::unregister_viewport(RID p_viewport) {
	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);

	rendering_server->call_on_render_thread(callable_mp(this, &OpenXRFoveatedInsetExtension::_unregister_viewport_rt).bind(p_viewport));
}

void OpenXRFoveatedInsetExtension::_register_viewport_rt(RID p_viewport) {
	render_state.viewports.push_back(p_viewport);
}

void OpenXRFoveatedInsetExtension::_unregister_viewport_rt(RID p_viewport) {
	render_state.viewports.erase(p_viewport);
}

void OpenXRFoveatedInsetExtension::on_pre_render() {
	if (render_state.viewports.size() == 0) {
		return;
	} else if (render_state.viewports.size() > 1) {
		WARN_PRINT("Multiple OpenXRFoveatedInsetViewport nodes are active, using the first.");
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	if (openxr_api->get_view_configuration() != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET) {
		return;
	}

	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);

	RID render_target = rendering_server->viewport_get_render_target(render_state.viewports[0]);
	if (!render_target.is_valid()) {
		return;
	}

	// Check if we need to update our swap chains.
	Size2i new_size = openxr_api->get_recommended_target_size(2);
	if (render_state.size != new_size) {
		_free_swapchains();

		int64_t color_swapchain_format = openxr_api->get_color_swapchain_format();
		if (color_swapchain_format != 0) {
			if (!render_state.swapchains[SWAPCHAIN_COLOR].create(0, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT, color_swapchain_format, new_size.width, new_size.height, 1, 2, true)) {
				return;
			}

			openxr_api->set_object_name(XR_OBJECT_TYPE_SWAPCHAIN, uint64_t(render_state.swapchains[SWAPCHAIN_COLOR].get_swapchain()), "Inset color swapchain");
		}

		// TODO check depth swapchain.

		// Our new size is now applicable.
		render_state.size = new_size;

		// Set projections for views 2 and 3.
		for (uint32_t i = 2; i < 4; i++) {
			openxr_api->set_projection_view_swapchain_rt(i, render_state.swapchains[SWAPCHAIN_COLOR].get_swapchain(), i - 2, new_size);
		}

		// TODO Set depth for views 2 and 3.
	}

	// Check if we need to acquire our swap chains.
	for (int i = 0; i < SWAPCHAIN_MAX; i++) {
		if (!render_state.swapchains[i].is_image_acquired() && render_state.swapchains[i].get_swapchain() != XR_NULL_HANDLE) {
			bool should_render = true;
			if (!render_state.swapchains[i].acquire(should_render)) {
				return;
			}
		}
	}

	// Set our swapchains on our render target.
	RID color_texture = render_state.swapchains[SWAPCHAIN_COLOR].get_image();
	RID depth_texture = render_state.swapchains[SWAPCHAIN_DEPTH].get_image();
	RSG::texture_storage->render_target_set_override(render_target, color_texture, depth_texture, RID(), RID());
}

void OpenXRFoveatedInsetExtension::on_post_render() {
	if (render_state.viewports.size() == 0) {
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	if (openxr_api->get_view_configuration() != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET) {
		return;
	}

	// Release our swapchain image if we acquired it.
	for (int i = 0; i < SWAPCHAIN_MAX; i++) {
		if (render_state.swapchains[i].is_image_acquired()) {
			render_state.swapchains[i].release();
		}
	}
}

void OpenXRFoveatedInsetExtension::_free_swapchains() {
	for (int i = 0; i < SWAPCHAIN_MAX; i++) {
		render_state.swapchains[i].queue_free();
	}
}
