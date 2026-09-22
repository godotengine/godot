/**************************************************************************/
/*  openxr_spatial_container_self_rendering_extension.cpp                 */
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

#include "openxr_spatial_container_self_rendering_extension.h"

#include "../../openxr_api.h"

#include "core/object/class_db.h"

OpenXRSpatialContainerSelfRenderingExtension *OpenXRSpatialContainerSelfRenderingExtension::singleton = nullptr;

OpenXRSpatialContainerSelfRenderingExtension *OpenXRSpatialContainerSelfRenderingExtension::get_singleton() {
	return singleton;
}

OpenXRSpatialContainerSelfRenderingExtension::OpenXRSpatialContainerSelfRenderingExtension() {
	singleton = this;
}

OpenXRSpatialContainerSelfRenderingExtension::~OpenXRSpatialContainerSelfRenderingExtension() {
	_cleanup();
	singleton = nullptr;
}

void OpenXRSpatialContainerSelfRenderingExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_enabled"), &OpenXRSpatialContainerSelfRenderingExtension::is_enabled);
}

bool OpenXRSpatialContainerSelfRenderingExtension::is_enabled() const {
	return spatial_container_self_rendering_ext;
}

XrSpatialContainerGraphicsPresentationEXT OpenXRSpatialContainerSelfRenderingExtension::get_spatial_container_graphics_presentation() const {
	return XR_SPATIAL_CONTAINER_GRAPHICS_PRESENTATION_SELF_RENDERING_EXT;
}

HashMap<String, bool *> OpenXRSpatialContainerSelfRenderingExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_EXT_SPATIAL_CONTAINER_SELF_RENDERING_EXTENSION_NAME] = &spatial_container_self_rendering_ext;
	return request_extensions;
}

bool OpenXRSpatialContainerSelfRenderingExtension::_initialize_spatial_container_self_rendering_extension(const XrInstance &p_instance) {
	EXT_INIT_XR_FUNC_V(xrBeginSpatialContainerRenderingEXT);
	EXT_INIT_XR_FUNC_V(xrEndSpatialContainerRenderingEXT);
	EXT_INIT_XR_FUNC_V(xrLocateSpatialContainerViewsEXT);

	return true;
}

void OpenXRSpatialContainerSelfRenderingExtension::_cleanup() {
	spatial_container_self_rendering_ext = false;
}

void OpenXRSpatialContainerSelfRenderingExtension::on_instance_created(const XrInstance p_instance) {
	if (!spatial_container_self_rendering_ext) {
		return;
	}

	if (!_initialize_spatial_container_self_rendering_extension(p_instance)) {
		print_error("OpenXR: Failed to initialize spatial container self rendering extension");
		spatial_container_self_rendering_ext = false;
	}
}

void OpenXRSpatialContainerSelfRenderingExtension::on_session_created(const XrSession p_session) {
	if (!is_enabled()) {
		return;
	}

	ERR_FAIL_NULL(OpenXRAPI::get_singleton());
	OpenXRAPI::get_singleton()->register_frame_info_extension(this);
	OpenXRAPI::get_singleton()->register_projection_layer_extension(this);
}

void OpenXRSpatialContainerSelfRenderingExtension::on_session_destroyed() {
	if (!is_enabled()) {
		return;
	}

	ERR_FAIL_NULL(OpenXRAPI::get_singleton());
	OpenXRAPI::get_singleton()->unregister_frame_info_extension(this);
	OpenXRAPI::get_singleton()->unregister_projection_layer_extension(this);
}

void OpenXRSpatialContainerSelfRenderingExtension::on_process() {
	if (!is_enabled()) {
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	if (openxr_api->is_running() && has_pending_container) {
		print_verbose("OpenXR: Adding pending container to render set...");
		_add_spatial_container_to_render_set(pending_container_data);
	}
}

void OpenXRSpatialContainerSelfRenderingExtension::on_spatial_container_created(const OpenXRSpatialContainerExtension::SpatialContainerData &p_container_data) {
	if (!is_enabled()) {
		return;
	}

	_add_spatial_container_to_render_set(p_container_data);
}

void OpenXRSpatialContainerSelfRenderingExtension::on_spatial_container_visibility_changed(const OpenXRSpatialContainerExtension::SpatialContainerData &p_container_data, bool p_visible) {
	if (!is_enabled()) {
		return;
	}

	if (p_visible) {
		_add_spatial_container_to_render_set(p_container_data);
	} else {
		_remove_spatial_container_from_render_set(p_container_data.spatial_container_handle);
	}
}

void OpenXRSpatialContainerSelfRenderingExtension::on_spatial_container_closed(const OpenXRSpatialContainerExtension::SpatialContainerData &p_container_data) {
	if (!is_enabled()) {
		return;
	}

	_remove_spatial_container_from_render_set(p_container_data.spatial_container_handle);
}

void OpenXRSpatialContainerSelfRenderingExtension::_add_spatial_container_to_render_set(const OpenXRSpatialContainerExtension::SpatialContainerData &p_container_data) {
	// Check if the spatial container is already part of the active render set.
	if (active_spatial_container == p_container_data.spatial_container_handle) {
		print_verbose("OpenXR: Spatial container is already part of the active render set.");
		return;
	}

	pending_container_data = p_container_data;
	has_pending_container = true;

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	if (!openxr_api->is_running()) {
		print_verbose("OpenXR: Session is not running yet...");
		return;
	}
	const XrSession session = openxr_api->get_session();

	if (active_spatial_container != XR_NULL_HANDLE) {
		// Shouldn't happen, but in case it does...
		print_verbose("OpenXR: Pre-existing active spatial container. Cleaning it up.");
		_remove_spatial_container_from_render_set(active_spatial_container);
	}

	XrSpatialContainerBeginInfoEXT begin_info = {
		XR_TYPE_SPATIAL_CONTAINER_BEGIN_INFO_EXT, // type
		nullptr, // next
		p_container_data.spatial_container_handle, // spatialContainer
		openxr_api->get_view_configuration() // primaryViewConfigurationType
	};
	XrResult result = xrBeginSpatialContainerRenderingEXT(session, &begin_info);
	if (XR_FAILED(result)) {
		ERR_PRINT("OpenXR: Failed to begin spatial container rendering.");
		has_pending_container = false;
		return;
	}

	has_pending_container = false;

	// Add the spatial container to the active render set.
	print_verbose("OpenXR: Setting active spatial container.");
	active_spatial_container = p_container_data.spatial_container_handle;

	active_view_locate_info.viewConfigurationType = openxr_api->get_view_configuration();
	active_view_locate_info.space = p_container_data.space_handle;
	active_view_locate_info.spatialContainer = active_spatial_container;

	active_view_state.viewConfigurationType = openxr_api->get_view_configuration();

	active_layer.spatialContainer = active_spatial_container;
}

void OpenXRSpatialContainerSelfRenderingExtension::_remove_spatial_container_from_render_set(const XrSpatialContainerEXT &p_container) {
	// Check if the spatial container is in the active render set.
	if (active_spatial_container != p_container) {
		print_verbose("OpenXR: Spatial container is not part of the active render set.");
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	const XrSession session = openxr_api->get_session();

	XrSpatialContainerEndInfoEXT end_info = {
		XR_TYPE_SPATIAL_CONTAINER_END_INFO_EXT, // type
		nullptr, // next
		p_container, // spatialContainer
	};
	XrResult result = xrEndSpatialContainerRenderingEXT(session, &end_info);
	if (XR_FAILED(result)) {
		ERR_PRINT("OpenXR: Failed to end spatial container rendering.");
		return;
	}

	// Reset the active spatial container.
	print_verbose("OpenXR: Resetting active spatial container.");
	active_spatial_container = XR_NULL_HANDLE;

	active_view_locate_info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
	active_view_locate_info.space = XR_NULL_HANDLE;
	active_view_locate_info.spatialContainer = XR_NULL_HANDLE;

	active_view_state.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
	active_view_state.viewStateFlags = 0;
	active_view_state.shouldSubmitLayers = false;
	active_view_state.recommendedImageExtent = { 0, 0 };

	active_layer.spatialContainer = XR_NULL_HANDLE;
	active_layer.retainPreviousSubmission = false;
	active_layer.layerCount = 0;
	active_layer.layers = nullptr;
}

void *OpenXRSpatialContainerSelfRenderingExtension::set_frame_end_info_and_get_next_pointer(void *p_next_pointer) {
	if (!is_enabled() || active_spatial_container == XR_NULL_HANDLE) {
		return p_next_pointer;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, p_next_pointer);

	if (active_view_state.shouldSubmitLayers) {
		submit_layers[0] = reinterpret_cast<const XrCompositionLayerBaseHeader *>(openxr_api->get_projection_layer());
		active_layer.layerCount = 1;
		active_layer.layers = submit_layers;
	} else {
		active_layer.layerCount = 0;
		active_layer.layers = nullptr;
	}

	// Update the container layers.
	frame_end_info.containerLayerCount = 1;
	frame_end_info.containerLayers = &active_layer;

	frame_end_info.next = p_next_pointer;
	return &frame_end_info;
}

void *OpenXRSpatialContainerSelfRenderingExtension::set_projection_layer_and_get_next_pointer(void *p_next_pointer) {
	if (!is_enabled() || active_spatial_container == XR_NULL_HANDLE) {
		return p_next_pointer;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, p_next_pointer);

	container_view_config.viewConfigurationType = openxr_api->get_view_configuration();
	container_view_config.next = p_next_pointer;
	return &container_view_config;
}

XrResult OpenXRSpatialContainerSelfRenderingExtension::locate_spatial_container_views(RID p_spatial_container_rid, XrView *p_views, bool &r_view_pose_valid, bool &r_should_submit_layers) {
	if (!is_enabled() || active_spatial_container == XR_NULL_HANDLE) {
		return XR_ERROR_INITIALIZATION_FAILED;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, XR_ERROR_VALIDATION_FAILURE);
	const XrSession session = openxr_api->get_session();

	void *spatial_container_views_locate_info_next_pointer = nullptr;
	for (OpenXRExtensionWrapper *extension : openxr_api->frame_info_extensions) {
		void *np = extension->set_spatial_container_views_locate_info_and_get_next_pointer(p_spatial_container_rid, spatial_container_views_locate_info_next_pointer);
		if (np != nullptr) {
			spatial_container_views_locate_info_next_pointer = np;
		}
	}

	XrSpatialContainerViewsLocateInfoEXT locate_info = {
		XR_TYPE_SPATIAL_CONTAINER_VIEWS_LOCATE_INFO_EXT, // type
		spatial_container_views_locate_info_next_pointer, // next
		openxr_api->get_predicted_display_time(), // displayTime
		1, // viewLocateInfoCount
		&active_view_locate_info, // viewLocateInfos
	};

	uint32_t total_view_count = openxr_api->get_view_count();

	XrResult result = xrLocateSpatialContainerViewsEXT(session, &locate_info, 1, &active_view_state, total_view_count, p_views);
	if (XR_FAILED(result)) {
		ERR_FAIL_V_MSG(result, "OpenXR: Couldn't locate spatial container views [" + openxr_api->get_error_string(result) + "]");
	}

	r_should_submit_layers = active_view_state.shouldSubmitLayers;
	r_view_pose_valid = !((active_view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) == 0 ||
			(active_view_state.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) == 0);

	return result;
}
