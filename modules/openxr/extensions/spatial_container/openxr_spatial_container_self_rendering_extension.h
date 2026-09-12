/**************************************************************************/
/*  openxr_spatial_container_self_rendering_extension.h                   */
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

#include "../../openxr_util.h"
#include "../../util.h"
#include "../openxr_extension_wrapper.h"
#include "openxr_spatial_container_extension.h"
#include "openxr_spatial_container_state.h"

#include <openxr/openxr.h>

// Spatial container self rendering extension
class OpenXRSpatialContainerSelfRenderingExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRSpatialContainerSelfRenderingExtension, OpenXRExtensionWrapper);

public:
	static OpenXRSpatialContainerSelfRenderingExtension *get_singleton();

	OpenXRSpatialContainerSelfRenderingExtension();
	virtual ~OpenXRSpatialContainerSelfRenderingExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;
	virtual void on_process() override;

	virtual void *set_frame_end_info_and_get_next_pointer(void *p_next_pointer) override;
	virtual void *set_projection_layer_and_get_next_pointer(void *p_next_pointer) override;

	bool is_enabled() const;

	// Spatial container rendering mechanism APIs.
	void on_spatial_container_created(const OpenXRSpatialContainerExtension::SpatialContainerData &p_container_data);
	void on_spatial_container_visibility_changed(const OpenXRSpatialContainerExtension::SpatialContainerData &p_container, bool p_visible);
	void on_spatial_container_closed(const OpenXRSpatialContainerExtension::SpatialContainerData &p_container);
	XrSpatialContainerGraphicsPresentationEXT get_spatial_container_graphics_presentation() const;
	XrResult locate_spatial_container_views(RID p_spatial_container_rid, XrView *p_views, bool &r_view_pose_valid, bool &r_should_submit_layers);

protected:
	static void _bind_methods();

private:
	// OpenXR API call wrappers
	EXT_PROTO_XRRESULT_FUNC2(xrBeginSpatialContainerRenderingEXT,
			(XrSession), session,
			(const XrSpatialContainerBeginInfoEXT *), begin_info);
	EXT_PROTO_XRRESULT_FUNC2(xrEndSpatialContainerRenderingEXT,
			(XrSession), session,
			(const XrSpatialContainerEndInfoEXT *), end_info);
	EXT_PROTO_XRRESULT_FUNC6(xrLocateSpatialContainerViewsEXT,
			(XrSession), session,
			(const XrSpatialContainerViewsLocateInfoEXT *), locate_info,
			(uint32_t), view_state_count,
			(XrSpatialContainerViewStateEXT *), view_states,
			(uint32_t), view_count,
			(XrView *), views);

	void _add_spatial_container_to_render_set(const OpenXRSpatialContainerExtension::SpatialContainerData &p_container_data);
	void _remove_spatial_container_from_render_set(const XrSpatialContainerEXT &p_container);
	void _cleanup();

	bool _initialize_spatial_container_self_rendering_extension(const XrInstance &p_instance);

	static OpenXRSpatialContainerSelfRenderingExtension *singleton;

	bool spatial_container_self_rendering_ext = false;

	XrSpatialContainerEXT active_spatial_container = XR_NULL_HANDLE;
	XrSpatialContainerViewLocateInfoEXT active_view_locate_info = {
		XR_TYPE_SPATIAL_CONTAINER_VIEW_LOCATE_INFO_EXT, // type
		nullptr, // next
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, // viewConfigurationType
		XR_NULL_HANDLE, // space
		XR_NULL_HANDLE, // spatialContainer
	};
	XrSpatialContainerViewStateEXT active_view_state = {
		XR_TYPE_SPATIAL_CONTAINER_VIEW_STATE_EXT, // type
		nullptr, // next
		0, // viewStateFlags
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, // viewConfigurationType
		false, // shouldSubmitLayers
		{ 0, 0 }, // recommendedImageExtent
	};
	XrSpatialContainerLayerEXT active_layer = {
		XR_TYPE_SPATIAL_CONTAINER_LAYER_EXT, // type
		nullptr, // next
		XR_NULL_HANDLE, // spatialContainer
		false, // retainPreviousSubmission
		0, // layerCount
		nullptr, // layers
	};

	XrSpatialContainerLayerFrameEndInfoEXT frame_end_info = {
		XR_TYPE_SPATIAL_CONTAINER_LAYER_FRAME_END_INFO_EXT, // type
		nullptr, // next
		0, // containerLayerCount
		nullptr // containerLayers
	};

	XrSpatialContainerCompositionLayerViewConfigurationEXT container_view_config = {
		XR_TYPE_SPATIAL_CONTAINER_COMPOSITION_LAYER_VIEW_CONFIGURATION_EXT, // type
		nullptr, // next
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, // viewConfigurationType
	};

	OpenXRSpatialContainerExtension::SpatialContainerData pending_container_data;
	bool has_pending_container = false;
	const XrCompositionLayerBaseHeader *submit_layers[1] = { nullptr };
};
