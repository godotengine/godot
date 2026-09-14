/**************************************************************************/
/*  openxr_spatial_container_extension.h                                  */
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
#include "openxr_spatial_container_state.h"

#include "core/templates/rid_owner.h"

#include <openxr/openxr.h>

class OpenXRSpatialContainerSelfRenderingExtension;

// Spatial container extension
class OpenXRSpatialContainerExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRSpatialContainerExtension, OpenXRExtensionWrapper);

public:
	struct SpatialContainerData {
		RID spatial_container_rid = RID();
		XrSpatialContainerEXT spatial_container_handle = XR_NULL_HANDLE;
		XrSpace space_handle = XR_NULL_HANDLE;
	};

	static OpenXRSpatialContainerExtension *get_singleton();

	OpenXRSpatialContainerExtension();
	virtual ~OpenXRSpatialContainerExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_instance_destroyed() override;
	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;
	virtual bool on_event_polled(const XrEventDataBuffer &event) override;
	virtual void on_state_loss_pending() override;

	virtual void *set_session_create_and_get_next_pointer(void *p_next_pointer) override;
	virtual void *set_system_properties_and_get_next_pointer(void *p_next_pointer) override;

	bool is_enabled() const;

	bool is_spatial_container_active() const;

	Array get_supported_bounds_modes();

	Ref<OpenXRSpatialContainerState> get_spatial_container_state();
	Vector3 get_spatial_container_bounds();
	bool request_spatial_container_bounds_mode(OpenXRSpatialContainerState::BoundsMode p_mode);
	bool request_spatial_container_visible(bool p_visible);

	XrResult locate_spatial_container_views(XrView *p_views, bool &r_view_pose_valid, bool &r_should_submit_layers);

protected:
	static void _bind_methods();

private:
	// OpenXR API call wrappers
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialContainerEXT,
			(XrSession), session,
			(const XrSpatialContainerCreateInfoEXT *), create_info,
			(XrSpatialContainerEXT *), spatial_container);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroySpatialContainerEXT,
			(XrSpatialContainerEXT), spatial_container);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialContainerSpaceEXT,
			(XrSession), session,
			(const XrSpatialContainerSpaceCreateInfoEXT *), create_info,
			(XrSpace *), space);
	EXT_PROTO_XRRESULT_FUNC2(xrRequestSpatialContainerVisibleEXT,
			(XrSpatialContainerEXT), spatial_container,
			(const XrSpatialContainerVisibleRequestInfoEXT *), info);
	EXT_PROTO_XRRESULT_FUNC2(xrRequestSpatialContainerBoundsModeEXT,
			(XrSpatialContainerEXT), spatial_container,
			(const XrSpatialContainerBoundsModeRequestInfoEXT *), info);
	EXT_PROTO_XRRESULT_FUNC3(xrGetSpatialContainerBoundsEXT,
			(XrSpatialContainerEXT), spatial_container,
			(const XrSpatialContainerBoundsGetInfoEXT *), get_info,
			(XrSpatialContainerBoundsEXT *), bounds);
	EXT_PROTO_XRRESULT_FUNC3(xrGetSpatialContainerStateEXT,
			(XrSpatialContainerEXT), spatial_container,
			(const XrSpatialContainerStateGetInfoEXT *), get_info,
			(XrSpatialContainerStateEXT *), state);
	EXT_PROTO_XRRESULT_FUNC5(xrEnumerateSupportedSpatialContainerGraphicsPresentationsEXT,
			(XrInstance), instance,
			(XrSystemId), system_id,
			(uint32_t), graphics_presentation_capacity_input,
			(uint32_t *), graphics_presentation_count_output,
			(XrSpatialContainerGraphicsPresentationEXT *), graphics_presentations);

	bool can_create_spatial_container() const;

	void create_spatial_container(Vector3 p_suggested_bounds);
	void destroy_spatial_container();
	void _cleanup();

	mutable RID_Owner<SpatialContainerData> spatial_container_owner;

	bool _initialize_spatial_container_extension(const XrInstance &p_instance);

	static OpenXRSpatialContainerExtension *singleton;

	bool spatial_container_ext = false;
	XrSessionCreateInfoSpatialContainersEXT session_create_info = {
		XR_TYPE_SESSION_CREATE_INFO_SPATIAL_CONTAINERS_EXT, // type
		nullptr, // next
	};
	XrSystemSpatialContainerPropertiesEXT spatial_container_properties = {
		XR_TYPE_SYSTEM_SPATIAL_CONTAINER_PROPERTIES_EXT, // type
		nullptr, // next
		0, // maxSpatialContainerCount
		XR_FALSE, // supportsBounded
		XR_FALSE, // supportsImmersive
	};
	SpatialContainerData spatial_container_data;

	// TODO: For now we are hardcoding self rendering as the rendering mechanism.
	OpenXRSpatialContainerSelfRenderingExtension *rendering_mechanism = nullptr;
};
