/**************************************************************************/
/*  openxr_spatial_container_extension.cpp                                */
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

#include "openxr_spatial_container_extension.h"

#include "../../openxr_api.h"
#include "openxr_spatial_container_self_rendering_extension.h"

#include "core/config/project_settings.h"
#include "core/object/class_db.h"

OpenXRSpatialContainerExtension *OpenXRSpatialContainerExtension::singleton = nullptr;

OpenXRSpatialContainerExtension *OpenXRSpatialContainerExtension::get_singleton() {
	return singleton;
}

OpenXRSpatialContainerExtension::OpenXRSpatialContainerExtension() {
	singleton = this;
}

OpenXRSpatialContainerExtension::~OpenXRSpatialContainerExtension() {
	_cleanup();
	singleton = nullptr;
}

bool OpenXRSpatialContainerExtension::is_enabled() const {
	return spatial_container_ext && rendering_mechanism && rendering_mechanism->is_enabled();
}

bool OpenXRSpatialContainerExtension::is_spatial_container_active() const {
	return is_enabled() && spatial_container_data.spatial_container_handle != XR_NULL_HANDLE;
}

bool OpenXRSpatialContainerExtension::can_create_spatial_container() const {
	print_line("OpenXR: Max spatial container count: ", spatial_container_properties.maxSpatialContainerCount);
	return is_enabled() && spatial_container_properties.maxSpatialContainerCount > 0;
}

void OpenXRSpatialContainerExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_enabled"), &OpenXRSpatialContainerExtension::is_enabled);
	ClassDB::bind_method(D_METHOD("is_spatial_container_active"), &OpenXRSpatialContainerExtension::is_spatial_container_active);
	ClassDB::bind_method(D_METHOD("get_supported_bounds_modes"), &OpenXRSpatialContainerExtension::get_supported_bounds_modes);

	ClassDB::bind_method(D_METHOD("get_spatial_container_state"), &OpenXRSpatialContainerExtension::get_spatial_container_state);
	ClassDB::bind_method(D_METHOD("get_spatial_container_bounds"), &OpenXRSpatialContainerExtension::get_spatial_container_bounds);
	ClassDB::bind_method(D_METHOD("request_spatial_container_bounds_mode", "bounds_mode"), &OpenXRSpatialContainerExtension::request_spatial_container_bounds_mode);
	ClassDB::bind_method(D_METHOD("request_spatial_container_visible", "visible"), &OpenXRSpatialContainerExtension::request_spatial_container_visible);

	ADD_SIGNAL(MethodInfo("spatial_container_bounds_changed", PropertyInfo(Variant::RID, "spatial_container_rid"), PropertyInfo(Variant::BOOL, "spatial_container_infinite_bounds"), PropertyInfo(Variant::INT, "spatial_container_bounds_mode", PROPERTY_HINT_ENUM, "Bounded,Immersive"), PropertyInfo(Variant::VECTOR3, "spatial_container_bounds")));
	ADD_SIGNAL(MethodInfo("spatial_container_bounds_mode_request_denied", PropertyInfo(Variant::RID, "spatial_container_rid")));
	ADD_SIGNAL(MethodInfo("spatial_container_closed", PropertyInfo(Variant::RID, "spatial_container_rid")));
	ADD_SIGNAL(MethodInfo("spatial_container_interactable_changed", PropertyInfo(Variant::RID, "spatial_container_rid"), PropertyInfo(Variant::BOOL, "spatial_container_interactable")));
	ADD_SIGNAL(MethodInfo("spatial_container_visible_changed", PropertyInfo(Variant::RID, "spatial_container_rid"), PropertyInfo(Variant::BOOL, "spatial_container_visible")));
	ADD_SIGNAL(MethodInfo("spatial_container_visible_request_denied", PropertyInfo(Variant::RID, "spatial_container_rid")));
}

HashMap<String, bool *> OpenXRSpatialContainerExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_EXT_SPATIAL_CONTAINER_EXTENSION_NAME] = &spatial_container_ext;

	return request_extensions;
}

bool OpenXRSpatialContainerExtension::_initialize_spatial_container_extension(const XrInstance &p_instance) {
	EXT_INIT_XR_FUNC_V(xrCreateSpatialContainerEXT);
	EXT_INIT_XR_FUNC_V(xrDestroySpatialContainerEXT);
	EXT_INIT_XR_FUNC_V(xrCreateSpatialContainerSpaceEXT);
	EXT_INIT_XR_FUNC_V(xrRequestSpatialContainerVisibleEXT);
	EXT_INIT_XR_FUNC_V(xrRequestSpatialContainerBoundsModeEXT);
	EXT_INIT_XR_FUNC_V(xrGetSpatialContainerBoundsEXT);
	EXT_INIT_XR_FUNC_V(xrGetSpatialContainerStateEXT);
	EXT_INIT_XR_FUNC_V(xrEnumerateSupportedSpatialContainerGraphicsPresentationsEXT);

	return true;
}

void OpenXRSpatialContainerExtension::_cleanup() {
	spatial_container_ext = false;
}

void OpenXRSpatialContainerExtension::on_instance_created(const XrInstance p_instance) {
	if (!spatial_container_ext) {
		return;
	}

	if (!_initialize_spatial_container_extension(p_instance)) {
		print_error("OpenXR: Failed to initialize spatial container extension.");
		spatial_container_ext = false;
		return;
	}

	rendering_mechanism = OpenXRSpatialContainerSelfRenderingExtension::get_singleton();
}

void OpenXRSpatialContainerExtension::on_instance_destroyed() {
	rendering_mechanism = nullptr;
}

Array OpenXRSpatialContainerExtension::get_supported_bounds_modes() {
	Array modes;

	if (spatial_container_properties.supportsBounded) {
		modes.push_back(OpenXRSpatialContainerState::BOUNDS_MODE_BOUNDED);
	}
	if (spatial_container_properties.supportsImmersive) {
		modes.push_back(OpenXRSpatialContainerState::BOUNDS_MODE_IMMERSIVE);
	}
	return modes;
}

void OpenXRSpatialContainerExtension::on_session_created(const XrSession p_session) {
	if (!can_create_spatial_container()) {
		print_verbose("OpenXR: Unable to create spatial container!");
		return;
	}

	// Retrieve the bounds from the project settings.
	Vector3 bounds = GLOBAL_GET_CACHED(Vector3, "xr/openxr/extensions/spatial_container/bounds");
	create_spatial_container(bounds);
}

void OpenXRSpatialContainerExtension::on_session_destroyed() {
	if (!can_create_spatial_container()) {
		print_verbose("OpenXR: Unable to destroy spatial container!");
		return;
	}

	destroy_spatial_container();
}

void *OpenXRSpatialContainerExtension::set_session_create_and_get_next_pointer(void *p_next_pointer) {
	if (!is_enabled()) {
		return p_next_pointer;
	}

	session_create_info.next = p_next_pointer;
	return &session_create_info;
}

void *OpenXRSpatialContainerExtension::set_system_properties_and_get_next_pointer(void *p_next_pointer) {
	if (!is_enabled()) {
		return p_next_pointer;
	}

	spatial_container_properties.next = p_next_pointer;
	return &spatial_container_properties;
}

bool OpenXRSpatialContainerExtension::on_event_polled(const XrEventDataBuffer &event) {
	if (!is_enabled()) {
		return false;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, false);

	switch ((int)event.type) {
		case XR_TYPE_EVENT_DATA_SPATIAL_CONTAINER_BOUNDS_CHANGED_EXT: {
			const XrEventDataSpatialContainerBoundsChangedEXT *event_data = (XrEventDataSpatialContainerBoundsChangedEXT *)&event;
			if (event_data->spatialContainer == spatial_container_data.spatial_container_handle) {
				print_verbose("OpenXR: Spatial container bounds changed...");
				emit_signal(SNAME("spatial_container_bounds_changed"), spatial_container_data.spatial_container_rid, event_data->infiniteBounds, OpenXRSpatialContainerState::_to_bounds_mode(event_data->boundsMode), Vector3(event_data->bounds.width, event_data->bounds.height, event_data->bounds.depth));
				return true;
			}
			return false;
		} break;

		case XR_TYPE_EVENT_DATA_SPATIAL_CONTAINER_BOUNDS_MODE_REQUEST_DENIED_EXT: {
			const XrEventDataSpatialContainerBoundsModeRequestDeniedEXT *event_data = (XrEventDataSpatialContainerBoundsModeRequestDeniedEXT *)&event;
			if (event_data->spatialContainer == spatial_container_data.spatial_container_handle) {
				print_verbose("OpenXR: Spatial container bounds mode request denied...");
				emit_signal(SNAME("spatial_container_bounds_mode_request_denied"), spatial_container_data.spatial_container_rid);
				return true;
			}
			return false;
		} break;

		case XR_TYPE_EVENT_DATA_SPATIAL_CONTAINER_CLOSED_EXT: {
			const XrEventDataSpatialContainerClosedEXT *event_data = (XrEventDataSpatialContainerClosedEXT *)&event;
			if (event_data->spatialContainer == spatial_container_data.spatial_container_handle) {
				print_verbose("OpenXR: Spatial container closed...");
				emit_signal(SNAME("spatial_container_closed"), spatial_container_data.spatial_container_rid);

				// Backward compatibility logic.
				openxr_api->on_state_exiting();

				destroy_spatial_container();
				return true;
			}
			return false;
		} break;

		case XR_TYPE_EVENT_DATA_SPATIAL_CONTAINER_INTERACTABLE_CHANGED_EXT: {
			const XrEventDataSpatialContainerInteractableChangedEXT *event_data = (XrEventDataSpatialContainerInteractableChangedEXT *)&event;
			if (event_data->spatialContainer == spatial_container_data.spatial_container_handle) {
				print_verbose("OpenXR: Spatial container interactability changed...");
				emit_signal(SNAME("spatial_container_interactable_changed"), spatial_container_data.spatial_container_rid, event_data->interactable);

				// Backward compatibility logic.
				// TODO: When we add support for multiple spatial containers within a session, this will need to be revisited as follow:
				// - Switch to state focused when AT LEAST ONE spatial container is interactable
				// - Switch to state visible when NONE of the spatial containers are interactable
				if (event_data->interactable) {
					openxr_api->on_state_focused();
				} else {
					openxr_api->on_state_visible();
				}
				return true;
			}
			return false;
		} break;

		case XR_TYPE_EVENT_DATA_SPATIAL_CONTAINER_VISIBLE_CHANGED_EXT: {
			const XrEventDataSpatialContainerVisibleChangedEXT *event_data = (XrEventDataSpatialContainerVisibleChangedEXT *)&event;
			if (event_data->spatialContainer == spatial_container_data.spatial_container_handle) {
				print_verbose("OpenXR: Spatial container visibility changed...");
				emit_signal(SNAME("spatial_container_visible_changed"), spatial_container_data.spatial_container_rid, event_data->visible);

				if (rendering_mechanism) {
					rendering_mechanism->on_spatial_container_visibility_changed(spatial_container_data, event_data->visible);
				}

				// Backward compatibility logic.
				// TODO: When we add support for multiple spatial containers within a session, this will need to be revisited as follow:
				// - Switch to state visible when AT LEAST ONE spatial container is visible
				// - Switch to state synchronized when NONE of the spatial containers are visible
				if (event_data->visible) {
					openxr_api->on_state_visible();
				} else {
					openxr_api->on_state_synchronized();
				}
				return true;
			}
			return false;
		} break;

		case XR_TYPE_EVENT_DATA_SPATIAL_CONTAINER_VISIBLE_REQUEST_DENIED_EXT: {
			const XrEventDataSpatialContainerVisibleRequestDeniedEXT *event_data = (XrEventDataSpatialContainerVisibleRequestDeniedEXT *)&event;
			if (event_data->spatialContainer == spatial_container_data.spatial_container_handle) {
				print_verbose("OpenXR: Spatial container visibility request denied...");
				emit_signal(SNAME("spatial_container_visible_request_denied"), spatial_container_data.spatial_container_rid);
				return true;
			}
			return false;
		} break;

		default: {
			return false;
		} break;
	}
}

void OpenXRSpatialContainerExtension::on_state_loss_pending() {
	if (!is_enabled()) {
		return;
	}

	destroy_spatial_container();
}

Ref<OpenXRSpatialContainerState> OpenXRSpatialContainerExtension::get_spatial_container_state() {
	Ref<OpenXRSpatialContainerState> container_state;
	if (!is_spatial_container_active()) {
		return container_state;
	}

	print_verbose("OpenXR: Retrieving Spatial container state...");
	XrSpatialContainerStateGetInfoEXT get_info = {
		XR_TYPE_SPATIAL_CONTAINER_STATE_GET_INFO_EXT, // type
		nullptr, // next
	};
	XrSpatialContainerStateEXT state = {
		XR_TYPE_SPATIAL_CONTAINER_STATE_EXT, // type
		nullptr, // next
		false, // visible
		false, // anyInteractable
		XR_SPATIAL_CONTAINER_BOUNDS_MODE_MAX_ENUM_EXT // boundsMode
	};
	XrResult result = xrGetSpatialContainerStateEXT(spatial_container_data.spatial_container_handle, &get_info, &state);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), container_state, vformat("Failed to retrieve spatial container state: %d", result));

	container_state.instantiate(state);
	return container_state;
}

Vector3 OpenXRSpatialContainerExtension::get_spatial_container_bounds() {
	if (!is_spatial_container_active()) {
		return Vector3();
	}

	print_verbose("OpenXR: Retrieving spatial container bounds...");
	XrSpatialContainerBoundsGetInfoEXT bounds_get_info = {
		XR_TYPE_SPATIAL_CONTAINER_BOUNDS_GET_INFO_EXT, // type
		nullptr, // next
	};
	XrSpatialContainerBoundsEXT bounds = {
		XR_TYPE_SPATIAL_CONTAINER_BOUNDS_EXT, // type
		nullptr, // next
		{ 0.0, 0.0, 0.0 }, // bounds
		false, // infiniteBounds
	};
	XrResult result = xrGetSpatialContainerBoundsEXT(spatial_container_data.spatial_container_handle, &bounds_get_info, &bounds);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), Vector3(), "Failed to retrieve spatial container bounds.");

	return Vector3(bounds.bounds.width, bounds.bounds.height, bounds.bounds.depth);
}

bool OpenXRSpatialContainerExtension::request_spatial_container_bounds_mode(OpenXRSpatialContainerState::BoundsMode p_mode) {
	if (!is_spatial_container_active()) {
		return false;
	}

	// Is bounds mode supported?
	if (p_mode == OpenXRSpatialContainerState::BOUNDS_MODE_BOUNDED && !spatial_container_properties.supportsBounded) {
		ERR_FAIL_V_MSG(false, "Bounded mode is not supported.");
	}
	if (p_mode == OpenXRSpatialContainerState::BOUNDS_MODE_IMMERSIVE && !spatial_container_properties.supportsImmersive) {
		ERR_FAIL_V_MSG(false, "Immersive mode is not supported.");
	}

	XrSpatialContainerBoundsModeRequestInfoEXT request_info = {
		XR_TYPE_SPATIAL_CONTAINER_BOUNDS_MODE_REQUEST_INFO_EXT, // type
		nullptr, // next
		OpenXRSpatialContainerState::_from_bounds_mode(p_mode), // boundsMode
	};
	XrResult result = xrRequestSpatialContainerBoundsModeEXT(spatial_container_data.spatial_container_handle, &request_info);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "Failed requesting bounds mode update.");

	return true;
}

XrResult OpenXRSpatialContainerExtension::locate_spatial_container_views(XrView *p_views, bool &r_view_pose_valid, bool &r_should_submit_layers) {
	if (!is_spatial_container_active()) {
		return XR_ERROR_INITIALIZATION_FAILED;
	}

	return rendering_mechanism->locate_spatial_container_views(spatial_container_data.spatial_container_rid, p_views, r_view_pose_valid, r_should_submit_layers);
}

bool OpenXRSpatialContainerExtension::request_spatial_container_visible(bool p_visible) {
	if (!is_spatial_container_active()) {
		return false;
	}

	XrSpatialContainerVisibleRequestInfoEXT request_info = {
		XR_TYPE_SPATIAL_CONTAINER_VISIBLE_REQUEST_INFO_EXT, // type
		nullptr, // next
		p_visible, // visible
	};
	XrResult result = xrRequestSpatialContainerVisibleEXT(spatial_container_data.spatial_container_handle, &request_info);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "Failed requesting visibility update.");

	return true;
}

void OpenXRSpatialContainerExtension::create_spatial_container(Vector3 p_suggested_bounds) {
	if (!can_create_spatial_container()) {
		ERR_FAIL_MSG("OpenXR: Unable to create spatial container!");
	}

	if (is_spatial_container_active()) {
		ERR_FAIL_MSG("OpenXR: Spatial container is already active.");
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	const XrSession session = openxr_api->get_session();

	print_verbose("OpenXR: Creating spatial container...");

	// Store this as a RID so we can keep track of it.
	spatial_container_data.spatial_container_rid = spatial_container_owner.make_rid(spatial_container_data);

	void *create_info_next_pointer = nullptr;
	for (OpenXRExtensionWrapper *extension : openxr_api->get_registered_extension_wrappers()) {
		void *np = extension->set_spatial_container_create_info_and_get_next_pointer(spatial_container_data.spatial_container_rid, create_info_next_pointer);
		if (np != nullptr) {
			create_info_next_pointer = np;
		}
	}

	XrSpatialContainerCreateInfoEXT create_info = {
		XR_TYPE_SPATIAL_CONTAINER_CREATE_INFO_EXT, // type
		create_info_next_pointer, // next
		rendering_mechanism->get_spatial_container_graphics_presentation(), // graphicsPresentation
		{ (float)p_suggested_bounds.x, (float)p_suggested_bounds.y, (float)p_suggested_bounds.z } // suggestedBounds
	};
	XrResult result = xrCreateSpatialContainerEXT(session, &create_info, &spatial_container_data.spatial_container_handle);
	if (XR_FAILED(result)) {
		ERR_PRINT("OpenXR: Failed to create spatial container.");
		destroy_spatial_container();
		return;
	}

	print_verbose("OpenXR: Creating spatial container space...");
	XrSpatialContainerSpaceCreateInfoEXT space_create_info = {
		XR_TYPE_SPATIAL_CONTAINER_SPACE_CREATE_INFO_EXT, // type
		nullptr, // next
		spatial_container_data.spatial_container_handle // spatialContainer
	};
	result = xrCreateSpatialContainerSpaceEXT(session, &space_create_info, &spatial_container_data.space_handle);
	if (XR_FAILED(result)) {
		ERR_PRINT("OpenXR: Failed to create spatial container space.. Destroying spatial container.");
		destroy_spatial_container();
		return;
	}

	openxr_api->set_custom_play_space(spatial_container_data.space_handle);

	print_verbose("OpenXR: Updating spatial container bounds mode...");
	OpenXRSpatialContainerState::BoundsMode bounds_mode = (OpenXRSpatialContainerState::BoundsMode)GLOBAL_GET_CACHED(int, "xr/openxr/extensions/spatial_container/bounds_mode");
	if (!request_spatial_container_bounds_mode(bounds_mode)) {
		ERR_PRINT("OpenXR: Failed updating spatial container bounds mode.");
	}

	print_verbose("OpenXR: Updating spatial container visibility...");
	if (!request_spatial_container_visible(true)) {
		ERR_PRINT("OpenXR: Failed updating spatial container visibility.");
	}

	if (rendering_mechanism) {
		rendering_mechanism->on_spatial_container_created(spatial_container_data);
	}
}

void OpenXRSpatialContainerExtension::destroy_spatial_container() {
	if (is_spatial_container_active()) {
		request_spatial_container_visible(false);
		// TODO: Ideally we should wait until the visibility request has been honored before continuing with the logic below.

		if (rendering_mechanism) {
			rendering_mechanism->on_spatial_container_closed(spatial_container_data);
		}

		ERR_FAIL_NULL(OpenXRAPI::get_singleton());
		OpenXRAPI::get_singleton()->set_custom_play_space(XR_NULL_HANDLE);

		print_verbose("OpenXR: Destroying spatial container...");
		XrResult result = xrDestroySpatialContainerEXT(spatial_container_data.spatial_container_handle);
		ERR_FAIL_COND_MSG(XR_FAILED(result), "Failed to destroy spatial container.");

		spatial_container_data.spatial_container_handle = XR_NULL_HANDLE;
		spatial_container_data.space_handle = XR_NULL_HANDLE;

		spatial_container_owner.free(spatial_container_data.spatial_container_rid);
		spatial_container_data.spatial_container_rid = RID();
	}
}
