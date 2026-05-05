/**************************************************************************/
/*  openxr_foveated_inset_viewport.cpp                                    */
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

#include "openxr_foveated_inset_viewport.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "servers/xr/xr_server.h"

#include "modules/openxr/extensions/openxr_foveated_inset_extension.h"

OpenXRFoveatedInsetViewport::OpenXRFoveatedInsetViewport() {
	set_process_internal(true);
	set_process_priority(99999);
	set_update_mode(SubViewport::UPDATE_DISABLED);

	OpenXRFoveatedInsetExtension *fi_ext = OpenXRFoveatedInsetExtension::get_singleton();
	ERR_FAIL_NULL(fi_ext);

	fi_ext->register_viewport(get_viewport_rid());

	xr_origin = memnew(XROrigin3D);
	xr_origin->set_current(false);
	add_child(xr_origin, false, Node3D::INTERNAL_MODE_BACK);

	xr_camera = memnew(XRCamera3D);
	xr_camera->set_tracker(XR_TRACKER_INSET);
	xr_origin->add_child(xr_camera, false, Node3D::INTERNAL_MODE_BACK);
}

OpenXRFoveatedInsetViewport::~OpenXRFoveatedInsetViewport() {
	OpenXRFoveatedInsetExtension *fi_ext = OpenXRFoveatedInsetExtension::get_singleton();
	ERR_FAIL_NULL(fi_ext);

	fi_ext->unregister_viewport(get_viewport_rid());
}

void OpenXRFoveatedInsetViewport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_xr_origin3d"), &OpenXRFoveatedInsetViewport::get_xr_origin3d);
	ClassDB::bind_method(D_METHOD("get_xr_camera3d"), &OpenXRFoveatedInsetViewport::get_xr_camera3d);
}

void OpenXRFoveatedInsetViewport::_validate_property(PropertyInfo &p_property) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}
	// Hide properties that are managed by us.
	if (p_property.name == "size" || p_property.name == "size_2d_override" || p_property.name == "size_2d_override_stretch" || p_property.name == "view_count" || p_property.name == "use_xr" || p_property.name == "render_target_update_mode") {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void OpenXRFoveatedInsetViewport::_notification(int p_what) {
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			// Update the size of our viewport
			_update_size();

			// Apply our world origin
			XRServer *xr_server = XRServer::get_singleton();
			if (xr_server) {
				xr_origin->set_transform(xr_server->get_world_origin());
			}
		} break;
	}
}

void OpenXRFoveatedInsetViewport::_update_size() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	OpenXRFoveatedInsetExtension *fi_ext = OpenXRFoveatedInsetExtension::get_singleton();
	if (openxr_api == nullptr || fi_ext == nullptr) {
		set_update_mode(SubViewport::UPDATE_DISABLED);
		return;
	}

	if (!openxr_api->is_initialized()) {
		set_update_mode(SubViewport::UPDATE_DISABLED);
		return;
	}

	if (openxr_api->get_view_configuration() != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET) {
		WARN_PRINT_ONCE("View configuration is not set to stereo with foveated inset, or this is not supported.");
		set_update_mode(SubViewport::UPDATE_DISABLED);
		return;
	}

	set_update_mode(SubViewport::UPDATE_ALWAYS);

	Size2i new_size = fi_ext->get_render_size();
	if (get_size() != new_size) {
		set_size(new_size);
	}

	if (get_view_count() != 2) {
		set_view_count(2);
	}
}
