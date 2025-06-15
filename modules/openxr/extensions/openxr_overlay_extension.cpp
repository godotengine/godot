/**************************************************************************/
/*  openxr_overlay_extension.cpp                                          */
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

#include "openxr_overlay_extension.h"

#include "core/config/project_settings.h"
#include "modules/openxr/extensions/openxr_composition_layer_depth_extension.h"
#include "modules/openxr/openxr_api.h"

#include "openxr/openxr.h"

OpenXROverlayExtension *OpenXROverlayExtension::singleton = nullptr;

OpenXROverlayExtension *OpenXROverlayExtension::get_singleton() {
	ERR_FAIL_NULL_V(singleton, nullptr);
	return singleton;
}

OpenXROverlayExtension::OpenXROverlayExtension() {
	singleton = this;

	enabled = GLOBAL_GET("xr/openxr/extensions/overlay/enabled");
	session_layers_placement = GLOBAL_GET("xr/openxr/extensions/overlay/session_layers_placement");
}

OpenXROverlayExtension::~OpenXROverlayExtension() {
	singleton = nullptr;
}

// TODO: Add project options for this
HashMap<String, bool *> OpenXROverlayExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;
	request_extensions[XR_EXTX_OVERLAY_EXTENSION_NAME] = &available;

	return request_extensions;
}

void *OpenXROverlayExtension::set_session_create_and_get_next_pointer(void *p_next_pointer) {
	if (!available || !enabled) {
		return p_next_pointer;
	}
	XrSessionCreateInfoOverlayEXTX *session_create_info_overlay = new XrSessionCreateInfoOverlayEXTX();
	session_create_info_overlay->type = XR_TYPE_SESSION_CREATE_INFO_OVERLAY_EXTX;
	session_create_info_overlay->next = p_next_pointer;
	session_create_info_overlay->sessionLayersPlacement = session_layers_placement;
	if (OpenXRAPI::get_singleton()->get_submit_depth_buffer() && OpenXRCompositionLayerDepthExtension::get_singleton()->is_available()) {
		session_create_info_overlay->createFlags = XR_OVERLAY_MAIN_SESSION_ENABLED_COMPOSITION_LAYER_INFO_DEPTH_BIT_EXTX;
	}
	return session_create_info_overlay;
}

bool OpenXROverlayExtension::is_available() {
	return available;
}

bool OpenXROverlayExtension::is_enabled() {
	return enabled;
}

uint32_t OpenXROverlayExtension::get_session_layers_placement() const {
	return session_layers_placement;
}

void OpenXROverlayExtension::set_session_layers_placement(uint32_t p_session_layers_placement) {
	session_layers_placement = p_session_layers_placement;
}
