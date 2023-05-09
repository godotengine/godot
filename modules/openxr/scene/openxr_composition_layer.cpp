/**************************************************************************/
/*  openxr_composition_layer.cpp                                          */
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

#include "../extensions/openxr_composition_layer_extension.h"
#include "../openxr_api.h"

#include "openxr_composition_layer.h"

#include "servers/rendering/rendering_server_default.h"
#include "servers/rendering_server.h"

// TODOs:
// - make settings for pose, radius, scale and bias setable, possibly linking pose to player position (or leave that up to user)
// - add gizmo to visualise positioning/output of composition layer
// - add support for other composition layers

OpenXRCompositionLayer::OpenXRCompositionLayer() {
	openxr_api = OpenXRAPI::get_singleton();
	openxr_layer_provider = memnew(ViewportCompositionLayerProvider);

	if (openxr_api != nullptr) {
		set_process_internal(true);
	}
}

OpenXRCompositionLayer::~OpenXRCompositionLayer() {
	if (openxr_layer_provider != nullptr) {
		memdelete(openxr_layer_provider);
		openxr_layer_provider = nullptr;
	}
}

void OpenXRCompositionLayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_composition_layer_type", "composition_layer_type"), &OpenXRCompositionLayer::set_composition_layer_type);
	ClassDB::bind_method(D_METHOD("get_composition_layer_type"), &OpenXRCompositionLayer::get_composition_layer_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "composition_layer_type", PROPERTY_HINT_ENUM, "Equirect"), "set_composition_layer_type", "get_composition_layer_type");

	ClassDB::bind_method(D_METHOD("is_supported"), &OpenXRCompositionLayer::is_supported);

	BIND_ENUM_CONSTANT(COMPOSITION_LAYER_EQUIRECT2);
	BIND_ENUM_CONSTANT(COMPOSITION_LAYER_MAX);
}

void OpenXRCompositionLayer::set_composition_layer_type(const CompositionLayerTypes p_type) {
	composition_layer_type = p_type;

	switch (composition_layer_type) {
		case COMPOSITION_LAYER_EQUIRECT2: {
			openxr_layer_provider->setup_for_type(XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR);
		} break;
		default: {
			// this will clear it and make it inactive.
			openxr_layer_provider->setup_for_type(XR_TYPE_UNKNOWN);
		} break;
	}
}

bool OpenXRCompositionLayer::is_supported() {
	return openxr_layer_provider->is_supported();
}

void OpenXRCompositionLayer::_notification(int p_what) {
	if (openxr_api == nullptr) {
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rs);

	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			// TODO, if our update mode will result in us not updating our viewport,
			// we should skip this and reuse our last result.

			if (openxr_layer_provider->is_supported()) {
				// Update our XR swapchain
				Size2i vp_size = get_size();
				openxr_layer_provider->update_swapchain(vp_size.width, vp_size.height);

				// Render to our XR swapchain image
				RID vp = get_viewport_rid();
				RID rt = rs->viewport_get_render_target(vp);
				RSG::texture_storage->render_target_set_override(rt, openxr_layer_provider->get_image(), RID(), RID());
			} else {
				// We still render our viewport, this will allow the user to display the viewport with a Quadmesh.
				RID vp = get_viewport_rid();
				RID rt = rs->viewport_get_render_target(vp);
				RSG::texture_storage->render_target_set_override(rt, RID(), RID(), RID());
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
			// Add our composition layer provider to our OpenXR API,
			// we do this even if not supported as the user may change type.
			openxr_api->register_composition_layer_provider(openxr_layer_provider);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			// Remove our composition layer provider from our OpenXR API
			openxr_api->unregister_composition_layer_provider(openxr_layer_provider);

			// reset our viewport
			RID vp = get_viewport_rid();
			RID rt = rs->viewport_get_render_target(vp);
			RSG::texture_storage->render_target_set_override(rt, RID(), RID(), RID());

			// free our swapchain
			openxr_layer_provider->free_swapchain();
		} break;
	}
}
