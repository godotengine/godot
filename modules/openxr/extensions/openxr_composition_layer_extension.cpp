/**************************************************************************/
/*  openxr_composition_layer_extension.cpp                                */
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

#include "openxr_composition_layer_extension.h"

#include "scene/main/viewport.h"
#include "servers/rendering/rendering_server_default.h"

////////////////////////////////////////////////////////////////////////////
// OpenXRCompositionLayerExtension

OpenXRCompositionLayerExtension *OpenXRCompositionLayerExtension::singleton = nullptr;

OpenXRCompositionLayerExtension *OpenXRCompositionLayerExtension::get_singleton() {
	return singleton;
}

OpenXRCompositionLayerExtension::OpenXRCompositionLayerExtension() {
	singleton = this;
}

OpenXRCompositionLayerExtension::~OpenXRCompositionLayerExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRCompositionLayerExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_KHR_COMPOSITION_LAYER_CYLINDER_EXTENSION_NAME] = &cylinder_ext_available;
	request_extensions[XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME] = &equirect_ext_available;

	return request_extensions;
}

bool OpenXRCompositionLayerExtension::is_available(XrStructureType p_which) {
	switch (p_which) {
		case XR_TYPE_COMPOSITION_LAYER_QUAD: {
			// Doesn't require an extension.
			return true;
		} break;
		case XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR: {
			return cylinder_ext_available;
		} break;
		case XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR: {
			return equirect_ext_available;
		} break;
		default: {
			ERR_PRINT(vformat("Unsupported composition layer type: %s", p_which));
			return false;
		}
	};
}

////////////////////////////////////////////////////////////////////////////
// OpenXRViewportCompositionLayerProvider

OpenXRViewportCompositionLayerProvider::OpenXRViewportCompositionLayerProvider(XrCompositionLayerBaseHeader *p_composition_layer) {
	composition_layer = p_composition_layer;
	openxr_api = OpenXRAPI::get_singleton();
	composition_layer_extension = OpenXRCompositionLayerExtension::get_singleton();
}

OpenXRViewportCompositionLayerProvider::~OpenXRViewportCompositionLayerProvider() {
	// This will reset the viewport and free the swapchain too.
	set_viewport(nullptr);
}

int OpenXRViewportCompositionLayerProvider::get_composition_layer_count() {
	return 1;
}

XrCompositionLayerBaseHeader *OpenXRViewportCompositionLayerProvider::get_composition_layer(int p_index) {
	if (openxr_api == nullptr || composition_layer_extension == nullptr) {
		// OpenXR not initialised or we're in the editor?
		return nullptr;
	}

	if (!composition_layer_extension->is_available(composition_layer->type)) {
		// Selected type is not supported, ignore our layer.
		return nullptr;
	}

	if (swapchain_info.swapchain == XR_NULL_HANDLE) {
		// Don't have a swapchain to display? Ignore our layer.
		return nullptr;
	}

	if (swapchain_info.image_acquired) {
		openxr_api->release_image(swapchain_info);
	}

	// Update the layer struct for the swapchain.
	switch (composition_layer->type) {
		case XR_TYPE_COMPOSITION_LAYER_QUAD: {
			XrCompositionLayerQuad *quad_layer = (XrCompositionLayerQuad *)composition_layer;
			quad_layer->subImage.swapchain = swapchain_info.swapchain;
			quad_layer->subImage.imageArrayIndex = 0;
			quad_layer->subImage.imageRect.offset.x = 0;
			quad_layer->subImage.imageRect.offset.y = 0;
			quad_layer->subImage.imageRect.extent.width = width;
			quad_layer->subImage.imageRect.extent.height = height;
		} break;

		case XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR: {
			XrCompositionLayerCylinderKHR *cylinder_layer = (XrCompositionLayerCylinderKHR *)composition_layer;
			cylinder_layer->subImage.swapchain = swapchain_info.swapchain;
			cylinder_layer->subImage.imageArrayIndex = 0;
			cylinder_layer->subImage.imageRect.offset.x = 0;
			cylinder_layer->subImage.imageRect.offset.y = 0;
			cylinder_layer->subImage.imageRect.extent.width = width;
			cylinder_layer->subImage.imageRect.extent.height = height;
		} break;

		case XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR: {
			XrCompositionLayerEquirect2KHR *equirect_layer = (XrCompositionLayerEquirect2KHR *)composition_layer;
			equirect_layer->subImage.swapchain = swapchain_info.swapchain;
			equirect_layer->subImage.imageArrayIndex = 0;
			equirect_layer->subImage.imageRect.offset.x = 0;
			equirect_layer->subImage.imageRect.offset.y = 0;
			equirect_layer->subImage.imageRect.extent.width = width;
			equirect_layer->subImage.imageRect.extent.height = height;
		} break;

		default: {
			return nullptr;
		} break;
	}

	return composition_layer;
}

int OpenXRViewportCompositionLayerProvider::get_composition_layer_order(int p_index) {
	return sort_order;
}

bool OpenXRViewportCompositionLayerProvider::set_viewport(SubViewport *p_viewport) {
	RenderingServer *rs = RenderingServer::get_singleton();
	ERR_FAIL_NULL_V(rs, false);

	if (viewport != p_viewport) {
		if (viewport) {
			RID vp = viewport->get_viewport_rid();
			RID rt = rs->viewport_get_render_target(vp);
			RSG::texture_storage->render_target_set_override(rt, RID(), RID(), RID());
		}

		viewport = p_viewport;

		if (!viewport) {
			free_swapchain();
		}

		return true;
	}

	return false;
}

void OpenXRViewportCompositionLayerProvider::process() {
	RenderingServer *rs = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rs);

	if (viewport && openxr_api && openxr_api->is_running()) {
		RID vp = viewport->get_viewport_rid();
		RS::ViewportUpdateMode update_mode = rs->viewport_get_update_mode(vp);
		if (update_mode == RS::VIEWPORT_UPDATE_WHEN_VISIBLE || update_mode == RS::VIEWPORT_UPDATE_WHEN_PARENT_VISIBLE) {
			WARN_PRINT_ONCE("OpenXR composition layers cannot use Viewports with UPDATE_WHEN_VISIBLE or UPDATE_WHEN_PARENT_VISIBLE. Switching to UPDATE_ALWAYS.");
			viewport->set_update_mode(SubViewport::UPDATE_ALWAYS);
		}
		if (update_mode == RS::VIEWPORT_UPDATE_ONCE || update_mode == RS::VIEWPORT_UPDATE_ALWAYS) {
			// Update our XR swapchain
			Size2i vp_size = viewport->get_size();
			if (update_and_acquire_swapchain(vp_size.width, vp_size.height, update_mode == RS::VIEWPORT_UPDATE_ONCE)) {
				// Render to our XR swapchain image.
				RID rt = rs->viewport_get_render_target(vp);
				RSG::texture_storage->render_target_set_override(rt, get_current_swapchain_texture(), RID(), RID());
			}
		}
	}
}

bool OpenXRViewportCompositionLayerProvider::update_and_acquire_swapchain(uint32_t p_width, uint32_t p_height, bool p_static_image) {
	if (openxr_api == nullptr || composition_layer_extension == nullptr) {
		// OpenXR not initialised or we're in the editor?
		return false;
	}
	if (!composition_layer_extension->is_available(composition_layer->type)) {
		// Selected type is not supported?
		return false;
	}

	// See if our current swapchain is outdated.
	if (swapchain_info.swapchain != XR_NULL_HANDLE) {
		// If this swap chain, or the previous one, were static, then we can't reuse it.
		if (width == p_width && height == p_height && !p_static_image && !static_image) {
			// We're all good! Just acquire it.
			return openxr_api->acquire_image(swapchain_info);
		}

		openxr_api->free_swapchain(swapchain_info);
	}

	// Create our new swap chain
	int64_t swapchain_format = openxr_api->get_color_swapchain_format();
	const uint32_t sample_count = 3;
	const uint32_t array_size = 1;
	XrSwapchainCreateFlags create_flags = 0;
	if (p_static_image) {
		create_flags |= XR_SWAPCHAIN_CREATE_STATIC_IMAGE_BIT;
	}
	if (!openxr_api->create_swapchain(create_flags, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT, swapchain_format, p_width, p_height, sample_count, array_size, swapchain_info.swapchain, &swapchain_info.swapchain_graphics_data)) {
		width = 0;
		height = 0;
		return false;
	}

	// Acquire our image so we can start rendering into it
	bool ret = openxr_api->acquire_image(swapchain_info);

	width = p_width;
	height = p_height;
	static_image = p_static_image;
	return ret;
}

void OpenXRViewportCompositionLayerProvider::free_swapchain() {
	if (swapchain_info.swapchain != XR_NULL_HANDLE) {
		openxr_api->free_swapchain(swapchain_info);
	}

	width = 0;
	height = 0;
	static_image = false;
}

RID OpenXRViewportCompositionLayerProvider::get_current_swapchain_texture() {
	if (openxr_api == nullptr) {
		return RID();
	}

	return openxr_api->get_image(swapchain_info);
}
