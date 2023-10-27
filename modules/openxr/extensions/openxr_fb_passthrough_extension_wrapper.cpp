/**************************************************************************/
/*  openxr_fb_passthrough_extension_wrapper.cpp                           */
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

#include "openxr_fb_passthrough_extension_wrapper.h"

#include "core/os/os.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"

using namespace godot;

OpenXRFbPassthroughExtensionWrapper *OpenXRFbPassthroughExtensionWrapper::singleton = nullptr;

OpenXRFbPassthroughExtensionWrapper *OpenXRFbPassthroughExtensionWrapper::get_singleton() {
	return singleton;
}

OpenXRFbPassthroughExtensionWrapper::OpenXRFbPassthroughExtensionWrapper() {
	singleton = this;
}

OpenXRFbPassthroughExtensionWrapper::~OpenXRFbPassthroughExtensionWrapper() {
	cleanup();
}

HashMap<String, bool *> OpenXRFbPassthroughExtensionWrapper::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_FB_PASSTHROUGH_EXTENSION_NAME] = &fb_passthrough_ext;
	request_extensions[XR_FB_TRIANGLE_MESH_EXTENSION_NAME] = &fb_triangle_mesh_ext;

	return request_extensions;
}

void OpenXRFbPassthroughExtensionWrapper::cleanup() {
	fb_passthrough_ext = false;
	fb_triangle_mesh_ext = false;
}

Viewport *OpenXRFbPassthroughExtensionWrapper::get_main_viewport() {
	MainLoop *main_loop = OS::get_singleton()->get_main_loop();
	if (!main_loop) {
		print_error("Unable to retrieve main loop");
		return nullptr;
	}

	SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);
	if (!scene_tree) {
		print_error("Unable to retrieve scene tree");
		return nullptr;
	}

	Viewport *viewport = scene_tree->get_root()->get_viewport();
	return viewport;
}

void OpenXRFbPassthroughExtensionWrapper::on_instance_created(const XrInstance instance) {
	if (fb_passthrough_ext) {
		bool result = initialize_fb_passthrough_extension(instance);
		if (!result) {
			print_error("Failed to initialize fb_passthrough extension");
			fb_passthrough_ext = false;
		}
	}

	if (fb_triangle_mesh_ext) {
		bool result = initialize_fb_triangle_mesh_extension(instance);
		if (!result) {
			print_error("Failed to initialize fb_triangle_mesh extension");
			fb_triangle_mesh_ext = false;
		}
	}

	if (fb_passthrough_ext) {
		OpenXRAPI::get_singleton()->register_composition_layer_provider(this);
	}
}

bool OpenXRFbPassthroughExtensionWrapper::is_passthrough_enabled() {
	return fb_passthrough_ext && passthrough_handle != XR_NULL_HANDLE && passthrough_layer != XR_NULL_HANDLE;
}

bool OpenXRFbPassthroughExtensionWrapper::start_passthrough() {
	if (passthrough_handle == XR_NULL_HANDLE) {
		return false;
	}

	if (is_passthrough_enabled()) {
		return true;
	}

	// Start the passthrough feature
	XrResult result = xrPassthroughStartFB(passthrough_handle);
	if (!is_valid_passthrough_result(result, "Failed to start passthrough")) {
		stop_passthrough();
		return false;
	}

	// Create the passthrough layer
	XrPassthroughLayerCreateInfoFB passthrough_layer_config = {
		XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB,
		nullptr,
		passthrough_handle,
		XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB,
		XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB,
	};
	result = xrCreatePassthroughLayerFB(OpenXRAPI::get_singleton()->get_session(), &passthrough_layer_config, &passthrough_layer);
	if (!is_valid_passthrough_result(result, "Failed to create the passthrough layer")) {
		stop_passthrough();
		return false;
	}

	// Check if the the viewport has transparent background
	Viewport *viewport = get_main_viewport();
	if (viewport && !viewport->has_transparent_background()) {
		print_error("Main viewport doesn't have transparent background! Passthrough may not properly render.");
	}

	return true;
}

void OpenXRFbPassthroughExtensionWrapper::on_session_created(const XrSession session) {
	if (fb_passthrough_ext) {
		// Create the passthrough feature and start it.
		XrPassthroughCreateInfoFB passthrough_create_info = {
			XR_TYPE_PASSTHROUGH_CREATE_INFO_FB,
			nullptr,
			0,
		};

		XrResult result = xrCreatePassthroughFB(OpenXRAPI::get_singleton()->get_session(), &passthrough_create_info, &passthrough_handle);
		if (!OpenXRAPI::get_singleton()->xr_result(result, "Failed to create passthrough")) {
			passthrough_handle = XR_NULL_HANDLE;
			return;
		}
	}
}

XrCompositionLayerBaseHeader *OpenXRFbPassthroughExtensionWrapper::get_composition_layer() {
	if (is_passthrough_enabled()) {
		composition_passthrough_layer.layerHandle = passthrough_layer;
		return (XrCompositionLayerBaseHeader *)&composition_passthrough_layer;
	} else {
		return nullptr;
	}
}

void OpenXRFbPassthroughExtensionWrapper::stop_passthrough() {
	if (!fb_passthrough_ext) {
		return;
	}

	XrResult result;
	if (passthrough_layer != XR_NULL_HANDLE) {
		// Destroy the layer
		result = xrDestroyPassthroughLayerFB(passthrough_layer);
		OpenXRAPI::get_singleton()->xr_result(result, "Unable to destroy passthrough layer");
		passthrough_layer = XR_NULL_HANDLE;
	}

	if (passthrough_handle != XR_NULL_HANDLE) {
		result = xrPassthroughPauseFB(passthrough_handle);
		OpenXRAPI::get_singleton()->xr_result(result, "Unable to stop passthrough feature");
	}
}

void OpenXRFbPassthroughExtensionWrapper::on_session_destroyed() {
	if (fb_passthrough_ext) {
		stop_passthrough();

		XrResult result;
		if (passthrough_handle != XR_NULL_HANDLE) {
			result = xrDestroyPassthroughFB(passthrough_handle);
			OpenXRAPI::get_singleton()->xr_result(result, "Unable to destroy passthrough feature");
			passthrough_handle = XR_NULL_HANDLE;
		}
	}
}

void OpenXRFbPassthroughExtensionWrapper::on_instance_destroyed() {
	if (fb_passthrough_ext) {
		OpenXRAPI::get_singleton()->unregister_composition_layer_provider(this);
	}
	cleanup();
}

bool OpenXRFbPassthroughExtensionWrapper::initialize_fb_passthrough_extension(const XrInstance p_instance) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), false);

	EXT_INIT_XR_FUNC_V(xrCreatePassthroughFB);
	EXT_INIT_XR_FUNC_V(xrDestroyPassthroughFB);
	EXT_INIT_XR_FUNC_V(xrPassthroughStartFB);
	EXT_INIT_XR_FUNC_V(xrPassthroughPauseFB);
	EXT_INIT_XR_FUNC_V(xrCreatePassthroughLayerFB);
	EXT_INIT_XR_FUNC_V(xrDestroyPassthroughLayerFB);
	EXT_INIT_XR_FUNC_V(xrPassthroughLayerPauseFB);
	EXT_INIT_XR_FUNC_V(xrPassthroughLayerResumeFB);
	EXT_INIT_XR_FUNC_V(xrPassthroughLayerSetStyleFB);
	EXT_INIT_XR_FUNC_V(xrCreateGeometryInstanceFB);
	EXT_INIT_XR_FUNC_V(xrDestroyGeometryInstanceFB);
	EXT_INIT_XR_FUNC_V(xrGeometryInstanceSetTransformFB);

	return true;
}

bool OpenXRFbPassthroughExtensionWrapper::initialize_fb_triangle_mesh_extension(const XrInstance p_instance) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), false);

	EXT_INIT_XR_FUNC_V(xrCreateTriangleMeshFB);
	EXT_INIT_XR_FUNC_V(xrDestroyTriangleMeshFB);
	EXT_INIT_XR_FUNC_V(xrTriangleMeshGetVertexBufferFB);
	EXT_INIT_XR_FUNC_V(xrTriangleMeshGetIndexBufferFB);
	EXT_INIT_XR_FUNC_V(xrTriangleMeshBeginUpdateFB);
	EXT_INIT_XR_FUNC_V(xrTriangleMeshEndUpdateFB);
	EXT_INIT_XR_FUNC_V(xrTriangleMeshBeginVertexBufferUpdateFB);
	EXT_INIT_XR_FUNC_V(xrTriangleMeshEndVertexBufferUpdateFB);

	return true;
}
