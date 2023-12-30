/**************************************************************************/
/*  openxr_htc_passthrough_extension.cpp                                  */
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

#include "openxr_htc_passthrough_extension.h"

#include "core/os/os.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"

using namespace godot;

OpenXRHTCPassthroughExtension *OpenXRHTCPassthroughExtension::singleton = nullptr;

OpenXRHTCPassthroughExtension *OpenXRHTCPassthroughExtension::get_singleton() {
	return singleton;
}

OpenXRHTCPassthroughExtension::OpenXRHTCPassthroughExtension() {
	singleton = this;
}

OpenXRHTCPassthroughExtension::~OpenXRHTCPassthroughExtension() {
	cleanup();
}

HashMap<String, bool *> OpenXRHTCPassthroughExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_HTC_PASSTHROUGH_EXTENSION_NAME] = &htc_passthrough_ext;

	return request_extensions;
}

void OpenXRHTCPassthroughExtension::cleanup() {
	htc_passthrough_ext = false;
}

Viewport *OpenXRHTCPassthroughExtension::get_main_viewport() {
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

void OpenXRHTCPassthroughExtension::on_instance_created(const XrInstance instance) {
	if (htc_passthrough_ext) {
		bool result = initialize_htc_passthrough_extension(instance);
		if (!result) {
			print_error("Failed to initialize htc_passthrough extension");
			htc_passthrough_ext = false;
		}
	}

	if (htc_passthrough_ext) {
		OpenXRAPI::get_singleton()->register_composition_layer_provider(this);
	}
}

bool OpenXRHTCPassthroughExtension::is_passthrough_enabled() {
	return htc_passthrough_ext && passthrough_handle != XR_NULL_HANDLE;
}

bool OpenXRHTCPassthroughExtension::start_passthrough() {
	if (passthrough_handle == XR_NULL_HANDLE) {
		return false;
	}

	if (is_passthrough_enabled()) {
		return true;
	}

	const XrPassthroughCreateInfoHTC create_info = {
		XR_TYPE_PASSTHROUGH_CREATE_INFO_HTC,
		nullptr,
		XR_PASSTHROUGH_FORM_PLANAR_HTC
	};

	XrResult result = xrCreatePassthroughHTC(OpenXRAPI::get_singleton()->get_session(),
			&create_info,
			&passthrough_handle);

	if (result != XR_SUCCESS) {
		stop_passthrough();
		return false;
	}

	// Check if the the viewport has transparent background
	Viewport *viewport = get_main_viewport();
	if (viewport && !viewport->has_transparent_background()) {
		print_error("Main viewport doesn't have transparent background! Passthrough may not properly render.");
	}

	// HACK: to behave similar to Meta Passthrough variant
	set_alpha(1.0f);

	return true;
}

void OpenXRHTCPassthroughExtension::on_session_created(const XrSession session) {
	if (session != nullptr && htc_passthrough_ext) {
		start_passthrough();
	}
}

XrCompositionLayerBaseHeader *OpenXRHTCPassthroughExtension::get_composition_layer() {
	if (is_passthrough_enabled()) {
		composition_passthrough_layer.passthrough = passthrough_handle;

		return (XrCompositionLayerBaseHeader *)&composition_passthrough_layer;
	} else {
		return nullptr;
	}
}

void OpenXRHTCPassthroughExtension::stop_passthrough() {
	if (!htc_passthrough_ext) {
		return;
	}

	if (passthrough_handle != XR_NULL_HANDLE) {
		XrResult result = xrDestroyPassthroughHTC(passthrough_handle);
		if (OpenXRAPI::get_singleton()->xr_result(result, "Unable to destroy passthrough feature")) {
			passthrough_handle = XR_NULL_HANDLE;
		}
	}
}

void OpenXRHTCPassthroughExtension::on_session_destroyed() {
	if (htc_passthrough_ext) {
		stop_passthrough();
	}
}

void OpenXRHTCPassthroughExtension::on_instance_destroyed() {
	if (htc_passthrough_ext) {
		OpenXRAPI::get_singleton()->unregister_composition_layer_provider(this);
	}
	cleanup();
}

void OpenXRHTCPassthroughExtension::set_alpha(float alpha) {
	composition_passthrough_layer.color.alpha = std::clamp(alpha, 0.0f, 1.0f);
}

float OpenXRHTCPassthroughExtension::get_alpha() const {
	return composition_passthrough_layer.color.alpha;
}

bool OpenXRHTCPassthroughExtension::initialize_htc_passthrough_extension(const XrInstance p_instance) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), false);

	EXT_INIT_XR_FUNC_V(xrCreatePassthroughHTC);
	EXT_INIT_XR_FUNC_V(xrDestroyPassthroughHTC);

	return true;
}
