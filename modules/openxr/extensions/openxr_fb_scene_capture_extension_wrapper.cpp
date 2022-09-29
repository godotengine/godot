/*************************************************************************/
/*  openxr_fb_scene_capture_extension_wrapper.cpp                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "openxr_fb_scene_capture_extension_wrapper.h"
#include "modules/openxr/openxr_interface.h"

using namespace godot;

OpenXRFbSceneCaptureExtensionWrapper *OpenXRFbSceneCaptureExtensionWrapper::singleton = nullptr;

OpenXRFbSceneCaptureExtensionWrapper *OpenXRFbSceneCaptureExtensionWrapper::get_singleton() {
	return singleton;
}

OpenXRFbSceneCaptureExtensionWrapper::OpenXRFbSceneCaptureExtensionWrapper(OpenXRAPI *p_openxr_api) :
		OpenXRExtensionWrapper(p_openxr_api) {
	request_extensions[XR_FB_SCENE_CAPTURE_EXTENSION_NAME] = &fb_scene_capture_ext;
	singleton = this;
}

OpenXRFbSceneCaptureExtensionWrapper::~OpenXRFbSceneCaptureExtensionWrapper() {
	cleanup();
}

void OpenXRFbSceneCaptureExtensionWrapper::cleanup() {
	fb_scene_capture_ext = false;
}

void OpenXRFbSceneCaptureExtensionWrapper::on_instance_created(const XrInstance instance) {
	if (fb_scene_capture_ext) {
		bool result = initialize_fb_scene_capture_extension(instance);
		if (!result) {
			print_error("Failed to initialize fb_scene_capture extension");
			fb_scene_capture_ext = false;
		}
	}
}

void OpenXRFbSceneCaptureExtensionWrapper::on_instance_destroyed() {
	cleanup();
}

bool OpenXRFbSceneCaptureExtensionWrapper::request_scene_capture() {
	XrAsyncRequestIdFB requestId;
	XrSceneCaptureRequestInfoFB request;
	request.type = XR_TYPE_SCENE_CAPTURE_REQUEST_INFO_FB;
	request.next = nullptr;
	request.requestByteCount = 0;
	request.request = nullptr;
	XrResult result = xrRequestSceneCaptureFB(openxr_api->get_session(), &request, &requestId);
	scene_capture_enabled = (result == XR_SUCCESS);
	return scene_capture_enabled;
}

bool OpenXRFbSceneCaptureExtensionWrapper::is_scene_capture_enabled() {
	return scene_capture_enabled;
}

bool OpenXRFbSceneCaptureExtensionWrapper::initialize_fb_scene_capture_extension(const XrInstance p_instance) {
	ERR_FAIL_NULL_V(openxr_api, false);

	EXT_INIT_XR_FUNC_V(xrRequestSceneCaptureFB);

	return true;
}

bool OpenXRFbSceneCaptureExtensionWrapper::on_event_polled(const XrEventDataBuffer &event) {
	ERR_FAIL_NULL_V(openxr_api, false);

	if (event.type == XR_TYPE_EVENT_DATA_SCENE_CAPTURE_COMPLETE_FB) {
		scene_capture_enabled = false;
		OpenXRInterface *xr_interface = openxr_api->get_xr_interface();
		if (xr_interface) {
			xr_interface->on_scene_capture_completed();
		}
		return true;
	}
	return false;
}
