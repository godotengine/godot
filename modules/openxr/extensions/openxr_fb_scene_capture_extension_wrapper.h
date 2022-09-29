/*************************************************************************/
/*  openxr_fb_scene_capture_extension_wrapper.h                          */
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

#ifndef OPENXR_FB_SCENE_CAPTURE_EXTENSION_WRAPPER_H
#define OPENXR_FB_SCENE_CAPTURE_EXTENSION_WRAPPER_H

#include "../openxr_api.h"
#include "../util.h"
#include "openxr_extension_wrapper.h"

#include <openxr/fb_scene_capture.h>

class Viewport;

// Wrapper for the set of Facebook XR scene capture extension.
class OpenXRFbSceneCaptureExtensionWrapper : public OpenXRExtensionWrapper {
	friend class OpenXRAPI;

public:
	void on_instance_created(const XrInstance instance) override;

	void on_instance_destroyed() override;

	bool is_scene_capture_supported() {
		return fb_scene_capture_ext;
	}

	bool request_scene_capture();
	bool is_scene_capture_enabled();

	virtual bool on_event_polled(const XrEventDataBuffer &event) override;

	static OpenXRFbSceneCaptureExtensionWrapper *get_singleton();

protected:
	OpenXRFbSceneCaptureExtensionWrapper(OpenXRAPI *p_openxr_api);
	~OpenXRFbSceneCaptureExtensionWrapper();

private:
	EXT_PROTO_XRRESULT_FUNC3(xrRequestSceneCaptureFB,
			(XrSession), session,
			(const XrSceneCaptureRequestInfoFB *), request,
			(XrAsyncRequestIdFB *), requestId)

	bool initialize_fb_scene_capture_extension(const XrInstance instance);

	void cleanup();

	static OpenXRFbSceneCaptureExtensionWrapper *singleton;

	bool fb_scene_capture_ext = false;

	bool scene_capture_enabled = false;
};

#endif // OPENXR_FB_SCENE_CAPTURE_EXTENSION_WRAPPER_H
