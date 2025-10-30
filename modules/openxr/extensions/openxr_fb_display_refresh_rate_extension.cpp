/**************************************************************************/
/*  openxr_fb_display_refresh_rate_extension.cpp                          */
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

#include "openxr_fb_display_refresh_rate_extension.h"
#include "../openxr_interface.h"

OpenXRDisplayRefreshRateExtension *OpenXRDisplayRefreshRateExtension::singleton = nullptr;

OpenXRDisplayRefreshRateExtension *OpenXRDisplayRefreshRateExtension::get_singleton() {
	return singleton;
}

OpenXRDisplayRefreshRateExtension::OpenXRDisplayRefreshRateExtension() {
	singleton = this;
}

OpenXRDisplayRefreshRateExtension::~OpenXRDisplayRefreshRateExtension() {
	display_refresh_rate_ext = false;
}

HashMap<String, bool *> OpenXRDisplayRefreshRateExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME] = &display_refresh_rate_ext;

	return request_extensions;
}

void OpenXRDisplayRefreshRateExtension::on_instance_created(const XrInstance p_instance) {
	if (display_refresh_rate_ext) {
		EXT_INIT_XR_FUNC(xrEnumerateDisplayRefreshRatesFB);
		EXT_INIT_XR_FUNC(xrGetDisplayRefreshRateFB);
		EXT_INIT_XR_FUNC(xrRequestDisplayRefreshRateFB);
	}
}

void OpenXRDisplayRefreshRateExtension::on_instance_destroyed() {
	display_refresh_rate_ext = false;
}

bool OpenXRDisplayRefreshRateExtension::on_event_polled(const XrEventDataBuffer &event) {
	switch (event.type) {
		case XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB: {
			const XrEventDataDisplayRefreshRateChangedFB *event_fb = (XrEventDataDisplayRefreshRateChangedFB *)&event;

			OpenXRInterface *xr_interface = OpenXRAPI::get_singleton()->get_xr_interface();
			if (xr_interface) {
				xr_interface->on_refresh_rate_changes(event_fb->toDisplayRefreshRate);
			}

			return true;
		} break;
		default:
			return false;
	}
}

float OpenXRDisplayRefreshRateExtension::get_refresh_rate() const {
	float refresh_rate = 0.0;

	if (display_refresh_rate_ext) {
		float rate;
		XrResult result = xrGetDisplayRefreshRateFB(OpenXRAPI::get_singleton()->get_session(), &rate);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to obtain refresh rate [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
		} else {
			refresh_rate = rate;
		}
	}

	return refresh_rate;
}

void OpenXRDisplayRefreshRateExtension::set_refresh_rate(float p_refresh_rate) {
	if (display_refresh_rate_ext) {
		XrResult result = xrRequestDisplayRefreshRateFB(OpenXRAPI::get_singleton()->get_session(), p_refresh_rate);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to set refresh rate [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
		}
	}
}

Array OpenXRDisplayRefreshRateExtension::get_available_refresh_rates() const {
	Array arr;
	XrResult result;

	if (display_refresh_rate_ext) {
		uint32_t display_refresh_rate_count = 0;
		result = xrEnumerateDisplayRefreshRatesFB(OpenXRAPI::get_singleton()->get_session(), 0, &display_refresh_rate_count, nullptr);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to obtain refresh rates count [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
		}

		if (display_refresh_rate_count > 0) {
			float *display_refresh_rates = (float *)memalloc(sizeof(float) * display_refresh_rate_count);
			if (display_refresh_rates == nullptr) {
				print_line("OpenXR: Failed to obtain refresh rates memory buffer [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
				return arr;
			}

			result = xrEnumerateDisplayRefreshRatesFB(OpenXRAPI::get_singleton()->get_session(), display_refresh_rate_count, &display_refresh_rate_count, display_refresh_rates);
			if (XR_FAILED(result)) {
				print_line("OpenXR: Failed to obtain refresh rates count [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
				memfree(display_refresh_rates);
				return arr;
			}

			for (uint32_t i = 0; i < display_refresh_rate_count; i++) {
				float refresh_rate = display_refresh_rates[i];
				arr.push_back(Variant(refresh_rate));
			}

			memfree(display_refresh_rates);
		}
	}

	return arr;
}
