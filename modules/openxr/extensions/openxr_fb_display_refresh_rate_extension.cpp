/*************************************************************************/
/*  openxr_fb_display_refresh_rate_extension.cpp                         */
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

#include "openxr_fb_display_refresh_rate_extension.h"

OpenXRDisplayRefreshRateExtension *OpenXRDisplayRefreshRateExtension::singleton = nullptr;

OpenXRDisplayRefreshRateExtension *OpenXRDisplayRefreshRateExtension::get_singleton() {
	return singleton;
}

OpenXRDisplayRefreshRateExtension::OpenXRDisplayRefreshRateExtension(OpenXRAPI *p_openxr_api) :
		OpenXRExtensionWrapper(p_openxr_api) {
	singleton = this;

	// Extensions we use for our hand tracking.
	request_extensions[XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME] = &display_refresh_rate_ext;
}

OpenXRDisplayRefreshRateExtension::~OpenXRDisplayRefreshRateExtension() {
	display_refresh_rate_ext = false;
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

float OpenXRDisplayRefreshRateExtension::get_refresh_rate() const {
	float refresh_rate = 0.0;

	if (display_refresh_rate_ext) {
		float rate;
		XrResult result = xrGetDisplayRefreshRateFB(openxr_api->get_session(), &rate);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to obtain refresh rate [", openxr_api->get_error_string(result), "]");
		} else {
			refresh_rate = rate;
		}
	}

	return refresh_rate;
}

void OpenXRDisplayRefreshRateExtension::set_refresh_rate(float p_refresh_rate) {
	if (display_refresh_rate_ext) {
		XrResult result = xrRequestDisplayRefreshRateFB(openxr_api->get_session(), p_refresh_rate);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to set refresh rate [", openxr_api->get_error_string(result), "]");
		}
	}
}

Array OpenXRDisplayRefreshRateExtension::get_available_refresh_rates() const {
	Array arr;
	XrResult result;

	if (display_refresh_rate_ext) {
		uint32_t display_refresh_rate_count = 0;
		result = xrEnumerateDisplayRefreshRatesFB(openxr_api->get_session(), 0, &display_refresh_rate_count, nullptr);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to obtain refresh rates count [", openxr_api->get_error_string(result), "]");
		}

		if (display_refresh_rate_count > 0) {
			float *display_refresh_rates = (float *)memalloc(sizeof(float) * display_refresh_rate_count);
			if (display_refresh_rates == nullptr) {
				print_line("OpenXR: Failed to obtain refresh rates memory buffer [", openxr_api->get_error_string(result), "]");
				return arr;
			}

			result = xrEnumerateDisplayRefreshRatesFB(openxr_api->get_session(), display_refresh_rate_count, &display_refresh_rate_count, display_refresh_rates);
			if (XR_FAILED(result)) {
				print_line("OpenXR: Failed to obtain refresh rates count [", openxr_api->get_error_string(result), "]");
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
