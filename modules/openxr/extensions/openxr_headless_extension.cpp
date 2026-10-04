/**************************************************************************/
/*  openxr_headless_extension.cpp                                         */
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

#include <ctime>

#include "../openxr_api.h"
#include "../openxr_platform_inc.h"

#include "openxr_headless_extension.h"

// Implementation for:
// https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_MND_headless

///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenXRHeadlessExtension

OpenXRHeadlessExtension *OpenXRHeadlessExtension::singleton = nullptr;

OpenXRHeadlessExtension *OpenXRHeadlessExtension::get_singleton() {
	return singleton;
}

OpenXRHeadlessExtension::OpenXRHeadlessExtension() {
	singleton = this;
}

OpenXRHeadlessExtension::~OpenXRHeadlessExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRHeadlessExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_MND_HEADLESS_EXTENSION_NAME] = &headless_ext;
#ifdef UNIX_ENABLED
	request_extensions[XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME] = &convert_timespec_time_ext;
#endif

	return request_extensions;
}

void OpenXRHeadlessExtension::on_instance_created(const XrInstance p_instance) {
#ifdef UNIX_ENABLED
	if (convert_timespec_time_ext) {
		EXT_INIT_XR_FUNC(xrConvertTimespecTimeToTimeKHR);
	}
#endif
}

void OpenXRHeadlessExtension::get_current_xrtime(XrTime* result) {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	XrInstance instance = openxr_api->get_instance();
	ERR_FAIL_COND(instance == XR_NULL_HANDLE);

#ifdef UNIX_ENABLED
	timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	xrConvertTimespecTimeToTimeKHR(instance, &time, result);
#endif
}

bool OpenXRHeadlessExtension::is_available() {
#ifdef UNIX_ENABLED
	return headless_ext && convert_timespec_time_ext;
#else
	return false;
#endif
}
