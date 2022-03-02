/*************************************************************************/
/*  openxr_android_extension.cpp                                         */
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

#include "openxr_android_extension.h"

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

OpenXRAndroidExtension *OpenXRAndroidExtension::singleton = nullptr;

OpenXRAndroidExtension *OpenXRAndroidExtension::get_singleton() {
	return singleton;
}

OpenXRAndroidExtension::OpenXRAndroidExtension(OpenXRAPI *p_openxr_api) :
		OpenXRExtensionWrapper(p_openxr_api) {
	singleton = this;

	request_extensions[XR_KHR_ANDROID_THREAD_SETTINGS_EXTENSION_NAME] = nullptr; // must be available

	// Initialize the loader
	PFN_xrInitializeLoaderKHR xrInitializeLoaderKHR;
	result = xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", (PFN_xrVoidFunction *)(&xrInitializeLoaderKHR));
	ERR_FAIL_COND_MSG(XR_FAILED(result), "Failed to retrieve pointer to xrInitializeLoaderKHR");

	// TODO fix this code, this is still code from GDNative!
	JNIEnv *env = android_api->godot_android_get_env();
	JavaVM *vm;
	env->GetJavaVM(&vm);
	jobject activity_object = env->NewGlobalRef(android_api->godot_android_get_activity());

	XrLoaderInitInfoAndroidKHR loader_init_info_android = {
		.type = XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR,
		.next = nullptr,
		.applicationVM = vm,
		.applicationContext = activity_object
	};
	xrInitializeLoaderKHR((const XrLoaderInitInfoBaseHeaderKHR *)&loader_init_info_android);
	ERR_FAIL_COND_MSG(XR_FAILED(result), "Failed to call xrInitializeLoaderKHR");
}

OpenXRAndroidExtension::~OpenXRAndroidExtension() {
	singleton = nullptr;
}
