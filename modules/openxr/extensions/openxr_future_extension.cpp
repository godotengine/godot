/**************************************************************************/
/*  openxr_future_extension.cpp                                           */
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

#include "openxr_future_extension.h"

#include "../openxr_api.h"

OpenXRFutureExtension *OpenXRFutureExtension::singleton = nullptr;

OpenXRFutureExtension *OpenXRFutureExtension::get_singleton() {
	return singleton;
}

OpenXRFutureExtension::OpenXRFutureExtension() {
	singleton = this;
}

OpenXRFutureExtension::~OpenXRFutureExtension() {
	singleton = nullptr;
}

void OpenXRFutureExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_active"), &OpenXRFutureExtension::get_active);
	ClassDB::bind_method(D_METHOD("register_future", "future", "callable"), &OpenXRFutureExtension::_register_future);
	ClassDB::bind_method(D_METHOD("cancel_future", "future"), &OpenXRFutureExtension::_cancel_future);
}

HashMap<String, bool *> OpenXRFutureExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_EXT_FUTURE_EXTENSION_NAME] = &future_ext;

	return request_extensions;
}

void OpenXRFutureExtension::on_instance_created(const XrInstance p_instance) {
	if (future_ext) {
		EXT_INIT_XR_FUNC(xrPollFutureEXT);
		EXT_INIT_XR_FUNC(xrCancelFutureEXT);

		future_ext = xrPollFutureEXT_ptr && xrCancelFutureEXT_ptr;
	}
}

void OpenXRFutureExtension::on_instance_destroyed() {
	xrPollFutureEXT_ptr = nullptr;
	xrCancelFutureEXT_ptr = nullptr;
}

void OpenXRFutureExtension::on_session_destroyed() {
	if (!get_active()) {
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Cancel any running futures.
	for (const KeyValue<XrFutureEXT, Callable> &element : futures) {
		XrFutureCancelInfoEXT cancel_info = {
			XR_TYPE_FUTURE_CANCEL_INFO_EXT, // type
			nullptr, // next
			element.key // future
		};
		XrResult result = xrCancelFutureEXT_ptr(openxr_api->get_instance(), &cancel_info);
		if (XR_FAILED(result)) {
			WARN_PRINT("OpenXR: Failed to cancel future [" + openxr_api->get_error_string(result) + "]");
		}
	}
	futures.clear();
}

bool OpenXRFutureExtension::get_active() const {
	return future_ext;
}

void OpenXRFutureExtension::register_future(XrFutureEXT p_future, Callable p_callable) {
	ERR_FAIL_COND(futures.has(p_future));

	futures[p_future] = p_callable;
}

void OpenXRFutureExtension::_register_future(uint64_t p_future, Callable p_callable) {
	register_future((XrFutureEXT)p_future, p_callable);
}

void OpenXRFutureExtension::cancel_future(XrFutureEXT p_future) {
	ERR_FAIL_COND(!futures.has(p_future));

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	XrFutureCancelInfoEXT cancel_info = {
		XR_TYPE_FUTURE_CANCEL_INFO_EXT, // type
		nullptr, // next
		p_future // future
	};
	XrResult result = xrCancelFutureEXT_ptr(openxr_api->get_instance(), &cancel_info);
	if (XR_FAILED(result)) {
		WARN_PRINT("OpenXR: Failed to cancel future [" + openxr_api->get_error_string(result) + "]");
	}

	futures.erase(p_future);
}

void OpenXRFutureExtension::_cancel_future(uint64_t p_future) {
	cancel_future((XrFutureEXT)p_future);
}

void OpenXRFutureExtension::on_process() {
	if (!get_active()) {
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Process futures
	Vector<XrFutureEXT> completed;
	for (const KeyValue<XrFutureEXT, Callable> &element : futures) {
		XrFuturePollInfoEXT poll_info = {
			XR_TYPE_FUTURE_POLL_INFO_EXT, // type
			nullptr, // next
			element.key // future
		};
		XrFuturePollResultEXT poll_result = {
			XR_TYPE_FUTURE_POLL_RESULT_EXT, // type
			nullptr, // next
			XR_FUTURE_STATE_MAX_ENUM_EXT // state
		};
		XrResult result = xrPollFutureEXT_ptr(openxr_api->get_instance(), &poll_info, &poll_result);
		if (XR_FAILED(result)) {
			ERR_PRINT("OpenXR: Failed to obtain future status [" + openxr_api->get_error_string(result) + "]");
			// Maybe remove this depending on the error?
			continue;
		}
		if (poll_result.state == XR_FUTURE_STATE_READY_EXT) {
			// Call our callable.
			element.value.call((uint64_t)element.key);

			// Queue removing this
			completed.push_back(element.key);
		}
	}

	// Now clean up completed futures.
	for (const XrFutureEXT &future : completed) {
		futures.erase(future);
	}
}
