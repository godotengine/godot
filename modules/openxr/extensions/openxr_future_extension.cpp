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

////////////////////////////////////////////////////////////////////////////
// OpenXRFutureResult

void OpenXRFutureResult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_status"), &OpenXRFutureResult::get_status);
	ClassDB::bind_method(D_METHOD("get_future"), &OpenXRFutureResult::_get_future);
	ClassDB::bind_method(D_METHOD("cancel_future"), &OpenXRFutureResult::cancel_future);

	ADD_SIGNAL(MethodInfo("completed", PropertyInfo(Variant::OBJECT, "result", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRFutureResult")));

	BIND_ENUM_CONSTANT(RESULT_RUNNING);
	BIND_ENUM_CONSTANT(RESULT_FINISHED);
	BIND_ENUM_CONSTANT(RESULT_CANCELLED);
}

void OpenXRFutureResult::_mark_as_finished() {
	// Update our status
	status = RESULT_FINISHED;

	// Perform our callback
	on_success_callback.call((uint64_t)future);

	// Emit our signal
	emit_signal(SNAME("completed"), this);
}

void OpenXRFutureResult::_mark_as_cancelled() {
	// Update our status
	status = RESULT_CANCELLED;

	// There is no point in doing a callback for cancellation as its always user invoked.

	// But we do emit our signal to make sure any await finishes.
	emit_signal(SNAME("completed"), this);
}

OpenXRFutureResult::ResultStatus OpenXRFutureResult::get_status() const {
	return status;
}

XrFutureEXT OpenXRFutureResult::get_future() const {
	return future;
}

uint64_t OpenXRFutureResult::_get_future() const {
	return (uint64_t)future;
}

void OpenXRFutureResult::cancel_future() {
	ERR_FAIL_COND(status != RESULT_RUNNING);

	OpenXRFutureExtension *future_extension = OpenXRFutureExtension::get_singleton();
	ERR_FAIL_NULL(future_extension);

	future_extension->cancel_future(future);
}

OpenXRFutureResult::OpenXRFutureResult(XrFutureEXT p_future, const Callable &p_on_success) {
	future = p_future;
	on_success_callback = p_on_success;
}

////////////////////////////////////////////////////////////////////////////
// OpenXRFutureExtension

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
	ClassDB::bind_method(D_METHOD("is_active"), &OpenXRFutureExtension::is_active);
	ClassDB::bind_method(D_METHOD("register_future", "future", "on_success"), &OpenXRFutureExtension::_register_future, DEFVAL(Callable()));
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
	if (!is_active()) {
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Cancel any running futures.
	for (const KeyValue<XrFutureEXT, Ref<OpenXRFutureResult>> &element : futures) {
		XrFutureCancelInfoEXT cancel_info = {
			XR_TYPE_FUTURE_CANCEL_INFO_EXT, // type
			nullptr, // next
			element.key // future
		};
		XrResult result = xrCancelFutureEXT_ptr(openxr_api->get_instance(), &cancel_info);
		if (XR_FAILED(result)) {
			WARN_PRINT("OpenXR: Failed to cancel future [" + openxr_api->get_error_string(result) + "]");
		}

		// Make sure we mark our future result as cancelled
		element.value->_mark_as_cancelled();
	}
	futures.clear();
}

bool OpenXRFutureExtension::is_active() const {
	return future_ext;
}

Ref<OpenXRFutureResult> OpenXRFutureExtension::register_future(XrFutureEXT p_future, const Callable &p_on_success) {
	ERR_FAIL_COND_V(futures.has(p_future), nullptr);

	Ref<OpenXRFutureResult> future_result;
	future_result.instantiate(p_future, p_on_success);

	futures[p_future] = future_result;

	return future_result;
}

Ref<OpenXRFutureResult> OpenXRFutureExtension::_register_future(uint64_t p_future, const Callable &p_on_success) {
	return register_future((XrFutureEXT)p_future, p_on_success);
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

	// Make sure we mark our future result as cancelled
	futures[p_future]->_mark_as_cancelled();

	// And erase it from the futures we track
	futures.erase(p_future);
}

void OpenXRFutureExtension::_cancel_future(uint64_t p_future) {
	cancel_future((XrFutureEXT)p_future);
}

void OpenXRFutureExtension::on_process() {
	if (!is_active()) {
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Process futures
	Vector<XrFutureEXT> completed;
	for (const KeyValue<XrFutureEXT, Ref<OpenXRFutureResult>> &element : futures) {
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
			// Mark our future result as finished (this will invoke our callback).
			element.value->_mark_as_finished();

			// Queue removing this
			completed.push_back(element.key);
		}
	}

	// Now clean up completed futures.
	for (const XrFutureEXT &future : completed) {
		futures.erase(future);
	}
}
