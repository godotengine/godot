/**************************************************************************/
/*  openxr_future_extension.h                                             */
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

#pragma once

/*
	The OpenXR future extension forms the basis of OpenXRs ability to
	execute logic asynchronously.

	Asynchronous functions will return a future object which can be
	polled each frame to determine if the asynchronous function has
	been completed.

	If so the future can be used to obtain final return values.
	The API call for this is often part of the extension that utilizes
	the future.

	We will be using Godot Callables to drive responses on futures.
*/

#include "../util.h"
#include "core/object/ref_counted.h"
#include "openxr_extension_wrapper.h"

#include <openxr/openxr.h>

class OpenXRFutureExtension;

class OpenXRFutureResult : public RefCounted {
	GDCLASS(OpenXRFutureResult, RefCounted);

	friend class OpenXRFutureExtension;

protected:
	static void _bind_methods();

	void _mark_as_finished();
	void _mark_as_cancelled();

public:
	enum ResultStatus {
		RESULT_RUNNING,
		RESULT_FINISHED,
		RESULT_CANCELLED,
	};

	ResultStatus get_status() const;
	XrFutureEXT get_future() const;

	void cancel_future();

	OpenXRFutureResult(XrFutureEXT p_future, const Callable &p_on_success);

private:
	ResultStatus status = RESULT_RUNNING;
	XrFutureEXT future;
	Callable on_success_callback;

	uint64_t _get_future() const;
};

VARIANT_ENUM_CAST(OpenXRFutureResult::ResultStatus);

class OpenXRFutureExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRFutureExtension, OpenXRExtensionWrapper);

protected:
	static void _bind_methods();

public:
	static OpenXRFutureExtension *get_singleton();

	OpenXRFutureExtension();
	virtual ~OpenXRFutureExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions() override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_instance_destroyed() override;
	virtual void on_session_destroyed() override;

	virtual void on_process() override;

	bool is_active() const;

	Ref<OpenXRFutureResult> register_future(XrFutureEXT p_future, const Callable &p_on_success = Callable());
	void cancel_future(XrFutureEXT p_future);

private:
	static OpenXRFutureExtension *singleton;

	bool future_ext = false;

	HashMap<XrFutureEXT, Ref<OpenXRFutureResult>> futures;

	// Make these accessible from GDExtension and/or GDScript
	Ref<OpenXRFutureResult> _register_future(uint64_t p_future, const Callable &p_on_success = Callable());
	void _cancel_future(uint64_t p_future);

	// OpenXR API call wrappers

	// Futures
	EXT_PROTO_XRRESULT_FUNC3(xrPollFutureEXT, (XrInstance), instance, (const XrFuturePollInfoEXT *), poll_info, (XrFuturePollResultEXT *), poll_result);
	EXT_PROTO_XRRESULT_FUNC2(xrCancelFutureEXT, (XrInstance), instance, (const XrFutureCancelInfoEXT *), cancel_info);
};
