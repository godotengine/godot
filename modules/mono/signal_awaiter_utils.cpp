/*************************************************************************/
/*  signal_awaiter_utils.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "signal_awaiter_utils.h"

#include "mono_gd/gd_mono_utils.h"

namespace SignalAwaiterUtils {

Error connect_signal_awaiter(Object *p_source, const String &p_signal, Object *p_target, MonoObject *p_awaiter) {

	ERR_FAIL_NULL_V(p_source, ERR_INVALID_DATA);
	ERR_FAIL_NULL_V(p_target, ERR_INVALID_DATA);

	uint32_t awaiter_handle = MonoGCHandle::make_strong_handle(p_awaiter);
	Ref<SignalAwaiterHandle> sa_con = memnew(SignalAwaiterHandle(awaiter_handle));
	Vector<Variant> binds;
	binds.push_back(sa_con);
	Error err = p_source->connect(p_signal, p_target, "_AwaitedSignalCallback", binds, Object::CONNECT_ONESHOT);

	if (err != OK) {
		// set it as completed to prevent it from calling the failure callback when deleted
		// the awaiter will be aware of the failure by checking the returned error
		sa_con->set_completed(true);
	}

	return err;
}
}

SignalAwaiterHandle::SignalAwaiterHandle(uint32_t p_handle)
	: MonoGCHandle(p_handle) {
}

SignalAwaiterHandle::~SignalAwaiterHandle() {
	if (!completed) {
		GDMonoUtils::SignalAwaiter_FailureCallback thunk = CACHED_METHOD_THUNK(SignalAwaiter, FailureCallback);

		MonoObject *awaiter = get_target();

		if (awaiter) {
			MonoObject *ex = NULL;
			thunk(awaiter, &ex);

			if (ex) {
				mono_print_unhandled_exception(ex);
				ERR_FAIL_V();
			}
		}
	}
}
