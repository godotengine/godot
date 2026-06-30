/**************************************************************************/
/*  socket_monitor_gcd.mm                                                 */
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

#include "socket_monitor_gcd.h"

#include "core/variant/variant.h"

#include <atomic>

// State we capture, to prevent use after free
struct SocketMonitorGCDState {
	Callable callable;
	std::atomic<bool> cancelled = false;

	explicit SocketMonitorGCDState(const Callable &p_callable) :
			callable(p_callable) {}
};

void SocketMonitorGCD::start(int p_fd, const Callable &p_on_readable) {
	stop();

	_source = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, p_fd, 0, dispatch_get_main_queue());
	if (!_source) {
		return;
	}

	SocketMonitorGCDState *state = memnew(SocketMonitorGCDState(p_on_readable));
	_state = state;

	dispatch_source_set_event_handler(_source, ^{
		// Once cancellation has been requested, stop delivering: the callable's
		// target may already be tearing down. 
		if (!state->cancelled.load()) {
			state->callable.call_deferred();
		}
	});
	dispatch_source_set_cancel_handler(_source, ^{
		// Runs after the last event handler completes; sole owner that frees state.
		memdelete(state);
	});

	dispatch_resume(_source);
	_active = true;
}

void SocketMonitorGCD::stop() {
	if (_source) {
		if (_state) {
			// Signal in-flight/queued event handlers before triggering the
			// asynchronous cancellation that will eventually free the state.
			_state->cancelled.store(true);
			_state = nullptr;
		}
		dispatch_source_cancel(_source);
		_source = nullptr;
	}
	_active = false;
}

SocketMonitorGCD::~SocketMonitorGCD() {
	stop();
}
