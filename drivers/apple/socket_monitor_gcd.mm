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

void SocketMonitorGCD::start(int p_fd, const Callable &p_on_readable) {
	stop();

	_source = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, p_fd, 0, dispatch_get_main_queue());
	if (!_source) {
		return;
	}

	Callable cb = p_on_readable;
	dispatch_source_set_event_handler(_source, ^{
		cb.call_deferred();
	});

	dispatch_resume(_source);
	_active = true;
}

void SocketMonitorGCD::stop() {
	if (_source) {
		dispatch_source_cancel(_source);
		_source = nullptr;
	}
	_active = false;
}

SocketMonitorGCD::~SocketMonitorGCD() {
	stop();
}
