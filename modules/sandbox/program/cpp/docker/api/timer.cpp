/**************************************************************************/
/*  timer.cpp                                                             */
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

#include "timer.hpp"

#include "object.hpp"
#include "syscalls.h"

using TimerEngineCallback = Variant (*)(Variant, Variant);
using TimerEngineNativeCallback = Variant (*)(Object, PackedArray<uint8_t>);
MAKE_SYSCALL(ECALL_TIMER_PERIODIC, void, sys_timer_periodic, CallbackTimer::period_t, bool, TimerEngineCallback, void *, Variant *);
MAKE_SYSCALL(ECALL_TIMER_PERIODIC, void, sys_timer_periodic_native, CallbackTimer::period_t, bool, TimerEngineNativeCallback, void *, Variant *);
MAKE_SYSCALL(ECALL_TIMER_STOP, void, sys_timer_stop, unsigned);

// clang-format off
Variant CallbackTimer::create(period_t period, bool oneshot, TimerCallback callback) {
	Variant timer;
	sys_timer_periodic(period, oneshot, [](Variant timer, Variant storage) -> Variant {
		std::vector<uint8_t> callback = storage.as_byte_array().fetch();
		CallbackTimer::TimerCallback *timerfunc = (CallbackTimer::TimerCallback *)callback.data();
		return (*timerfunc)(timer);
	}, &callback, &timer);
	return timer;
}

Variant CallbackTimer::create_native(period_t period, bool oneshot, TimerNativeCallback callback) {
	Variant timer;
	sys_timer_periodic_native(period, oneshot, [](Object timer, PackedArray<uint8_t> storage) -> Variant {
		std::vector<uint8_t> callback = storage.fetch();
		CallbackTimer::TimerNativeCallback *timerfunc = (CallbackTimer::TimerNativeCallback *)callback.data();
		return (*timerfunc)(timer);
	}, &callback, &timer);
	return timer;
}
// clang-format on
