/**************************************************************************/
/*  timer.hpp                                                             */
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

#include "function.hpp"
#include "variant.hpp"
struct Object;

struct CallbackTimer {
	using period_t = double;
	using TimerCallback = Function<Variant(Variant)>;
	using TimerNativeCallback = Function<Variant(Object)>;

	// For when all arguments are Variants
	static Variant oneshot(period_t secs, TimerCallback callback);

	static Variant periodic(period_t period, TimerCallback callback);

	// For when native/register-based arguments are enabled
	static Variant native_oneshot(period_t secs, TimerNativeCallback callback);

	static Variant native_periodic(period_t period, TimerNativeCallback callback);

private:
	static Variant create(period_t p, bool oneshot, TimerCallback callback);
	static Variant create_native(period_t p, bool oneshot, TimerNativeCallback callback);
};

inline Variant CallbackTimer::oneshot(period_t secs, TimerCallback callback) {
	return create(secs, true, callback);
}

inline Variant CallbackTimer::periodic(period_t period, TimerCallback callback) {
	return create(period, false, callback);
}

inline Variant CallbackTimer::native_oneshot(period_t secs, TimerNativeCallback callback) {
	return create_native(secs, true, callback);
}

inline Variant CallbackTimer::native_periodic(period_t period, TimerNativeCallback callback) {
	return create_native(period, false, callback);
}
