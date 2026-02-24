/**************************************************************************/
/*  signal_watcher.h                                                      */
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

#include "core/object/object.h"
#include "tests/test_macros.h"

// Utility class / macros for testing signals
//
// Use SIGNAL_WATCH(*object, "signal_name") to start watching
// Makes sure to call SIGNAL_UNWATCH(*object, "signal_name") to stop watching in cleanup, this is not done automatically.
//
// The SignalWatcher will capture all signals and their args sent between checks.
//
// Use SIGNAL_CHECK("signal_name"), Vector<Vector<Variant>>), to check the arguments of all fired signals.
// The outer vector is each fired signal, the inner vector the list of arguments for that signal. Order does matter.
//
// Use SIGNAL_CHECK_FALSE("signal_name") to check if a signal was not fired.
//
// Use SIGNAL_DISCARD("signal_name") to discard records all of the given signal, use only in placed you don't need to check.
//
// All signals are automatically discarded between test/sub test cases.

class SignalWatcher : public Object {
	inline static SignalWatcher *singleton = nullptr;

	HashMap<String, Array> _signals;

	void _add_signal_entry(const Array &p_args, const String &p_name);
	void _signal_callback_zero(const String &p_name);
	void _signal_callback_one(Variant p_arg1, const String &p_name);
	void _signal_callback_two(Variant p_arg1, Variant p_arg2, const String &p_name);
	void _signal_callback_three(Variant p_arg1, Variant p_arg2, Variant p_arg3, const String &p_name);

public:
	static SignalWatcher *get_singleton() { return singleton; }

	void watch_signal(Object *p_object, const String &p_signal);
	void unwatch_signal(Object *p_object, const String &p_signal);
	bool check(const String &p_name, const Array &p_args);
	bool check_false(const String &p_name);
	void discard_signal(const String &p_name);

	void _clear_signals();

	SignalWatcher();
	~SignalWatcher();
};

#define SIGNAL_WATCH(m_object, m_signal) SignalWatcher::get_singleton()->watch_signal(m_object, m_signal);
#define SIGNAL_UNWATCH(m_object, m_signal) SignalWatcher::get_singleton()->unwatch_signal(m_object, m_signal);

#define SIGNAL_CHECK(m_signal, m_args) CHECK(SignalWatcher::get_singleton()->check(m_signal, m_args));
#define SIGNAL_CHECK_FALSE(m_signal) CHECK(SignalWatcher::get_singleton()->check_false(m_signal));
#define SIGNAL_DISCARD(m_signal) SignalWatcher::get_singleton()->discard_signal(m_signal);
