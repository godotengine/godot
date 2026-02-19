/**************************************************************************/
/*  signal_watcher.cpp                                                    */
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

#include "signal_watcher.h"

#include "core/object/class_db.h"

void SignalWatcher::_add_signal_entry(const Array &p_args, const String &p_name) {
	if (!_signals.has(p_name)) {
		_signals[p_name] = Array();
	}
	_signals[p_name].push_back(p_args);
}

void SignalWatcher::_signal_callback_zero(const String &p_name) {
	Array args;
	_add_signal_entry(args, p_name);
}

void SignalWatcher::_signal_callback_one(Variant p_arg1, const String &p_name) {
	Array args = { p_arg1 };
	_add_signal_entry(args, p_name);
}

void SignalWatcher::_signal_callback_two(Variant p_arg1, Variant p_arg2, const String &p_name) {
	Array args = { p_arg1, p_arg2 };
	_add_signal_entry(args, p_name);
}

void SignalWatcher::_signal_callback_three(Variant p_arg1, Variant p_arg2, Variant p_arg3, const String &p_name) {
	Array args = { p_arg1, p_arg2, p_arg3 };
	_add_signal_entry(args, p_name);
}

void SignalWatcher::watch_signal(Object *p_object, const String &p_signal) {
	MethodInfo method_info;
	ClassDB::get_signal(p_object->get_class(), p_signal, &method_info);
	switch (method_info.arguments.size()) {
		case 0: {
			p_object->connect(p_signal, callable_mp(this, &SignalWatcher::_signal_callback_zero).bind(p_signal));
		} break;
		case 1: {
			p_object->connect(p_signal, callable_mp(this, &SignalWatcher::_signal_callback_one).bind(p_signal));
		} break;
		case 2: {
			p_object->connect(p_signal, callable_mp(this, &SignalWatcher::_signal_callback_two).bind(p_signal));
		} break;
		case 3: {
			p_object->connect(p_signal, callable_mp(this, &SignalWatcher::_signal_callback_three).bind(p_signal));
		} break;
		default: {
			MESSAGE("Signal ", p_signal, " arg count not supported.");
		} break;
	}
}

void SignalWatcher::unwatch_signal(Object *p_object, const String &p_signal) {
	MethodInfo method_info;
	ClassDB::get_signal(p_object->get_class(), p_signal, &method_info);
	switch (method_info.arguments.size()) {
		case 0: {
			p_object->disconnect(p_signal, callable_mp(this, &SignalWatcher::_signal_callback_zero));
		} break;
		case 1: {
			p_object->disconnect(p_signal, callable_mp(this, &SignalWatcher::_signal_callback_one));
		} break;
		case 2: {
			p_object->disconnect(p_signal, callable_mp(this, &SignalWatcher::_signal_callback_two));
		} break;
		case 3: {
			p_object->disconnect(p_signal, callable_mp(this, &SignalWatcher::_signal_callback_three));
		} break;
		default: {
			MESSAGE("Signal ", p_signal, " arg count not supported.");
		} break;
	}
}

bool SignalWatcher::check(const String &p_name, const Array &p_args) {
	if (!_signals.has(p_name)) {
		MESSAGE("Signal ", p_name, " not emitted");
		return false;
	}

	if (p_args.size() != _signals[p_name].size()) {
		MESSAGE("Signal has " << _signals[p_name] << " expected " << p_args);
		discard_signal(p_name);
		return false;
	}

	bool match = true;
	for (int i = 0; i < p_args.size(); i++) {
		if (((Array)p_args[i]).size() != ((Array)_signals[p_name][i]).size()) {
			MESSAGE("Signal has " << _signals[p_name][i] << " expected " << p_args[i]);
			match = false;
			continue;
		}

		for (int j = 0; j < ((Array)p_args[i]).size(); j++) {
			if (((Array)p_args[i])[j] != ((Array)_signals[p_name][i])[j]) {
				MESSAGE("Signal has " << _signals[p_name][i] << " expected " << p_args[i]);
				match = false;
				break;
			}
		}
	}

	discard_signal(p_name);
	return match;
}

bool SignalWatcher::check_false(const String &p_name) {
	bool has = _signals.has(p_name);
	if (has) {
		MESSAGE("Signal has " << _signals[p_name] << " expected none.");
	}
	discard_signal(p_name);
	return !has;
}

void SignalWatcher::discard_signal(const String &p_name) {
	if (_signals.has(p_name)) {
		_signals.erase(p_name);
	}
}

void SignalWatcher::_clear_signals() {
	_signals.clear();
}

SignalWatcher::SignalWatcher() {
	ERR_FAIL_COND_MSG(singleton, "Singleton in SignalWatcher already exists.");
	singleton = this;
}

SignalWatcher::~SignalWatcher() {
	if (this == singleton) {
		singleton = nullptr;
	}
}
