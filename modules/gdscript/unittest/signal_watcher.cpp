/*************************************************************************/
/*  signal_watcher.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "signal_watcher.h"

void SignalWatcher::watch(const Object *object, const String &signal) {
}

bool SignalWatcher::called(const Object *object, const String &signal) const {
	return false;
}

bool SignalWatcher::called_once(const Object *object, const String &signal) const {
	return false;
}

Variant SignalWatcher::_called_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	return false;
}

Variant SignalWatcher::_called_once_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	return false;
}

Variant SignalWatcher::_any_call(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	return false;
}

bool SignalWatcher::has_calls(const Object *object, const String &signal, const Array &calls, bool any_order) const {
	return false;
}

bool SignalWatcher::not_called(const Object *object, const String &signal) const {
	return false;
}

void SignalWatcher::reset() {
}

int SignalWatcher::call_count(const Object *object, const String &signal) const {
	return 0;
}

Array SignalWatcher::calls(const Object *object, const String &signal) const {
	Array calls;
	return calls;
}

Variant SignalWatcher::_handler(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	return NULL;
}

void SignalWatcher::_bind_methods() {
	ClassDB::bind_method(D_METHOD("watch", "object", "signal"), &SignalWatcher::watch);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "called_with", &SignalWatcher::_called_with, MethodInfo("called_with"));

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_handler", &SignalWatcher::_handler, MethodInfo("_handler"));
}
