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
#include "test_compare.h"

SignalWatcher::Args *SignalWatcher::write(const Object *p_object, const String &p_signal) {
	ObjectSignalArgs::Element *object_signal_args = m_signals.find(ObjectSignal(p_object, p_signal));
	if (object_signal_args) {
		return &object_signal_args->value();
	}
	return NULL;
}

const SignalWatcher::Args *SignalWatcher::read(const Object *p_object, const String &p_signal) const {
	const ObjectSignalArgs::Element *object_signal_args = m_signals.find(ObjectSignal(p_object, p_signal));
	if (object_signal_args) {
		return &object_signal_args->value();
	}
	return NULL;
}

void SignalWatcher::touch(const Object *p_object, const String &p_signal) {
	ObjectSignalArgs::Element *object_signal_args = m_signals.find(ObjectSignal(p_object, p_signal));
	if (!object_signal_args) {
		m_signals.insert(ObjectSignal(p_object, p_signal), Args());
	}
}

void SignalWatcher::watch(Object *p_object, const String &p_signal) {
	touch(p_object, p_signal);
	Vector<Variant> binds;
	binds.resize(2);
	binds.set(0, p_object);
	binds.set(1, p_signal);
	p_object->connect(p_signal, this, "_handler", binds);
}

void SignalWatcher::watch_all(Object *p_object) {
}

bool SignalWatcher::called(const Object *p_object, const String &p_signal) const {
	const Args *args = read(p_object, p_signal);
	return args != NULL && args->size() > 0;
}

bool SignalWatcher::called_once(const Object *p_object, const String &p_signal) const {
	const Args *args = read(p_object, p_signal);
	return args != NULL && args->size() == 1;
}

bool SignalWatcher::parse_params(const Variant **p_args, int p_argcount, Variant::CallError &r_error, Params &r_params) {
	if (p_argcount < 2) {
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = p_argcount;
		return false;
	}
	if (p_args[0]->get_type() != Variant::OBJECT) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::OBJECT;
		return false;
	}
	if (p_args[1]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = Variant::STRING;
		return false;
	}
	r_error.error = Variant::CallError::CALL_OK;

	r_params.m_object = *p_args[0];
	r_params.m_signal = *p_args[1];
	int size = p_argcount - 2;
	r_params.m_arguments.resize(size);
	for (int i = 0; i < size; i++) {
		r_params.m_arguments[i] = *p_args[i + 2];
	}
	return true;
}

bool SignalWatcher::called_with(const Object *p_object, const String &p_signal, const Array &p_arguments) const {
	const Args *args = read(p_object, p_signal);
	int size = args->size();
	if (args && size > 0) {
		return TestCompare::deep_equal((*args)[size - 1], p_arguments);
	}
	return false;
}

Variant SignalWatcher::_called_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	Params params;
	if (parse_params(p_args, p_argcount, r_error, params)) {
		return called_with(params.m_object, params.m_signal, params.m_arguments);
	}
	return Variant();
}

bool SignalWatcher::called_once_with(const Object *p_object, const String &p_signal, const Array &p_arguments) const {
	return called_once(p_object, p_signal) && called_with(p_object, p_signal, p_arguments);
}

Variant SignalWatcher::_called_once_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	Params params;
	if (parse_params(p_args, p_argcount, r_error, params)) {
		return called_once_with(params.m_object, params.m_signal, params.m_arguments);
	}
	return Variant();
}

bool SignalWatcher::any_call(const Object *p_object, const String &p_signal, const Array &p_arguments) const {
	const Args *args = read(p_object, p_signal);
	if (args) {
		return args->find(p_arguments) != -1;
	}
	return false;
}

Variant SignalWatcher::_any_call(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	Params params;
	if (parse_params(p_args, p_argcount, r_error, params)) {
		return any_call(params.m_object, params.m_signal, params.m_arguments);
	}
	return Variant();
}

const Array to_array(const Variant &variant) {
	if (variant.is_array()) {
		return variant;
	} else {
		Array arguments;
		arguments.resize(1);
		arguments[0] = variant;
		return arguments;
	}
}

int SignalWatcher::has_calls(const Object *p_object, const String &p_signal, const Array &calls, bool any_order) const {
	const Args *args = read(p_object, p_signal);
	if (args) {
		if (any_order) {
			int call_count = calls.size();
			for (int i = 0; i < call_count; i++) {
				if (args->find(to_array(calls[i])) == -1) {
					return i;
				}
			}
			return call_count;
		} else {
			int arg_count = args->size();
			int call_count = calls.size();
			int call_index = 0;
			for (int arg_index = 0; call_index < call_count && arg_index < arg_count; arg_index++) {
				if (args->get(arg_index) == to_array(calls[call_index])) {
					call_index++;
				}
			}
			return call_index;
		}
	}
	return -1;
}

bool SignalWatcher::not_called(const Object *p_object, const String &p_signal) const {
	const Args *args = read(p_object, p_signal);
	return args && args->size() == 0;
}

void SignalWatcher::reset() {
	m_signals.clear();
}

int SignalWatcher::call_count(const Object *p_object, const String &p_signal) const {
	const Args *args = read(p_object, p_signal);
	return args ? args->size() : -1;
}

Array SignalWatcher::calls(const Object *p_object, const String &p_signal) const {
	Array calls;
	const Args *args = read(p_object, p_signal);
	if (args) {
		int size = args->size();
		calls.resize(size);
		for (int i = 0; i < size; i++) {
			calls[i] = (*args)[i];
		}
	}
	return calls;
}

Variant SignalWatcher::_handler(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	if (p_argcount < 2) {
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = p_argcount;
		return Variant();
	}
	if (p_args[p_argcount - 2]->get_type() != Variant::OBJECT) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = p_argcount - 2;
		r_error.expected = Variant::OBJECT;
		return Variant();
	}
	if (p_args[p_argcount - 1]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = p_argcount - 1;
		r_error.expected = Variant::STRING;
		return Variant();
	}
	r_error.error = Variant::CallError::CALL_OK;

	Object *object = *p_args[p_argcount - 2];
	String signal = *p_args[p_argcount - 1];
	Array arguments;
	int size = p_argcount - 2;
	arguments.resize(size);
	for (int i = 0; i < size; i++) {
		arguments[i] = *p_args[i];
	}
	Args *args = write(object, signal);
	if (args) {
		args->push_back(arguments);
	}
	return Variant();
}

void SignalWatcher::_bind_methods() {
	ClassDB::bind_method(D_METHOD("watch", "object", "signal"), &SignalWatcher::watch);
	ClassDB::bind_method(D_METHOD("watch_all", "object"), &SignalWatcher::watch_all);

	ClassDB::bind_method(D_METHOD("called", "object", "signal"), &SignalWatcher::called);
	ClassDB::bind_method(D_METHOD("called_once", "object", "signal"), &SignalWatcher::called_once);
	MethodInfo mi;
	mi.name = "called_with";
	mi.arguments.push_back(PropertyInfo(Variant::OBJECT, "object"));
	mi.arguments.push_back(PropertyInfo(Variant::STRING, "signal"));
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "called_with", &SignalWatcher::_called_with, mi);
	mi.name = "called_once_with";
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "called_once_with", &SignalWatcher::_called_once_with, mi);
	mi.name = "any_call";
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "any_call", &SignalWatcher::_any_call, mi);
	ClassDB::bind_method(D_METHOD("has_calls", "object", "signal", "calls", "any_order"), &SignalWatcher::has_calls, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("not_called", "object", "signal"), &SignalWatcher::not_called);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_handler", &SignalWatcher::_handler, MethodInfo("_handler"));
}
