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

SignalWatcher::Args *SignalWatcher::write(const Object *object, const String &signal) {
	ObjectSignalArgs::Element *object_signal_args = m_signals.find(object);
	if (object_signal_args) {
		SignalArgs::Element *signal_args = object_signal_args->value().find(signal);
		if (signal_args) {
			return &signal_args->value();
		}
	}
	return NULL;
}

const SignalWatcher::Args *SignalWatcher::read(const Object *object, const String &signal) const {
	const ObjectSignalArgs::Element *object_signal_args = m_signals.find(object);
	if (object_signal_args) {
		const SignalArgs::Element *signal_args = object_signal_args->value().find(signal);
		if (signal_args) {
			return &signal_args->value();
		}
	}
	return NULL;
}

void SignalWatcher::touch(const Object *object, const String &signal) {
	ObjectSignalArgs::Element *object_signal_args = m_signals.find(object);
	if (object_signal_args) {
		SignalArgs &signal_args = object_signal_args->value();
		SignalArgs::Element *signal_args_element = signal_args.find(signal);
		if (!signal_args_element) {
			signal_args.insert(signal, Args());
		}
	} else {
		m_signals.insert(object, SignalArgs())->value().insert(signal, Args());
	}
}

void SignalWatcher::watch(Object *object, const String &signal) {
	touch(object, signal);
	Vector<Variant> binds;
	binds.resize(2);
	binds.set(0, object);
	binds.set(1, signal);
	object->connect(signal, this, "_handler", binds);
}

bool SignalWatcher::called(const Object *object, const String &signal) const {
	const Args *args = read(object, signal);
	return args != NULL && args->size() > 0;
}

bool SignalWatcher::called_once(const Object *object, const String &signal) const {
	const Args *args = read(object, signal);
	return args != NULL && args->size() == 1;
}

Variant SignalWatcher::_check_arguments(const Variant **p_args, int p_argcount, Variant::CallError &r_error, CheckArg check) {
	if (p_argcount < 2) {
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = p_argcount;
		return Variant();
	}
	if (p_args[0]->get_type() != Variant::OBJECT) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::OBJECT;
		return Variant();
	}
	if (p_args[1]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = Variant::STRING;
		return Variant();
	}
	r_error.error = Variant::CallError::CALL_OK;

	Object *object = *p_args[0];
	String signal = *p_args[1];
	Array arguments;
	int size = p_argcount - 2;
	arguments.resize(size);
	for (int i = 0; i < size; i++) {
		arguments[i] = *p_args[i + 2];
	}
	return (this->*check)(object, signal, arguments);
}

bool SignalWatcher::called_with(const Object *object, const String &signal, const Array &arguments) const {
	const Args *args = read(object, signal);
	int size = args->size();
	if (args && size > 0) {
		return (*args)[size - 1] == arguments;
	}
	return false;
}

Variant SignalWatcher::_called_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	return _check_arguments(p_args, p_argcount, r_error, &SignalWatcher::called_with);
}

bool SignalWatcher::called_once_with(const Object *object, const String &signal, const Array &arguments) const {
	return called_once(object, signal) && called_with(object, signal, arguments);
}

Variant SignalWatcher::_called_once_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	return _check_arguments(p_args, p_argcount, r_error, &SignalWatcher::called_once_with);
}

bool SignalWatcher::any_call(const Object *object, const String &signal, const Array &arguments) const {
	return false;
}

Variant SignalWatcher::_any_call(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	return _check_arguments(p_args, p_argcount, r_error, &SignalWatcher::any_call);
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
	const Args *args = read(object, signal);
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

	ClassDB::bind_method(D_METHOD("called", "object", "signal"), &SignalWatcher::called);
	ClassDB::bind_method(D_METHOD("called_once", "object", "signal"), &SignalWatcher::called_once);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "called_with", &SignalWatcher::_called_with);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "called_once_with", &SignalWatcher::_called_once_with);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "any_call", &SignalWatcher::_any_call);
	ClassDB::bind_method(D_METHOD("has_calls", "object", "signal", "calls", "any_order"), &SignalWatcher::has_calls, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("not_called", "object", "signal"), &SignalWatcher::not_called);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_handler", &SignalWatcher::_handler, MethodInfo("_handler"));
}
