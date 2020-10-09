/*************************************************************************/
/*  callable_bind.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "callable_bind.h"

//////////////////////////////////

uint32_t CallableCustomBind::hash() const {
	return callable.hash();
}
String CallableCustomBind::get_as_text() const {
	return callable.operator String();
}

bool CallableCustomBind::_equal_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomBind *a = (const CallableCustomBind *)p_a;
	const CallableCustomBind *b = (const CallableCustomBind *)p_b;

	if (!(a->callable != b->callable)) {
		return false;
	}

	if (a->binds.size() != b->binds.size()) {
		return false;
	}

	return true;
}

bool CallableCustomBind::_less_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomBind *a = (const CallableCustomBind *)p_a;
	const CallableCustomBind *b = (const CallableCustomBind *)p_b;

	if (a->callable < b->callable) {
		return true;
	} else if (b->callable < a->callable) {
		return false;
	}

	return a->binds.size() < b->binds.size();
}

CallableCustom::CompareEqualFunc CallableCustomBind::get_compare_equal_func() const {
	return _equal_func;
}
CallableCustom::CompareLessFunc CallableCustomBind::get_compare_less_func() const {
	return _less_func;
}
ObjectID CallableCustomBind::get_object() const {
	return callable.get_object_id();
}
const Callable *CallableCustomBind::get_base_comparator() const {
	return &callable;
}

void CallableCustomBind::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	const Variant **args = (const Variant **)alloca(sizeof(const Variant **) * (binds.size() + p_argcount));
	for (int i = 0; i < p_argcount; i++) {
		args[i] = (const Variant *)p_arguments[i];
	}
	for (int i = 0; i < binds.size(); i++) {
		args[i + p_argcount] = (const Variant *)&binds[i];
	}

	callable.call(args, p_argcount + binds.size(), r_return_value, r_call_error);
}

CallableCustomBind::CallableCustomBind(const Callable &p_callable, const Vector<Variant> &p_binds) {
	callable = p_callable;
	binds = p_binds;
}

CallableCustomBind::~CallableCustomBind() {
}

//////////////////////////////////

uint32_t CallableCustomUnbind::hash() const {
	return callable.hash();
}
String CallableCustomUnbind::get_as_text() const {
	return callable.operator String();
}

bool CallableCustomUnbind::_equal_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomUnbind *a = (const CallableCustomUnbind *)p_a;
	const CallableCustomUnbind *b = (const CallableCustomUnbind *)p_b;

	if (!(a->callable != b->callable)) {
		return false;
	}

	if (a->argcount != b->argcount) {
		return false;
	}

	return true;
}

bool CallableCustomUnbind::_less_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomUnbind *a = (const CallableCustomUnbind *)p_a;
	const CallableCustomUnbind *b = (const CallableCustomUnbind *)p_b;

	if (a->callable < b->callable) {
		return true;
	} else if (b->callable < a->callable) {
		return false;
	}

	return a->argcount < b->argcount;
}

CallableCustom::CompareEqualFunc CallableCustomUnbind::get_compare_equal_func() const {
	return _equal_func;
}
CallableCustom::CompareLessFunc CallableCustomUnbind::get_compare_less_func() const {
	return _less_func;
}
ObjectID CallableCustomUnbind::get_object() const {
	return callable.get_object_id();
}
const Callable *CallableCustomUnbind::get_base_comparator() const {
	return &callable;
}

void CallableCustomUnbind::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	if (argcount > p_argcount) {
		r_call_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_call_error.argument = 0;
		r_call_error.expected = argcount;
		return;
	}
	callable.call(p_arguments, p_argcount - argcount, r_return_value, r_call_error);
}

CallableCustomUnbind::CallableCustomUnbind(const Callable &p_callable, int p_argcount) {
	callable = p_callable;
	argcount = p_argcount;
}

CallableCustomUnbind::~CallableCustomUnbind() {
}

Callable callable_bind(const Callable &p_callable, const Variant &p_arg1) {
	return p_callable.bind((const Variant **)&p_arg1, 1);
}

Callable callable_bind(const Callable &p_callable, const Variant &p_arg1, const Variant &p_arg2) {
	const Variant *args[2] = { &p_arg1, &p_arg2 };
	return p_callable.bind(args, 2);
}

Callable callable_bind(const Callable &p_callable, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3) {
	const Variant *args[3] = { &p_arg1, &p_arg2, &p_arg3 };
	return p_callable.bind(args, 3);
}

Callable callable_bind(const Callable &p_callable, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4) {
	const Variant *args[4] = { &p_arg1, &p_arg2, &p_arg3, &p_arg4 };
	return p_callable.bind(args, 4);
}

Callable callable_bind(const Callable &p_callable, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5) {
	const Variant *args[5] = { &p_arg1, &p_arg2, &p_arg3, &p_arg4, &p_arg5 };
	return p_callable.bind(args, 5);
}
