/**************************************************************************/
/*  callable_bind.cpp                                                     */
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

#include "callable_bind.h"

//////////////////////////////////

uint32_t CallableCustomBind::hash() const {
	return callable.hash();
}
String CallableCustomBind::get_as_text() const {
	return callable.operator String();
}

bool CallableCustomBind::_equal_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomBind *a = static_cast<const CallableCustomBind *>(p_a);
	const CallableCustomBind *b = static_cast<const CallableCustomBind *>(p_b);

	if (!(a->callable != b->callable)) {
		return false;
	}

	if (a->binds.size() != b->binds.size()) {
		return false;
	}

	return true;
}

bool CallableCustomBind::_less_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomBind *a = static_cast<const CallableCustomBind *>(p_a);
	const CallableCustomBind *b = static_cast<const CallableCustomBind *>(p_b);

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

bool CallableCustomBind::is_valid() const {
	return callable.is_valid();
}

StringName CallableCustomBind::get_method() const {
	return callable.get_method();
}

ObjectID CallableCustomBind::get_object() const {
	return callable.get_object_id();
}

const Callable *CallableCustomBind::get_base_comparator() const {
	return callable.get_base_comparator();
}

int CallableCustomBind::get_bound_arguments_count() const {
	return callable.get_bound_arguments_count() + binds.size();
}

void CallableCustomBind::get_bound_arguments(Vector<Variant> &r_arguments, int &r_argcount) const {
	Vector<Variant> sub_args;
	int sub_count;
	callable.get_bound_arguments_ref(sub_args, sub_count);

	if (sub_count == 0) {
		r_arguments = binds;
		r_argcount = binds.size();
		return;
	}

	int new_count = sub_count + binds.size();
	r_argcount = new_count;

	if (new_count <= 0) {
		// Removed more arguments than it adds.
		r_arguments = Vector<Variant>();
		return;
	}

	r_arguments.resize(new_count);

	if (sub_count > 0) {
		for (int i = 0; i < sub_count; i++) {
			r_arguments.write[i] = sub_args[i];
		}
		for (int i = 0; i < binds.size(); i++) {
			r_arguments.write[i + sub_count] = binds[i];
		}
		r_argcount = new_count;
	} else {
		for (int i = 0; i < binds.size() + sub_count; i++) {
			r_arguments.write[i] = binds[i - sub_count];
		}
	}
}

void CallableCustomBind::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	const Variant **args = (const Variant **)alloca(sizeof(Variant *) * (binds.size() + p_argcount));
	for (int i = 0; i < p_argcount; i++) {
		args[i] = (const Variant *)p_arguments[i];
	}
	for (int i = 0; i < binds.size(); i++) {
		args[i + p_argcount] = (const Variant *)&binds[i];
	}

	callable.callp(args, p_argcount + binds.size(), r_return_value, r_call_error);
}

Error CallableCustomBind::rpc(int p_peer_id, const Variant **p_arguments, int p_argcount, Callable::CallError &r_call_error) const {
	const Variant **args = (const Variant **)alloca(sizeof(Variant *) * (binds.size() + p_argcount));
	for (int i = 0; i < p_argcount; i++) {
		args[i] = (const Variant *)p_arguments[i];
	}
	for (int i = 0; i < binds.size(); i++) {
		args[i + p_argcount] = (const Variant *)&binds[i];
	}

	return callable.rpcp(p_peer_id, args, p_argcount + binds.size(), r_call_error);
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
	const CallableCustomUnbind *a = static_cast<const CallableCustomUnbind *>(p_a);
	const CallableCustomUnbind *b = static_cast<const CallableCustomUnbind *>(p_b);

	if (!(a->callable != b->callable)) {
		return false;
	}

	if (a->argcount != b->argcount) {
		return false;
	}

	return true;
}

bool CallableCustomUnbind::_less_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomUnbind *a = static_cast<const CallableCustomUnbind *>(p_a);
	const CallableCustomUnbind *b = static_cast<const CallableCustomUnbind *>(p_b);

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

bool CallableCustomUnbind::is_valid() const {
	return callable.is_valid();
}

StringName CallableCustomUnbind::get_method() const {
	return callable.get_method();
}

ObjectID CallableCustomUnbind::get_object() const {
	return callable.get_object_id();
}

const Callable *CallableCustomUnbind::get_base_comparator() const {
	return callable.get_base_comparator();
}

int CallableCustomUnbind::get_bound_arguments_count() const {
	return callable.get_bound_arguments_count() - argcount;
}

void CallableCustomUnbind::get_bound_arguments(Vector<Variant> &r_arguments, int &r_argcount) const {
	Vector<Variant> sub_args;
	int sub_count;
	callable.get_bound_arguments_ref(sub_args, sub_count);

	r_argcount = sub_args.size() - argcount;

	if (argcount >= sub_args.size()) {
		r_arguments = Vector<Variant>();
	} else {
		sub_args.resize(sub_args.size() - argcount);
		r_arguments = sub_args;
	}
}

void CallableCustomUnbind::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	if (p_argcount < argcount) {
		r_call_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_call_error.expected = argcount;
		return;
	}
	callable.callp(p_arguments, p_argcount - argcount, r_return_value, r_call_error);
}

Error CallableCustomUnbind::rpc(int p_peer_id, const Variant **p_arguments, int p_argcount, Callable::CallError &r_call_error) const {
	if (p_argcount < argcount) {
		r_call_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_call_error.expected = argcount;
		return ERR_UNCONFIGURED;
	}
	return callable.rpcp(p_peer_id, p_arguments, p_argcount - argcount, r_call_error);
}

CallableCustomUnbind::CallableCustomUnbind(const Callable &p_callable, int p_argcount) {
	callable = p_callable;
	argcount = p_argcount;
}

CallableCustomUnbind::~CallableCustomUnbind() {
}
