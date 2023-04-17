/**************************************************************************/
/*  callable_bind_unbind.cpp                                              */
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

#include "callable_bind_unbind.h"

uint32_t CallableCustomBindUnbind::hash() const {
	return callable.hash();
}
String CallableCustomBindUnbind::get_as_text() const {
	return callable.operator String();
}

bool CallableCustomBindUnbind::_equal_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomBindUnbind *a = static_cast<const CallableCustomBindUnbind *>(p_a);
	const CallableCustomBindUnbind *b = static_cast<const CallableCustomBindUnbind *>(p_b);

	if (!(a->callable != b->callable)) {
		return false;
	}

	if (a->binds.size() != b->binds.size()) { // `a->binds != b->binds`?
		return false;
	}

	if (a->unbind_argcount != b->unbind_argcount) {
		return false;
	}

	return true;
}

bool CallableCustomBindUnbind::_less_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomBindUnbind *a = static_cast<const CallableCustomBindUnbind *>(p_a);
	const CallableCustomBindUnbind *b = static_cast<const CallableCustomBindUnbind *>(p_b);

	if (a->callable < b->callable) {
		return true;
	} else if (b->callable < a->callable) {
		return false;
	}

	if (a->binds.size() < b->binds.size()) {
		return true;
	} else if (b->binds.size() < a->binds.size()) {
		return false;
	}

	return a->unbind_argcount < b->unbind_argcount;
}

CallableCustom::CompareEqualFunc CallableCustomBindUnbind::get_compare_equal_func() const {
	return _equal_func;
}

CallableCustom::CompareLessFunc CallableCustomBindUnbind::get_compare_less_func() const {
	return _less_func;
}

bool CallableCustomBindUnbind::is_valid() const {
	return callable.is_valid();
}

StringName CallableCustomBindUnbind::get_method() const {
	return callable.get_method();
}

ObjectID CallableCustomBindUnbind::get_object() const {
	return callable.get_object_id();
}

const Callable *CallableCustomBindUnbind::get_base_comparator() const {
	return &callable;
}

bool CallableCustomBindUnbind::is_bind_unbind() const {
	return true;
}

int CallableCustomBindUnbind::get_bound_arguments_count() const {
	return MAX(0, callable.get_bound_arguments_count() - unbind_argcount) + binds.size();
}

void CallableCustomBindUnbind::get_bound_arguments(Vector<Variant> &r_arguments, int &r_argcount) const {
	Vector<Variant> sub_args;
	int sub_count;
	callable.get_bound_arguments_ref(sub_args, sub_count);

	int new_count = sub_count - unbind_argcount;

	if (new_count <= 0) {
		r_arguments = binds;
		r_argcount = binds.size();
		return;
	}

	int all_count = new_count + binds.size();

	r_arguments.resize(all_count);
	for (int i = 0; i < all_count; i++) {
		if (i < new_count) {
			r_arguments.write[i] = sub_args[i];
		} else {
			r_arguments.write[i] = binds[i - new_count];
		}
	}

	r_argcount = all_count;
}

void CallableCustomBindUnbind::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	int new_count = p_argcount - unbind_argcount;

	if (new_count < 0) {
		r_call_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_call_error.argument = 0;
		r_call_error.expected = -new_count; // !
		return;
	}

	int all_count = new_count + binds.size();

	const Variant **args = (const Variant **)alloca(sizeof(const Variant **) * all_count);
	for (int i = 0; i < all_count; i++) {
		if (i < new_count) {
			args[i] = (const Variant *)p_arguments[i];
		} else {
			args[i] = (const Variant *)&binds[i - new_count];
		}
	}

	callable.callp(args, all_count, r_return_value, r_call_error);
}

CallableCustomBindUnbind::CallableCustomBindUnbind(const Callable &p_callable, const Vector<Variant> &p_binds, int p_unbind_argcount) {
	callable = p_callable;
	binds = p_binds;
	unbind_argcount = p_unbind_argcount;
}

CallableCustomBindUnbind::CallableCustomBindUnbind(const CallableCustomBindUnbind *p_other, const Vector<Variant> &p_binds, int p_unbind_argcount) {
	callable = p_other->callable;
	binds = p_other->binds;
	unbind_argcount = p_other->unbind_argcount;

	if (!binds.is_empty() && p_unbind_argcount > 0) {
		int n = MIN(binds.size(), p_unbind_argcount);
		binds = binds.slice(n);
		p_unbind_argcount -= n;
	}

	binds.append_array(p_binds);
	unbind_argcount += p_unbind_argcount;
}
