/*************************************************************************/
/*  callable_bind.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CALLABLE_BIND_H
#define CALLABLE_BIND_H

#include "core/variant/callable.h"
#include "core/variant/variant.h"

class CallableCustomBind : public CallableCustom {
	Callable callable;
	Vector<Variant> binds;

	static bool _equal_func(const CallableCustom *p_a, const CallableCustom *p_b);
	static bool _less_func(const CallableCustom *p_a, const CallableCustom *p_b);

public:
	//for every type that inherits, these must always be the same for this type
	virtual uint32_t hash() const;
	virtual String get_as_text() const;
	virtual CompareEqualFunc get_compare_equal_func() const;
	virtual CompareLessFunc get_compare_less_func() const;
	virtual ObjectID get_object() const; //must always be able to provide an object
	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const;
	virtual const Callable *get_base_comparator() const;

	Callable get_callable() { return callable; }
	Vector<Variant> get_binds() { return binds; }

	CallableCustomBind(const Callable &p_callable, const Vector<Variant> &p_binds);
	virtual ~CallableCustomBind();
};

class CallableCustomUnbind : public CallableCustom {
	Callable callable;
	int argcount;

	static bool _equal_func(const CallableCustom *p_a, const CallableCustom *p_b);
	static bool _less_func(const CallableCustom *p_a, const CallableCustom *p_b);

public:
	//for every type that inherits, these must always be the same for this type
	virtual uint32_t hash() const;
	virtual String get_as_text() const;
	virtual CompareEqualFunc get_compare_equal_func() const;
	virtual CompareLessFunc get_compare_less_func() const;
	virtual ObjectID get_object() const; //must always be able to provide an object
	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const;
	virtual const Callable *get_base_comparator() const;

	Callable get_callable() { return callable; }
	int get_unbinds() { return argcount; }

	CallableCustomUnbind(const Callable &p_callable, int p_argcount);
	virtual ~CallableCustomUnbind();
};

Callable callable_bind(const Callable &p_callable, const Variant &p_arg1);
Callable callable_bind(const Callable &p_callable, const Variant &p_arg1, const Variant &p_arg2);
Callable callable_bind(const Callable &p_callable, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3);
Callable callable_bind(const Callable &p_callable, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4);
Callable callable_bind(const Callable &p_callable, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5);

#endif // CALLABLE_BIND_H
