/**************************************************************************/
/*  callable_bind_unbind.h                                                */
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

#ifndef CALLABLE_BIND_UNBIND_H
#define CALLABLE_BIND_UNBIND_H

#include "core/variant/callable.h"
#include "core/variant/variant.h"

class CallableCustomBindUnbind : public CallableCustom {
	Callable callable;
	Vector<Variant> binds;
	int unbind_argcount;

	static bool _equal_func(const CallableCustom *p_a, const CallableCustom *p_b);
	static bool _less_func(const CallableCustom *p_a, const CallableCustom *p_b);

public:
	// For every type that inherits, these must always be the same for this type.
	virtual uint32_t hash() const override;
	virtual String get_as_text() const override;
	virtual CompareEqualFunc get_compare_equal_func() const override;
	virtual CompareLessFunc get_compare_less_func() const override;
	virtual bool is_valid() const override;
	virtual StringName get_method() const override;
	virtual ObjectID get_object() const override;
	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const override;
	virtual const Callable *get_base_comparator() const override;
	virtual bool is_bind_unbind() const override;
	virtual int get_bound_arguments_count() const override;
	virtual void get_bound_arguments(Vector<Variant> &r_arguments, int &r_argcount) const override;

	_FORCE_INLINE_ Callable get_callable() const { return callable; }
	_FORCE_INLINE_ Vector<Variant> get_binds() const { return binds; }
	_FORCE_INLINE_ int get_unbinds() const { return unbind_argcount; }

	CallableCustomBindUnbind(const Callable &p_callable, const Vector<Variant> &p_binds, int p_unbind_argcount);
	CallableCustomBindUnbind(const CallableCustomBindUnbind *p_other, const Vector<Variant> &p_binds, int p_unbind_argcount);
};

#endif // CALLABLE_BIND_UNBIND_H
