/**************************************************************************/
/*  callable_custom.hpp                                                   */
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

#include <godot_cpp/core/object_id.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

class Object;

class CallableCustomBase {
public:
	virtual ObjectID get_object() const = 0;
	virtual int get_argument_count(bool &r_is_valid) const;
	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, GDExtensionCallError &r_call_error) const = 0;
	virtual ~CallableCustomBase() {}
};

class CallableCustom : public CallableCustomBase {
public:
	typedef bool (*CompareEqualFunc)(const CallableCustom *p_a, const CallableCustom *p_b);
	typedef bool (*CompareLessFunc)(const CallableCustom *p_a, const CallableCustom *p_b);

	virtual uint32_t hash() const = 0;
	virtual String get_as_text() const = 0;
	virtual CompareEqualFunc get_compare_equal_func() const = 0;
	virtual CompareLessFunc get_compare_less_func() const = 0;
	virtual bool is_valid() const;
	virtual ObjectID get_object() const = 0;
	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, GDExtensionCallError &r_call_error) const = 0;
};

} // namespace godot
