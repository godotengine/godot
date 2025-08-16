/**************************************************************************/
/*  callable_custom.cpp                                                   */
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

#include <godot_cpp/variant/callable_custom.hpp>

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

int CallableCustomBase::get_argument_count(bool &r_is_valid) const {
	r_is_valid = false;
	return 0;
}

static void callable_custom_call(void *p_userdata, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError *r_error) {
	CallableCustom *callable_custom = (CallableCustom *)p_userdata;
	callable_custom->call((const Variant **)p_args, p_argument_count, *(Variant *)r_return, *r_error);
}

static GDExtensionBool callable_custom_is_valid(void *p_userdata) {
	CallableCustom *callable_custom = (CallableCustom *)p_userdata;
	return callable_custom->is_valid();
}

static void callable_custom_free(void *p_userdata) {
	CallableCustom *callable_custom = (CallableCustom *)p_userdata;
	memdelete(callable_custom);
}

static uint32_t callable_custom_hash(void *p_userdata) {
	CallableCustom *callable_custom = (CallableCustom *)p_userdata;
	return callable_custom->hash();
}

static void callable_custom_to_string(void *p_userdata, GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) {
	CallableCustom *callable_custom = (CallableCustom *)p_userdata;
	*((String *)r_out) = callable_custom->get_as_text();
	*r_is_valid = true;
}

static GDExtensionBool callable_custom_equal_func(void *p_a, void *p_b) {
	CallableCustom *a = (CallableCustom *)p_a;
	CallableCustom *b = (CallableCustom *)p_b;
	CallableCustom::CompareEqualFunc func_a = a->get_compare_equal_func();
	CallableCustom::CompareEqualFunc func_b = b->get_compare_equal_func();
	if (func_a != func_b) {
		return false;
	}
	return func_a(a, b);
}

static GDExtensionBool callable_custom_less_than_func(void *p_a, void *p_b) {
	CallableCustom *a = (CallableCustom *)p_a;
	CallableCustom *b = (CallableCustom *)p_b;
	CallableCustom::CompareEqualFunc func_a = a->get_compare_less_func();
	CallableCustom::CompareEqualFunc func_b = b->get_compare_less_func();
	if (func_a != func_b) {
		// Just compare the addresses.
		return p_a < p_b;
	}
	return func_a(a, b);
}

static GDExtensionInt custom_callable_get_argument_count_func(void *p_userdata, GDExtensionBool *r_is_valid) {
	CallableCustom *callable_custom = (CallableCustom *)p_userdata;
	bool valid = false;
	int ret = callable_custom->get_argument_count(valid);
	*r_is_valid = valid;
	return ret;
}

bool CallableCustom::is_valid() const {
	// The same default implementation as in Godot.
	return ObjectDB::get_instance(get_object());
}

Callable::Callable(CallableCustom *p_callable_custom) {
	GDExtensionCallableCustomInfo2 info = {};
	info.callable_userdata = p_callable_custom;
	info.token = internal::token;
	info.object_id = p_callable_custom->get_object();
	info.call_func = &callable_custom_call;
	info.is_valid_func = &callable_custom_is_valid;
	info.free_func = &callable_custom_free;
	info.hash_func = &callable_custom_hash;
	info.equal_func = &callable_custom_equal_func;
	info.less_than_func = &callable_custom_less_than_func;
	info.to_string_func = &callable_custom_to_string;
	info.get_argument_count_func = &custom_callable_get_argument_count_func;

	::godot::internal::gdextension_interface_callable_custom_create2(_native_ptr(), &info);
}

CallableCustom *Callable::get_custom() const {
	CallableCustomBase *callable_custom = (CallableCustomBase *)::godot::internal::gdextension_interface_callable_custom_get_userdata(_native_ptr(), internal::token);
	return dynamic_cast<CallableCustom *>(callable_custom);
}

} // namespace godot
