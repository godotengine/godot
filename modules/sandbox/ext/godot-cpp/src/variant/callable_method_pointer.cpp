/**************************************************************************/
/*  callable_method_pointer.cpp                                           */
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

#include <godot_cpp/variant/callable_method_pointer.hpp>

#include <godot_cpp/templates/hashfuncs.hpp>

namespace godot {

static void custom_callable_mp_call(void *p_userdata, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError *r_error) {
	CallableCustomMethodPointerBase *callable_method_pointer = (CallableCustomMethodPointerBase *)p_userdata;
	callable_method_pointer->call((const Variant **)p_args, p_argument_count, *(Variant *)r_return, *r_error);
}

static GDExtensionBool custom_callable_mp_is_valid(void *p_userdata) {
	CallableCustomMethodPointerBase *callable_method_pointer = (CallableCustomMethodPointerBase *)p_userdata;
	ObjectID object = callable_method_pointer->get_object();
	return object == ObjectID() || ObjectDB::get_instance(object);
}

static void custom_callable_mp_free(void *p_userdata) {
	CallableCustomMethodPointerBase *callable_method_pointer = (CallableCustomMethodPointerBase *)p_userdata;
	memdelete(callable_method_pointer);
}

static uint32_t custom_callable_mp_hash(void *p_userdata) {
	CallableCustomMethodPointerBase *callable_method_pointer = (CallableCustomMethodPointerBase *)p_userdata;
	return callable_method_pointer->get_hash();
}

static GDExtensionBool custom_callable_mp_equal_func(void *p_a, void *p_b) {
	CallableCustomMethodPointerBase *a = (CallableCustomMethodPointerBase *)p_a;
	CallableCustomMethodPointerBase *b = (CallableCustomMethodPointerBase *)p_b;

	if (a->get_comp_size() != b->get_comp_size()) {
		return false;
	}

	return memcmp(a->get_comp_ptr(), b->get_comp_ptr(), a->get_comp_size() * 4) == 0;
}

static GDExtensionBool custom_callable_mp_less_than_func(void *p_a, void *p_b) {
	CallableCustomMethodPointerBase *a = (CallableCustomMethodPointerBase *)p_a;
	CallableCustomMethodPointerBase *b = (CallableCustomMethodPointerBase *)p_b;

	if (a->get_comp_size() != b->get_comp_size()) {
		return a->get_comp_size() < b->get_comp_size();
	}

	return memcmp(a->get_comp_ptr(), b->get_comp_ptr(), a->get_comp_size() * 4) < 0;
}

static GDExtensionInt custom_callable_mp_get_argument_count_func(void *p_userdata, GDExtensionBool *r_is_valid) {
	CallableCustomMethodPointerBase *callable_method_pointer = (CallableCustomMethodPointerBase *)p_userdata;
	bool valid = false;
	int ret = callable_method_pointer->get_argument_count(valid);
	*r_is_valid = valid;
	return ret;
}

void CallableCustomMethodPointerBase::_setup(uint32_t *p_base_ptr, uint32_t p_ptr_size) {
	comp_ptr = p_base_ptr;
	comp_size = p_ptr_size / 4;

	for (uint32_t i = 0; i < comp_size; i++) {
		if (i == 0) {
			h = hash_murmur3_one_32(comp_ptr[i]);
		} else {
			h = hash_murmur3_one_32(comp_ptr[i], h);
		}
	}
}

namespace internal {

Callable create_callable_from_ccmp(CallableCustomMethodPointerBase *p_callable_method_pointer) {
	GDExtensionCallableCustomInfo2 info = {};
	info.callable_userdata = p_callable_method_pointer;
	info.token = internal::token;
	info.object_id = p_callable_method_pointer->get_object();
	info.call_func = &custom_callable_mp_call;
	info.is_valid_func = &custom_callable_mp_is_valid;
	info.free_func = &custom_callable_mp_free;
	info.hash_func = &custom_callable_mp_hash;
	info.equal_func = &custom_callable_mp_equal_func;
	info.less_than_func = &custom_callable_mp_less_than_func;
	info.get_argument_count_func = &custom_callable_mp_get_argument_count_func;

	Callable callable;
	::godot::internal::gdextension_interface_callable_custom_create2(callable._native_ptr(), &info);
	return callable;
}

} // namespace internal

} // namespace godot
