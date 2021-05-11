/*************************************************************************/
/*  gd_mono_marshal.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gd_mono_marshal.h"

#include "../signal_awaiter_utils.h"
#include "gd_mono.h"
#include "gd_mono_cache.h"
#include "gd_mono_class.h"

namespace GDMonoMarshal {

// TODO: Those are just temporary until the code that needs them is moved to C#

Variant::Type managed_to_variant_type(const ManagedType &p_type, bool *r_nil_is_variant) {
	MonoReflectionType *refltype = mono_type_get_object(mono_domain_get(), p_type.type_class->get_mono_type());
	MonoBoolean nil_is_variant = false;

	MonoException *exc = nullptr;
	int32_t ret = CACHED_METHOD_THUNK(Marshaling, managed_to_variant_type)
						  .invoke(refltype, &nil_is_variant, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		return Variant::NIL;
	}

	if (r_nil_is_variant) {
		*r_nil_is_variant = (bool)nil_is_variant;
	}

	return (Variant::Type)ret;
}

bool try_get_array_element_type(const ManagedType &p_array_type, ManagedType &r_elem_type) {
	MonoReflectionType *array_refltype = mono_type_get_object(mono_domain_get(), p_array_type.type_class->get_mono_type());
	MonoReflectionType *elem_refltype = nullptr;

	MonoException *exc = nullptr;
	MonoBoolean ret = CACHED_METHOD_THUNK(Marshaling, try_get_array_element_type)
							  .invoke(array_refltype, &elem_refltype, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		return Variant::NIL;
	}

	r_elem_type = ManagedType::from_reftype(elem_refltype);
	return ret;
}

MonoObject *variant_to_mono_object_of_type(const Variant &p_var, const ManagedType &p_type) {
	MonoReflectionType *refltype = mono_type_get_object(mono_domain_get(), p_type.type_class->get_mono_type());

	MonoException *exc = nullptr;
	MonoObject *ret = CACHED_METHOD_THUNK(Marshaling, variant_to_mono_object_of_type)
							  .invoke(&p_var, refltype, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		return nullptr;
	}

	return ret;
}

MonoObject *variant_to_mono_object(const Variant &p_var) {
	MonoException *exc = nullptr;
	MonoObject *ret = CACHED_METHOD_THUNK(Marshaling, variant_to_mono_object)
							  .invoke(&p_var, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		return nullptr;
	}

	return ret;
}

static Variant mono_object_to_variant_impl(MonoObject *p_obj, bool p_fail_with_err) {
	if (!p_obj) {
		return Variant();
	}

	MonoBoolean fail_with_error = p_fail_with_err;

	Variant ret;

	MonoException *exc = nullptr;
	CACHED_METHOD_THUNK(Marshaling, mono_object_to_variant_out)
			.invoke(p_obj, fail_with_error, &ret, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		return Variant();
	}

	return ret;
}

Variant mono_object_to_variant(MonoObject *p_obj) {
	return mono_object_to_variant_impl(p_obj, /* fail_with_err: */ true);
}

Variant mono_object_to_variant_no_err(MonoObject *p_obj) {
	return mono_object_to_variant_impl(p_obj, /* fail_with_err: */ false);
}

String mono_object_to_variant_string(MonoObject *p_obj, MonoException **r_exc) {
	if (p_obj == nullptr) {
		return String("null");
	}

	Variant var = GDMonoMarshal::mono_object_to_variant_no_err(p_obj);

	if (var.get_type() == Variant::NIL) { // `&& p_obj != nullptr` but omitted because always true
		// Cannot convert MonoObject* to Variant; fallback to 'ToString()'.
		MonoException *exc = nullptr;
		MonoString *mono_str = GDMonoUtils::object_to_string(p_obj, &exc);

		if (exc) {
			if (r_exc) {
				*r_exc = exc;
			}
			return String();
		}

		return GDMonoMarshal::mono_string_to_godot(mono_str);
	} else {
		return var.operator String();
	}
}

MonoArray *Array_to_mono_array(const Array &p_array) {
	int length = p_array.size();
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(MonoObject), length);

	for (int i = 0; i < length; i++) {
		MonoObject *boxed = variant_to_mono_object(p_array[i]);
		mono_array_setref(ret, i, boxed);
	}

	return ret;
}

Array mono_array_to_Array(MonoArray *p_array) {
	Array ret;
	if (!p_array) {
		return ret;
	}
	int length = mono_array_length(p_array);
	ret.resize(length);

	for (int i = 0; i < length; i++) {
		MonoObject *elem = mono_array_get(p_array, MonoObject *, i);
		ret[i] = mono_object_to_variant(elem);
	}

	return ret;
}

MonoArray *PackedStringArray_to_mono_array(const PackedStringArray &p_array) {
	const String *r = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(String), length);

	for (int i = 0; i < length; i++) {
		MonoString *boxed = mono_string_from_godot(r[i]);
		mono_array_setref(ret, i, boxed);
	}

	return ret;
}
} // namespace GDMonoMarshal
