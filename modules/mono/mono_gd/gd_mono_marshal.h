/*************************************************************************/
/*  gd_mono_marshal.h                                                    */
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

#ifndef GDMONOMARSHAL_H
#define GDMONOMARSHAL_H

#include "core/variant/variant.h"

#include "gd_mono.h"
#include "gd_mono_utils.h"

namespace GDMonoMarshal {

template <typename T>
T unbox(MonoObject *p_obj) {
	return *(T *)mono_object_unbox(p_obj);
}

Variant::Type managed_to_variant_type(const ManagedType &p_type, bool *r_nil_is_variant = nullptr);

bool try_get_array_element_type(const ManagedType &p_array_type, ManagedType &r_elem_type);

// String

_FORCE_INLINE_ String mono_string_to_godot_not_null(MonoString *p_mono_string) {
	char32_t *utf32 = (char32_t *)mono_string_to_utf32(p_mono_string);
	String ret = String(utf32);
	mono_free(utf32);
	return ret;
}

_FORCE_INLINE_ String mono_string_to_godot(MonoString *p_mono_string) {
	if (p_mono_string == nullptr) {
		return String();
	}

	return mono_string_to_godot_not_null(p_mono_string);
}

_FORCE_INLINE_ MonoString *mono_string_from_godot(const String &p_string) {
	return mono_string_from_utf32((mono_unichar4 *)(p_string.get_data()));
}

// Variant

MonoObject *variant_to_mono_object_of_type(const Variant &p_var, const ManagedType &p_type);

MonoObject *variant_to_mono_object(const Variant &p_var);

// These overloads were added to avoid passing a `const Variant *` to the `const Variant &`
// parameter. That would result in the `Variant(bool)` copy constructor being called as
// pointers are implicitly converted to bool. Implicit conversions are f-ing evil.

_FORCE_INLINE_ MonoObject *variant_to_mono_object_of_type(const Variant *p_var, const ManagedType &p_type) {
	return variant_to_mono_object_of_type(*p_var, p_type);
}
_FORCE_INLINE_ MonoObject *variant_to_mono_object(const Variant *p_var) {
	return variant_to_mono_object(*p_var);
}

Variant mono_object_to_variant(MonoObject *p_obj);
Variant mono_object_to_variant_no_err(MonoObject *p_obj);

/// Tries to convert the MonoObject* to Variant and then convert the Variant to String.
/// If the MonoObject* cannot be converted to Variant, then 'ToString()' is called instead.
String mono_object_to_variant_string(MonoObject *p_obj, MonoException **r_exc);

// Array

MonoArray *Array_to_mono_array(const Array &p_array);
Array mono_array_to_Array(MonoArray *p_array);

// PackedStringArray

MonoArray *PackedStringArray_to_mono_array(const PackedStringArray &p_array);

} // namespace GDMonoMarshal

#endif // GDMONOMARSHAL_H
