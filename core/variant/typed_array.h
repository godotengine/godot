/**************************************************************************/
/*  typed_array.h                                                         */
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

#ifndef TYPED_ARRAY_H
#define TYPED_ARRAY_H

#include "core/object/object.h"
#include "core/variant/array.h"
#include "core/variant/binder_common.h"
#include "core/variant/method_ptrcall.h"
#include "core/variant/type_info.h"
#include "core/variant/variant.h"

template <class T>
class TypedArray : public Array {
public:
	typedef T Type;
	_FORCE_INLINE_ void operator=(const Array &p_array) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign an array with a different element type.");
		_ref(p_array);
	}
	_FORCE_INLINE_ TypedArray(const Variant &p_variant) :
			Array(Array(p_variant), GetTypeInfo<T>::VARIANT_TYPE, GetTypeInfo<T>::get_class_info().class_name, Variant()) {
	}
	_FORCE_INLINE_ TypedArray(const Array &p_array) :
			Array(p_array, GetTypeInfo<T>::VARIANT_TYPE, GetTypeInfo<T>::get_class_info().class_name, Variant()) {
	}
	_FORCE_INLINE_ TypedArray() {
		set_typed(GetTypeInfo<T>::VARIANT_TYPE, GetTypeInfo<T>::get_class_info().class_name, Variant());
	}
};

template <class T>
struct VariantInternalAccessor<T, EnableIf_t<types_are_same_v<TypedArray<typename RemoveRefPointerConst_t<T>::Type>, RemoveRefPointerConst_t<T>>>> {
	typedef RemoveRefPointerConst_t<T> TStripped;
	static _FORCE_INLINE_ TStripped get(const Variant *v) { return *VariantInternal::get_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const TStripped &p_array) { *VariantInternal::get_array(v) = p_array; }
};

template <class T>
struct PtrToArg<T, EnableIf_t<types_are_same_v<TypedArray<typename RemoveRefPointerConst_t<T>::Type>, RemoveRefPointerConst_t<T>>>> {
	typedef RemoveRefPointerConst_t<T> TStripped;
	_FORCE_INLINE_ static TStripped convert(const void *p_ptr) {
		return TStripped(*reinterpret_cast<const Array *>(p_ptr));
	}
	typedef Array EncodeT;
	_FORCE_INLINE_ static void encode(TStripped p_val, void *p_ptr) {
		*(Array *)p_ptr = p_val;
	}
};

template <class T>
struct GetTypeInfo<T, EnableIf_t<types_are_same_v<TypedArray<typename RemoveRefPointerConst_t<T>::Type>, RemoveRefPointerConst_t<T>>>> {
	static const Variant::Type VARIANT_TYPE = Variant::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, GetTypeInfo<typename RemoveRefPointerConst_t<T>::Type>::get_type_name());
	}
};

#endif // TYPED_ARRAY_H
