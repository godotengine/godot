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

template <typename T>
class TypedArray : public Array {
public:
	_FORCE_INLINE_ void operator=(const Array &p_array) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign an array with a different element type.");
		_ref(p_array);
	}
	_FORCE_INLINE_ TypedArray() {
		static_assert(is_variant_type_v<T>, "Only Variant types are allowed for typed arrays.");
		set_typed(get_variant_type_v<T>, GetObjectClassName<T>::self(), Variant());
	}
	_FORCE_INLINE_ TypedArray(const Variant &p_variant) :
			TypedArray(Array(p_variant)) {
	}
	_FORCE_INLINE_ TypedArray(const Array &p_array) :
			TypedArray() {
		if (is_same_typed(p_array)) {
			_ref(p_array);
		} else {
			assign(p_array);
		}
	}
};

template <typename T>
struct VariantInternalAccessor<TypedArray<T>> {
	static _FORCE_INLINE_ TypedArray<T> get(const Variant *v) { return *VariantInternal::get_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const TypedArray<T> &p_array) { *VariantInternal::get_array(v) = p_array; }
};
template <typename T>
struct VariantInternalAccessor<const TypedArray<T> &> {
	static _FORCE_INLINE_ TypedArray<T> get(const Variant *v) { return *VariantInternal::get_array(v); }
	static _FORCE_INLINE_ void set(Variant *v, const TypedArray<T> &p_array) { *VariantInternal::get_array(v) = p_array; }
};

template <typename T>
struct PtrToArg<TypedArray<T>> {
	_FORCE_INLINE_ static TypedArray<T> convert(const void *p_ptr) {
		return TypedArray<T>(*reinterpret_cast<const Array *>(p_ptr));
	}
	typedef Array EncodeT;
	_FORCE_INLINE_ static void encode(TypedArray<T> p_val, void *p_ptr) {
		*(Array *)p_ptr = p_val;
	}
};

template <typename T>
struct PtrToArg<const TypedArray<T> &> {
	typedef Array EncodeT;
	_FORCE_INLINE_ static TypedArray<T> convert(const void *p_ptr) {
		return TypedArray<T>(*reinterpret_cast<const Array *>(p_ptr));
	}
};

template <typename T>
struct GetTypeInfo<TypedArray<T>> {
	static const Variant::Type VARIANT_TYPE = Variant::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE,
				get_variant_type_v<T> == Variant::OBJECT ? GetObjectClassName<T>::self() : Variant::get_type_name(get_variant_type_v<T>));
	}
};

template <typename T>
struct GetTypeInfo<const TypedArray<T> &> {
	static const Variant::Type VARIANT_TYPE = Variant::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE,
				get_variant_type_v<T> == Variant::OBJECT ? GetObjectClassName<T>::self() : Variant::get_type_name(get_variant_type_v<T>));
	}
};

#endif // TYPED_ARRAY_H
