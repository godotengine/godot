/**************************************************************************/
/*  struct.h                                                              */
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

#ifndef STRUCT_H
#define STRUCT_H

#include "core/object/object.h"
#include "core/variant/array.h"
#include "core/variant/binder_common.h"
#include "core/variant/method_ptrcall.h"
#include "core/variant/type_info.h"
#include "core/variant/variant.h"

class Array;

template <class T>
class Struct : public Array {
public:
	_FORCE_INLINE_ void operator=(const Array &p_array) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign a Struct from array with a different format.");
		_ref(p_array);
	}
	_FORCE_INLINE_ Variant &operator[](const StringName &p_struct_member) {
		return get_named(p_struct_member);
	}
	_FORCE_INLINE_ const Variant &operator[](const StringName &p_struct_member) const {
		return get_named(p_struct_member);
	}
	template <typename StructMember>
	_FORCE_INLINE_ int get_member_index() const {
		return T::Layout::template get_member_index<StructMember>();
	}
	template <typename StructMember>
	_FORCE_INLINE_ Variant get_member_value() const {
		return get(get_member_index<StructMember>());
	}
	template <typename StructMember>
	_FORCE_INLINE_ void set_member_value(const typename StructMember::Type &p_value) {
		set(get_member_index<StructMember>(), p_value);
	}
	template <typename StructMember>
	_FORCE_INLINE_ typename StructMember::Type get_member() const {
		return StructMember::from_variant(get(get_member_index<StructMember>()));
	}
	template <typename StructMember>
	_FORCE_INLINE_ void set_member(const typename StructMember::Type &p_struct_member) {
		set(get_member_index<StructMember>(), StructMember::to_variant(p_struct_member));
	}
	_FORCE_INLINE_ Struct(const T &p_struct) :
			Array(T::Layout::to_array(p_struct), T::get_struct_info()) {
	}
	_FORCE_INLINE_ Struct(const Variant &p_variant) :
			Array(Array(p_variant), T::get_struct_info()) {
	}
	_FORCE_INLINE_ Struct(const Array &p_array) :
			Array(p_array, T::get_struct_info()) {
	}
	_FORCE_INLINE_ Struct() :
			Array(T::get_struct_info()) {
	}
	_FORCE_INLINE_ operator Dictionary() {
		Dictionary dict;
		for (int i = 0; i < size(); i++) {
			dict[get_member_name(i)] = get(i);
		}
		return dict;
	}
};

template <class T>
struct VariantInternalAccessor<Struct<T>> {
	_FORCE_INLINE_ static Struct<T> get(const Variant *v) { return *VariantInternal::get_array(v); }
	_FORCE_INLINE_ static void set(Variant *v, const Struct<T> &p_array) { *VariantInternal::get_array(v) = p_array; }
};

template <class T>
struct VariantInternalAccessor<const Struct<T> &> {
	_FORCE_INLINE_ static Struct<T> get(const Variant *v) { return *VariantInternal::get_array(v); }
	_FORCE_INLINE_ static void set(Variant *v, const Struct<T> &p_array) { *VariantInternal::get_array(v) = p_array; }
};

template <class T>
struct PtrToArg<Struct<T>> {
	_FORCE_INLINE_ static Struct<T> convert(const void *p_ptr) {
		return Struct<T>(*reinterpret_cast<const Array *>(p_ptr));
	}
	typedef Array EncodeT;
	_FORCE_INLINE_ static void encode(Struct<T> p_val, void *p_ptr) {
		*(Array *)p_ptr = p_val;
	}
};

template <class T>
struct PtrToArg<const Struct<T> &> {
	typedef Array EncodeT;
	_FORCE_INLINE_ static Struct<T> convert(const void *p_ptr) {
		return Struct<T>(*reinterpret_cast<const Array *>(p_ptr));
	}
};

template <class T>
struct GetTypeInfo<Struct<T>> {
	static const Variant::Type VARIANT_TYPE = Variant::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static PropertyInfo get_class_info() {
		return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, T::get_struct_name());
	}
};

template <class T>
struct GetTypeInfo<const Struct<T> &> {
	static const Variant::Type VARIANT_TYPE = Variant::ARRAY;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;
	static PropertyInfo get_class_info() {
		return PropertyInfo(Variant::ARRAY, String(), PROPERTY_HINT_ARRAY_TYPE, T::get_struct_name());
	}
};

#endif // STRUCT_H
