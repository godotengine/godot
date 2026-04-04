/**************************************************************************/
/*  packed_byte_array.hpp                                                 */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/core/defs.hpp>

#include <godot_cpp/core/error_macros.hpp>
#include <initializer_list>

#include <godot_cpp/variant/string.hpp>

#include <gdextension_interface.h>

namespace godot {

class Array;
class Dictionary;
class PackedColorArray;
class PackedFloat32Array;
class PackedFloat64Array;
class PackedInt32Array;
class PackedInt64Array;
class PackedVector2Array;
class PackedVector3Array;
class PackedVector4Array;
class Variant;

class PackedByteArray {
	static constexpr size_t PACKED_BYTE_ARRAY_SIZE = 16;
	alignas(8) uint8_t opaque[PACKED_BYTE_ARRAY_SIZE] = {};

	friend class Variant;

	static struct _MethodBindings {
		GDExtensionTypeFromVariantConstructorFunc from_variant_constructor;
		GDExtensionPtrConstructor constructor_0;
		GDExtensionPtrConstructor constructor_1;
		GDExtensionPtrConstructor constructor_2;
		GDExtensionPtrDestructor destructor;
		GDExtensionPtrBuiltInMethod method_get;
		GDExtensionPtrBuiltInMethod method_set;
		GDExtensionPtrBuiltInMethod method_size;
		GDExtensionPtrBuiltInMethod method_is_empty;
		GDExtensionPtrBuiltInMethod method_push_back;
		GDExtensionPtrBuiltInMethod method_append;
		GDExtensionPtrBuiltInMethod method_append_array;
		GDExtensionPtrBuiltInMethod method_remove_at;
		GDExtensionPtrBuiltInMethod method_insert;
		GDExtensionPtrBuiltInMethod method_fill;
		GDExtensionPtrBuiltInMethod method_resize;
		GDExtensionPtrBuiltInMethod method_clear;
		GDExtensionPtrBuiltInMethod method_has;
		GDExtensionPtrBuiltInMethod method_reverse;
		GDExtensionPtrBuiltInMethod method_slice;
		GDExtensionPtrBuiltInMethod method_sort;
		GDExtensionPtrBuiltInMethod method_bsearch;
		GDExtensionPtrBuiltInMethod method_duplicate;
		GDExtensionPtrBuiltInMethod method_find;
		GDExtensionPtrBuiltInMethod method_rfind;
		GDExtensionPtrBuiltInMethod method_count;
		GDExtensionPtrBuiltInMethod method_erase;
		GDExtensionPtrBuiltInMethod method_get_string_from_ascii;
		GDExtensionPtrBuiltInMethod method_get_string_from_utf8;
		GDExtensionPtrBuiltInMethod method_get_string_from_utf16;
		GDExtensionPtrBuiltInMethod method_get_string_from_utf32;
		GDExtensionPtrBuiltInMethod method_get_string_from_wchar;
		GDExtensionPtrBuiltInMethod method_get_string_from_multibyte_char;
		GDExtensionPtrBuiltInMethod method_hex_encode;
		GDExtensionPtrBuiltInMethod method_compress;
		GDExtensionPtrBuiltInMethod method_decompress;
		GDExtensionPtrBuiltInMethod method_decompress_dynamic;
		GDExtensionPtrBuiltInMethod method_decode_u8;
		GDExtensionPtrBuiltInMethod method_decode_s8;
		GDExtensionPtrBuiltInMethod method_decode_u16;
		GDExtensionPtrBuiltInMethod method_decode_s16;
		GDExtensionPtrBuiltInMethod method_decode_u32;
		GDExtensionPtrBuiltInMethod method_decode_s32;
		GDExtensionPtrBuiltInMethod method_decode_u64;
		GDExtensionPtrBuiltInMethod method_decode_s64;
		GDExtensionPtrBuiltInMethod method_decode_half;
		GDExtensionPtrBuiltInMethod method_decode_float;
		GDExtensionPtrBuiltInMethod method_decode_double;
		GDExtensionPtrBuiltInMethod method_has_encoded_var;
		GDExtensionPtrBuiltInMethod method_decode_var;
		GDExtensionPtrBuiltInMethod method_decode_var_size;
		GDExtensionPtrBuiltInMethod method_to_int32_array;
		GDExtensionPtrBuiltInMethod method_to_int64_array;
		GDExtensionPtrBuiltInMethod method_to_float32_array;
		GDExtensionPtrBuiltInMethod method_to_float64_array;
		GDExtensionPtrBuiltInMethod method_to_vector2_array;
		GDExtensionPtrBuiltInMethod method_to_vector3_array;
		GDExtensionPtrBuiltInMethod method_to_vector4_array;
		GDExtensionPtrBuiltInMethod method_to_color_array;
		GDExtensionPtrBuiltInMethod method_bswap16;
		GDExtensionPtrBuiltInMethod method_bswap32;
		GDExtensionPtrBuiltInMethod method_bswap64;
		GDExtensionPtrBuiltInMethod method_encode_u8;
		GDExtensionPtrBuiltInMethod method_encode_s8;
		GDExtensionPtrBuiltInMethod method_encode_u16;
		GDExtensionPtrBuiltInMethod method_encode_s16;
		GDExtensionPtrBuiltInMethod method_encode_u32;
		GDExtensionPtrBuiltInMethod method_encode_s32;
		GDExtensionPtrBuiltInMethod method_encode_u64;
		GDExtensionPtrBuiltInMethod method_encode_s64;
		GDExtensionPtrBuiltInMethod method_encode_half;
		GDExtensionPtrBuiltInMethod method_encode_float;
		GDExtensionPtrBuiltInMethod method_encode_double;
		GDExtensionPtrBuiltInMethod method_encode_var;
		GDExtensionPtrIndexedSetter indexed_setter;
		GDExtensionPtrIndexedGetter indexed_getter;
		GDExtensionPtrOperatorEvaluator operator_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not;
		GDExtensionPtrOperatorEvaluator operator_in_Dictionary;
		GDExtensionPtrOperatorEvaluator operator_in_Array;
		GDExtensionPtrOperatorEvaluator operator_equal_PackedByteArray;
		GDExtensionPtrOperatorEvaluator operator_not_equal_PackedByteArray;
		GDExtensionPtrOperatorEvaluator operator_add_PackedByteArray;
	} _method_bindings;

	static void init_bindings();
	static void _init_bindings_constructors_destructor();

	PackedByteArray(const Variant *p_variant);

public:
	_FORCE_INLINE_ GDExtensionTypePtr _native_ptr() const { return const_cast<uint8_t(*)[PACKED_BYTE_ARRAY_SIZE]>(&opaque); }
	PackedByteArray();
	PackedByteArray(const PackedByteArray &p_from);
	PackedByteArray(const Array &p_from);
	PackedByteArray(PackedByteArray &&p_other);
	~PackedByteArray();
	int64_t get(int64_t p_index) const;
	void set(int64_t p_index, int64_t p_value);
	int64_t size() const;
	bool is_empty() const;
	bool push_back(int64_t p_value);
	bool append(int64_t p_value);
	void append_array(const PackedByteArray &p_array);
	void remove_at(int64_t p_index);
	int64_t insert(int64_t p_at_index, int64_t p_value);
	void fill(int64_t p_value);
	int64_t resize(int64_t p_new_size);
	void clear();
	bool has(int64_t p_value) const;
	void reverse();
	PackedByteArray slice(int64_t p_begin, int64_t p_end = 2147483647) const;
	void sort();
	int64_t bsearch(int64_t p_value, bool p_before = true) const;
	PackedByteArray duplicate() const;
	int64_t find(int64_t p_value, int64_t p_from = 0) const;
	int64_t rfind(int64_t p_value, int64_t p_from = -1) const;
	int64_t count(int64_t p_value) const;
	bool erase(int64_t p_value);
	String get_string_from_ascii() const;
	String get_string_from_utf8() const;
	String get_string_from_utf16() const;
	String get_string_from_utf32() const;
	String get_string_from_wchar() const;
	String get_string_from_multibyte_char(const String &p_encoding = String()) const;
	String hex_encode() const;
	PackedByteArray compress(int64_t p_compression_mode = 0) const;
	PackedByteArray decompress(int64_t p_buffer_size, int64_t p_compression_mode = 0) const;
	PackedByteArray decompress_dynamic(int64_t p_max_output_size, int64_t p_compression_mode = 0) const;
	int64_t decode_u8(int64_t p_byte_offset) const;
	int64_t decode_s8(int64_t p_byte_offset) const;
	int64_t decode_u16(int64_t p_byte_offset) const;
	int64_t decode_s16(int64_t p_byte_offset) const;
	int64_t decode_u32(int64_t p_byte_offset) const;
	int64_t decode_s32(int64_t p_byte_offset) const;
	int64_t decode_u64(int64_t p_byte_offset) const;
	int64_t decode_s64(int64_t p_byte_offset) const;
	double decode_half(int64_t p_byte_offset) const;
	double decode_float(int64_t p_byte_offset) const;
	double decode_double(int64_t p_byte_offset) const;
	bool has_encoded_var(int64_t p_byte_offset, bool p_allow_objects = false) const;
	Variant decode_var(int64_t p_byte_offset, bool p_allow_objects = false) const;
	int64_t decode_var_size(int64_t p_byte_offset, bool p_allow_objects = false) const;
	PackedInt32Array to_int32_array() const;
	PackedInt64Array to_int64_array() const;
	PackedFloat32Array to_float32_array() const;
	PackedFloat64Array to_float64_array() const;
	PackedVector2Array to_vector2_array() const;
	PackedVector3Array to_vector3_array() const;
	PackedVector4Array to_vector4_array() const;
	PackedColorArray to_color_array() const;
	void bswap16(int64_t p_offset = 0, int64_t p_count = -1);
	void bswap32(int64_t p_offset = 0, int64_t p_count = -1);
	void bswap64(int64_t p_offset = 0, int64_t p_count = -1);
	void encode_u8(int64_t p_byte_offset, int64_t p_value);
	void encode_s8(int64_t p_byte_offset, int64_t p_value);
	void encode_u16(int64_t p_byte_offset, int64_t p_value);
	void encode_s16(int64_t p_byte_offset, int64_t p_value);
	void encode_u32(int64_t p_byte_offset, int64_t p_value);
	void encode_s32(int64_t p_byte_offset, int64_t p_value);
	void encode_u64(int64_t p_byte_offset, int64_t p_value);
	void encode_s64(int64_t p_byte_offset, int64_t p_value);
	void encode_half(int64_t p_byte_offset, double p_value);
	void encode_float(int64_t p_byte_offset, double p_value);
	void encode_double(int64_t p_byte_offset, double p_value);
	int64_t encode_var(int64_t p_byte_offset, const Variant &p_value, bool p_allow_objects = false);
	bool operator==(const Variant &p_other) const;
	bool operator!=(const Variant &p_other) const;
	bool operator!() const;
	bool operator==(const PackedByteArray &p_other) const;
	bool operator!=(const PackedByteArray &p_other) const;
	PackedByteArray operator+(const PackedByteArray &p_other) const;
	PackedByteArray &operator=(const PackedByteArray &p_other);
	PackedByteArray &operator=(PackedByteArray &&p_other);
	const uint8_t &operator[](int64_t p_index) const;
	uint8_t &operator[](int64_t p_index);
	const uint8_t *ptr() const;
	uint8_t *ptrw();

	struct Iterator {
		_FORCE_INLINE_ uint8_t &operator*() const {
			return *elem_ptr;
		}
		_FORCE_INLINE_ uint8_t *operator->() const { return elem_ptr; }
		_FORCE_INLINE_ Iterator &operator++() {
			elem_ptr++;
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			elem_ptr--;
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return elem_ptr == b.elem_ptr; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return elem_ptr != b.elem_ptr; }

		Iterator(uint8_t *p_ptr) { elem_ptr = p_ptr; }
		Iterator() {}
		Iterator(const Iterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		uint8_t *elem_ptr = nullptr;
	};

	struct ConstIterator {
		_FORCE_INLINE_ const uint8_t &operator*() const {
			return *elem_ptr;
		}
		_FORCE_INLINE_ const uint8_t *operator->() const { return elem_ptr; }
		_FORCE_INLINE_ ConstIterator &operator++() {
			elem_ptr++;
			return *this;
		}
		_FORCE_INLINE_ ConstIterator &operator--() {
			elem_ptr--;
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return elem_ptr == b.elem_ptr; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return elem_ptr != b.elem_ptr; }

		ConstIterator(const uint8_t *p_ptr) { elem_ptr = p_ptr; }
		ConstIterator() {}
		ConstIterator(const ConstIterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		const uint8_t *elem_ptr = nullptr;
	};

	_FORCE_INLINE_ Iterator begin() {
		return Iterator(ptrw());
	}
	_FORCE_INLINE_ Iterator end() {
		return Iterator(ptrw() + size());
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return ConstIterator(ptr());
	}
	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator(ptr() + size());
	}

	_FORCE_INLINE_ PackedByteArray(std::initializer_list<uint8_t> p_init) {
		ERR_FAIL_COND(resize(p_init.size()) != 0);

		size_t i = 0;
		for (const uint8_t &element : p_init) {
			set(i++, element);
		}
	}
};

} // namespace godot
