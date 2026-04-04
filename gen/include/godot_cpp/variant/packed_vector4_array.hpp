/**************************************************************************/
/*  packed_vector4_array.hpp                                              */
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

#include <godot_cpp/variant/vector4.hpp>

#include <godot_cpp/core/error_macros.hpp>
#include <initializer_list>

#include <gdextension_interface.h>

namespace godot {

class Array;
class Dictionary;
class PackedByteArray;
class Variant;
struct Vector4;

class PackedVector4Array {
	static constexpr size_t PACKED_VECTOR4_ARRAY_SIZE = 16;
	alignas(8) uint8_t opaque[PACKED_VECTOR4_ARRAY_SIZE] = {};

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
		GDExtensionPtrBuiltInMethod method_to_byte_array;
		GDExtensionPtrBuiltInMethod method_sort;
		GDExtensionPtrBuiltInMethod method_bsearch;
		GDExtensionPtrBuiltInMethod method_duplicate;
		GDExtensionPtrBuiltInMethod method_find;
		GDExtensionPtrBuiltInMethod method_rfind;
		GDExtensionPtrBuiltInMethod method_count;
		GDExtensionPtrBuiltInMethod method_erase;
		GDExtensionPtrIndexedSetter indexed_setter;
		GDExtensionPtrIndexedGetter indexed_getter;
		GDExtensionPtrOperatorEvaluator operator_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not;
		GDExtensionPtrOperatorEvaluator operator_in_Dictionary;
		GDExtensionPtrOperatorEvaluator operator_in_Array;
		GDExtensionPtrOperatorEvaluator operator_equal_PackedVector4Array;
		GDExtensionPtrOperatorEvaluator operator_not_equal_PackedVector4Array;
		GDExtensionPtrOperatorEvaluator operator_add_PackedVector4Array;
	} _method_bindings;

	static void init_bindings();
	static void _init_bindings_constructors_destructor();

	PackedVector4Array(const Variant *p_variant);

public:
	_FORCE_INLINE_ GDExtensionTypePtr _native_ptr() const { return const_cast<uint8_t(*)[PACKED_VECTOR4_ARRAY_SIZE]>(&opaque); }
	PackedVector4Array();
	PackedVector4Array(const PackedVector4Array &p_from);
	PackedVector4Array(const Array &p_from);
	PackedVector4Array(PackedVector4Array &&p_other);
	~PackedVector4Array();
	Vector4 get(int64_t p_index) const;
	void set(int64_t p_index, const Vector4 &p_value);
	int64_t size() const;
	bool is_empty() const;
	bool push_back(const Vector4 &p_value);
	bool append(const Vector4 &p_value);
	void append_array(const PackedVector4Array &p_array);
	void remove_at(int64_t p_index);
	int64_t insert(int64_t p_at_index, const Vector4 &p_value);
	void fill(const Vector4 &p_value);
	int64_t resize(int64_t p_new_size);
	void clear();
	bool has(const Vector4 &p_value) const;
	void reverse();
	PackedVector4Array slice(int64_t p_begin, int64_t p_end = 2147483647) const;
	PackedByteArray to_byte_array() const;
	void sort();
	int64_t bsearch(const Vector4 &p_value, bool p_before = true) const;
	PackedVector4Array duplicate() const;
	int64_t find(const Vector4 &p_value, int64_t p_from = 0) const;
	int64_t rfind(const Vector4 &p_value, int64_t p_from = -1) const;
	int64_t count(const Vector4 &p_value) const;
	bool erase(const Vector4 &p_value);
	bool operator==(const Variant &p_other) const;
	bool operator!=(const Variant &p_other) const;
	bool operator!() const;
	bool operator==(const PackedVector4Array &p_other) const;
	bool operator!=(const PackedVector4Array &p_other) const;
	PackedVector4Array operator+(const PackedVector4Array &p_other) const;
	PackedVector4Array &operator=(const PackedVector4Array &p_other);
	PackedVector4Array &operator=(PackedVector4Array &&p_other);
	const Vector4 &operator[](int64_t p_index) const;
	Vector4 &operator[](int64_t p_index);
	const Vector4 *ptr() const;
	Vector4 *ptrw();

	struct Iterator {
		_FORCE_INLINE_ Vector4 &operator*() const {
			return *elem_ptr;
		}
		_FORCE_INLINE_ Vector4 *operator->() const { return elem_ptr; }
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

		Iterator(Vector4 *p_ptr) { elem_ptr = p_ptr; }
		Iterator() {}
		Iterator(const Iterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		Vector4 *elem_ptr = nullptr;
	};

	struct ConstIterator {
		_FORCE_INLINE_ const Vector4 &operator*() const {
			return *elem_ptr;
		}
		_FORCE_INLINE_ const Vector4 *operator->() const { return elem_ptr; }
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

		ConstIterator(const Vector4 *p_ptr) { elem_ptr = p_ptr; }
		ConstIterator() {}
		ConstIterator(const ConstIterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		const Vector4 *elem_ptr = nullptr;
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

	_FORCE_INLINE_ PackedVector4Array(std::initializer_list<Vector4> p_init) {
		ERR_FAIL_COND(resize(p_init.size()) != 0);

		size_t i = 0;
		for (const Vector4 &element : p_init) {
			set(i++, element);
		}
	}
};

} // namespace godot
