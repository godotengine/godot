/**************************************************************************/
/*  packed_vector2_array.hpp                                              */
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

#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/error_macros.hpp>
#include <initializer_list>

#include <gdextension_interface.h>

namespace godot {

class Array;
class Dictionary;
class PackedByteArray;
struct Transform2D;
class Variant;
struct Vector2;

class PackedVector2Array {
	static constexpr size_t PACKED_VECTOR2_ARRAY_SIZE = 16;
	alignas(8) uint8_t opaque[PACKED_VECTOR2_ARRAY_SIZE] = {};

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
		GDExtensionPtrOperatorEvaluator operator_multiply_Transform2D;
		GDExtensionPtrOperatorEvaluator operator_in_Dictionary;
		GDExtensionPtrOperatorEvaluator operator_in_Array;
		GDExtensionPtrOperatorEvaluator operator_equal_PackedVector2Array;
		GDExtensionPtrOperatorEvaluator operator_not_equal_PackedVector2Array;
		GDExtensionPtrOperatorEvaluator operator_add_PackedVector2Array;
	} _method_bindings;

	static void init_bindings();
	static void _init_bindings_constructors_destructor();

	PackedVector2Array(const Variant *p_variant);

public:
	_FORCE_INLINE_ GDExtensionTypePtr _native_ptr() const { return const_cast<uint8_t(*)[PACKED_VECTOR2_ARRAY_SIZE]>(&opaque); }
	PackedVector2Array();
	PackedVector2Array(const PackedVector2Array &p_from);
	PackedVector2Array(const Array &p_from);
	PackedVector2Array(PackedVector2Array &&p_other);
	~PackedVector2Array();
	Vector2 get(int64_t p_index) const;
	void set(int64_t p_index, const Vector2 &p_value);
	int64_t size() const;
	bool is_empty() const;
	bool push_back(const Vector2 &p_value);
	bool append(const Vector2 &p_value);
	void append_array(const PackedVector2Array &p_array);
	void remove_at(int64_t p_index);
	int64_t insert(int64_t p_at_index, const Vector2 &p_value);
	void fill(const Vector2 &p_value);
	int64_t resize(int64_t p_new_size);
	void clear();
	bool has(const Vector2 &p_value) const;
	void reverse();
	PackedVector2Array slice(int64_t p_begin, int64_t p_end = 2147483647) const;
	PackedByteArray to_byte_array() const;
	void sort();
	int64_t bsearch(const Vector2 &p_value, bool p_before = true) const;
	PackedVector2Array duplicate() const;
	int64_t find(const Vector2 &p_value, int64_t p_from = 0) const;
	int64_t rfind(const Vector2 &p_value, int64_t p_from = -1) const;
	int64_t count(const Vector2 &p_value) const;
	bool erase(const Vector2 &p_value);
	bool operator==(const Variant &p_other) const;
	bool operator!=(const Variant &p_other) const;
	bool operator!() const;
	PackedVector2Array operator*(const Transform2D &p_other) const;
	bool operator==(const PackedVector2Array &p_other) const;
	bool operator!=(const PackedVector2Array &p_other) const;
	PackedVector2Array operator+(const PackedVector2Array &p_other) const;
	PackedVector2Array &operator=(const PackedVector2Array &p_other);
	PackedVector2Array &operator=(PackedVector2Array &&p_other);
	const Vector2 &operator[](int64_t p_index) const;
	Vector2 &operator[](int64_t p_index);
	const Vector2 *ptr() const;
	Vector2 *ptrw();

	struct Iterator {
		_FORCE_INLINE_ Vector2 &operator*() const {
			return *elem_ptr;
		}
		_FORCE_INLINE_ Vector2 *operator->() const { return elem_ptr; }
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

		Iterator(Vector2 *p_ptr) { elem_ptr = p_ptr; }
		Iterator() {}
		Iterator(const Iterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		Vector2 *elem_ptr = nullptr;
	};

	struct ConstIterator {
		_FORCE_INLINE_ const Vector2 &operator*() const {
			return *elem_ptr;
		}
		_FORCE_INLINE_ const Vector2 *operator->() const { return elem_ptr; }
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

		ConstIterator(const Vector2 *p_ptr) { elem_ptr = p_ptr; }
		ConstIterator() {}
		ConstIterator(const ConstIterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		const Vector2 *elem_ptr = nullptr;
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

	_FORCE_INLINE_ PackedVector2Array(std::initializer_list<Vector2> p_init) {
		ERR_FAIL_COND(resize(p_init.size()) != 0);

		size_t i = 0;
		for (const Vector2 &element : p_init) {
			set(i++, element);
		}
	}
};

} // namespace godot
