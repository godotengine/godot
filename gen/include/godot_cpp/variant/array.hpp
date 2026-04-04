/**************************************************************************/
/*  array.hpp                                                             */
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

#include <godot_cpp/variant/array_helpers.hpp>

#include <gdextension_interface.h>

namespace godot {

class Callable;
class Dictionary;
class PackedByteArray;
class PackedColorArray;
class PackedFloat32Array;
class PackedFloat64Array;
class PackedInt32Array;
class PackedInt64Array;
class PackedStringArray;
class PackedVector2Array;
class PackedVector3Array;
class PackedVector4Array;
class StringName;
class Variant;

class Array {
	static constexpr size_t ARRAY_SIZE = 8;
	alignas(8) uint8_t opaque[ARRAY_SIZE] = {};

	friend class Variant;

	static struct _MethodBindings {
		GDExtensionTypeFromVariantConstructorFunc from_variant_constructor;
		GDExtensionPtrConstructor constructor_0;
		GDExtensionPtrConstructor constructor_1;
		GDExtensionPtrConstructor constructor_2;
		GDExtensionPtrConstructor constructor_3;
		GDExtensionPtrConstructor constructor_4;
		GDExtensionPtrConstructor constructor_5;
		GDExtensionPtrConstructor constructor_6;
		GDExtensionPtrConstructor constructor_7;
		GDExtensionPtrConstructor constructor_8;
		GDExtensionPtrConstructor constructor_9;
		GDExtensionPtrConstructor constructor_10;
		GDExtensionPtrConstructor constructor_11;
		GDExtensionPtrConstructor constructor_12;
		GDExtensionPtrDestructor destructor;
		GDExtensionPtrBuiltInMethod method_size;
		GDExtensionPtrBuiltInMethod method_is_empty;
		GDExtensionPtrBuiltInMethod method_clear;
		GDExtensionPtrBuiltInMethod method_hash;
		GDExtensionPtrBuiltInMethod method_assign;
		GDExtensionPtrBuiltInMethod method_get;
		GDExtensionPtrBuiltInMethod method_set;
		GDExtensionPtrBuiltInMethod method_push_back;
		GDExtensionPtrBuiltInMethod method_push_front;
		GDExtensionPtrBuiltInMethod method_append;
		GDExtensionPtrBuiltInMethod method_append_array;
		GDExtensionPtrBuiltInMethod method_resize;
		GDExtensionPtrBuiltInMethod method_insert;
		GDExtensionPtrBuiltInMethod method_remove_at;
		GDExtensionPtrBuiltInMethod method_fill;
		GDExtensionPtrBuiltInMethod method_erase;
		GDExtensionPtrBuiltInMethod method_front;
		GDExtensionPtrBuiltInMethod method_back;
		GDExtensionPtrBuiltInMethod method_pick_random;
		GDExtensionPtrBuiltInMethod method_find;
		GDExtensionPtrBuiltInMethod method_find_custom;
		GDExtensionPtrBuiltInMethod method_rfind;
		GDExtensionPtrBuiltInMethod method_rfind_custom;
		GDExtensionPtrBuiltInMethod method_count;
		GDExtensionPtrBuiltInMethod method_has;
		GDExtensionPtrBuiltInMethod method_pop_back;
		GDExtensionPtrBuiltInMethod method_pop_front;
		GDExtensionPtrBuiltInMethod method_pop_at;
		GDExtensionPtrBuiltInMethod method_sort;
		GDExtensionPtrBuiltInMethod method_sort_custom;
		GDExtensionPtrBuiltInMethod method_shuffle;
		GDExtensionPtrBuiltInMethod method_bsearch;
		GDExtensionPtrBuiltInMethod method_bsearch_custom;
		GDExtensionPtrBuiltInMethod method_reverse;
		GDExtensionPtrBuiltInMethod method_duplicate;
		GDExtensionPtrBuiltInMethod method_duplicate_deep;
		GDExtensionPtrBuiltInMethod method_slice;
		GDExtensionPtrBuiltInMethod method_filter;
		GDExtensionPtrBuiltInMethod method_map;
		GDExtensionPtrBuiltInMethod method_reduce;
		GDExtensionPtrBuiltInMethod method_any;
		GDExtensionPtrBuiltInMethod method_all;
		GDExtensionPtrBuiltInMethod method_max;
		GDExtensionPtrBuiltInMethod method_min;
		GDExtensionPtrBuiltInMethod method_is_typed;
		GDExtensionPtrBuiltInMethod method_is_same_typed;
		GDExtensionPtrBuiltInMethod method_get_typed_builtin;
		GDExtensionPtrBuiltInMethod method_get_typed_class_name;
		GDExtensionPtrBuiltInMethod method_get_typed_script;
		GDExtensionPtrBuiltInMethod method_make_read_only;
		GDExtensionPtrBuiltInMethod method_is_read_only;
		GDExtensionPtrIndexedSetter indexed_setter;
		GDExtensionPtrIndexedGetter indexed_getter;
		GDExtensionPtrOperatorEvaluator operator_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not;
		GDExtensionPtrOperatorEvaluator operator_in_Dictionary;
		GDExtensionPtrOperatorEvaluator operator_equal_Array;
		GDExtensionPtrOperatorEvaluator operator_not_equal_Array;
		GDExtensionPtrOperatorEvaluator operator_less_Array;
		GDExtensionPtrOperatorEvaluator operator_less_equal_Array;
		GDExtensionPtrOperatorEvaluator operator_greater_Array;
		GDExtensionPtrOperatorEvaluator operator_greater_equal_Array;
		GDExtensionPtrOperatorEvaluator operator_add_Array;
		GDExtensionPtrOperatorEvaluator operator_in_Array;
	} _method_bindings;

	static void init_bindings();
	static void _init_bindings_constructors_destructor();

	Array(const Variant *p_variant);

	const Variant *ptr() const;
	Variant *ptrw();

public:
	_FORCE_INLINE_ GDExtensionTypePtr _native_ptr() const { return const_cast<uint8_t(*)[ARRAY_SIZE]>(&opaque); }
	Array();
	Array(const Array &p_from);
	Array(const Array &p_base, int64_t p_type, const StringName &p_class_name, const Variant &p_script);
	Array(const PackedByteArray &p_from);
	Array(const PackedInt32Array &p_from);
	Array(const PackedInt64Array &p_from);
	Array(const PackedFloat32Array &p_from);
	Array(const PackedFloat64Array &p_from);
	Array(const PackedStringArray &p_from);
	Array(const PackedVector2Array &p_from);
	Array(const PackedVector3Array &p_from);
	Array(const PackedColorArray &p_from);
	Array(const PackedVector4Array &p_from);
	Array(Array &&p_other);
	~Array();
	int64_t size() const;
	bool is_empty() const;
	void clear();
	int64_t hash() const;
	void assign(const Array &p_array);
	Variant get(int64_t p_index) const;
	void set(int64_t p_index, const Variant &p_value);
	void push_back(const Variant &p_value);
	void push_front(const Variant &p_value);
	void append(const Variant &p_value);
	void append_array(const Array &p_array);
	int64_t resize(int64_t p_size);
	int64_t insert(int64_t p_position, const Variant &p_value);
	void remove_at(int64_t p_position);
	void fill(const Variant &p_value);
	void erase(const Variant &p_value);
	Variant front() const;
	Variant back() const;
	Variant pick_random() const;
	int64_t find(const Variant &p_what, int64_t p_from = 0) const;
	int64_t find_custom(const Callable &p_method, int64_t p_from = 0) const;
	int64_t rfind(const Variant &p_what, int64_t p_from = -1) const;
	int64_t rfind_custom(const Callable &p_method, int64_t p_from = -1) const;
	int64_t count(const Variant &p_value) const;
	bool has(const Variant &p_value) const;
	Variant pop_back();
	Variant pop_front();
	Variant pop_at(int64_t p_position);
	void sort();
	void sort_custom(const Callable &p_func);
	void shuffle();
	int64_t bsearch(const Variant &p_value, bool p_before = true) const;
	int64_t bsearch_custom(const Variant &p_value, const Callable &p_func, bool p_before = true) const;
	void reverse();
	Array duplicate(bool p_deep = false) const;
	Array duplicate_deep(int64_t p_deep_subresources_mode = 1) const;
	Array slice(int64_t p_begin, int64_t p_end = 2147483647, int64_t p_step = 1, bool p_deep = false) const;
	Array filter(const Callable &p_method) const;
	Array map(const Callable &p_method) const;
	Variant reduce(const Callable &p_method, const Variant &p_accum) const;
	bool any(const Callable &p_method) const;
	bool all(const Callable &p_method) const;
	Variant max() const;
	Variant min() const;
	bool is_typed() const;
	bool is_same_typed(const Array &p_array) const;
	int64_t get_typed_builtin() const;
	StringName get_typed_class_name() const;
	Variant get_typed_script() const;
	void make_read_only();
	bool is_read_only() const;
	bool operator==(const Variant &p_other) const;
	bool operator!=(const Variant &p_other) const;
	bool operator!() const;
	bool operator==(const Array &p_other) const;
	bool operator!=(const Array &p_other) const;
	bool operator<(const Array &p_other) const;
	bool operator<=(const Array &p_other) const;
	bool operator>(const Array &p_other) const;
	bool operator>=(const Array &p_other) const;
	Array operator+(const Array &p_other) const;
	Array &operator=(const Array &p_other);
	Array &operator=(Array &&p_other);
	template <typename... Args>
	static Array make(Args... p_args) {
		return helpers::append_all(Array(), p_args...);
	}
	const Variant &operator[](int64_t p_index) const;
	Variant &operator[](int64_t p_index);
	void set_typed(uint32_t p_type, const StringName &p_class_name, const Variant &p_script);

	struct Iterator {
		_FORCE_INLINE_ Variant &operator*() const;
		_FORCE_INLINE_ Variant *operator->() const;
		_FORCE_INLINE_ Iterator &operator++();
		_FORCE_INLINE_ Iterator &operator--();

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return elem_ptr == b.elem_ptr; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return elem_ptr != b.elem_ptr; }

		Iterator(Variant *p_ptr) { elem_ptr = p_ptr; }
		Iterator() {}
		Iterator(const Iterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		Variant *elem_ptr = nullptr;
	};

	struct ConstIterator {
		_FORCE_INLINE_ const Variant &operator*() const;
		_FORCE_INLINE_ const Variant *operator->() const;
		_FORCE_INLINE_ ConstIterator &operator++();
		_FORCE_INLINE_ ConstIterator &operator--();

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return elem_ptr == b.elem_ptr; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return elem_ptr != b.elem_ptr; }

		ConstIterator(const Variant *p_ptr) { elem_ptr = p_ptr; }
		ConstIterator() {}
		ConstIterator(const ConstIterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		const Variant *elem_ptr = nullptr;
	};

	_FORCE_INLINE_ Iterator begin();
	_FORCE_INLINE_ Iterator end();

	_FORCE_INLINE_ ConstIterator begin() const;
	_FORCE_INLINE_ ConstIterator end() const;
	
	_FORCE_INLINE_ Array(std::initializer_list<Variant> p_init);
};

} // namespace godot
