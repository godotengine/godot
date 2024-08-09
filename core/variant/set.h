/**************************************************************************/
/*  set.h                                                                 */
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

#ifndef SET_H
#define SET_H

#include "core/string/ustring.h"
#include "core/templates/list.h"
#include "core/variant/array.h"

class Variant;

struct SetPrivate;

class Set {
	mutable SetPrivate *_p;

protected:
	void _ref(const Set &p_from) const;
	void _unref() const;

public:
	struct Iterator {
		_FORCE_INLINE_ const Variant &operator*() const { return *element_ptr; }
		_FORCE_INLINE_ const Variant *operator->() const { return element_ptr; }

		Iterator &operator++();

		_FORCE_INLINE_ bool operator==(const Iterator &p_other) const { return element_ptr == p_other.element_ptr; }
		_FORCE_INLINE_ bool operator!=(const Iterator &p_other) const { return element_ptr != p_other.element_ptr; }

		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_other) :
				element_ptr(p_other.element_ptr), private_data(p_other.private_data) {}

	private:
		const Variant *element_ptr = nullptr;
		const SetPrivate *private_data = nullptr;
		friend class Set;
		_FORCE_INLINE_ Iterator(const Variant *p_element, const SetPrivate *p_private_data) :
				element_ptr(p_element), private_data(p_private_data) {}
	};

	Iterator begin() const;
	Iterator end() const;

	int size() const;
	bool is_empty() const;
	void clear();

	void merge(const Set &p_set);
	Set merged(const Set &p_set) const;

	void assign(const Set &p_set);
	void set_typed(uint32_t p_type, const StringName &p_class_name, const Variant &p_script);
	bool is_typed() const;
	bool is_same_typed(const Set &p_other) const;
	uint32_t get_typed_builtin() const;
	StringName get_typed_class_name() const;
	Variant get_typed_script() const;

	void add(const Variant &p_value) { insert(p_value); }
	void remove(const Variant &p_value) { erase(p_value); }

	bool has(const Variant &p_value) const;
	bool has_all(const Array &p_values) const;

	bool erase(const Variant &p_value);
	void insert(const Variant &p_value);

	bool operator==(const Set &p_set) const;
	bool operator!=(const Set &p_set) const;
	bool recursive_equal(const Set &p_set, int p_recursion_count) const;

	uint32_t hash() const;
	uint32_t recursive_hash(int p_recursion_count) const;
	void operator=(const Set &p_set);

	Variant get_value_at_index(int p_idx) const;
	const Variant *next(const Variant *p_value = nullptr) const;

	Array values() const;

	Set duplicate(bool p_deep = false) const;
	Set recursive_duplicate(bool p_deep, int recursion_count) const;

	void make_read_only();
	bool is_read_only() const;

	bool is_disjoint(const Set &p_set) const { return !is_overlapping(p_set); }
	bool is_overlapping(const Set &p_set) const;

	void difference(const Set &p_set);
	Set differentiated(const Set &p_set) const;
	void intersect(const Set &p_set);
	Set intersected(const Set &p_set) const;
	void symmetric_difference(const Set &p_set);
	Set symmetric_differentiated(const Set &p_set) const;

	bool includes(const Set &p_set) const;

	Set operator+(const Set &p_set) const { return merged(p_set); }
	Set operator-(const Set &p_set) const { return differentiated(p_set); }

	const void *id() const;

	Set(const Set &p_base, uint32_t p_type, const StringName &p_class_name, const Variant &p_script);
	Set(const Set &p_from);
	Set();
	~Set();
};

#endif // SET_H
