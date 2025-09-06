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

#pragma once

#include "variant.hpp"

class ArrayIterator;
class ArrayProxy;

struct Array {
	constexpr Array() {} // DON'T TOUCH
	Array(unsigned size);
	Array(const std::vector<Variant> &values);
	static Array Create(unsigned size = 0) { return Array(size); }

	Array &operator=(const std::vector<Variant> &values);
	Array &operator=(const Array &other);

	// Array operations
	void append(const Variant &value) { push_back(value); }
	void push_back(const Variant &value);
	void push_front(const Variant &value);
	void pop_at(int idx);
	void pop_back();
	void pop_front();
	void insert(int idx, const Variant &value);
	void erase(const Variant &value);
	void resize(int size);
	void clear();
	void sort();

	// Array access
	ArrayProxy operator[](int idx) const;
	Variant at(int idx) const;
	Variant at_or(int idx, const Variant &default_value) const;
	Variant front() const;
	Variant back() const;
	bool has(const Variant &value) const;

	std::vector<Variant> to_vector() const;

	// Array size
	int size() const;
	bool is_empty() const { return size() == 0; }

	METHOD(bool, all);
	METHOD(bool, any);
	VMETHOD(append_array);
	VMETHOD(assign);
	METHOD(int64_t, bsearch);
	METHOD(int64_t, bsearch_custom);
	METHOD(int64_t, count);
	METHOD(Array, duplicate);
	VMETHOD(fill);
	METHOD(Array, filter);
	METHOD(int64_t, find);
	METHOD(int64_t, hash);
	METHOD(bool, is_read_only);
	METHOD(bool, is_same_typed);
	METHOD(bool, is_typed);
	VMETHOD(make_read_only);
	METHOD(Array, map);
	METHOD(Variant, max);
	METHOD(Variant, min);
	METHOD(Variant, pick_random);
	METHOD(Variant, reduce);
	VMETHOD(remove_at);
	METHOD(int64_t, reverse);
	METHOD(int64_t, rfind);
	VMETHOD(shuffle);
	METHOD(Array, slice);
	VMETHOD(sort_custom);

	// Call methods on the Array
	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	inline auto begin();
	inline auto end();
	inline auto rbegin();
	inline auto rend();

	template <typename... Args>
	static Array make(Args... p_args) {
		return Array(std::vector<Variant>{ Variant(p_args)... });
	}

	static Array from_variant_index(unsigned idx) {
		Array a;
		a.m_idx = idx;
		return a;
	}
	unsigned get_variant_index() const noexcept { return m_idx; }
	bool is_permanent() const { return Variant::is_permanent_index(m_idx); }

private:
	unsigned m_idx = INT32_MIN;
};

inline Array Variant::as_array() const {
	if (m_type != ARRAY) {
		api_throw("std::bad_cast", "Failed to cast Variant to Array", this);
	}
	return Array::from_variant_index(v.i);
}

inline Variant::Variant(const Array &a) {
	m_type = ARRAY;
	v.i = a.get_variant_index();
}

inline Variant::operator Array() const {
	return as_array();
}

inline Variant Array::at_or(int idx, const Variant &default_value) const {
	return (idx >= 0 && idx < size()) ? at(idx) : default_value;
}

struct ArrayProxy {
	ArrayProxy(const Array &array, int idx) :
			m_array(Array::from_variant_index(array.get_variant_index())), m_idx(idx) {}

	ArrayProxy &operator=(const Variant &value);

	template <typename T>
	ArrayProxy &operator=(const T &value) {
		return operator=(Variant(value));
	}

	template <typename T>
	operator T() const { return get(); }

	operator Variant() const { return get(); }

	template <typename T>
	bool operator==(const T &value) const { return get() == Variant(value); }
	template <typename T>
	bool operator!=(const T &value) const { return get() != Variant(value); }
	template <typename T>
	bool operator<(const T &value) const { return get() < Variant(value); }

	/// @brief Get the value at the given index in the array.
	/// @return The value at the given index.
	Variant get() const;

	/// @brief Get the value at the given index in the array, or a default value if the index is out of bounds.
	/// @param default_value The default value to return if the index is out of bounds.
	Variant get_or(const Variant &default_value = {}) const { return (m_idx >= 0 && m_idx < m_array.size()) ? get() : default_value; }

	/// @brief Get the value as a specific type, by storing it in the given reference.
	/// @tparam T The type to convert the value to.
	/// @param type  The expected Variant type of the value.
	/// @param value The reference to store the value in. It will be converted to the expected type from a Variant.
	/// @return True if the value was successfully converted, false otherwise.
	template <typename T>
	bool get_as_type(Variant::Type type, T &value) const {
		if (m_idx >= 0 && m_idx < m_array.size()) {
			Variant v = get();
			if (v.get_type() == type) {
				value = v;
				return true;
			}
		}
		return false;
	}

private:
	Array m_array;
	const int m_idx;
};

inline ArrayProxy Array::operator[](int idx) const {
	return ArrayProxy(*this, idx);
}
inline Variant Array::at(int idx) const {
	return ArrayProxy(*this, idx).get();
}
inline Variant Array::front() const {
	return ArrayProxy(*this, 0).get();
}
inline Variant Array::back() const {
	return ArrayProxy(*this, size() - 1).get();
}

class ArrayIterator {
public:
	ArrayIterator(const Array &array, unsigned idx) :
			m_array(array), m_idx(idx) {}

	bool operator!=(const ArrayIterator &other) const { return m_idx != other.m_idx; }
	ArrayIterator &operator++() {
		m_idx++;
		return *this;
	}
	Variant operator*() const { return m_array[m_idx]; }

private:
	const Array m_array;
	unsigned m_idx;
};

inline auto Array::begin() {
	return ArrayIterator(*this, 0);
}
inline auto Array::end() {
	return ArrayIterator(*this, size());
}
inline auto Array::rbegin() {
	return ArrayIterator(*this, size() - 1);
}
inline auto Array::rend() {
	return ArrayIterator(*this, -1);
}

template <typename... Args>
inline Variant Array::operator()(std::string_view method, Args &&...args) {
	return Variant(*this).method_call(method, std::forward<Args>(args)...);
}
