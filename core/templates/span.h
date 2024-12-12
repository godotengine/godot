/**************************************************************************/
/*  span.h                                                                */
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

#ifndef SPAN_H
#define SPAN_H

#include "core/typedefs.h"

// Equivalent of std::span.
// Represents a view into a contiguous memory space.
template <typename T>
class Span {
public:
	using pointer = T *;
	using const_pointer = T const *;
	using iterator = pointer;
	using const_iterator = const_pointer;

	Span() = default;
	Span(pointer start, size_t n) :
			data_(start), count_(n) {}

	// Convenience function to create a span from a single element.
	explicit Span(T &t) :
			data_(&t), count_(1) {}

	pointer ptrw() { return data_; }
	const_pointer ptr() const { return data_; }

	iterator begin() { return data_; }
	const_iterator begin() const { return data_; }

	iterator end() { return data_ + count_; }
	const_iterator end() const { return data_ + count_; }

	T &operator[](int i) { return data_[i]; }
	const T &operator[](int i) const { return data_[i]; }

	T &front() { return *data_; }
	const T &front() const { return *data_; }

	T &back() { return *(data_ + (count_ - 1)); }
	const T &back() const { return *(data_ + (count_ - 1)); }

	size_t size() const { return count_; }
	bool empty() const { return count_ == 0; }

	pointer data() { return data_; }
	const_pointer data() const { return data_; }

	Span subspan(std::ptrdiff_t p_from, std::ptrdiff_t p_len = -1) {
		if (p_len == -1) {
			p_len = count_ - p_from;
		}

		if (count_ == 0 || p_from < 0 || p_from >= (std::ptrdiff_t)count_ || p_len <= 0) {
			return Span();
		}

		if (p_from + p_len > (std::ptrdiff_t)count_) {
			p_len = count_ - p_from;
		}

		return Span(data_ + p_from, p_len);
	}

	// Algorithms.
	size_t find(const T &p_val, std::ptrdiff_t p_from = 0) const;
	size_t rfind(const T &p_val, std::ptrdiff_t p_from = -1) const;
	size_t count(const T &p_val) const;

private:
	pointer data_ = {};
	size_t count_ = 0;
};

// Convenience function for Span<const T>(x), i.e. to omit the type.
template <typename T>
Span<const T> const_span(const T &p_val) {
	return Span(p_val);
}

template <typename T>
size_t Span<T>::find(const T &p_val, std::ptrdiff_t p_from) const {
	if (p_from < 0 || count_ == 0) {
		return -1;
	}

	for (size_t i = p_from; i < count_; i++) {
		if (data_[i] == p_val) {
			return i;
		}
	}

	return -1;
}

template <typename T>
size_t Span<T>::rfind(const T &p_val, std::ptrdiff_t p_from) const {
	if (p_from < 0) {
		p_from = count_ + p_from;
	}
	if (p_from < 0 || p_from >= (std::ptrdiff_t)count_) {
		p_from = count_ - 1;
	}

	for (std::ptrdiff_t i = p_from; i >= 0; i--) {
		if (data_[i] == p_val) {
			return i;
		}
	}
	return -1;
}

template <typename T>
size_t Span<T>::count(const T &p_val) const {
	size_t amount = 0;
	for (size_t i = 0; i < count_; i++) {
		if (data_[i] == p_val) {
			amount++;
		}
	}
	return amount;
}

#endif // SPAN_H
