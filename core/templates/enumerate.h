/**************************************************************************/
/*  enumerate.h                                                           */
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

namespace Internal {
template <typename T> // Like std::begin, but without requiring any additional includes.
[[nodiscard]] constexpr auto std_begin(T &p_container) -> decltype(p_container.begin()) {
	return p_container.begin();
}
template <typename T> // Like std::begin, but without requiring any additional includes.
[[nodiscard]] constexpr auto std_begin(const T &p_container) -> decltype(p_container.begin()) {
	return p_container.begin();
}
template <typename T, size_t SIZE> // Like std::begin, but without requiring any additional includes.
[[nodiscard]] constexpr T *std_begin(T (&p_carray)[SIZE]) {
	return p_carray;
}
template <typename T> // Like std::end, but without requiring any additional includes.
[[nodiscard]] constexpr auto std_end(T &p_container) -> decltype(p_container.end()) {
	return p_container.end();
}
template <typename T> // Like std::end, but without requiring any additional includes.
[[nodiscard]] constexpr auto std_end(const T &p_container) -> decltype(p_container.end()) {
	return p_container.end();
}
template <typename T, size_t SIZE> // Like std::end, but without requiring any additional includes.
[[nodiscard]] constexpr T *std_end(T (&p_carray)[SIZE]) {
	return p_carray + SIZE;
}
} // namespace Internal

/**
 * An enumeration wrapper to supply range loops with an index.
 */
template <typename T, typename I>
class Enumerate {
	T &_iterable;
	I &_index;

public:
	using iterator_t = decltype(Internal::std_begin(_iterable));
	using element_t = decltype(*Internal::std_begin(_iterable));

	struct Iterator {
		[[nodiscard]] constexpr element_t &operator*() const { return *_iterator; }
		[[nodiscard]] constexpr element_t operator->() const { return _iterator; }
		constexpr Iterator &operator++() {
			++_iterator;
			++_index;
			return *this;
		}
		constexpr Iterator &operator--() {
			--_iterator;
			--_index;
			return *this;
		}

		[[nodiscard]] constexpr bool operator==(const Iterator &p_other) const { return _iterator == p_other._iterator; }
		[[nodiscard]] constexpr bool operator!=(const Iterator &p_other) const { return _iterator != p_other._iterator; }

		constexpr Iterator(iterator_t p_iterator, I &p_index) :
				_iterator(p_iterator), _index(p_index) {}

	private:
		iterator_t _iterator;
		I &_index;
	};

	struct ConstIterator {
		[[nodiscard]] constexpr const element_t &operator*() const { return *_iterator; }
		[[nodiscard]] constexpr const element_t operator->() const { return _iterator; }
		constexpr ConstIterator &operator++() {
			++_iterator;
			++_index;
			return *this;
		}
		constexpr ConstIterator &operator--() {
			--_iterator;
			--_index;
			return *this;
		}
		[[nodiscard]] constexpr bool operator==(const ConstIterator &p_other) const { return _iterator == p_other._iterator; }
		[[nodiscard]] constexpr bool operator!=(const ConstIterator &p_other) const { return _iterator != p_other._iterator; }

		constexpr ConstIterator(const iterator_t p_iterator, I &p_index) :
				_iterator(p_iterator), _index(p_index) {}

	private:
		const iterator_t _iterator;
		I &_index;
	};

	[[nodiscard]] constexpr Iterator begin() { return Iterator(Internal::std_begin(_iterable), _index); }
	[[nodiscard]] constexpr Iterator end() { return Iterator(Internal::std_end(_iterable), _index); }

	[[nodiscard]] constexpr ConstIterator begin() const { return ConstIterator(Internal::std_begin(_iterable), _index); }
	[[nodiscard]] constexpr ConstIterator end() const { return ConstIterator(Internal::std_end(_iterable), _index); }

	constexpr Enumerate(T &p_iterable, I &p_index) :
			_iterable(p_iterable), _index(p_index) {}
};
