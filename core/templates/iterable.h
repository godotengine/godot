/**************************************************************************/
/*  iterable.h                                                            */
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

#include "core/templates/tuple.h"
#include "core/typedefs.h"

// Like std::begin, but without an expensive include.
template <class T, size_t SIZE>
T *std_begin(T (&array)[SIZE]) {
	return array;
}

template <class T, size_t SIZE>
T *std_end(T (&array)[SIZE]) {
	return array + SIZE;
}

template <class T, size_t SIZE>
const T *std_begin(const T (&array)[SIZE]) {
	return array;
}

template <class T, size_t SIZE>
const T *std_end(const T (&array)[SIZE]) {
	return array + SIZE;
}

template <class T>
auto std_begin(T &t) -> decltype(t.begin()) {
	return t.begin();
}

template <class T>
auto std_end(T &t) -> decltype(t.end()) {
	return t.end();
}

template <class T>
auto std_begin(const T &t) -> decltype(t.begin()) {
	return t.begin();
}

template <class T>
auto std_end(const T &t) -> decltype(t.end()) {
	return t.end();
}

/// Can be returned from a function and directly iterated.
/// Example usage:
/// for (int i : object.get_iterable()) { ... }
template <typename I>
class Iterable {
	I _begin;
	I _end;

public:
	I begin() { return _begin; }
	I end() { return _end; }

	Iterable(I &&begin, I &&end) :
			_begin(std::move(begin)), _end(std::move(end)) {}
	Iterable(const I &begin, const I &end) :
			_begin(begin), _end(end) {}
};

template <typename T>
struct IteratorType {
	using type = decltype(std_begin(std::declval<T>()));
};

template <typename T>
using IteratorTypeT = typename IteratorType<T>::type;

template <typename T, typename... Rest>
bool _tuple_any_elements_equal(const Tuple<T, Rest...> &p_lhs, const Tuple<T, Rest...> &p_rhs) {
	if constexpr (sizeof...(Rest) == 0) {
		return p_lhs.value == p_rhs.value;
	} else {
		return p_lhs.value == p_rhs.value || _tuple_any_elements_equal(static_cast<const Tuple<Rest...> &>(p_lhs), static_cast<const Tuple<Rest...> &>(p_rhs));
	}
}

template <typename T, typename... Rest>
void _tuple_increment(Tuple<T, Rest...> &p_tuple) {
	p_tuple.value++;
	if constexpr (sizeof...(Rest) > 0) {
		_tuple_increment<Rest...>(p_tuple);
	}
}

template <typename... T, typename... T1, size_t... Is>
Tuple<T...> _tuple_dereference_impl(Tuple<T1...> &p_tuple, IndexSequence<Is...>) {
	return Tuple<T...>{ *p_tuple.template get<Is>()... };
}

template <typename TUPLE, typename... T>
struct ZipShortestIterator {
	TUPLE iterators;

	Tuple<T...> operator*() {
		return _tuple_dereference_impl<T...>(iterators, BuildIndexSequence<sizeof...(T)>{});
	}
	ZipShortestIterator &operator++() {
		_tuple_increment(iterators);
		return *this;
	}
	bool operator!=(const ZipShortestIterator &iter) const {
		return !_tuple_any_elements_equal(iterators, iter.iterators);
	}
};

/// Can be used to iterate multiple iterables together.
/// The iteration stops with the shortest iterator.
/// Example usage:
/// for (auto [ai, bi, ci] : zip_shortest<A, B, C>(a, b, c)) { ... }
template <typename... T, typename... ITER>
Iterable<ZipShortestIterator<Tuple<IteratorTypeT<ITER>...>, T...>> zip_shortest(ITER &&...t) {
	static_assert(sizeof...(T) == sizeof...(ITER));
	using TUPLE = Tuple<IteratorTypeT<ITER>...>;
	return Iterable<ZipShortestIterator<TUPLE, T...>>{
		ZipShortestIterator<TUPLE, T...>{ TUPLE{ std_begin(t)... } },
		ZipShortestIterator<TUPLE, T...>{ TUPLE{ std_end(t)... } }
	};
}

template <typename IDX, typename VALUE, typename T>
struct EnumerateIterator {
	size_t index;
	T iterator;

	Tuple<IDX, VALUE> operator*() {
		return { index, *iterator };
	}
	EnumerateIterator &operator++() {
		index++;
		iterator++;
		return *this;
	}
	bool operator!=(const EnumerateIterator &iter) {
		return iterator != iter.iterator;
	}
};

/// Can be used to count an index with an iterable.
/// Example usage:
/// for (auto [index, ai] : enumerate<size_t, A>(a)) { ... }
template <typename IDX, typename VALUE, typename ITER>
Iterable<EnumerateIterator<IDX, VALUE, IteratorTypeT<ITER>>> enumerate(ITER &&t) {
	return Iterable{
		EnumerateIterator<IDX, VALUE, IteratorTypeT<ITER>>{ 0, std_begin(t) },
		EnumerateIterator<IDX, VALUE, IteratorTypeT<ITER>>{ 0, std_end(t) }
	};
}
