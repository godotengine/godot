/**************************************************************************/
/*  tuple.h                                                               */
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

// Simple recursive Tuple type that has no runtime overhead.
//
// The compile-time recursion works as follows:
// Assume the following: Tuple<int, float> my_tuple(42, 3.14f);
// This expands to a class hierarchy that inherits from the previous step.
// So in this case this leads to:
//  - struct Tuple<int> : Tuple<float>   <--- This contains the int value.
//  - struct Tuple<float>                <--- This contains the float value.
// where each of the classes has a single field of the type for that step in the
// recursion. So: float value;  int value; etc.
//
// This works by splitting up the parameter pack for each step in the recursion minus the first.
// so the first step creates the "T value" from the first template parameter.
// any further template arguments end up in "Rest", which we then use to instantiate a new
// tuple, but now minus the first argument. To write this all out:
//
// Tuple<int, float>
// step 1: Tuple T = int, Rest = float. Results in a Tuple<int> : Tuple<float>
// step 2: Tuple T = float, no Rest. Results in a Tuple<float>

#include "core/typedefs.h"

template <typename... Types>
struct Tuple;

template <>
struct Tuple<> {};

template <typename T, typename... Rest>
struct Tuple<T, Rest...> : Tuple<Rest...> {
	T value;

	Tuple() = default;

	template <typename F, typename... R>
	_FORCE_INLINE_ Tuple(F &&f, R &&...rest) :
			Tuple<Rest...>(std::forward<R>(rest)...),
			value(std::forward<F>(f)) {}

	template <std::size_t Index>
	std::tuple_element_t<Index, Tuple> &get() {
		if constexpr (Index == 0) {
			return value;
		} else {
			return Tuple<Rest...>::template get<Index - 1>();
		}
	}

	template <std::size_t Index>
	const std::tuple_element_t<Index, Tuple> &get() const {
		if constexpr (Index == 0) {
			return value;
		} else {
			return Tuple<Rest...>::template get<Index - 1>();
		}
	}
};

namespace std {
template <typename... Args>
struct tuple_size<Tuple<Args...>> : std::integral_constant<std::size_t, sizeof...(Args)> {};

template <typename T, typename... Rest>
struct tuple_element<0, Tuple<T, Rest...>> {
	using type = T;
};

template <std::size_t Index, typename T, typename... Rest>
struct tuple_element<Index, Tuple<T, Rest...>>
		: tuple_element<Index - 1, Tuple<Rest...>> {};
} //namespace std

// Tuple is zero-constructible if and only if all constrained types are zero-constructible.
template <typename... Types>
struct is_zero_constructible<Tuple<Types...>> : std::conjunction<is_zero_constructible<Types>...> {};
