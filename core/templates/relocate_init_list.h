/**************************************************************************/
/*  relocate_init_list.h                                                  */
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

#include "core/typedefs.h"

// Kind of like std::initializer_list, but with required relocation semantics:
// Whoever is passed the RelocateInitList is expected to relocate the data into itself.
template <typename T>
struct RelocateInitList {
	T *ptr;
	size_t size;
};

// If you want Container to support init list relocation, declare a relocation list constructor:
// Constructor(RelocateInitList<T> p_init) {
//     ptr = p_init.ptr;
//     size = p_init.size;
// }
// A factory method is recommended for simpler use:
// template<typename... Args> make(Args... args) {
//     RelocateInitData<T, sizeof...(Args)> data(std::forward<Args>(args)...);
//     return Container(data);
// }
// To use, simply call `Container<T>::make(a, b, c, ...)`.
template <typename T, size_t CAPACITY>
struct RelocateInitData {
	alignas(T) uint8_t _data[CAPACITY * sizeof(T)];

	_FORCE_INLINE_ T *ptr() { return reinterpret_cast<T *>(_data); }

	template <typename... Args>
	_FORCE_INLINE_ RelocateInitData(Args &&...args) {
		static_assert(sizeof...(Args) == CAPACITY);
		size_t i = -1;
		(memnew_placement(ptr() + (++i), T(std::forward<Args>(args))), ...);
	}

	_FORCE_INLINE_ operator RelocateInitList<T>() { return RelocateInitList<T>{ ptr(), CAPACITY }; }
};
