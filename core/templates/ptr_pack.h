/**************************************************************************/
/*  ptr_pack.h                                                            */
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

#include <memory>

template <typename T, size_t N>
struct PtrPack {
	const T *pointers[N];

	_ALWAYS_INLINE_ const T **ptr() _LIFETIME_BOUND_ { return pointers; }
	_ALWAYS_INLINE_ const T *const *ptr() const _LIFETIME_BOUND_ { return pointers; }
};

template <typename T>
struct PtrPack<T, 0> {
	_ALWAYS_INLINE_ const T **ptr() { return nullptr; }
	_ALWAYS_INLINE_ const T *const *ptr() const { return nullptr; }
};

// Transforms addresses of arguments into array of pointers. If used with temporaries
// must be called directly inside the function call site and used immediately.
template <typename T, typename... Args>
_ALWAYS_INLINE_ static constexpr PtrPack<T, sizeof...(Args)> make_ptr_pack(const Args &...p_args _LIFETIME_BOUND_) {
	if constexpr (sizeof...(Args) > 0) {
		return PtrPack<T, sizeof...(Args)>{ { std::addressof(p_args)... } };
	} else {
		return PtrPack<T, 0>{};
	}
}
