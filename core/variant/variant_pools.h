/**************************************************************************/
/*  variant_pools.h                                                       */
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

#include "core/math/math_defs.h"
#include "core/os/memory.h"

namespace VariantPools {
inline constexpr size_t BUCKET_SMALL = 2 * 3 * sizeof(real_t);
inline constexpr size_t BUCKET_MEDIUM = 4 * 3 * sizeof(real_t);
inline constexpr size_t BUCKET_LARGE = 4 * 4 * sizeof(real_t);

void *alloc_small();
void *alloc_medium();
void *alloc_large();

template <typename T>
_FORCE_INLINE_ T *alloc() {
	if constexpr (sizeof(T) <= BUCKET_SMALL && alignof(real_t) % alignof(T) == 0) {
		return static_cast<T *>(alloc_small());
	} else if constexpr (sizeof(T) <= BUCKET_MEDIUM && alignof(real_t) % alignof(T) == 0) {
		return static_cast<T *>(alloc_medium());
	} else if constexpr (sizeof(T) <= BUCKET_LARGE && alignof(real_t) % alignof(T) == 0) {
		return static_cast<T *>(alloc_large());
	} else {
		return memnew(T);
	}
}

void free_small(void *p_ptr);
void free_medium(void *p_ptr);
void free_large(void *p_ptr);

template <typename T>
_FORCE_INLINE_ void free(T *p_ptr) {
	if constexpr (sizeof(T) <= BUCKET_SMALL && alignof(real_t) % alignof(T) == 0) {
		free_small(p_ptr);
	} else if constexpr (sizeof(T) <= BUCKET_MEDIUM && alignof(real_t) % alignof(T) == 0) {
		free_medium(p_ptr);
	} else if constexpr (sizeof(T) <= BUCKET_LARGE && alignof(real_t) % alignof(T) == 0) {
		free_large(p_ptr);
	} else {
		memdelete(p_ptr);
	}
}
}; //namespace VariantPools
