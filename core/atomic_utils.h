/*************************************************************************/
/*  atomic_utils.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef ATOMIC_UTILS_H
#define ATOMIC_UTILS_H

#include "core/typedefs.h"

#include <atomic>

template <class T>
static _ALWAYS_INLINE_ T atomic_conditional_increment(std::atomic<T> *p_target) {

	while (true) {
		T tmp = *p_target;
		if (tmp == 0)
			return 0; // if zero, can't add to it anymore
		if (p_target->compare_exchange_strong(tmp, tmp + 1))
			return tmp + 1;
	}
}

template <class T>
static _ALWAYS_INLINE_ T atomic_exchange_if_greater(std::atomic<T> *p_target, T p_value) {

	while (true) {
		T tmp = *p_target;
		if (tmp >= p_value)
			return tmp; // already greater, or equal
		if (p_target->compare_exchange_strong(tmp, p_value))
			return p_value;
	}
}

#endif
