/*************************************************************************/
/*  baked_light_baker_cmpxchg.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "typedefs.h"

#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100

void baked_light_baker_add_64f(double *dst, double value) {

	union {
		int64_t i;
		double f;
	} swapy;

	while (true) {
		swapy.f = *dst;
		int64_t from = swapy.i;
		swapy.f += value;
		int64_t to = swapy.i;
		if (__sync_bool_compare_and_swap((int64_t *)dst, from, to))
			break;
	}
}

void baked_light_baker_add_64i(int64_t *dst, int64_t value) {

	while (!__sync_bool_compare_and_swap(dst, *dst, (*dst) + value)) {
	}
}

#elif defined(WINDOWS_ENABLED)

#include "windows.h"

void baked_light_baker_add_64f(double *dst, double value) {

	union {
		int64_t i;
		double f;
	} swapy;

	while (true) {
		swapy.f = *dst;
		int64_t from = swapy.i;
		swapy.f += value;
		int64_t to = swapy.i;
		int64_t result = InterlockedCompareExchange64((int64_t *)dst, to, from);
		if (result == from)
			break;
	}
}

void baked_light_baker_add_64i(int64_t *dst, int64_t value) {

	while (true) {
		int64_t from = *dst;
		int64_t to = from + value;
		int64_t result = InterlockedCompareExchange64(dst, to, from);
		if (result == from)
			break;
	}
}

#else

//in goder (the god of programmers) we trust
#warning seems this platform or compiler does not support safe cmpxchg, your baked lighting may be funny

void baked_light_baker_add_64f(double *dst, double value) {

	*dst += value;
}

void baked_light_baker_add_64i(int64_t *dst, int64_t value) {

	*dst += value;
}

#endif
