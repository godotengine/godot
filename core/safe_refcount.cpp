/*************************************************************************/
/*  safe_refcount.cpp                                                    */
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
#include "safe_refcount.h"

// Atomic functions, these are used for multithread safe reference counters!

#ifdef NO_THREADS

uint32_t atomic_conditional_increment(register uint32_t *pw) {

	if (*pw == 0)
		return 0;

	(*pw)++;

	return *pw;
}

uint32_t atomic_increment(register uint32_t *pw) {

	(*pw)++;

	return *pw;
}

uint32_t atomic_decrement(register uint32_t *pw) {

	(*pw)--;

	return *pw;
}

#else

#ifdef _MSC_VER

// don't pollute my namespace!
#include <windows.h>
uint32_t atomic_conditional_increment(register uint32_t *pw) {

	/* try to increment until it actually works */
	// taken from boost

	while (true) {
		uint32_t tmp = static_cast<uint32_t const volatile &>(*pw);
		if (tmp == 0)
			return 0; // if zero, can't add to it anymore
		if (InterlockedCompareExchange((LONG volatile *)pw, tmp + 1, tmp) == tmp)
			return tmp + 1;
	}
}

uint32_t atomic_decrement(register uint32_t *pw) {
	return InterlockedDecrement((LONG volatile *)pw);
}

uint32_t atomic_increment(register uint32_t *pw) {
	return InterlockedIncrement((LONG volatile *)pw);
}
#elif defined(__GNUC__)

uint32_t atomic_conditional_increment(register uint32_t *pw) {

	while (true) {
		uint32_t tmp = static_cast<uint32_t const volatile &>(*pw);
		if (tmp == 0)
			return 0; // if zero, can't add to it anymore
		if (__sync_val_compare_and_swap(pw, tmp, tmp + 1) == tmp)
			return tmp + 1;
	}
}

uint32_t atomic_decrement(register uint32_t *pw) {

	return __sync_sub_and_fetch(pw, 1);
}

uint32_t atomic_increment(register uint32_t *pw) {

	return __sync_add_and_fetch(pw, 1);
}

#else
//no threads supported?
#error Must provide atomic functions for this platform or compiler!

#endif

#endif
