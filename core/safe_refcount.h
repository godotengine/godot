/*************************************************************************/
/*  safe_refcount.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef SAFE_REFCOUNT_H
#define SAFE_REFCOUNT_H

#include "os/mutex.h"
/* x86/x86_64 GCC */

#include "platform_config.h"
#include "typedefs.h"

// Atomic functions, these are used for multithread safe reference counters!

#ifdef NO_THREADS

/* Bogus implementation unaware of multiprocessing */

template <class T>
static _ALWAYS_INLINE_ T atomic_conditional_increment(register T *pw) {

	if (*pw == 0)
		return 0;

	(*pw)++;

	return *pw;
}

template <class T>
static _ALWAYS_INLINE_ T atomic_decrement(register T *pw) {

	(*pw)--;

	return *pw;
}

template <class T>
static _ALWAYS_INLINE_ T atomic_increment(register T *pw) {

	(*pw)++;

	return *pw;
}

template <class T, class V>
static _ALWAYS_INLINE_ T atomic_sub(register T *pw, register V val) {

	(*pw) -= val;

	return *pw;
}

template <class T, class V>
static _ALWAYS_INLINE_ T atomic_add(register T *pw, register V val) {

	(*pw) += val;

	return *pw;
}

template <class T, class V>
static _ALWAYS_INLINE_ T atomic_exchange_if_greater(register T *pw, register V val) {

	if (val > *pw)
		*pw = val;

	return *pw;
}

#elif defined(__GNUC__)

/* Implementation for GCC & Clang */

// GCC guarantees atomic intrinsics for sizes of 1, 2, 4 and 8 bytes.
// Clang states it supports GCC atomic builtins.

template <class T>
static _ALWAYS_INLINE_ T atomic_conditional_increment(register T *pw) {

	while (true) {
		T tmp = static_cast<T const volatile &>(*pw);
		if (tmp == 0)
			return 0; // if zero, can't add to it anymore
		if (__sync_val_compare_and_swap(pw, tmp, tmp + 1) == tmp)
			return tmp + 1;
	}
}

template <class T>
static _ALWAYS_INLINE_ T atomic_decrement(register T *pw) {

	return __sync_sub_and_fetch(pw, 1);
}

template <class T>
static _ALWAYS_INLINE_ T atomic_increment(register T *pw) {

	return __sync_add_and_fetch(pw, 1);
}

template <class T, class V>
static _ALWAYS_INLINE_ T atomic_sub(register T *pw, register V val) {

	return __sync_sub_and_fetch(pw, val);
}

template <class T, class V>
static _ALWAYS_INLINE_ T atomic_add(register T *pw, register V val) {

	return __sync_add_and_fetch(pw, val);
}

template <class T, class V>
static _ALWAYS_INLINE_ T atomic_exchange_if_greater(register T *pw, register V val) {

	while (true) {
		T tmp = static_cast<T const volatile &>(*pw);
		if (tmp >= val)
			return tmp; // already greater, or equal
		if (__sync_val_compare_and_swap(pw, tmp, val) == tmp)
			return val;
	}
}

#elif defined(_MSC_VER)

/* Implementation for MSVC-Windows */

// don't pollute my namespace!
#include <windows.h>

#define ATOMIC_CONDITIONAL_INCREMENT_BODY(m_pw, m_win_type, m_win_cmpxchg, m_cpp_type) \
	/* try to increment until it actually works */                                     \
	/* taken from boost */                                                             \
	while (true) {                                                                     \
		m_cpp_type tmp = static_cast<m_cpp_type const volatile &>(*(m_pw));            \
		if (tmp == 0)                                                                  \
			return 0; /* if zero, can't add to it anymore */                           \
		if (m_win_cmpxchg((m_win_type volatile *)(m_pw), tmp + 1, tmp) == tmp)         \
			return tmp + 1;                                                            \
	}

#define ATOMIC_EXCHANGE_IF_GREATER_BODY(m_pw, m_val, m_win_type, m_win_cmpxchg, m_cpp_type) \
	while (true) {                                                                          \
		m_cpp_type tmp = static_cast<m_cpp_type const volatile &>(*(m_pw));                 \
		if (tmp >= m_val)                                                                   \
			return tmp; /* already greater, or equal */                                     \
		if (m_win_cmpxchg((m_win_type volatile *)(m_pw), m_val, tmp) == tmp)                \
			return m_val;                                                                   \
	}

static _ALWAYS_INLINE_ uint32_t atomic_conditional_increment(register uint32_t *pw) {

	ATOMIC_CONDITIONAL_INCREMENT_BODY(pw, LONG, InterlockedCompareExchange, uint32_t)
}

static _ALWAYS_INLINE_ uint32_t atomic_decrement(register uint32_t *pw) {

	return InterlockedDecrement((LONG volatile *)pw);
}

static _ALWAYS_INLINE_ uint32_t atomic_increment(register uint32_t *pw) {

	return InterlockedIncrement((LONG volatile *)pw);
}

static _ALWAYS_INLINE_ uint32_t atomic_sub(register uint32_t *pw, register uint32_t val) {

	return InterlockedExchangeAdd((LONG volatile *)pw, -(int32_t)val) - val;
}

static _ALWAYS_INLINE_ uint32_t atomic_add(register uint32_t *pw, register uint32_t val) {

	return InterlockedAdd((LONG volatile *)pw, val);
}

static _ALWAYS_INLINE_ uint32_t atomic_exchange_if_greater(register uint32_t *pw, register uint32_t val) {

	ATOMIC_EXCHANGE_IF_GREATER_BODY(pw, val, LONG, InterlockedCompareExchange, uint32_t)
}

static _ALWAYS_INLINE_ uint64_t atomic_conditional_increment(register uint64_t *pw) {

	ATOMIC_CONDITIONAL_INCREMENT_BODY(pw, LONGLONG, InterlockedCompareExchange64, uint64_t)
}

static _ALWAYS_INLINE_ uint64_t atomic_decrement(register uint64_t *pw) {

	return InterlockedDecrement64((LONGLONG volatile *)pw);
}

static _ALWAYS_INLINE_ uint64_t atomic_increment(register uint64_t *pw) {

	return InterlockedIncrement64((LONGLONG volatile *)pw);
}

static _ALWAYS_INLINE_ uint64_t atomic_sub(register uint64_t *pw, register uint64_t val) {

	return InterlockedExchangeAdd64((LONGLONG volatile *)pw, -(int64_t)val) - val;
}

static _ALWAYS_INLINE_ uint64_t atomic_add(register uint64_t *pw, register uint64_t val) {

	return InterlockedAdd64((LONGLONG volatile *)pw, val);
}

static _ALWAYS_INLINE_ uint64_t atomic_exchange_if_greater(register uint64_t *pw, register uint64_t val) {

	ATOMIC_EXCHANGE_IF_GREATER_BODY(pw, val, LONGLONG, InterlockedCompareExchange64, uint64_t)
}

#else

//no threads supported?
#error Must provide atomic functions for this platform or compiler!

#endif

struct SafeRefCount {

	uint32_t count;

public:
	// destroy() is called when weak_count_ drops to zero.

	_ALWAYS_INLINE_ bool ref() { //true on success

		return atomic_conditional_increment(&count) != 0;
	}

	_ALWAYS_INLINE_ uint32_t refval() { //true on success

		return atomic_conditional_increment(&count);
	}

	_ALWAYS_INLINE_ bool unref() { // true if must be disposed of

		if (atomic_decrement(&count) == 0) {
			return true;
		}

		return false;
	}

	_ALWAYS_INLINE_ uint32_t get() const { // nothrow

		return count;
	}

	_ALWAYS_INLINE_ void init(uint32_t p_value = 1) {

		count = p_value;
	}
};

#endif
