/*************************************************************************/
/*  safe_refcount.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#if defined(_MSC_VER)

/* Implementation for MSVC-Windows */

// don't pollute my namespace!
#include <windows.h>

#define ATOMIC_CONDITIONAL_INCREMENT_BODY(m_pw, m_win_type, m_win_cmpxchg, m_cpp_type) \
	/* try to increment until it actually works */                                     \
	/* taken from boost */                                                             \
	while (true) {                                                                     \
		m_cpp_type tmp = static_cast<m_cpp_type const volatile &>(*(m_pw));            \
		if (tmp == 0) {                                                                \
			return 0; /* if zero, can't add to it anymore */                           \
		}                                                                              \
		if (m_win_cmpxchg((m_win_type volatile *)(m_pw), tmp + 1, tmp) == tmp) {       \
			return tmp + 1;                                                            \
		}                                                                              \
	}

#define ATOMIC_EXCHANGE_IF_GREATER_BODY(m_pw, m_val, m_win_type, m_win_cmpxchg, m_cpp_type) \
	while (true) {                                                                          \
		m_cpp_type tmp = static_cast<m_cpp_type const volatile &>(*(m_pw));                 \
		if (tmp >= m_val) {                                                                 \
			return tmp; /* already greater, or equal */                                     \
		}                                                                                   \
		if (m_win_cmpxchg((m_win_type volatile *)(m_pw), m_val, tmp) == tmp) {              \
			return m_val;                                                                   \
		}                                                                                   \
	}

_ALWAYS_INLINE_ uint32_t _atomic_conditional_increment_impl(volatile uint32_t *pw) {
	ATOMIC_CONDITIONAL_INCREMENT_BODY(pw, LONG, InterlockedCompareExchange, uint32_t);
}

_ALWAYS_INLINE_ uint32_t _atomic_decrement_impl(volatile uint32_t *pw) {
	return InterlockedDecrement((LONG volatile *)pw);
}

_ALWAYS_INLINE_ uint32_t _atomic_increment_impl(volatile uint32_t *pw) {
	return InterlockedIncrement((LONG volatile *)pw);
}

_ALWAYS_INLINE_ uint32_t _atomic_sub_impl(volatile uint32_t *pw, volatile uint32_t val) {
	return InterlockedExchangeAdd((LONG volatile *)pw, -(int32_t)val) - val;
}

_ALWAYS_INLINE_ uint32_t _atomic_add_impl(volatile uint32_t *pw, volatile uint32_t val) {
	return InterlockedAdd((LONG volatile *)pw, val);
}

_ALWAYS_INLINE_ uint32_t _atomic_exchange_if_greater_impl(volatile uint32_t *pw, volatile uint32_t val) {
	ATOMIC_EXCHANGE_IF_GREATER_BODY(pw, val, LONG, InterlockedCompareExchange, uint32_t);
}

_ALWAYS_INLINE_ uint64_t _atomic_conditional_increment_impl(volatile uint64_t *pw) {
	ATOMIC_CONDITIONAL_INCREMENT_BODY(pw, LONGLONG, InterlockedCompareExchange64, uint64_t);
}

_ALWAYS_INLINE_ uint64_t _atomic_decrement_impl(volatile uint64_t *pw) {
	return InterlockedDecrement64((LONGLONG volatile *)pw);
}

_ALWAYS_INLINE_ uint64_t _atomic_increment_impl(volatile uint64_t *pw) {
	return InterlockedIncrement64((LONGLONG volatile *)pw);
}

_ALWAYS_INLINE_ uint64_t _atomic_sub_impl(volatile uint64_t *pw, volatile uint64_t val) {
	return InterlockedExchangeAdd64((LONGLONG volatile *)pw, -(int64_t)val) - val;
}

_ALWAYS_INLINE_ uint64_t _atomic_add_impl(volatile uint64_t *pw, volatile uint64_t val) {
	return InterlockedAdd64((LONGLONG volatile *)pw, val);
}

_ALWAYS_INLINE_ uint64_t _atomic_exchange_if_greater_impl(volatile uint64_t *pw, volatile uint64_t val) {
	ATOMIC_EXCHANGE_IF_GREATER_BODY(pw, val, LONGLONG, InterlockedCompareExchange64, uint64_t);
}

// The actual advertised functions; they'll call the right implementation

uint32_t atomic_conditional_increment(volatile uint32_t *pw) {
	return _atomic_conditional_increment_impl(pw);
}

uint32_t atomic_decrement(volatile uint32_t *pw) {
	return _atomic_decrement_impl(pw);
}

uint32_t atomic_increment(volatile uint32_t *pw) {
	return _atomic_increment_impl(pw);
}

uint32_t atomic_sub(volatile uint32_t *pw, volatile uint32_t val) {
	return _atomic_sub_impl(pw, val);
}

uint32_t atomic_add(volatile uint32_t *pw, volatile uint32_t val) {
	return _atomic_add_impl(pw, val);
}

uint32_t atomic_exchange_if_greater(volatile uint32_t *pw, volatile uint32_t val) {
	return _atomic_exchange_if_greater_impl(pw, val);
}

uint64_t atomic_conditional_increment(volatile uint64_t *pw) {
	return _atomic_conditional_increment_impl(pw);
}

uint64_t atomic_decrement(volatile uint64_t *pw) {
	return _atomic_decrement_impl(pw);
}

uint64_t atomic_increment(volatile uint64_t *pw) {
	return _atomic_increment_impl(pw);
}

uint64_t atomic_sub(volatile uint64_t *pw, volatile uint64_t val) {
	return _atomic_sub_impl(pw, val);
}

uint64_t atomic_add(volatile uint64_t *pw, volatile uint64_t val) {
	return _atomic_add_impl(pw, val);
}

uint64_t atomic_exchange_if_greater(volatile uint64_t *pw, volatile uint64_t val) {
	return _atomic_exchange_if_greater_impl(pw, val);
}
#endif
