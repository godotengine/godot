/*************************************************************************/
/*  util_macros.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef UTIL_MACROS_H
#define UTIL_MACROS_H

// noreturn

#if __cpp_static_assert
#define GD_STATIC_ASSERT(m_cond) static_assert((m_cond), "Condition '" #m_cond "' failed")
#else
#define _GD_STATIC_ASSERT_VARNAME_CONCAT_B(m_ignore, m_name) m_name
#define _GD_STATIC_ASSERT_VARNAME_CONCAT_A(m_a, m_b) GD_STATIC_ASSERT_VARNAME_CONCAT_B(hello there, m_a##m_b)
#define _GD_STATIC_ASSERT_VARNAME_CONCAT(m_a, m_b) GD_STATIC_ASSERT_VARNAME_CONCAT_A(m_a, m_b)
#define GD_STATIC_ASSERT(m_cond) typedef int GD_STATIC_ASSERT_VARNAME_CONCAT(godot_static_assert_, __COUNTER__)[((m_cond) ? 1 : -1)]
#endif

#undef _NO_RETURN_

#ifdef __GNUC__
#define _NO_RETURN_ __attribute__((noreturn))
#elif _MSC_VER
#define _NO_RETURN_ __declspec(noreturn)
#else
#error Platform or compiler not supported
#endif

// unreachable

#if defined(_MSC_VER)
#define _UNREACHABLE_() __assume(0)
#elif defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__) >= 405
#define _UNREACHABLE_() __builtin_unreachable()
#else
#define _UNREACHABLE_() \
	CRASH_NOW();        \
	do {                \
	} while (true);
#endif

#endif // UTIL_MACROS_H
