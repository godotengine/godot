/*************************************************************************/
/*  typetraits.h                                                         */
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
#ifndef TYPETRAITS_H
#define TYPETRAITS_H

template <bool val>
struct static_constant {
public:
	static const bool value = val;
};

typedef static_constant<false> static_false;
typedef static_constant<true> static_true;

template <typename T>
struct is_integral : public static_false {};

template <>
struct is_integral<signed char> : public static_true {};
template <>
struct is_integral<unsigned char> : public static_true {};
template <>
struct is_integral<char> : public static_true {};

template <>
struct is_integral<unsigned short> : public static_true {};
template <>
struct is_integral<unsigned int> : public static_true {};
template <>
struct is_integral<unsigned long> : public static_true {};

template <>
struct is_integral<short> : public static_true {};
template <>
struct is_integral<int> : public static_true {};
template <>
struct is_integral<long> : public static_true {};

template <>
struct is_integral<bool> : public static_true {};

template <typename T>
struct is_floating_point : public static_false {};

template <>
struct is_floating_point<float> : public static_true {};
template <>
struct is_floating_point<double> : public static_true {};

template <typename T>
struct is_arithmetic : public static_constant<is_integral<T>::value || is_floating_point<T>::value> {};

template <typename T>
struct is_pointer : public static_false {};

template <typename T>
struct is_pointer<T *> : public static_true {};
template <typename T>
struct is_pointer<T *const> : public static_true {};
template <typename T>
struct is_pointer<T *const volatile> : public static_true {};
template <typename T>
struct is_pointer<T *volatile> : public static_true {};

template <typename T>
struct is_scalar : public static_constant<is_arithmetic<T>::value || is_pointer<T>::value> {};

#endif
