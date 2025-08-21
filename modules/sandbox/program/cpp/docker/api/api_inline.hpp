/**************************************************************************/
/*  api_inline.hpp                                                        */
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

#include "syscalls.h"

/// Math and interpolation operations.
// clang-format off
template <typename Float>
static inline Float perform_math_op(Math_Op math_op, Float x) {
	register Float fa0 asm("fa0") = x;
	register int   iop asm("a0") = static_cast<int>(math_op);
	register long snum asm("a7") = (sizeof(Float) == 4) ? ECALL_MATH_OP32 : ECALL_MATH_OP64;

	asm volatile ("ecall"
		: "+f"(fa0) : "r"(iop), "r"(snum));
	return Float(fa0);
}

template <typename Float>
static inline Float perform_math_op2(Math_Op math_op, Float x, Float y) {
	register Float fa0 asm("fa0") = x;
	register Float fa1 asm("fa1") = y;
	register int   iop asm("a0") = static_cast<int>(math_op);
	register long snum asm("a7") = (sizeof(Float) == 4) ? ECALL_MATH_OP32 : ECALL_MATH_OP64;

	asm volatile ("ecall"
		: "+f"(fa0) : "f"(fa1), "r"(iop), "r"(snum));
	return Float(fa0);
}

template <typename Float>
static inline Float perform_lerp_op(Lerp_Op lerp_op, Float x, Float y, Float t) {
	register Float fa0 asm("fa0") = x;
	register Float fa1 asm("fa1") = y;
	register Float fa2 asm("fa2") = t;
	register int   iop asm("a0") = static_cast<int>(lerp_op);
	register long snum asm("a7") = (sizeof(Float) == 4) ? ECALL_LERP_OP32 : ECALL_LERP_OP64;

	asm volatile ("ecall"
		: "+f"(fa0) : "f"(fa1), "f"(fa2), "r"(iop), "r"(snum));
	return Float(fa0);
}
// clang-format on

// 64-bit FP math operations.

inline double Math::sin(double x) {
	return perform_math_op<double>(Math_Op::SIN, x);
}

inline double Math::cos(double x) {
	return perform_math_op<double>(Math_Op::COS, x);
}

inline double Math::tan(double x) {
	return perform_math_op<double>(Math_Op::TAN, x);
}

inline double Math::asin(double x) {
	return perform_math_op<double>(Math_Op::ASIN, x);
}

inline double Math::acos(double x) {
	return perform_math_op<double>(Math_Op::ACOS, x);
}

inline double Math::atan(double x) {
	return perform_math_op<double>(Math_Op::ATAN, x);
}

inline double Math::atan2(double y, double x) {
	return perform_math_op2<double>(Math_Op::ATAN2, y, x);
}

inline double Math::pow(double x, double y) {
	return perform_math_op2<double>(Math_Op::POW, x, y);
}

// 32-bit FP math operations.

inline float Math::sinf(float x) {
	return perform_math_op<float>(Math_Op::SIN, x);
}

inline float Math::cosf(float x) {
	return perform_math_op<float>(Math_Op::COS, x);
}

inline float Math::tanf(float x) {
	return perform_math_op<float>(Math_Op::TAN, x);
}

inline float Math::asinf(float x) {
	return perform_math_op<float>(Math_Op::ASIN, x);
}

inline float Math::acosf(float x) {
	return perform_math_op<float>(Math_Op::ACOS, x);
}

inline float Math::atanf(float x) {
	return perform_math_op<float>(Math_Op::ATAN, x);
}

inline float Math::atan2f(float y, float x) {
	return perform_math_op2<float>(Math_Op::ATAN2, y, x);
}

inline float Math::powf(float x, float y) {
	return perform_math_op2<float>(Math_Op::POW, x, y);
}

// 64-bit FP interpolation operations.

inline double Math::lerp(double x, double y, double t) {
	return perform_lerp_op<double>(Lerp_Op::LERP, x, y, t);
}

inline double Math::smoothstep(double from, double to, double t) {
	return perform_lerp_op<double>(Lerp_Op::SMOOTHSTEP, from, to, t);
}

inline double Math::clamp(double x, double a, double b) {
	return perform_lerp_op<double>(Lerp_Op::CLAMP, x, a, b);
}

inline double Math::slerp(double a, double b, double t) {
	return perform_lerp_op<double>(Lerp_Op::SLERP, a, b, t);
}

// 32-bit FP interpolation operations.

inline float Math::lerpf(float x, float y, float t) {
	return perform_lerp_op<float>(Lerp_Op::LERP, x, y, t);
}

inline float Math::smoothstepf(float from, float to, float t) {
	return perform_lerp_op<float>(Lerp_Op::SMOOTHSTEP, from, to, t);
}

inline float Math::clampf(float x, float a, float b) {
	return perform_lerp_op<float>(Lerp_Op::CLAMP, x, a, b);
}

inline float Math::slerpf(float a, float b, float t) {
	return perform_lerp_op<float>(Lerp_Op::SLERP, a, b, t);
}
