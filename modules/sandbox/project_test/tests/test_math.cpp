/**************************************************************************/
/*  test_math.cpp                                                         */
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

#include "api.hpp"

PUBLIC Variant test_math_sin(double val) {
	return Math::sin(val);
}
PUBLIC Variant test_math_cos(double val) {
	return Math::cos(val);
}
PUBLIC Variant test_math_tan(double val) {
	return Math::tan(val);
}
PUBLIC Variant test_math_asin(double val) {
	return Math::asin(val);
}
PUBLIC Variant test_math_acos(double val) {
	return Math::acos(val);
}
PUBLIC Variant test_math_atan(double val) {
	return Math::atan(val);
}
PUBLIC Variant test_math_atan2(double x, double y) {
	return Math::atan2(x, y);
}
PUBLIC Variant test_math_pow(double x, double y) {
	return Math::pow(x, y);
}

// NOTE: We can only call with 64-bit floats from GDScript
PUBLIC Variant test_math_sinf(double val) {
	return Math::sinf(val);
}

PUBLIC Variant test_math_lerp(double a, double b, double t) {
	return Math::lerp(a, b, t);
}
PUBLIC Variant test_math_smoothstep(double a, double b, double t) {
	return Math::smoothstep(a, b, t);
}
PUBLIC Variant test_math_clamp(double x, double a, double b) {
	return Math::clamp(x, a, b);
}
PUBLIC Variant test_math_slerp(double a, double b, double t) {
	return Math::slerp(a, b, t);
}
