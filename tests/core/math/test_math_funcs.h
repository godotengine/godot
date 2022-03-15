/*************************************************************************/
/*  test_math_funcs.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_MATH_FUNCS_H
#define TEST_MATH_FUNCS_H

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "tests/test_macros.h"

namespace TestMathFuncs {
TEST_CASE("[MathFuncs] Angle methods") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Math::move_toward_angle(0.1, Math_TAU - 0.2, 0.2), -0.1),
			"MathFuncs move_toward_angle should wrap as expected.");
	CHECK_MESSAGE(
			Math::is_equal_approx(Math::move_toward_angle(Math_TAU - 0.1, 0.2, 0.2), 0.1),
			"MathFuncs move_toward_angle should wrap as expected.");
	CHECK_MESSAGE(
			Math::is_equal_approx(Math::move_toward_angle(0.1, 0.2, 0.3), 0.2),
			"MathFuncs move_toward_angle should stop as expected.");
	CHECK_MESSAGE(
			Math::is_equal_approx(Math::move_toward_angle(0.3, 0.2, 0.3), 0.2),
			"MathFuncs move_toward_angle should stop as expected.");
}
} // namespace TestMathFuncs

#endif // TEST_MATH_FUNCS_H
