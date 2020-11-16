/*************************************************************************/
/*  test_math.h                                                          */
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

#ifndef TEST_MATH_H
#define TEST_MATH_H

#include "core/math/math_funcs.h"
#include "core/math/transform.h"
#include "core/os/os.h"

#include <math.h>
#include <stdio.h>

#include "tests/test_macros.h"

namespace TestMath {

MainLoop *test();
}

const float FM_PI = 3.1415926535897932384626433832795028841971693993751f;

TEST_CASE("[Transform] Rotate around global origin") {
	// Start with the default orientation, but not centered on the origin.
	// Rotating should rotate both our basis and the origin.
	Transform transform = Transform();
	transform.origin = Vector3(0, 0, 1);

	Transform expected = Transform();
	expected.origin = Vector3(0, 0, -1);
	expected.basis.set_axis(0, Vector3(-1, 0, 0));
	expected.basis.set_axis(2, Vector3(0, 0, -1));

	Transform rotatedTransform = transform.rotated(Vector3(0, 1, 0), FM_PI);
	REQUIRE(rotatedTransform.is_equal_approx(expected));
}

TEST_CASE("[Transform] Rotate in-place") {
	// Start with the default orientation, but not centered on the origin.
	// Rotating in-place should only rotate the basis, leaving the origin alone.
	Transform transform = Transform();
	transform.origin = Vector3(0, 0, 1);

	Transform expected = Transform();
	expected.origin = Vector3(0, 0, 1);
	expected.basis.set_axis(0, Vector3(-1, 0, 0));
	expected.basis.set_axis(2, Vector3(0, 0, -1));

	Transform rotated_transform = transform.rotated_local(Vector3(0, 1, 0), FM_PI);
	REQUIRE(rotated_transform.is_equal_approx(expected));

	// Make sure that the rotatedTransform isn't sharing references with the original transform.
	REQUIRE(&rotated_transform.basis != &transform.basis);
	REQUIRE(&rotated_transform.origin != &transform.origin);
}

#endif
