/**************************************************************************/
/*  test_transform_interpolator.cpp                                       */
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

#include "core/math/transform_interpolator.h"
#include "tests/test_macros.h"

TEST_FORCE_LINK(test_transform_interpolator)

namespace TestTransform_interpolator {

TEST_CASE("[Transform_interpolator] Example test case") {
	// Populate some example Transform3D objects for testing
	Transform3D transform_a(Vector3(1, 2, 3), Vector3(4, 5, 6), Vector3(7, 8, 9), Vector3(10, 11, 12));
	Transform3D transform_b(Vector3(13, 14, 15), Vector3(16, 17, 18), Vector3(19, 20, 21), Vector3(22, 23, 24));
	Transform3D transform_c(Vector3(25, 26, 27), Vector3(28, 29, 30), Vector3(31, 32, 33), Vector3(34, 35, 36));
	Transform3D transform_d;

	//check checuksum_transform_3d function
	CHECK(TransformInterpolator::checksum_transform_3d(transform_a) == 18);

	//temporary trivial checks to ensure the Transform3D class is working as expected
	CHECK(transform_a.origin == Vector3(10, 11, 12));
	CHECK(transform_a.basis[0] == Vector3(1, 4, 7));
	CHECK(transform_a.basis[1] == Vector3(2, 5, 8));
	CHECK(transform_a.basis[2] == Vector3(3, 6, 9));
	CHECK(transform_d == Transform3D());

	//test the interpolate_transform_3d function with a simple case
	TransformInterpolator::interpolate_transform_3d(transform_a, transform_b, transform_c, 0.5);
	CHECK(transform_c.origin == Vector3(16, 17, 18)); //(10,11,12) + 0.5 * ((22,23,24) - (10,11,12)) = (16,17,18)
	CHECK(transform_c.basis[0] == Vector3(7, 10, 13)); //(1,4,7) + 0.5 * ((13,16,19) - (1,4,7)) = (7,10,13)
	CHECK(transform_c.basis[1] == Vector3(8, 11, 14)); //(2,5,8) + 0.5 * ((14,17,20) - (2,5,8)) = (8,11,14)
	CHECK(transform_c.basis[2] == Vector3(9, 12, 15)); //(3,6,9) + 0.5 * ((15,18,21) - (3,6,9)) = (9,12,15)

}

} // namespace TestTransform_interpolator
