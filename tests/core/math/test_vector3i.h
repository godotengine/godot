/*************************************************************************/
/*  test_vector3i.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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


#ifndef GODOT_TEST_VECTOR3I_H
#define GODOT_TEST_VECTOR3I_H

#include "core/math/Vector3i.h"

#include "tests/test_macros.h"

namespace TestVector3i {

TEST_CASE("[Vector3i] Constructor methods") {
	const Vector3i vector3i = Vector3i(1, 2, 3);
	const Vector3i vector3i_copy = Vector3i(vector3i);
	const Vector3i vector3i_from_float = Vector3i(Vector3(1.2, 2.6, 3.3));

	CHECK_MESSAGE(
			vector3i == vector3i_copy, "Vector3is created with the same values but different methods should be equal.");
	CHECK_MESSAGE(
			vector3i_from_float.x == 1, "Vector3is created from a Vector3 should round to correct integer values.");
	CHECK_MESSAGE(
			vector3i_from_float.y == 3, "Vector3is created from a Vector3 should round to correct integer values.");
	CHECK_MESSAGE(
			vector3i_from_float.z == 3, "Vector3is created from a Vector3 should round to correct integer values.");
}

TEST_CASE("[Vector3i] Transpose methods") {
	const Vector3i vector3i = Vector3i(1, 2, 3);
	const Vector3i xyz = vector3i.get_XYZ();
	const Vector3i xzy = vector3i.get_XZY();
	const Vector3i yxz = vector3i.get_YXZ();
	const Vector3i yzx = vector3i.get_YZX();
	const Vector3i zxy = vector3i.get_ZXY();
	const Vector3i zyx = vector3i.get_ZYX();
	const Vector2i xy = vector3i.get_XY();
	const Vector2i xz = vector3i.get_XZ();
	const Vector2i yx = vector3i.get_YX();
	const Vector2i yz = vector3i.get_YZ();
	const Vector2i zx = vector3i.get_ZX();
	const Vector2i zy = vector3i.get_ZY();

	CHECK_MESSAGE(
			&vector3i != &xyz, "Transposed vectors should be new objects");

	CHECK_MESSAGE(
			xyz.x == vector3i.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xyz.y == vector3i.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xyz.z == vector3i.z, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			xzy.x == vector3i.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xzy.y == vector3i.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xzy.z == vector3i.y, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			yxz.x == vector3i.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yxz.y == vector3i.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yxz.z == vector3i.z, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			yzx.x == vector3i.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yzx.y == vector3i.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yzx.z == vector3i.x, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			zxy.x == vector3i.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zxy.y == vector3i.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zxy.z == vector3i.y, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			zyx.x == vector3i.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zyx.z == vector3i.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zyx.z == vector3i.x, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			xy.x == vector3i.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xy.y == vector3i.y, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			xz.x == vector3i.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xz.y == vector3i.z, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			yx.x == vector3i.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yx.y == vector3i.x, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			yz.x == vector3i.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yz.y == vector3i.z, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			zx.x == vector3i.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zx.y == vector3i.x, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			zy.x == vector3i.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zy.y == vector3i.y, "Transposed vectors should have the correct values");
}

#endif //GODOT_TEST_VECTOR3I_H
