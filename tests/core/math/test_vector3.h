/*************************************************************************/
/*  test_vector3.h                                                       */
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

#ifndef GODOT_TEST_VECTOR3_H
#define GODOT_TEST_VECTOR3_H

#include "core/math/vector3.h"

#include "tests/test_macros.h"

namespace TestVector3 {

TEST_CASE("[Vector3] Constructor methods") {
	const Vector3 vector3 = Vector3(1.2, 2.1, 3.3);
	const Vector3 vector3_copy = Vector3(vector3);

	CHECK_MESSAGE(
			vector3 == vector3_copy, "Vector3s created with the same values but different methods should be equal.");
}

TEST_CASE("[Vector3] Transpose methods") {
	const Vector3 vector3 = Vector3(1.2, 2.1, 3.3);
	const Vector3 xyz = vector3.get_xyz();
	const Vector3 xzy = vector3.get_xzy();
	const Vector3 yxz = vector3.get_yxz();
	const Vector3 yzx = vector3.get_yzx();
	const Vector3 zxy = vector3.get_zxy();
	const Vector3 zyx = vector3.get_zyx();
	const Vector2 xy = vector3.get_xy();
	const Vector2 xz = vector3.get_xz();
	const Vector2 yx = vector3.get_yx();
	const Vector2 yz = vector3.get_yz();
	const Vector2 zx = vector3.get_zx();
	const Vector2 zy = vector3.get_zy();

	CHECK_MESSAGE(
			&vector3 != &xyz, "Transposed vectors should be new objects");

	CHECK_MESSAGE(
			xyz.x == vector3.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xyz.y == vector3.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xyz.z == vector3.z, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			xzy.x == vector3.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xzy.y == vector3.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xzy.z == vector3.y, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			yxz.x == vector3.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yxz.y == vector3.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yxz.z == vector3.z, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			yzx.x == vector3.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yzx.y == vector3.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yzx.z == vector3.x, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			zxy.x == vector3.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zxy.y == vector3.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zxy.z == vector3.y, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			zyx.x == vector3.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zyx.y == vector3.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zyx.z == vector3.x, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			xy.x == vector3.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xy.y == vector3.y, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			xz.x == vector3.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xz.y == vector3.z, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			yx.x == vector3.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yx.y == vector3.x, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			yz.x == vector3.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yz.y == vector3.z, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			zx.x == vector3.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zx.y == vector3.x, "Transposed vectors should have the correct values");

	CHECK_MESSAGE(
			zy.x == vector3.z, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			zy.y == vector3.y, "Transposed vectors should have the correct values");
}

} //namespace TestVector3

#endif //GODOT_TEST_VECTOR3_H
