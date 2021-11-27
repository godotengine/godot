/*************************************************************************/
/*  test_vector2i.h                                                      */
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

#ifndef GODOT_TEST_VECTOR2I_H
#define GODOT_TEST_VECTOR2I_H

#include "core/math/vector2.h"

#include "tests/test_macros.h"

namespace TestVector2i {

TEST_CASE("[Vector2i] Constructor methods") {
	const Vector2i vector2i = Vector2i(1, 2);
	const Vector2i vector2i_copy = Vector2i(vector2i);
	const Vector2i vector2i_from_float = Vector2i(Vector2(1.2, 2.6));

	CHECK_MESSAGE(
			vector2i == vector2i_copy, "Vector2is created with the same values but different methods should be equal.");
	CHECK_MESSAGE(
			vector2i_from_float.x == 1, "Vector2is created from a Vector2 should round to correct integer values.");
	CHECK_MESSAGE(
			vector2i_from_float.y == 2, "Vector2is created from a Vector2 should round to correct integer values.");
}

TEST_CASE("[Vector2i] Transpose methods") {
	const Vector2i vector2i = Vector2i(1, 2);
	const Vector2i xy = vector2i.get_xy();
	const Vector2i yx = vector2i.get_yx();

	CHECK_MESSAGE(
			&vector2i != &xy, "Transposed vectors should be new objects");
	CHECK_MESSAGE(
			xy.x == vector2i.x, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			xy.y == vector2i.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yx.x == vector2i.y, "Transposed vectors should have the correct values");
	CHECK_MESSAGE(
			yx.y == vector2i.x, "Transposed vectors should have the correct values");
}

} //namespace TestVector2i

#endif //GODOT_TEST_VECTOR2I_H
