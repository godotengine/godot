/*************************************************************************/
/*  test_compiler_sanity.h                                               */
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

#ifndef TEST_SANITY_H
#define TEST_SANITY_H

#include "core/os/os.h"
#include "core/string/ustring.h"
#include "scene/resources/material.h"
#include "tests/test_macros.h"

namespace TestSanity {

class ArrayInitializerTest : public Reference {
public:
	static const int max_size = 10;
	bool class_member_2[max_size] = {}; // will all be zero
	bool class_member_3[max_size] = { false }; // will all be zero
};

TEST_CASE("[Sanity] Check basic definition has zeros") {
	for (size_t x = 0; x < 100000; x++) {
		Ref<ArrayInitializerTest> mat;
		mat.instance();
		for (int y = 0; y < ArrayInitializerTest::max_size; y++) {
			bool yv1 = mat->class_member_2[y];
			bool yv2 = mat->class_member_3[y];

			CHECK(!yv1);
			CHECK(!yv2);
		}
	}
}

} // namespace TestSanity

#endif // TEST_SANITY_H
