/**************************************************************************/
/*  test_ctz.h                                                            */
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

#ifndef TEST_CTZ_H
#define TEST_CTZ_H

#include "core/typedefs.h"
#include "tests/test_macros.h"

namespace TestCtz {

TEST_CASE("[CTZ] Count trailing zeros") {
	CHECK(CTZ32(0) == 32);
	CHECK(CTZ32(0b00000001) == 0);
	CHECK(CTZ32(0b00001111) == 0);
	CHECK(CTZ32(0b00100000) == 5);
	CHECK(CTZ32(0b00000001'00000000) == 8);
	CHECK(CTZ32(0x8000'0000) == 31);

	// explicitly test software version
	CHECK(__CTZ32_software(0) == 32);
	CHECK(__CTZ32_software(0b00000001) == 0);
	CHECK(__CTZ32_software(0b00001111) == 0);
	CHECK(__CTZ32_software(0b00100000) == 5);
	CHECK(__CTZ32_software(0b00000001'00000000) == 8);
	CHECK(__CTZ32_software(0x8000'0000) == 31);

	CHECK(CTZ64(0) == 64);
	CHECK(CTZ64(0b00000001) == 0);
	CHECK(CTZ64(0b00001111) == 0);
	CHECK(CTZ64(0b00100000) == 5);
	CHECK(CTZ64(0b00000001'00000000) == 8);
	CHECK(CTZ64(0x8000'0000) == 31);
	CHECK(CTZ64(0x8000'0000'0000'0000) == 63);

	// explicitly test software version
	CHECK(__CTZ64_software(0) == 64);
	CHECK(__CTZ64_software(0b00000001) == 0);
	CHECK(__CTZ64_software(0b00001111) == 0);
	CHECK(__CTZ64_software(0b00100000) == 5);
	CHECK(__CTZ64_software(0b00000001'00000000) == 8);
	CHECK(__CTZ64_software(0x8000'0000) == 31);
	CHECK(__CTZ64_software(0x8000'0000'0000'0000) == 63);
}

} // namespace TestCtz

#endif // TEST_CTZ_H
