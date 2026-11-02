/**************************************************************************/
/*  tint_legacy_parse_test.cpp                                            */
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

#include "src/tint/utils/strconv/parse_num.h"

#include "drivers/webgpu/tests/check.h"

#include <cmath>
#include <cstdio>
#include <string_view>

int main() {
	using namespace tint::strconv;
	CHECK(ParseFloat("1.25") == 1.25f);
	CHECK(ParseDouble("-2.5e2") == -250.0);
	CHECK(ParseFloat("1e100") == ParseNumberError::kResultOutOfRange);
	CHECK(ParseFloat("1e-100") == ParseNumberError::kResultOutOfRange);
	CHECK(ParseFloat("1e-40") == tint::Success);
	CHECK(ParseDouble("1e-300") == tint::Success);
	CHECK(ParseDouble("0e-999") == 0.0);
	CHECK(ParseFloat("") == ParseNumberError::kUnparsable);
	CHECK(ParseFloat(" 1") == ParseNumberError::kUnparsable);
	CHECK(ParseFloat("\t1") == ParseNumberError::kUnparsable);
	CHECK(ParseFloat("+1") == ParseNumberError::kUnparsable);
	CHECK(ParseFloat("1x") == ParseNumberError::kUnparsable);
	CHECK(ParseFloat("0x1p2") == ParseNumberError::kUnparsable);
	CHECK(ParseFloat("-0x1p2") == ParseNumberError::kUnparsable);
	CHECK(ParseFloat(std::string_view("1.25suffix", 4)) == 1.25f);
	CHECK(ParseFloat(std::string_view("1\0x", 3)) == ParseNumberError::kUnparsable);
	CHECK(std::isinf(ParseFloat("inf").Get()));
	CHECK(std::isnan(ParseDouble("nan").Get()));
	CHECK(ParseInt("42") == 42);
	CHECK(ParseUint("-1") == ParseNumberError::kUnparsable);
	std::puts("PASS: 20 Tint number parsing checks");
}
