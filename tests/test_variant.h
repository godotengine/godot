/*************************************************************************/
/*  test_variant.h                                                       */
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

#ifndef TEST_VARIANT_H
#define TEST_VARIANT_H

#include "core/variant.h"
#include "core/variant_parser.h"

#include "tests/test_macros.h"

namespace TestVariant {

static inline Array build_array() {
	return Array();
}
template <typename... Targs>
static inline Array build_array(Variant item, Targs... Fargs) {
	Array a = build_array(Fargs...);
	a.push_front(item);
	return a;
}
static inline Dictionary build_dictionary() {
	return Dictionary();
}
template <typename... Targs>
static inline Dictionary build_dictionary(Variant key, Variant item, Targs... Fargs) {
	Dictionary d = build_dictionary(Fargs...);
	d[key] = item;
	return d;
}

TEST_CASE("[Variant] Writer and parser integer") {
	int64_t a32 = 2147483648; // 2^31, so out of bounds for 32-bit signed int [-2^31,-2^31-1].
	String a32_str;
	VariantWriter::write_to_string(a32, a32_str);

	CHECK_MESSAGE(a32_str != "-2147483648", "Should not wrap around");

	int64_t b64 = 9223372036854775807; // 2^63-1, upper bound for signed 64-bit int.
	String b64_str;
	VariantWriter::write_to_string(b64, b64_str);

	CHECK_MESSAGE(b64_str == "9223372036854775807", "Should not wrap around.");

	VariantParser::StreamString ss;
	String errs;
	int line;
	Variant b64_parsed;
	int64_t b64_int_parsed;

	ss.s = b64_str;
	VariantParser::parse(&ss, b64_parsed, errs, line);
	b64_int_parsed = b64_parsed;

	CHECK_MESSAGE(b64_int_parsed == 9223372036854775807, "Should parse back.");

	ss.s = "9223372036854775808"; // Overflowed by one.
	VariantParser::parse(&ss, b64_parsed, errs, line);
	b64_int_parsed = b64_parsed;

	CHECK_MESSAGE(b64_int_parsed == 9223372036854775807, "The result should be clamped to max value.");

	ss.s = "1e100"; // Googol! Scientific notation.
	VariantParser::parse(&ss, b64_parsed, errs, line);
	b64_int_parsed = b64_parsed;

	CHECK_MESSAGE(b64_int_parsed == 9223372036854775807, "The result should be clamped to max value.");
}

TEST_CASE("[Variant] Writer and parser float") {
	// Assuming real_t is double.
	real_t a64 = 340282346638528859811704183484516925440.0; // std::numeric_limits<real_t>::max()
	String a64_str;
	VariantWriter::write_to_string(a64, a64_str);

	CHECK_MESSAGE(a64_str == "3.40282e+38", "Writes in scientific notation.");
	CHECK_MESSAGE(a64_str != "inf", "Should not overflow.");
	CHECK_MESSAGE(a64_str != "nan", "The result should be defined.");

	VariantParser::StreamString ss;
	String errs;
	int line;
	Variant b64_parsed;
	real_t b64_float_parsed;

	ss.s = a64_str;
	VariantParser::parse(&ss, b64_parsed, errs, line);
	b64_float_parsed = b64_parsed;

	CHECK_MESSAGE(b64_float_parsed == 340282001837565597733306976381245063168.0, "Should parse back.");
	// Loses precision, but that's alright.

	ss.s = "1.0e+100"; // Float version of Googol!
	VariantParser::parse(&ss, b64_parsed, errs, line);
	b64_float_parsed = b64_parsed;

	CHECK_MESSAGE(b64_float_parsed == 340282001837565597733306976381245063168.0, "Should not overflow.");
}

TEST_CASE("[Variant] Writer and parser array") {
	Array a = build_array(1, String("hello"), build_array(Variant()));
	String a_str;
	VariantWriter::write_to_string(a, a_str);

	CHECK_EQ(a_str, "[ 1, \"hello\", [ null ] ]");

	VariantParser::StreamString ss;
	String errs;
	int line;
	Variant a_parsed;

	ss.s = a_str;
	VariantParser::parse(&ss, a_parsed, errs, line);

	CHECK_MESSAGE(a_parsed == Variant(a), "Should parse back.");
}

TEST_CASE("[Variant] Writer recursive array") {
	// There is no way to accurately represent a recursive array,
	// the only thing we can do is make sure the writer doesn't blow up

	// Self recursive
	Array a;
	a.push_back(a);

	// Writer should it recursion limit while visiting the array
	ERR_PRINT_OFF;
	String a_str;
	VariantWriter::write_to_string(a, a_str);
	ERR_PRINT_ON;

	// Nested recursive
	Array a1;
	Array a2;
	a1.push_back(a2);
	a2.push_back(a1);

	// Writer should it recursion limit while visiting the array
	ERR_PRINT_OFF;
	String a1_str;
	VariantWriter::write_to_string(a1, a1_str);
	ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary tearndown will leak memory
	a.clear();
	a1.clear();
	a2.clear();
}

TEST_CASE("[Variant] Writer and parser dictionary") {
	// d = {{1: 2}: 3, 4: "hello", 5: {null: []}}
	Dictionary d = build_dictionary(build_dictionary(1, 2), 3, 4, String("hello"), 5, build_dictionary(Variant(), build_array()));
	String d_str;
	VariantWriter::write_to_string(d, d_str);

	CHECK_EQ(d_str, "{\n4: \"hello\",\n5: {\nnull: [  ]\n},\n{\n1: 2\n}: 3\n}");

	VariantParser::StreamString ss;
	String errs;
	int line;
	Variant d_parsed;

	ss.s = d_str;
	VariantParser::parse(&ss, d_parsed, errs, line);

	CHECK_MESSAGE(d_parsed == Variant(d), "Should parse back.");
}

TEST_CASE("[Variant] Writer recursive dictionary") {
	// There is no way to accurately represent a recursive dictionary,
	// the only thing we can do is make sure the writer doesn't blow up

	// Self recursive
	Dictionary d;
	d[1] = d;

	// Writer should it recursion limit while visiting the dictionary
	ERR_PRINT_OFF;
	String d_str;
	VariantWriter::write_to_string(d, d_str);
	ERR_PRINT_ON;

	// Nested recursive
	Dictionary d1;
	Dictionary d2;
	d1[2] = d2;
	d2[1] = d1;

	// Writer should it recursion limit while visiting the dictionary
	ERR_PRINT_OFF;
	String d1_str;
	VariantWriter::write_to_string(d1, d1_str);
	ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary tearndown will leak memory
	d.clear();
	d1.clear();
	d2.clear();
}

#if 0 // TODO: recursion in dict key is currently buggy
TEST_CASE("[Variant] Writer recursive dictionary on keys") {
	// There is no way to accurately represent a recursive dictionary,
	// the only thing we can do is make sure the writer doesn't blow up

	// Self recursive
	Dictionary d;
	d[d] = 1;

	// Writer should it recursion limit while visiting the dictionary
	ERR_PRINT_OFF;
	String d_str;
	VariantWriter::write_to_string(d, d_str);
	ERR_PRINT_ON;

	// Nested recursive
	Dictionary d1;
	Dictionary d2;
	d1[d2] = 2;
	d2[d1] = 1;

	// Writer should it recursion limit while visiting the dictionary
	ERR_PRINT_OFF;
	String d1_str;
	VariantWriter::write_to_string(d1, d1_str);
	ERR_PRINT_ON;

	// Break the recursivity otherwise Dictionary tearndown will leak memory
	d.clear();
	d1.clear();
	d2.clear();
}
#endif

TEST_CASE("[Variant] Basic comparison") {
	CHECK_EQ(Variant(1), Variant(1));
	CHECK_FALSE(Variant(1) != Variant(1));
	CHECK_NE(Variant(1), Variant(2));
	CHECK_EQ(Variant(String("foo")), Variant(String("foo")));
	CHECK_NE(Variant(String("foo")), Variant(String("bar")));
	// Check "empty" version of different types are not equivalents
	CHECK_NE(Variant(0), Variant());
	CHECK_NE(Variant(String()), Variant());
	CHECK_NE(Variant(Array()), Variant());
	CHECK_NE(Variant(Dictionary()), Variant());
}

TEST_CASE("[Variant] Nested array comparison") {
	Array a1 = build_array(1, build_array(2, 3));
	Array a2 = build_array(1, build_array(2, 3));
	Array a_other = build_array(1, build_array(2, 4));
	Variant v_a1 = a1;
	Variant v_a1_ref2 = a1;
	Variant v_a2 = a2;
	Variant v_a_other = a_other;

	// test both operator== and operator!=
	CHECK_EQ(v_a1, v_a1);
	CHECK_FALSE(v_a1 != v_a1);
	CHECK_EQ(v_a1, v_a1_ref2);
	CHECK_FALSE(v_a1 != v_a1_ref2);
	CHECK_EQ(v_a1, v_a2);
	CHECK_FALSE(v_a1 != v_a2);
	CHECK_NE(v_a1, v_a_other);
	CHECK_FALSE(v_a1 == v_a_other);
}

TEST_CASE("[Variant] Nested dictionary comparison") {
	Dictionary d1 = build_dictionary(build_dictionary(1, 2), build_dictionary(3, 4));
	Dictionary d2 = build_dictionary(build_dictionary(1, 2), build_dictionary(3, 4));
	Dictionary d_other_key = build_dictionary(build_dictionary(1, 0), build_dictionary(3, 4));
	Dictionary d_other_val = build_dictionary(build_dictionary(1, 2), build_dictionary(3, 0));
	Variant v_d1 = d1;
	Variant v_d1_ref2 = d1;
	Variant v_d2 = d2;
	Variant v_d_other_key = d_other_key;
	Variant v_d_other_val = d_other_val;

	// test both operator== and operator!=
	CHECK_EQ(v_d1, v_d1);
	CHECK_FALSE(v_d1 != v_d1);
	CHECK_EQ(v_d1, v_d1_ref2);
	CHECK_FALSE(v_d1 != v_d1_ref2);
	CHECK_EQ(v_d1, v_d2);
	CHECK_FALSE(v_d1 != v_d2);
	CHECK_NE(v_d1, v_d_other_key);
	CHECK_FALSE(v_d1 == v_d_other_key);
	CHECK_NE(v_d1, v_d_other_val);
	CHECK_FALSE(v_d1 == v_d_other_val);
}

} // namespace TestVariant

#endif // TEST_VARIANT_H
