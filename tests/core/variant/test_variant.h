/*************************************************************************/
/*  test_variant.h                                                       */
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

#ifndef TEST_VARIANT_H
#define TEST_VARIANT_H

#include "core/variant/variant.h"
#include "core/variant/variant_parser.h"

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
	int64_t a32 = 2147483648; // 2^31, so out of bounds for 32-bit signed int [-2^31, +2^31-1].
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

TEST_CASE("[Variant] Writer and parser Variant::FLOAT") {
	// Variant::FLOAT is always 64-bit (C++ double).
	// This is the maximum non-infinity double-precision float.
	double a64 = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0;
	String a64_str;
	VariantWriter::write_to_string(a64, a64_str);

	CHECK_MESSAGE(a64_str == "1.79769e+308", "Writes in scientific notation.");
	CHECK_MESSAGE(a64_str != "inf", "Should not overflow.");
	CHECK_MESSAGE(a64_str != "nan", "The result should be defined.");

	String errs;
	int line;
	Variant variant_parsed;
	double float_parsed;

	VariantParser::StreamString bss;
	bss.s = a64_str;
	VariantParser::parse(&bss, variant_parsed, errs, line);
	float_parsed = variant_parsed;
	// Loses precision, but that's alright.
	CHECK_MESSAGE(float_parsed == 1.79769e+308, "Should parse back.");

	// Approximation of Googol with a double-precision float.
	VariantParser::StreamString css;
	css.s = "1.0e+100";
	VariantParser::parse(&css, variant_parsed, errs, line);
	float_parsed = variant_parsed;
	CHECK_MESSAGE(float_parsed == 1.0e+100, "Should match the double literal.");
}

TEST_CASE("[Variant] Assignment To Bool from Int,Float,String,Vec2,Vec2i,Vec3,Vec3i and Color") {
	Variant int_v = 0;
	Variant bool_v = true;
	int_v = bool_v; // int_v is now a bool
	CHECK(int_v == Variant(true));
	bool_v = false;
	int_v = bool_v;
	CHECK(int_v.get_type() == Variant::BOOL);

	Variant float_v = 0.0f;
	bool_v = true;
	float_v = bool_v;
	CHECK(float_v == Variant(true));
	bool_v = false;
	float_v = bool_v;
	CHECK(float_v.get_type() == Variant::BOOL);

	Variant string_v = "";
	bool_v = true;
	string_v = bool_v;
	CHECK(string_v == Variant(true));
	bool_v = false;
	string_v = bool_v;
	CHECK(string_v.get_type() == Variant::BOOL);

	Variant vec2_v = Vector2(0, 0);
	bool_v = true;
	vec2_v = bool_v;
	CHECK(vec2_v == Variant(true));
	bool_v = false;
	vec2_v = bool_v;
	CHECK(vec2_v.get_type() == Variant::BOOL);

	Variant vec2i_v = Vector2i(0, 0);
	bool_v = true;
	vec2i_v = bool_v;
	CHECK(vec2i_v == Variant(true));
	bool_v = false;
	vec2i_v = bool_v;
	CHECK(vec2i_v.get_type() == Variant::BOOL);

	Variant vec3_v = Vector3(0, 0, 0);
	bool_v = true;
	vec3_v = bool_v;
	CHECK(vec3_v == Variant(true));
	bool_v = false;
	vec3_v = bool_v;
	CHECK(vec3_v.get_type() == Variant::BOOL);

	Variant vec3i_v = Vector3i(0, 0, 0);
	bool_v = true;
	vec3i_v = bool_v;
	CHECK(vec3i_v == Variant(true));
	bool_v = false;
	vec3i_v = bool_v;
	CHECK(vec3i_v.get_type() == Variant::BOOL);

	Variant col_v = Color(0.5f, 0.2f, 0.75f);
	bool_v = true;
	col_v = bool_v;
	CHECK(col_v == Variant(true));
	bool_v = false;
	col_v = bool_v;
	CHECK(col_v.get_type() == Variant::BOOL);
}

TEST_CASE("[Variant] Assignment To Int from Bool,Float,String,Vec2,Vec2i,Vec3,Vec3i and Color") {
	Variant bool_v = false;
	Variant int_v = 2;
	bool_v = int_v; // Now bool_v is int
	CHECK(bool_v == Variant(2));
	int_v = -3;
	bool_v = int_v;
	CHECK(bool_v.get_type() == Variant::INT);

	Variant float_v = 0.0f;
	int_v = 2;
	float_v = int_v;
	CHECK(float_v == Variant(2));
	int_v = -3;
	float_v = int_v;
	CHECK(float_v.get_type() == Variant::INT);

	Variant string_v = "";
	int_v = 2;
	string_v = int_v;
	CHECK(string_v == Variant(2));
	int_v = -3;
	string_v = int_v;
	CHECK(string_v.get_type() == Variant::INT);

	Variant vec2_v = Vector2(0, 0);
	int_v = 2;
	vec2_v = int_v;
	CHECK(vec2_v == Variant(2));
	int_v = -3;
	vec2_v = int_v;
	CHECK(vec2_v.get_type() == Variant::INT);

	Variant vec2i_v = Vector2i(0, 0);
	int_v = 2;
	vec2i_v = int_v;
	CHECK(vec2i_v == Variant(2));
	int_v = -3;
	vec2i_v = int_v;
	CHECK(vec2i_v.get_type() == Variant::INT);

	Variant vec3_v = Vector3(0, 0, 0);
	int_v = 2;
	vec3_v = int_v;
	CHECK(vec3_v == Variant(2));
	int_v = -3;
	vec3_v = int_v;
	CHECK(vec3_v.get_type() == Variant::INT);

	Variant vec3i_v = Vector3i(0, 0, 0);
	int_v = 2;
	vec3i_v = int_v;
	CHECK(vec3i_v == Variant(2));
	int_v = -3;
	vec3i_v = int_v;
	CHECK(vec3i_v.get_type() == Variant::INT);

	Variant col_v = Color(0.5f, 0.2f, 0.75f);
	int_v = 2;
	col_v = int_v;
	CHECK(col_v == Variant(2));
	int_v = -3;
	col_v = int_v;
	CHECK(col_v.get_type() == Variant::INT);
}

TEST_CASE("[Variant] Assignment To Float from Bool,Int,String,Vec2,Vec2i,Vec3,Vec3i and Color") {
	Variant bool_v = false;
	Variant float_v = 1.5f;
	bool_v = float_v; // Now bool_v is float
	CHECK(bool_v == Variant(1.5f));
	float_v = -4.6f;
	bool_v = float_v;
	CHECK(bool_v.get_type() == Variant::FLOAT);

	Variant int_v = 1;
	float_v = 1.5f;
	int_v = float_v;
	CHECK(int_v == Variant(1.5f));
	float_v = -4.6f;
	int_v = float_v;
	CHECK(int_v.get_type() == Variant::FLOAT);

	Variant string_v = "";
	float_v = 1.5f;
	string_v = float_v;
	CHECK(string_v == Variant(1.5f));
	float_v = -4.6f;
	string_v = float_v;
	CHECK(string_v.get_type() == Variant::FLOAT);

	Variant vec2_v = Vector2(0, 0);
	float_v = 1.5f;
	vec2_v = float_v;
	CHECK(vec2_v == Variant(1.5f));
	float_v = -4.6f;
	vec2_v = float_v;
	CHECK(vec2_v.get_type() == Variant::FLOAT);

	Variant vec2i_v = Vector2i(0, 0);
	float_v = 1.5f;
	vec2i_v = float_v;
	CHECK(vec2i_v == Variant(1.5f));
	float_v = -4.6f;
	vec2i_v = float_v;
	CHECK(vec2i_v.get_type() == Variant::FLOAT);

	Variant vec3_v = Vector3(0, 0, 0);
	float_v = 1.5f;
	vec3_v = float_v;
	CHECK(vec3_v == Variant(1.5f));
	float_v = -4.6f;
	vec3_v = float_v;
	CHECK(vec3_v.get_type() == Variant::FLOAT);

	Variant vec3i_v = Vector3i(0, 0, 0);
	float_v = 1.5f;
	vec3i_v = float_v;
	CHECK(vec3i_v == Variant(1.5f));
	float_v = -4.6f;
	vec3i_v = float_v;
	CHECK(vec3i_v.get_type() == Variant::FLOAT);

	Variant col_v = Color(0.5f, 0.2f, 0.75f);
	float_v = 1.5f;
	col_v = float_v;
	CHECK(col_v == Variant(1.5f));
	float_v = -4.6f;
	col_v = float_v;
	CHECK(col_v.get_type() == Variant::FLOAT);
}

TEST_CASE("[Variant] Assignment To String from Bool,Int,Float,Vec2,Vec2i,Vec3,Vec3i and Color") {
	Variant bool_v = false;
	Variant string_v = "Hello";
	bool_v = string_v; // Now bool_v is string
	CHECK(bool_v == Variant("Hello"));
	string_v = "Hello there";
	bool_v = string_v;
	CHECK(bool_v.get_type() == Variant::STRING);

	Variant int_v = 0;
	string_v = "Hello";
	int_v = string_v;
	CHECK(int_v == Variant("Hello"));
	string_v = "Hello there";
	int_v = string_v;
	CHECK(int_v.get_type() == Variant::STRING);

	Variant float_v = 0.0f;
	string_v = "Hello";
	float_v = string_v;
	CHECK(float_v == Variant("Hello"));
	string_v = "Hello there";
	float_v = string_v;
	CHECK(float_v.get_type() == Variant::STRING);

	Variant vec2_v = Vector2(0, 0);
	string_v = "Hello";
	vec2_v = string_v;
	CHECK(vec2_v == Variant("Hello"));
	string_v = "Hello there";
	vec2_v = string_v;
	CHECK(vec2_v.get_type() == Variant::STRING);

	Variant vec2i_v = Vector2i(0, 0);
	string_v = "Hello";
	vec2i_v = string_v;
	CHECK(vec2i_v == Variant("Hello"));
	string_v = "Hello there";
	vec2i_v = string_v;
	CHECK(vec2i_v.get_type() == Variant::STRING);

	Variant vec3_v = Vector3(0, 0, 0);
	string_v = "Hello";
	vec3_v = string_v;
	CHECK(vec3_v == Variant("Hello"));
	string_v = "Hello there";
	vec3_v = string_v;
	CHECK(vec3_v.get_type() == Variant::STRING);

	Variant vec3i_v = Vector3i(0, 0, 0);
	string_v = "Hello";
	vec3i_v = string_v;
	CHECK(vec3i_v == Variant("Hello"));
	string_v = "Hello there";
	vec3i_v = string_v;
	CHECK(vec3i_v.get_type() == Variant::STRING);

	Variant col_v = Color(0.5f, 0.2f, 0.75f);
	string_v = "Hello";
	col_v = string_v;
	CHECK(col_v == Variant("Hello"));
	string_v = "Hello there";
	col_v = string_v;
	CHECK(col_v.get_type() == Variant::STRING);
}

TEST_CASE("[Variant] Assignment To Vec2 from Bool,Int,Float,String,Vec2i,Vec3,Vec3i and Color") {
	Variant bool_v = false;
	Variant vec2_v = Vector2(2.2f, 3.5f);
	bool_v = vec2_v; // Now bool_v is Vector2
	CHECK(bool_v == Variant(Vector2(2.2f, 3.5f)));
	vec2_v = Vector2(-5.4f, -7.9f);
	bool_v = vec2_v;
	CHECK(bool_v.get_type() == Variant::VECTOR2);

	Variant int_v = 0;
	vec2_v = Vector2(2.2f, 3.5f);
	int_v = vec2_v;
	CHECK(int_v == Variant(Vector2(2.2f, 3.5f)));
	vec2_v = Vector2(-5.4f, -7.9f);
	int_v = vec2_v;
	CHECK(int_v.get_type() == Variant::VECTOR2);

	Variant float_v = 0.0f;
	vec2_v = Vector2(2.2f, 3.5f);
	float_v = vec2_v;
	CHECK(float_v == Variant(Vector2(2.2f, 3.5f)));
	vec2_v = Vector2(-5.4f, -7.9f);
	float_v = vec2_v;
	CHECK(float_v.get_type() == Variant::VECTOR2);

	Variant string_v = "";
	vec2_v = Vector2(2.2f, 3.5f);
	string_v = vec2_v;
	CHECK(string_v == Variant(Vector2(2.2f, 3.5f)));
	vec2_v = Vector2(-5.4f, -7.9f);
	string_v = vec2_v;
	CHECK(string_v.get_type() == Variant::VECTOR2);

	Variant vec2i_v = Vector2i(0, 0);
	vec2_v = Vector2(2.2f, 3.5f);
	vec2i_v = vec2_v;
	CHECK(vec2i_v == Variant(Vector2(2.2f, 3.5f)));
	vec2_v = Vector2(-5.4f, -7.9f);
	vec2i_v = vec2_v;
	CHECK(vec2i_v.get_type() == Variant::VECTOR2);

	Variant vec3_v = Vector3(0, 0, 0);
	vec2_v = Vector2(2.2f, 3.5f);
	vec3_v = vec2_v;
	CHECK(vec3_v == Variant(Vector2(2.2f, 3.5f)));
	vec2_v = Vector2(-5.4f, -7.9f);
	vec3_v = vec2_v;
	CHECK(vec3_v.get_type() == Variant::VECTOR2);

	Variant vec3i_v = Vector3i(0, 0, 0);
	vec2_v = Vector2(2.2f, 3.5f);
	vec3i_v = vec2_v;
	CHECK(vec3i_v == Variant(Vector2(2.2f, 3.5f)));
	vec2_v = Vector2(-5.4f, -7.9f);
	vec3i_v = vec2_v;
	CHECK(vec3i_v.get_type() == Variant::VECTOR2);

	Variant col_v = Color(0.5f, 0.2f, 0.75f);
	vec2_v = Vector2(2.2f, 3.5f);
	col_v = vec2_v;
	CHECK(col_v == Variant(Vector2(2.2f, 3.5f)));
	vec2_v = Vector2(-5.4f, -7.9f);
	col_v = vec2_v;
	CHECK(col_v.get_type() == Variant::VECTOR2);
}

TEST_CASE("[Variant] Assignment To Vec2i from Bool,Int,Float,String,Vec2,Vec3,Vec3i and Color") {
	Variant bool_v = false;
	Variant vec2i_v = Vector2i(2, 3);
	bool_v = vec2i_v; // Now bool_v is Vector2i
	CHECK(bool_v == Variant(Vector2i(2, 3)));
	vec2i_v = Vector2i(-5, -7);
	bool_v = vec2i_v;
	CHECK(bool_v.get_type() == Variant::VECTOR2I);

	Variant int_v = 0;
	vec2i_v = Vector2i(2, 3);
	int_v = vec2i_v;
	CHECK(int_v == Variant(Vector2i(2, 3)));
	vec2i_v = Vector2i(-5, -7);
	int_v = vec2i_v;
	CHECK(int_v.get_type() == Variant::VECTOR2I);

	Variant float_v = 0.0f;
	vec2i_v = Vector2i(2, 3);
	float_v = vec2i_v;
	CHECK(float_v == Variant(Vector2i(2, 3)));
	vec2i_v = Vector2i(-5, -7);
	float_v = vec2i_v;
	CHECK(float_v.get_type() == Variant::VECTOR2I);

	Variant string_v = "";
	vec2i_v = Vector2i(2, 3);
	string_v = vec2i_v;
	CHECK(string_v == Variant(Vector2i(2, 3)));
	vec2i_v = Vector2i(-5, -7);
	string_v = vec2i_v;
	CHECK(string_v.get_type() == Variant::VECTOR2I);

	Variant vec2_v = Vector2(0, 0);
	vec2i_v = Vector2i(2, 3);
	vec2_v = vec2i_v;
	CHECK(vec2_v == Variant(Vector2i(2, 3)));
	vec2i_v = Vector2i(-5, -7);
	vec2_v = vec2i_v;
	CHECK(vec2i_v.get_type() == Variant::VECTOR2I);

	Variant vec3_v = Vector3(0, 0, 0);
	vec2i_v = Vector2i(2, 3);
	vec3_v = vec2i_v;
	CHECK(vec3_v == Variant(Vector2i(2, 3)));
	vec2i_v = Vector2i(-5, -7);
	vec3_v = vec2i_v;
	CHECK(vec3_v.get_type() == Variant::VECTOR2I);

	Variant vec3i_v = Vector3i(0, 0, 0);
	vec2i_v = Vector2i(2, 3);
	vec3i_v = vec2i_v;
	CHECK(vec3i_v == Variant(Vector2i(2, 3)));
	vec2i_v = Vector2i(-5, -7);
	vec3i_v = vec2i_v;
	CHECK(vec3i_v.get_type() == Variant::VECTOR2I);

	Variant col_v = Color(0.5f, 0.2f, 0.75f);
	vec2i_v = Vector2i(2, 3);
	col_v = vec2i_v;
	CHECK(col_v == Variant(Vector2i(2, 3)));
	vec2i_v = Vector2i(-5, -7);
	col_v = vec2i_v;
	CHECK(col_v.get_type() == Variant::VECTOR2I);
}

TEST_CASE("[Variant] Assignment To Vec3 from Bool,Int,Float,String,Vec2,Vec2i,Vec3i and Color") {
	Variant bool_v = false;
	Variant vec3_v = Vector3(2.2f, 3.5f, 5.3f);
	bool_v = vec3_v; // Now bool_v is Vector3
	CHECK(bool_v == Variant(Vector3(2.2f, 3.5f, 5.3f)));
	vec3_v = Vector3(-5.4f, -7.9f, -2.1f);
	bool_v = vec3_v;
	CHECK(bool_v.get_type() == Variant::VECTOR3);

	Variant int_v = 0;
	vec3_v = Vector3(2.2f, 3.5f, 5.3f);
	int_v = vec3_v;
	CHECK(int_v == Variant(Vector3(2.2f, 3.5f, 5.3f)));
	vec3_v = Vector3(-5.4f, -7.9f, -2.1f);
	int_v = vec3_v;
	CHECK(int_v.get_type() == Variant::VECTOR3);

	Variant float_v = 0.0f;
	vec3_v = Vector3(2.2f, 3.5f, 5.3f);
	float_v = vec3_v;
	CHECK(float_v == Variant(Vector3(2.2f, 3.5f, 5.3f)));
	vec3_v = Vector3(-5.4f, -7.9f, -2.1f);
	float_v = vec3_v;
	CHECK(float_v.get_type() == Variant::VECTOR3);

	Variant string_v = "";
	vec3_v = Vector3(2.2f, 3.5f, 5.3f);
	string_v = vec3_v;
	CHECK(string_v == Variant(Vector3(2.2f, 3.5f, 5.3f)));
	vec3_v = Vector3(-5.4f, -7.9f, -2.1f);
	string_v = vec3_v;
	CHECK(string_v.get_type() == Variant::VECTOR3);

	Variant vec2_v = Vector2(0, 0);
	vec3_v = Vector3(2.2f, 3.5f, 5.3f);
	vec2_v = vec3_v;
	CHECK(vec2_v == Variant(Vector3(2.2f, 3.5f, 5.3f)));
	vec3_v = Vector3(-5.4f, -7.9f, -2.1f);
	vec2_v = vec3_v;
	CHECK(vec2_v.get_type() == Variant::VECTOR3);

	Variant vec2i_v = Vector2i(0, 0);
	vec3_v = Vector3(2.2f, 3.5f, 5.3f);
	vec2i_v = vec3_v;
	CHECK(vec2i_v == Variant(Vector3(2.2f, 3.5f, 5.3f)));
	vec3_v = Vector3(-5.4f, -7.9f, -2.1f);
	vec2i_v = vec3_v;
	CHECK(vec2i_v.get_type() == Variant::VECTOR3);

	Variant vec3i_v = Vector3i(0, 0, 0);
	vec3_v = Vector3(2.2f, 3.5f, 5.3f);
	vec3i_v = vec3_v;
	CHECK(vec3i_v == Variant(Vector3(2.2f, 3.5f, 5.3f)));
	vec3_v = Vector3(-5.4f, -7.9f, -2.1f);
	vec3i_v = vec3_v;
	CHECK(vec3i_v.get_type() == Variant::VECTOR3);

	Variant col_v = Color(0.5f, 0.2f, 0.75f);
	vec3_v = Vector3(2.2f, 3.5f, 5.3f);
	col_v = vec3_v;
	CHECK(col_v == Variant(Vector3(2.2f, 3.5f, 5.3f)));
	vec3_v = Vector3(-5.4f, -7.9f, -2.1f);
	col_v = vec3_v;
	CHECK(col_v.get_type() == Variant::VECTOR3);
}

TEST_CASE("[Variant] Assignment To Vec3i from Bool,Int,Float,String,Vec2,Vec2i,Vec3 and Color") {
	Variant bool_v = false;
	Variant vec3i_v = Vector3i(2, 3, 5);
	bool_v = vec3i_v; // Now bool_v is Vector3i
	CHECK(bool_v == Variant(Vector3i(2, 3, 5)));
	vec3i_v = Vector3i(-5, -7, -2);
	bool_v = vec3i_v;
	CHECK(bool_v.get_type() == Variant::VECTOR3I);

	Variant int_v = 0;
	vec3i_v = Vector3i(2, 3, 5);
	int_v = vec3i_v;
	CHECK(int_v == Variant(Vector3i(2, 3, 5)));
	vec3i_v = Vector3i(-5, -7, -2);
	int_v = vec3i_v;
	CHECK(int_v.get_type() == Variant::VECTOR3I);

	Variant float_v = 0.0f;
	vec3i_v = Vector3i(2, 3, 5);
	float_v = vec3i_v;
	CHECK(float_v == Variant(Vector3i(2, 3, 5)));
	vec3i_v = Vector3i(-5, -7, -2);
	float_v = vec3i_v;
	CHECK(float_v.get_type() == Variant::VECTOR3I);

	Variant string_v = "";
	vec3i_v = Vector3i(2, 3, 5);
	string_v = vec3i_v;
	CHECK(string_v == Variant(Vector3i(2, 3, 5)));
	vec3i_v = Vector3i(-5, -7, -2);
	string_v = vec3i_v;
	CHECK(string_v.get_type() == Variant::VECTOR3I);

	Variant vec2_v = Vector2(0, 0);
	vec3i_v = Vector3i(2, 3, 5);
	vec2_v = vec3i_v;
	CHECK(vec2_v == Variant(Vector3i(2, 3, 5)));
	vec3i_v = Vector3i(-5, -7, -2);
	vec2_v = vec3i_v;
	CHECK(vec2_v.get_type() == Variant::VECTOR3I);

	Variant vec2i_v = Vector2i(0, 0);
	vec3i_v = Vector3i(2, 3, 5);
	vec2i_v = vec3i_v;
	CHECK(vec2i_v == Variant(Vector3i(2, 3, 5)));
	vec3i_v = Vector3i(-5, -7, -2);
	vec2i_v = vec3i_v;
	CHECK(vec2i_v.get_type() == Variant::VECTOR3I);

	Variant vec3_v = Vector3(0, 0, 0);
	vec3i_v = Vector3i(2, 3, 5);
	vec3_v = vec3i_v;
	CHECK(vec3_v == Variant(Vector3i(2, 3, 5)));
	vec3i_v = Vector3i(-5, -7, -2);
	vec3_v = vec3i_v;
	CHECK(vec3_v.get_type() == Variant::VECTOR3I);

	Variant col_v = Color(0.5f, 0.2f, 0.75f);
	vec3i_v = Vector3i(2, 3, 5);
	col_v = vec3i_v;
	CHECK(col_v == Variant(Vector3i(2, 3, 5)));
	vec3i_v = Vector3i(-5, -7, -2);
	col_v = vec3i_v;
	CHECK(col_v.get_type() == Variant::VECTOR3I);
}

TEST_CASE("[Variant] Assignment To Color from Bool,Int,Float,String,Vec2,Vec2i,Vec3 and Vec3i") {
	Variant bool_v = false;
	Variant col_v = Color(0.25f, 0.4f, 0.78f);
	bool_v = col_v; // Now bool_v is Color
	CHECK(bool_v == Variant(Color(0.25f, 0.4f, 0.78f)));
	col_v = Color(0.33f, 0.75f, 0.21f);
	bool_v = col_v;
	CHECK(bool_v.get_type() == Variant::COLOR);

	Variant int_v = 0;
	col_v = Color(0.25f, 0.4f, 0.78f);
	int_v = col_v;
	CHECK(int_v == Variant(Color(0.25f, 0.4f, 0.78f)));
	col_v = Color(0.33f, 0.75f, 0.21f);
	int_v = col_v;
	CHECK(int_v.get_type() == Variant::COLOR);

	Variant float_v = 0.0f;
	col_v = Color(0.25f, 0.4f, 0.78f);
	float_v = col_v;
	CHECK(float_v == Variant(Color(0.25f, 0.4f, 0.78f)));
	col_v = Color(0.33f, 0.75f, 0.21f);
	float_v = col_v;
	CHECK(float_v.get_type() == Variant::COLOR);

	Variant string_v = "";
	col_v = Color(0.25f, 0.4f, 0.78f);
	string_v = col_v;
	CHECK(string_v == Variant(Color(0.25f, 0.4f, 0.78f)));
	col_v = Color(0.33f, 0.75f, 0.21f);
	string_v = col_v;
	CHECK(string_v.get_type() == Variant::COLOR);

	Variant vec2_v = Vector2(0, 0);
	col_v = Color(0.25f, 0.4f, 0.78f);
	vec2_v = col_v;
	CHECK(vec2_v == Variant(Color(0.25f, 0.4f, 0.78f)));
	col_v = Color(0.33f, 0.75f, 0.21f);
	vec2_v = col_v;
	CHECK(vec2_v.get_type() == Variant::COLOR);

	Variant vec2i_v = Vector2i(0, 0);
	col_v = Color(0.25f, 0.4f, 0.78f);
	vec2i_v = col_v;
	CHECK(vec2i_v == Variant(Color(0.25f, 0.4f, 0.78f)));
	col_v = Color(0.33f, 0.75f, 0.21f);
	vec2i_v = col_v;
	CHECK(vec2i_v.get_type() == Variant::COLOR);

	Variant vec3_v = Vector3(0, 0, 0);
	col_v = Color(0.25f, 0.4f, 0.78f);
	vec3_v = col_v;
	CHECK(vec3_v == Variant(Color(0.25f, 0.4f, 0.78f)));
	col_v = Color(0.33f, 0.75f, 0.21f);
	vec3_v = col_v;
	CHECK(vec3_v.get_type() == Variant::COLOR);

	Variant vec3i_v = Vector3i(0, 0, 0);
	col_v = Color(0.25f, 0.4f, 0.78f);
	vec3i_v = col_v;
	CHECK(vec3i_v == Variant(Color(0.25f, 0.4f, 0.78f)));
	col_v = Color(0.33f, 0.75f, 0.21f);
	vec3i_v = col_v;
	CHECK(vec3i_v.get_type() == Variant::COLOR);
}
TEST_CASE("[Variant] Writer and parser array") {
	Array a = build_array(1, String("hello"), build_array(Variant()));
	String a_str;
	VariantWriter::write_to_string(a, a_str);

	CHECK_EQ(a_str, "[1, \"hello\", [null]]");

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

	CHECK_EQ(d_str, "{\n4: \"hello\",\n5: {\nnull: []\n},\n{\n1: 2\n}: 3\n}");

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
