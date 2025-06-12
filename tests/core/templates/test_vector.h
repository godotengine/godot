/**************************************************************************/
/*  test_vector.h                                                         */
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

#pragma once

#include "core/templates/vector.h"

#include "tests/test_macros.h"

namespace TestVector {

TEST_CASE("[Vector] List initialization") {
	Vector<int> vector{ 0, 1, 2, 3, 4 };

	CHECK(vector.size() == 5);
	CHECK(vector[0] == 0);
	CHECK(vector[1] == 1);
	CHECK(vector[2] == 2);
	CHECK(vector[3] == 3);
	CHECK(vector[4] == 4);
}

TEST_CASE("[Vector] Push back and append") {
	Vector<int> vector;
	vector.push_back(0);
	vector.push_back(1);
	vector.push_back(2);
	vector.push_back(3);
	// Alias for `push_back`.
	vector.append(4);

	CHECK(vector[0] == 0);
	CHECK(vector[1] == 1);
	CHECK(vector[2] == 2);
	CHECK(vector[3] == 3);
	CHECK(vector[4] == 4);
}

TEST_CASE("[Vector] Append array") {
	Vector<int> vector;
	vector.push_back(1);
	vector.push_back(2);

	Vector<int> vector_other;
	vector_other.push_back(128);
	vector_other.push_back(129);
	vector.append_array(vector_other);

	CHECK(vector.size() == 4);
	CHECK(vector[0] == 1);
	CHECK(vector[1] == 2);
	CHECK(vector[2] == 128);
	CHECK(vector[3] == 129);
}

TEST_CASE("[Vector] Insert") {
	Vector<int> vector;
	vector.insert(0, 2);
	vector.insert(0, 8);
	vector.insert(2, 5);
	vector.insert(1, 5);
	vector.insert(0, -2);

	CHECK(vector.size() == 5);
	CHECK(vector[0] == -2);
	CHECK(vector[1] == 8);
	CHECK(vector[2] == 5);
	CHECK(vector[3] == 2);
	CHECK(vector[4] == 5);
}

TEST_CASE("[Vector] Ordered insert") {
	Vector<int> vector;
	vector.ordered_insert(2);
	vector.ordered_insert(8);
	vector.ordered_insert(5);
	vector.ordered_insert(5);
	vector.ordered_insert(-2);

	CHECK(vector.size() == 5);
	CHECK(vector[0] == -2);
	CHECK(vector[1] == 2);
	CHECK(vector[2] == 5);
	CHECK(vector[3] == 5);
	CHECK(vector[4] == 8);
}

TEST_CASE("[Vector] Insert + Ordered insert") {
	Vector<int> vector;
	vector.ordered_insert(2);
	vector.ordered_insert(8);
	vector.insert(0, 5);
	vector.ordered_insert(5);
	vector.insert(1, -2);

	CHECK(vector.size() == 5);
	CHECK(vector[0] == 5);
	CHECK(vector[1] == -2);
	CHECK(vector[2] == 2);
	CHECK(vector[3] == 5);
	CHECK(vector[4] == 8);
}

TEST_CASE("[Vector] Fill large array and modify it") {
	Vector<int> vector;
	vector.resize(1'000'000);
	vector.fill(0x60d07);

	vector.write[200] = 0;
	CHECK(vector.size() == 1'000'000);
	CHECK(vector[0] == 0x60d07);
	CHECK(vector[200] == 0);
	CHECK(vector[499'999] == 0x60d07);
	CHECK(vector[999'999] == 0x60d07);
	vector.remove_at(200);
	CHECK(vector[200] == 0x60d07);

	vector.clear();
	CHECK(vector.size() == 0);
}

TEST_CASE("[Vector] Copy creation") {
	Vector<int> vector;
	vector.push_back(0);
	vector.push_back(1);
	vector.push_back(2);
	vector.push_back(3);
	vector.push_back(4);

	Vector<int> vector_other = Vector<int>(vector);
	vector_other.remove_at(0);
	CHECK(vector_other[0] == 1);
	CHECK(vector_other[1] == 2);
	CHECK(vector_other[2] == 3);
	CHECK(vector_other[3] == 4);

	// Make sure the original vector isn't modified.
	CHECK(vector[0] == 0);
	CHECK(vector[1] == 1);
	CHECK(vector[2] == 2);
	CHECK(vector[3] == 3);
	CHECK(vector[4] == 4);
}

TEST_CASE("[Vector] Duplicate") {
	Vector<int> vector;
	vector.push_back(0);
	vector.push_back(1);
	vector.push_back(2);
	vector.push_back(3);
	vector.push_back(4);

	Vector<int> vector_other = vector.duplicate();
	vector_other.remove_at(0);
	CHECK(vector_other[0] == 1);
	CHECK(vector_other[1] == 2);
	CHECK(vector_other[2] == 3);
	CHECK(vector_other[3] == 4);

	// Make sure the original vector isn't modified.
	CHECK(vector[0] == 0);
	CHECK(vector[1] == 1);
	CHECK(vector[2] == 2);
	CHECK(vector[3] == 3);
	CHECK(vector[4] == 4);
}

TEST_CASE("[Vector] Get, set") {
	Vector<int> vector;
	vector.push_back(0);
	vector.push_back(1);
	vector.push_back(2);
	vector.push_back(3);
	vector.push_back(4);

	CHECK(vector.get(0) == 0);
	CHECK(vector.get(1) == 1);
	vector.set(2, 256);
	CHECK(vector.get(2) == 256);
	CHECK(vector.get(3) == 3);

	ERR_PRINT_OFF;
	// Invalid (but should not crash): setting out of bounds.
	vector.set(6, 500);
	ERR_PRINT_ON;

	CHECK(vector.get(4) == 4);
}

TEST_CASE("[Vector] To byte array (variant call)") {
	// PackedInt32Array.
	{
		PackedInt32Array vector[] = { { 0, -1, 2008 }, {} };
		PackedByteArray out[] = { { /* 0 */ 0x00, 0x00, 0x00, 0x00, /* -1 */ 0xFF, 0xFF, 0xFF, 0xFF, /* 2008 */ 0xD8, 0x07, 0x00, 0x00 }, {} };

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}

	// PackedInt64Array.
	{
		PackedInt64Array vector[] = { { 0, -1, 2008 }, {} };
		PackedByteArray out[] = { { /* 0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* -1 */ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, /* 2008 */ 0xD8, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, {} };

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}

	// PackedFloat32Array.
	{
		PackedFloat32Array vector[] = { { 0.0, -1.0, 200e24 }, {} };
		PackedByteArray out[] = { { /* 0.0 */ 0x00, 0x00, 0x00, 0x00, /* -1.0 */ 0x00, 0x00, 0x80, 0xBF, /* 200e24 */ 0xA6, 0x6F, 0x25, 0x6B }, {} };

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}
	// PackedFloat64Array.
	{
		PackedFloat64Array vector[] = { { 0.0, -1.0, 200e24 }, {} };
		PackedByteArray out[] = { { /* 0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* -1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xBF, /* 200e24 */ 0x35, 0x03, 0x32, 0xB7, 0xF4, 0xAD, 0x64, 0x45 }, {} };

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}

	// PackedStringArray.
	{
		PackedStringArray vector[] = { { "test", "string" }, {}, { "", "test" } };
		PackedByteArray out[] = { { /* test */ 0x74, 0x65, 0x73, 0x74, /* null */ 0x00, /* string */ 0x73, 0x74, 0x72, 0x69, 0x6E, 0x67, /* null */ 0x00 }, {}, { /* null */ 0x00, /* test */ 0x74, 0x65, 0x73, 0x74, /* null */ 0x00 } };

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}

	// PackedVector2Array.
	{
		PackedVector2Array vector[] = { { Vector2(), Vector2(1, -1) }, {} };
#ifdef REAL_T_IS_DOUBLE
		PackedByteArray out[] = { { /* X=0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* Y=0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* X=1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, /* Y=-1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xBF }, {} };
#else
		PackedByteArray out[] = { { /* X=0.0 */ 0x00, 0x00, 0x00, 0x00, /* Y=0.0 */ 0x00, 0x00, 0x00, 0x00, /* X=1.0 */ 0x00, 0x00, 0x80, 0x3F, /* Y=-1.0 */ 0x00, 0x00, 0x80, 0xBF }, {} };
#endif

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}

	// PackedVector3Array.
	{
		PackedVector3Array vector[] = { { Vector3(), Vector3(1, 1, -1) }, {} };
#ifdef REAL_T_IS_DOUBLE
		PackedByteArray out[] = { { /* X=0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* Y=0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* Z=0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* X=1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, /* Y=1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, /* Z=-1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xBF }, {} };
#else
		PackedByteArray out[] = { { /* X=0.0 */ 0x00, 0x00, 0x00, 0x00, /* Y=0.0 */ 0x00, 0x00, 0x00, 0x00, /* Z=0.0 */ 0x00, 0x00, 0x00, 0x00, /* X=1.0 */ 0x00, 0x00, 0x80, 0x3F, /* Y=1.0 */ 0x00, 0x00, 0x80, 0x3F, /* Z=-1.0 */ 0x00, 0x00, 0x80, 0xBF }, {} };
#endif

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}

	// PackedColorArray.
	{
		PackedColorArray vector[] = { { Color(), Color(1, 1, 1) }, {} };
		PackedByteArray out[] = { { /* R=0.0 */ 0x00, 0x00, 0x00, 0x00, /* G=0.0 */ 0x00, 0x00, 0x00, 0x00, /* B=0.0 */ 0x00, 0x00, 0x00, 0x00, /* A=1.0 */ 0x00, 0x00, 0x80, 0x3F, /* R=1.0 */ 0x00, 0x00, 0x80, 0x3F, /* G=1.0 */ 0x00, 0x00, 0x80, 0x3F, /* B=1.0 */ 0x00, 0x00, 0x80, 0x3F, /* A=1.0 */ 0x00, 0x00, 0x80, 0x3F }, {} };

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}

	// PackedVector4Array.
	{
		PackedVector4Array vector[] = { { Vector4(), Vector4(1, -1, 1, -1) }, {} };
#ifdef REAL_T_IS_DOUBLE
		PackedByteArray out[] = { { /* X=0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* Y=0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* Z 0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* W=0.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* X=1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, /* Y=-1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xBF, /* Z=1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, /* W=-1.0 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xBF }, {} };
#else
		PackedByteArray out[] = { { /* X=0.0 */ 0x00, 0x00, 0x00, 0x00, /* Y=0.0 */ 0x00, 0x00, 0x00, 0x00, /* Z=0.0 */ 0x00, 0x00, 0x00, 0x00, /* W 0.0 */ 0x00, 0x00, 0x00, 0x00, /* X 1.0 */ 0x00, 0x00, 0x80, 0x3F, /* Y=-1.0 */ 0x00, 0x00, 0x80, 0xBF, /* Z=1.0 */ 0x00, 0x00, 0x80, 0x3F, /* W=-1.0 */ 0x00, 0x00, 0x80, 0xBF }, {} };
#endif

		for (size_t i = 0; i < std::size(vector); i++) {
			Callable::CallError err;
			Variant v_ret;
			Variant v_vector = vector[i];
			v_vector.callp("to_byte_array", nullptr, 0, v_ret, err);
			CHECK(v_ret.get_type() == Variant::PACKED_BYTE_ARRAY);
			CHECK(v_ret.operator PackedByteArray() == out[i]);
		}
	}
}

TEST_CASE("[Vector] To byte array") {
	Vector<int> vector;
	vector.push_back(0);
	vector.push_back(-1);
	vector.push_back(2008);
	vector.push_back(999999999);

	Vector<uint8_t> byte_array = vector.to_byte_array();
	CHECK(byte_array.size() == 16);
	// vector[0]
	CHECK(byte_array[0] == 0);
	CHECK(byte_array[1] == 0);
	CHECK(byte_array[2] == 0);
	CHECK(byte_array[3] == 0);

	// vector[1]
	CHECK(byte_array[4] == 255);
	CHECK(byte_array[5] == 255);
	CHECK(byte_array[6] == 255);
	CHECK(byte_array[7] == 255);

	// vector[2]
	CHECK(byte_array[8] == 216);
	CHECK(byte_array[9] == 7);
	CHECK(byte_array[10] == 0);
	CHECK(byte_array[11] == 0);

	// vector[3]
	CHECK(byte_array[12] == 255);
	CHECK(byte_array[13] == 201);
	CHECK(byte_array[14] == 154);
	CHECK(byte_array[15] == 59);
}

TEST_CASE("[Vector] PackedByteArray to PackedInt32Array") {
	PackedByteArray byte_array = {
		0x00, 0x00, 0x00, 0x00, // 0
		0xff, 0xff, 0xff, 0xff, // -1
		0xd8, 0x07, 0x00, 0x00, // 2008
		0xff, 0xc9, 0x9a, 0x3b, // 999999999
		0xff, 0xff, 0xff, 0x7f, // INT32_MAX
		0x00, 0x00, 0x00, 0x80, // INT32_MIN
	};

	PackedInt32Array int_array = Variant(byte_array).call("to_int32_array");

	CHECK(int_array[0] == 0);
	CHECK(int_array[1] == -1);
	CHECK(int_array[2] == 2008);
	CHECK(int_array[3] == 999999999);
	CHECK(int_array[4] == INT32_MAX);
	CHECK(int_array[5] == INT32_MIN);
}

TEST_CASE("[Vector] PackedByteArray to PackedVector2Array") {
	PackedByteArray vector2_byte_array = {
#ifdef REAL_T_IS_DOUBLE
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 0, 0
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xbf, // 1, -1
		0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x60, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x59, 0xc0, // 128.5, -100.125
		0x38, 0xdf, 0x06, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x90, 0x41, 0xf2, 0xff, 0xff, 0xff, 0xef, 0x3f, // 1.0 +/- CMP_EPSILON2
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xff, // +/- infinity
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, 0x20, 0x10, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, // various NaNs
#else
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 0, 0
		0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf, // 1, -1
		0x00, 0x80, 0x00, 0x43, 0x00, 0x40, 0xc8, 0xc2, // 128.5, -100.125
		0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, // 1.0 +/- CMP_EPSILON2
		0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0xff, // +/- infinity
		0x00, 0x00, 0xc0, 0x7f, 0x20, 0x10, 0xc0, 0x7f, // various NaNs
#endif // REAL_T_IS_DOUBLE
	};

	PackedVector2Array vector2_array = Variant(vector2_byte_array).call("to_vector2_array");
	CHECK(vector2_array[0] == Vector2{ 0, 0 });
	CHECK(vector2_array[1] == Vector2{ 1, -1 });
	CHECK(vector2_array[2] == Vector2{ 128.5, -100.125 });
	CHECK(vector2_array[3] == Vector2{ 1.0 + CMP_EPSILON2, 1.0 - CMP_EPSILON2 });
	CHECK(vector2_array[4] == Vector2{ INFINITY, -(INFINITY) });
	CHECK(isnan(vector2_array[5].x));
	CHECK(isnan(vector2_array[5].y));
}

TEST_CASE("[Vector] PackedByteArray to PackedVector3Array") {
	PackedByteArray vector3_byte_array = {
#ifdef REAL_T_IS_DOUBLE
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1, -1, 0
		0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x60, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x59, 0xc0, 0x0b, 0xb5, 0xa6, 0xf9, 0x81, 0xb3, 0xf5, 0x40, // 128.5, -100.125, 88888.12345
		0xbc, 0xbd, 0xd7, 0xd9, 0xdf, 0x7c, 0xdb, 0x3d, 0x38, 0xdf, 0x06, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x90, 0x41, 0xf2, 0xff, 0xff, 0xff, 0xef, 0x3f, // CMP_EPSILON2, 1.0 +/- CMP_EPSILON2
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, // +/- infinity, NaN
#else
		0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, // 1, -1, 0
		0x00, 0x80, 0x00, 0x43, 0x00, 0x40, 0xc8, 0xc2, 0x10, 0x9c, 0xad, 0x47, // 128.5, -100.125, 88888.12345
		0xff, 0xe6, 0xdb, 0x2e, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, // CMP_EPSILON2, 1.0 +/- CMP_EPSILON2
		0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0xc0, 0x7f, // +/- infinity, NaN
#endif // REAL_T_IS_DOUBLE
	};

	PackedVector3Array vector3_array = Variant(vector3_byte_array).call("to_vector3_array");
	CHECK(vector3_array[0] == Vector3{ 1, -1, 0 });
	CHECK(vector3_array[1] == Vector3{ 128.5, -100.125, 88888.12345 });
	CHECK(vector3_array[2] == Vector3{ CMP_EPSILON2, 1.0 + CMP_EPSILON2, 1.0 - CMP_EPSILON2 });
	CHECK(vector3_array[3].x == INFINITY);
	CHECK(vector3_array[3].y == -(INFINITY));
	CHECK(isnan(vector3_array[3].z));
}

TEST_CASE("[Vector] PackedByteArray to PackedColorArray") {
	PackedByteArray color_byte_array = {
		0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1, -1, 0, -0
		0x00, 0x80, 0x00, 0x43, 0x00, 0x40, 0xc8, 0xc2, 0x10, 0x9c, 0xad, 0x47, 0x05, 0xc5, 0x77, 0xbf, // 128.5, -100.125, 88888.12345, -0.96785
		0xff, 0xe6, 0xdb, 0x2e, 0xff, 0xe6, 0xdb, 0xae, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, // +/- CMP_EPSILON2, 1.0 +/- CMP_EPSILON2
		0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0xc0, 0x7f, 0x20, 0x10, 0xc0, 0x7f, // +/- infinity, various NaNs
	};

	PackedColorArray color_array = Variant(color_byte_array).call("to_color_array");
	CHECK(color_array[0] == Color{ 1, -1, 0, 0 });
	CHECK(color_array[1] == Color{ 128.5, -100.125, 88888.12345, -0.96785 });
	CHECK(color_array[2] == Color{ CMP_EPSILON2, -(CMP_EPSILON2), 1.0 + CMP_EPSILON2, 1.0 - CMP_EPSILON2 });
	CHECK(color_array[3].r == INFINITY);
	CHECK(color_array[3].g == -(INFINITY));
	CHECK(isnan(color_array[3].b));
	CHECK(isnan(color_array[3].a));
}

TEST_CASE("[Vector] Slice") {
	Vector<int> vector;
	vector.push_back(0);
	vector.push_back(1);
	vector.push_back(2);
	vector.push_back(3);
	vector.push_back(4);

	Vector<int> slice0 = vector.slice(0, 0);
	CHECK(slice0.size() == 0);

	Vector<int> slice1 = vector.slice(1, 3);
	CHECK(slice1.size() == 2);
	CHECK(slice1[0] == 1);
	CHECK(slice1[1] == 2);

	Vector<int> slice2 = vector.slice(1, -1);
	CHECK(slice2.size() == 3);
	CHECK(slice2[0] == 1);
	CHECK(slice2[1] == 2);
	CHECK(slice2[2] == 3);

	Vector<int> slice3 = vector.slice(3);
	CHECK(slice3.size() == 2);
	CHECK(slice3[0] == 3);
	CHECK(slice3[1] == 4);

	Vector<int> slice4 = vector.slice(2, -2);
	CHECK(slice4.size() == 1);
	CHECK(slice4[0] == 2);

	Vector<int> slice5 = vector.slice(-2);
	CHECK(slice5.size() == 2);
	CHECK(slice5[0] == 3);
	CHECK(slice5[1] == 4);

	Vector<int> slice6 = vector.slice(2, 42);
	CHECK(slice6.size() == 3);
	CHECK(slice6[0] == 2);
	CHECK(slice6[1] == 3);
	CHECK(slice6[2] == 4);

	ERR_PRINT_OFF;
	Vector<int> slice7 = vector.slice(5, 1);
	CHECK(slice7.size() == 0); // Expected to fail.
	ERR_PRINT_ON;
}

TEST_CASE("[Vector] Find, has") {
	Vector<int> vector;
	vector.push_back(3);
	vector.push_back(1);
	vector.push_back(4);
	vector.push_back(0);
	vector.push_back(2);

	CHECK(vector[0] == 3);
	CHECK(vector[1] == 1);
	CHECK(vector[2] == 4);
	CHECK(vector[3] == 0);
	CHECK(vector[4] == 2);

	CHECK(vector.find(0) == 3);
	CHECK(vector.find(1) == 1);
	CHECK(vector.find(2) == 4);
	CHECK(vector.find(3) == 0);
	CHECK(vector.find(4) == 2);

	CHECK(vector.find(-1) == -1);
	CHECK(vector.find(5) == -1);

	CHECK(vector.has(0));
	CHECK(vector.has(1));
	CHECK(vector.has(2));
	CHECK(vector.has(3));
	CHECK(vector.has(4));

	CHECK(!vector.has(-1));
	CHECK(!vector.has(5));
}

TEST_CASE("[Vector] Remove at") {
	Vector<int> vector;
	vector.push_back(0);
	vector.push_back(1);
	vector.push_back(2);
	vector.push_back(3);
	vector.push_back(4);

	vector.remove_at(0);

	CHECK(vector[0] == 1);
	CHECK(vector[1] == 2);
	CHECK(vector[2] == 3);
	CHECK(vector[3] == 4);

	vector.remove_at(2);

	CHECK(vector[0] == 1);
	CHECK(vector[1] == 2);
	CHECK(vector[2] == 4);

	vector.remove_at(1);

	CHECK(vector[0] == 1);
	CHECK(vector[1] == 4);

	vector.remove_at(0);

	CHECK(vector[0] == 4);
}

TEST_CASE("[Vector] Remove at and find") {
	Vector<int> vector;
	vector.push_back(0);
	vector.push_back(1);
	vector.push_back(2);
	vector.push_back(3);
	vector.push_back(4);

	CHECK(vector.size() == 5);

	vector.remove_at(0);

	CHECK(vector.size() == 4);

	CHECK(vector.find(0) == -1);
	CHECK(vector.find(1) != -1);
	CHECK(vector.find(2) != -1);
	CHECK(vector.find(3) != -1);
	CHECK(vector.find(4) != -1);

	vector.remove_at(vector.find(3));

	CHECK(vector.size() == 3);

	CHECK(vector.find(3) == -1);
	CHECK(vector.find(1) != -1);
	CHECK(vector.find(2) != -1);
	CHECK(vector.find(4) != -1);

	vector.remove_at(vector.find(2));

	CHECK(vector.size() == 2);

	CHECK(vector.find(2) == -1);
	CHECK(vector.find(1) != -1);
	CHECK(vector.find(4) != -1);

	vector.remove_at(vector.find(4));

	CHECK(vector.size() == 1);

	CHECK(vector.find(4) == -1);
	CHECK(vector.find(1) != -1);

	vector.remove_at(0);

	CHECK(vector.is_empty());
	CHECK(vector.size() == 0);
}

TEST_CASE("[Vector] Erase") {
	Vector<int> vector;
	vector.push_back(1);
	vector.push_back(3);
	vector.push_back(0);
	vector.push_back(2);
	vector.push_back(4);

	CHECK(vector.find(2) == 3);

	vector.erase(2);

	CHECK(vector.find(2) == -1);
	CHECK(vector.size() == 4);
}

TEST_CASE("[Vector] Size, resize, reserve") {
	Vector<int> vector;
	CHECK(vector.is_empty());
	CHECK(vector.size() == 0);

	vector.resize(10);

	CHECK(vector.size() == 10);

	vector.resize(5);

	CHECK(vector.size() == 5);

	vector.remove_at(0);
	vector.remove_at(0);
	vector.remove_at(0);

	CHECK(vector.size() == 2);

	vector.clear();

	CHECK(vector.size() == 0);
	CHECK(vector.is_empty());

	vector.push_back(0);
	vector.push_back(0);
	vector.push_back(0);

	CHECK(vector.size() == 3);

	vector.push_back(0);

	CHECK(vector.size() == 4);
}

TEST_CASE("[Vector] Sort") {
	Vector<int> vector;
	vector.push_back(2);
	vector.push_back(8);
	vector.push_back(-4);
	vector.push_back(5);
	vector.sort();

	CHECK(vector.size() == 4);
	CHECK(vector[0] == -4);
	CHECK(vector[1] == 2);
	CHECK(vector[2] == 5);
	CHECK(vector[3] == 8);
}

TEST_CASE("[Vector] Sort custom") {
	Vector<String> vector;
	vector.push_back("world");
	vector.push_back("World");
	vector.push_back("Hello");
	vector.push_back("10Hello");
	vector.push_back("12Hello");
	vector.push_back("01Hello");
	vector.push_back("1Hello");
	vector.push_back(".Hello");
	vector.sort_custom<NaturalNoCaseComparator>();

	CHECK(vector.size() == 8);
	CHECK(vector[0] == ".Hello");
	CHECK(vector[1] == "01Hello");
	CHECK(vector[2] == "1Hello");
	CHECK(vector[3] == "10Hello");
	CHECK(vector[4] == "12Hello");
	CHECK(vector[5] == "Hello");
	CHECK(vector[6] == "world");
	CHECK(vector[7] == "World");
}

TEST_CASE("[Vector] Search") {
	Vector<int> vector;
	vector.push_back(1);
	vector.push_back(2);
	vector.push_back(3);
	vector.push_back(5);
	vector.push_back(8);
	CHECK(vector.bsearch(2, true) == 1);
	CHECK(vector.bsearch(2, false) == 2);
	CHECK(vector.bsearch(5, true) == 3);
	CHECK(vector.bsearch(5, false) == 4);
}

TEST_CASE("[Vector] Operators") {
	Vector<int> vector;
	vector.push_back(2);
	vector.push_back(8);
	vector.push_back(-4);
	vector.push_back(5);

	Vector<int> vector_other;
	vector_other.push_back(2);
	vector_other.push_back(8);
	vector_other.push_back(-4);
	vector_other.push_back(5);

	CHECK(vector == vector_other);

	vector_other.push_back(10);
	CHECK(vector != vector_other);
}

struct CyclicVectorHolder {
	Vector<CyclicVectorHolder> *vector = nullptr;
	bool is_destructing = false;

	~CyclicVectorHolder() {
		if (is_destructing) {
			// The vector must exist and not expose its backing array at this point.
			CHECK_NE(vector, nullptr);
			CHECK_EQ(vector->ptr(), nullptr);
		}
	}
};

TEST_CASE("[Vector] Cyclic Reference") {
	// Create a stack-space vector.
	Vector<CyclicVectorHolder> vector;
	// Add a new (empty) element.
	vector.resize(1);
	// Expose the vector to its element through CyclicVectorHolder.
	// This is questionable behavior, but should still behave graciously.
	vector.ptrw()[0] = CyclicVectorHolder{ &vector };
	vector.ptrw()[0].is_destructing = true;
	// The vector goes out of scope and destructs, calling CyclicVectorHolder's destructor.
}

} // namespace TestVector
