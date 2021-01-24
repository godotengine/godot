/*************************************************************************/
/*  test_marshalls.h                                                     */
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

#ifndef TEST_MARSHALLS_H
#define TEST_MARSHALLS_H

#include "core/io/marshalls.h"

#include "tests/test_macros.h"

namespace TestMarshalls {

TEST_CASE("[Marshalls] Unsigned 16 bit integer encoding") {
	uint8_t arr[2];

	unsigned int actual_size = encode_uint16(0x1234, arr);
	CHECK(actual_size == sizeof(uint16_t));
	CHECK_MESSAGE(arr[0] == 0x34, "First encoded byte value should be equal to low order byte value.");
	CHECK_MESSAGE(arr[1] == 0x12, "Last encoded byte value should be equal to high order byte value.");
}

TEST_CASE("[Marshalls] Unsigned 32 bit integer encoding") {
	uint8_t arr[4];

	unsigned int actual_size = encode_uint32(0x12345678, arr);
	CHECK(actual_size == sizeof(uint32_t));
	CHECK_MESSAGE(arr[0] == 0x78, "First encoded byte value should be equal to low order byte value.");
	CHECK(arr[1] == 0x56);
	CHECK(arr[2] == 0x34);
	CHECK_MESSAGE(arr[3] == 0x12, "Last encoded byte value should be equal to high order byte value.");
}

TEST_CASE("[Marshalls] Unsigned 64 bit integer encoding") {
	uint8_t arr[8];

	unsigned int actual_size = encode_uint64(0x0f123456789abcdef, arr);
	CHECK(actual_size == sizeof(uint64_t));
	CHECK_MESSAGE(arr[0] == 0xef, "First encoded byte value should be equal to low order byte value.");
	CHECK(arr[1] == 0xcd);
	CHECK(arr[2] == 0xab);
	CHECK(arr[3] == 0x89);
	CHECK(arr[4] == 0x67);
	CHECK(arr[5] == 0x45);
	CHECK(arr[6] == 0x23);
	CHECK_MESSAGE(arr[7] == 0xf1, "Last encoded byte value should be equal to high order byte value.");
}

TEST_CASE("[Marshalls] Unsigned 16 bit integer decoding") {
	uint8_t arr[] = { 0x34, 0x12 };

	CHECK(decode_uint16(arr) == 0x1234);
}

TEST_CASE("[Marshalls] Unsigned 32 bit integer decoding") {
	uint8_t arr[] = { 0x78, 0x56, 0x34, 0x12 };

	CHECK(decode_uint32(arr) == 0x12345678);
}

TEST_CASE("[Marshalls] Unsigned 64 bit integer decoding") {
	uint8_t arr[] = { 0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0xf1 };

	CHECK(decode_uint64(arr) == 0x0f123456789abcdef);
}

TEST_CASE("[Marshalls] Floating point single precision encoding") {
	uint8_t arr[4];

	// Decimal: 0.15625
	// IEEE 754 single-precision binary floating-point format:
	// sign exponent (8 bits)    fraction (23 bits)
	//  0       01111100      01000000000000000000000
	// Hexadecimal: 0x3E200000
	unsigned int actual_size = encode_float(0.15625f, arr);
	CHECK(actual_size == sizeof(uint32_t));
	CHECK(arr[0] == 0x00);
	CHECK(arr[1] == 0x00);
	CHECK(arr[2] == 0x20);
	CHECK(arr[3] == 0x3e);
}

TEST_CASE("[Marshalls] Floating point double precision encoding") {
	uint8_t arr[8];

	// Decimal: 0.333333333333333314829616256247390992939472198486328125
	// IEEE 754 double-precision binary floating-point format:
	// sign exponent (11 bits)                  fraction (52 bits)
	//  0      01111111101     0101010101010101010101010101010101010101010101010101
	// Hexadecimal: 0x3FD5555555555555
	unsigned int actual_size = encode_double(0.33333333333333333, arr);
	CHECK(actual_size == sizeof(uint64_t));
	CHECK(arr[0] == 0x55);
	CHECK(arr[1] == 0x55);
	CHECK(arr[2] == 0x55);
	CHECK(arr[3] == 0x55);
	CHECK(arr[4] == 0x55);
	CHECK(arr[5] == 0x55);
	CHECK(arr[6] == 0xd5);
	CHECK(arr[7] == 0x3f);
}

TEST_CASE("[Marshalls] Floating point single precision decoding") {
	uint8_t arr[] = { 0x00, 0x00, 0x20, 0x3e };

	// See floating point encoding test case for details behind expected values
	CHECK(decode_float(arr) == 0.15625f);
}

TEST_CASE("[Marshalls] Floating point double precision decoding") {
	uint8_t arr[] = { 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0xd5, 0x3f };

	// See floating point encoding test case for details behind expected values
	CHECK(decode_double(arr) == 0.33333333333333333);
}

TEST_CASE("[Marshalls] C string encoding") {
	char cstring[] = "Godot"; // 5 characters
	uint8_t data[6];

	int actual_size = encode_cstring(cstring, data);
	CHECK(actual_size == 6);
	CHECK(data[0] == 'G');
	CHECK(data[1] == 'o');
	CHECK(data[2] == 'd');
	CHECK(data[3] == 'o');
	CHECK(data[4] == 't');
	CHECK(data[5] == '\0');
}

TEST_CASE("[Marshalls] NIL Variant encoding") {
	int r_len;
	Variant variant;
	uint8_t buffer[4];

	CHECK(encode_variant(variant, buffer, r_len) == OK);
	CHECK_MESSAGE(r_len == 4, "Length == 4 bytes for Variant::Type");
	CHECK_MESSAGE(buffer[0] == 0x00, "Variant::NIL");
	CHECK(buffer[1] == 0x00);
	CHECK(buffer[2] == 0x00);
	CHECK(buffer[3] == 0x00);
	// No value
}

TEST_CASE("[Marshalls] INT 32 bit Variant encoding") {
	int r_len;
	Variant variant(0x12345678);
	uint8_t buffer[8];

	CHECK(encode_variant(variant, buffer, r_len) == OK);
	CHECK_MESSAGE(r_len == 8, "Length == 4 bytes for Variant::Type + 4 bytes for int32_t");
	CHECK_MESSAGE(buffer[0] == 0x02, "Variant::INT");
	CHECK(buffer[1] == 0x00);
	CHECK(buffer[2] == 0x00);
	CHECK(buffer[3] == 0x00);
	// Check value
	CHECK(buffer[4] == 0x78);
	CHECK(buffer[5] == 0x56);
	CHECK(buffer[6] == 0x34);
	CHECK(buffer[7] == 0x12);
}

TEST_CASE("[Marshalls] INT 64 bit Variant encoding") {
	int r_len;
	Variant variant(uint64_t(0x0f123456789abcdef));
	uint8_t buffer[12];

	CHECK(encode_variant(variant, buffer, r_len) == OK);
	CHECK_MESSAGE(r_len == 12, "Length == 4 bytes for Variant::Type + 8 bytes for int64_t");
	CHECK_MESSAGE(buffer[0] == 0x02, "Variant::INT");
	CHECK(buffer[1] == 0x00);
	CHECK_MESSAGE(buffer[2] == 0x01, "ENCODE_FLAG_64");
	CHECK(buffer[3] == 0x00);
	// Check value
	CHECK(buffer[4] == 0xef);
	CHECK(buffer[5] == 0xcd);
	CHECK(buffer[6] == 0xab);
	CHECK(buffer[7] == 0x89);
	CHECK(buffer[8] == 0x67);
	CHECK(buffer[9] == 0x45);
	CHECK(buffer[10] == 0x23);
	CHECK(buffer[11] == 0xf1);
}

TEST_CASE("[Marshalls] FLOAT single precision Variant encoding") {
	int r_len;
	Variant variant(0.15625f);
	uint8_t buffer[8];

	CHECK(encode_variant(variant, buffer, r_len) == OK);
	CHECK_MESSAGE(r_len == 8, "Length == 4 bytes for Variant::Type + 4 bytes for float");
	CHECK_MESSAGE(buffer[0] == 0x03, "Variant::FLOAT");
	CHECK(buffer[1] == 0x00);
	CHECK(buffer[2] == 0x00);
	CHECK(buffer[3] == 0x00);
	// Check value
	CHECK(buffer[4] == 0x00);
	CHECK(buffer[5] == 0x00);
	CHECK(buffer[6] == 0x20);
	CHECK(buffer[7] == 0x3e);
}

TEST_CASE("[Marshalls] FLOAT double precision Variant encoding") {
	int r_len;
	Variant variant(0.33333333333333333);
	uint8_t buffer[12];

	CHECK(encode_variant(variant, buffer, r_len) == OK);
	CHECK_MESSAGE(r_len == 12, "Length == 4 bytes for Variant::Type + 8 bytes for double");
	CHECK_MESSAGE(buffer[0] == 0x03, "Variant::FLOAT");
	CHECK(buffer[1] == 0x00);
	CHECK_MESSAGE(buffer[2] == 0x01, "ENCODE_FLAG_64");
	CHECK(buffer[3] == 0x00);
	// Check value
	CHECK(buffer[4] == 0x55);
	CHECK(buffer[5] == 0x55);
	CHECK(buffer[6] == 0x55);
	CHECK(buffer[7] == 0x55);
	CHECK(buffer[8] == 0x55);
	CHECK(buffer[9] == 0x55);
	CHECK(buffer[10] == 0xd5);
	CHECK(buffer[11] == 0x3f);
}

TEST_CASE("[Marshalls] Invalid data Variant decoding") {
	Variant variant;
	int r_len = 0;
	uint8_t some_buffer[1] = { 0x00 };
	uint8_t out_of_range_type_buffer[4] = { 0xff }; // Greater than Variant::VARIANT_MAX

	CHECK(decode_variant(variant, some_buffer, /* less than 4 */ 1, &r_len) == ERR_INVALID_DATA);
	CHECK(r_len == 0);

	CHECK(decode_variant(variant, out_of_range_type_buffer, 4, &r_len) == ERR_INVALID_DATA);
	CHECK(r_len == 0);
}

TEST_CASE("[Marshalls] NIL Variant decoding") {
	Variant variant;
	int r_len;
	uint8_t buffer[] = {
		0x00, 0x00, 0x00, 0x00 // Variant::NIL
	};

	CHECK(decode_variant(variant, buffer, 4, &r_len) == OK);
	CHECK(r_len == 4);
	CHECK(variant == Variant());
}

TEST_CASE("[Marshalls] INT 32 bit Variant decoding") {
	Variant variant;
	int r_len;
	uint8_t buffer[] = {
		0x02, 0x00, 0x00, 0x00, // Variant::INT
		0x78, 0x56, 0x34, 0x12 // value
	};

	CHECK(decode_variant(variant, buffer, 8, &r_len) == OK);
	CHECK(r_len == 8);
	CHECK(variant == Variant(0x12345678));
}

TEST_CASE("[Marshalls] INT 64 bit Variant decoding") {
	Variant variant;
	int r_len;
	uint8_t buffer[] = {
		0x02, 0x00, 0x01, 0x00, // Variant::INT & ENCODE_FLAG_64
		0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0xf1 // value
	};

	CHECK(decode_variant(variant, buffer, 12, &r_len) == OK);
	CHECK(r_len == 12);
	CHECK(variant == Variant(uint64_t(0x0f123456789abcdef)));
}

TEST_CASE("[Marshalls] FLOAT single precision Variant decoding") {
	Variant variant;
	int r_len;
	uint8_t buffer[] = {
		0x03, 0x00, 0x00, 0x00, // Variant::FLOAT
		0x00, 0x00, 0x20, 0x3e // value
	};

	CHECK(decode_variant(variant, buffer, 8, &r_len) == OK);
	CHECK(r_len == 8);
	CHECK(variant == Variant(0.15625f));
}

TEST_CASE("[Marshalls] FLOAT double precision Variant decoding") {
	Variant variant;
	int r_len;
	uint8_t buffer[] = {
		0x03, 0x00, 0x01, 0x00, // Variant::FLOAT & ENCODE_FLAG_64
		0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0xd5, 0x3f // value
	};

	CHECK(decode_variant(variant, buffer, 12, &r_len) == OK);
	CHECK(r_len == 12);
	CHECK(variant == Variant(0.33333333333333333));
}
} // namespace TestMarshalls

#endif // TEST_MARSHALLS_H
