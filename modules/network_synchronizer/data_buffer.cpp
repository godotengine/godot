/*************************************************************************/
/*  data_buffer.cpp                                                      */
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

/**
	@author AndreaCatania
*/

#include "data_buffer.h"

#include "core/io/marshalls.h"

// TODO improve the allocation mechanism.

void DataBuffer::_bind_methods() {
	BIND_CONSTANT(DATA_TYPE_BOOL);
	BIND_CONSTANT(DATA_TYPE_INT);
	BIND_CONSTANT(DATA_TYPE_REAL);
	BIND_CONSTANT(DATA_TYPE_UNIT_REAL);
	BIND_CONSTANT(DATA_TYPE_VECTOR2);
	BIND_CONSTANT(DATA_TYPE_NORMALIZED_VECTOR2);
	BIND_CONSTANT(DATA_TYPE_VECTOR3);
	BIND_CONSTANT(DATA_TYPE_NORMALIZED_VECTOR3);

	BIND_CONSTANT(COMPRESSION_LEVEL_0);
	BIND_CONSTANT(COMPRESSION_LEVEL_1);
	BIND_CONSTANT(COMPRESSION_LEVEL_2);
	BIND_CONSTANT(COMPRESSION_LEVEL_3);

	ClassDB::bind_method(D_METHOD("size"), &DataBuffer::size);

	ClassDB::bind_method(D_METHOD("add_bool", "value"), &DataBuffer::add_bool);
	ClassDB::bind_method(D_METHOD("add_int", "value", "compression_level"), &DataBuffer::add_int, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_real", "value", "compression_level"), &DataBuffer::add_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_positive_unit_real", "value", "compression_level"), &DataBuffer::add_positive_unit_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_unit_real", "value", "compression_level"), &DataBuffer::add_unit_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_vector2", "value", "compression_level"), &DataBuffer::add_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_normalized_vector2", "value", "compression_level"), &DataBuffer::add_normalized_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_vector3", "value", "compression_level"), &DataBuffer::add_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_normalized_vector3", "value", "compression_level"), &DataBuffer::add_normalized_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_variant", "value"), &DataBuffer::add_variant);

	ClassDB::bind_method(D_METHOD("read_bool"), &DataBuffer::read_bool);
	ClassDB::bind_method(D_METHOD("read_int", "compression_level"), &DataBuffer::read_int, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_real", "compression_level"), &DataBuffer::read_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_unit_real", "compression_level"), &DataBuffer::read_unit_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_vector2", "compression_level"), &DataBuffer::read_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_normalized_vector2", "compression_level"), &DataBuffer::read_normalized_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_vector3", "compression_level"), &DataBuffer::read_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_normalized_vector3", "compression_level"), &DataBuffer::read_normalized_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_variant"), &DataBuffer::read_variant);

	ClassDB::bind_method(D_METHOD("skip_bool"), &DataBuffer::skip_bool);
	ClassDB::bind_method(D_METHOD("skip_int", "compression_level"), &DataBuffer::skip_int, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_real", "compression_level"), &DataBuffer::skip_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_unit_real", "compression_level"), &DataBuffer::skip_unit_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_vector2", "compression_level"), &DataBuffer::skip_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_normalized_vector2", "compression_level"), &DataBuffer::skip_normalized_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_vector3", "compression_level"), &DataBuffer::skip_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_normalized_vector3", "compression_level"), &DataBuffer::skip_normalized_vector3, DEFVAL(COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("get_bool_size"), &DataBuffer::get_bool_size);
	ClassDB::bind_method(D_METHOD("get_int_size", "compression_level"), &DataBuffer::get_int_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_real_size", "compression_level"), &DataBuffer::get_real_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_unit_real_size", "compression_level"), &DataBuffer::get_unit_real_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_vector2_size", "compression_level"), &DataBuffer::get_vector2_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_normalized_vector2_size", "compression_level"), &DataBuffer::get_normalized_vector2_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_vector3_size", "compression_level"), &DataBuffer::get_vector3_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_normalized_vector3_size", "compression_level"), &DataBuffer::get_normalized_vector3_size, DEFVAL(COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("read_bool_size"), &DataBuffer::read_bool_size);
	ClassDB::bind_method(D_METHOD("read_int_size", "compression_level"), &DataBuffer::read_int_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_real_size", "compression_level"), &DataBuffer::read_real_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_unit_real_size", "compression_level"), &DataBuffer::read_unit_real_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_vector2_size", "compression_level"), &DataBuffer::read_vector2_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_normalized_vector2_size", "compression_level"), &DataBuffer::read_normalized_vector2_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_vector3_size", "compression_level"), &DataBuffer::read_vector3_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_normalized_vector3_size", "compression_level"), &DataBuffer::read_normalized_vector3_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_variant_size"), &DataBuffer::read_variant_size);

	ClassDB::bind_method(D_METHOD("begin_read"), &DataBuffer::begin_read);
	ClassDB::bind_method(D_METHOD("begin_write", "meta_size"), &DataBuffer::begin_write);
	ClassDB::bind_method(D_METHOD("dry"), &DataBuffer::dry);
}

DataBuffer::DataBuffer(const DataBuffer &p_other) :
		Object(),
		metadata_size(p_other.metadata_size),
		bit_offset(p_other.bit_offset),
		bit_size(p_other.bit_size),
		is_reading(p_other.is_reading),
		buffer(p_other.buffer) {}

DataBuffer::DataBuffer(const BitArray &p_buffer) :
		Object(),
		bit_size(p_buffer.size_in_bits()),
		is_reading(true),
		buffer(p_buffer) {}

void DataBuffer::begin_write(int p_metadata_size) {
	metadata_size = p_metadata_size;
	bit_size = 0;
	bit_offset = 0;
	is_reading = false;
}

void DataBuffer::dry() {
	buffer.resize_in_bits(metadata_size + bit_size);
}

void DataBuffer::seek(int p_bits) {
	ERR_FAIL_COND((metadata_size + bit_size) < p_bits);
	bit_offset = p_bits;
}

void DataBuffer::force_set_size(int p_metadata_bit_size, int p_bit_size) {
	ERR_FAIL_COND_MSG(buffer.size_in_bits() < (p_metadata_bit_size + p_bit_size), "The buffer is smaller than the new given size.");
	metadata_size = p_metadata_bit_size;
	bit_size = p_bit_size;
}

int DataBuffer::get_metadata_size() const {
	return metadata_size;
}

int DataBuffer::size() const {
	return bit_size;
}

int DataBuffer::total_size() const {
	return bit_size + metadata_size;
}

int DataBuffer::get_bit_offset() const {
	return bit_offset;
}

void DataBuffer::skip(int p_bits) {
	ERR_FAIL_COND((metadata_size + bit_size) < (bit_offset + p_bits));
	bit_offset += p_bits;
}

void DataBuffer::begin_read() {
	bit_offset = 0;
	is_reading = true;
}

bool DataBuffer::add_bool(bool p_input) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const int bits = get_bit_taken(DATA_TYPE_BOOL, COMPRESSION_LEVEL_0);

	make_room_in_bits(bits);
	buffer.store_bits(bit_offset, p_input, bits);
	bit_offset += bits;

#ifdef DEBUG_ENABLED
	// Can't never happen because the buffer size is correctly handled.
	CRASH_COND((metadata_size + bit_size) > buffer.size_in_bits() && bit_offset > buffer.size_in_bits());
#endif

	return p_input;
}

bool DataBuffer::read_bool() {
	ERR_FAIL_COND_V(is_reading == false, false);

	const int bits = get_bit_taken(DATA_TYPE_BOOL, COMPRESSION_LEVEL_0);
	const bool d = buffer.read_bits(bit_offset, bits);
	bit_offset += bits;
	return d;
}

int64_t DataBuffer::add_int(int64_t p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const int bits = get_bit_taken(DATA_TYPE_INT, p_compression_level);

	int64_t value = p_input;

	// Clamp the value to the max that the bit can store.
	if (bits == 8) {
		value = CLAMP(value, INT8_MIN, INT8_MAX);
	} else if (bits == 16) {
		value = CLAMP(value, INT16_MIN, INT16_MAX);
	} else if (bits == 32) {
		value = CLAMP(value, INT32_MIN, INT32_MAX);
	} else {
		// Nothing to do here
	}

	make_room_in_bits(bits);
	buffer.store_bits(bit_offset, value, bits);
	bit_offset += bits;

#ifdef DEBUG_ENABLED
	// Can't never happen because the buffer size is correctly handled.
	CRASH_COND((metadata_size + bit_size) > buffer.size_in_bits() && bit_offset > buffer.size_in_bits());
#endif

	if (bits == 8) {
		return static_cast<int8_t>(value);
	} else if (bits == 16) {
		return static_cast<int16_t>(value);
	} else if (bits == 32) {
		return static_cast<int32_t>(value);
	} else {
		return value;
	}
}

int64_t DataBuffer::read_int(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, 0);

	const int bits = get_bit_taken(DATA_TYPE_INT, p_compression_level);

	const uint64_t value = buffer.read_bits(bit_offset, bits);
	bit_offset += bits;

	if (bits == 8) {
		return static_cast<int8_t>(value);
	} else if (bits == 16) {
		return static_cast<int16_t>(value);
	} else if (bits == 32) {
		return static_cast<int32_t>(value);
	} else {
		return static_cast<int64_t>(value);
	}
}

double DataBuffer::add_real(double p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	// Clamp the input value according to the compression level
	// Minifloat (compression level 0) have a special bias
	const int exponent_bits = get_exponent_bits(p_compression_level);
	const int mantissa_bits = get_mantissa_bits(p_compression_level);
	const double bias = p_compression_level == COMPRESSION_LEVEL_3 ? Math::pow(2.0, exponent_bits) - 3 : Math::pow(2.0, exponent_bits - 1) - 1;
	const double max_value = (2.0 - Math::pow(2.0, -(mantissa_bits - 1))) * Math::pow(2.0, bias);
	const double clamped_input = CLAMP(p_input, -max_value, max_value);

	// Split number according to IEEE 754 binary format.
	// Mantissa floating point value represented in range (-1;-0.5], [0.5; 1).
	int exponent;
	double mantissa = frexp(clamped_input, &exponent);

	// Extract sign.
	const bool sign = mantissa < 0;
	mantissa = Math::abs(mantissa);

	// Round mantissa into the specified number of bits (like float -> double conversion).
	double mantissa_scale = Math::pow(2.0, mantissa_bits);
	if (exponent <= 0) {
		// Subnormal value, apply exponent to mantissa and reduce power of scale by one.
		mantissa *= Math::pow(2.0, exponent);
		exponent = 0;
		mantissa_scale /= 2.0;
	}
	mantissa = round(mantissa * mantissa_scale) / mantissa_scale; // Round to specified number of bits. Math::round currently have an overflow with max double.
	if (mantissa < 0.5 && mantissa != 0) {
		// Check underflow, extract exponent from mantissa.
		exponent += ilogb(mantissa) + 1;
		mantissa /= Math::pow(2.0, exponent);
	} else if (mantissa == 1) {
		// Check overflow, increment the exponent.
		++exponent;
		mantissa = 0.5;
	}
	// Convert the mantissa to an integer that represents the offset index (IEE 754 floating point representation) to send over network safely.
	const uint64_t integer_mantissa = exponent <= 0 ? mantissa * mantissa_scale * Math::pow(2.0, exponent) : (mantissa - 0.5) * mantissa_scale;

	make_room_in_bits(mantissa_bits + exponent_bits);
	buffer.store_bits(bit_offset, sign, 1);
	bit_offset += 1;
	buffer.store_bits(bit_offset, integer_mantissa, mantissa_bits - 1);
	bit_offset += mantissa_bits - 1;
	// Send unsigned value (just shift it by bias) to avoid sign issues.
	buffer.store_bits(bit_offset, exponent + bias, exponent_bits);
	bit_offset += exponent_bits;

	return ldexp(sign ? -mantissa : mantissa, exponent);
}

double DataBuffer::read_real(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, 0.0);

	const bool sign = buffer.read_bits(bit_offset, 1);
	bit_offset += 1;

	const int mantissa_bits = get_mantissa_bits(p_compression_level);
	const uint64_t integer_mantissa = buffer.read_bits(bit_offset, mantissa_bits - 1);
	bit_offset += mantissa_bits - 1;

	const int exponent_bits = get_exponent_bits(p_compression_level);
	const double bias = p_compression_level == COMPRESSION_LEVEL_3 ? Math::pow(2.0, exponent_bits) - 3 : Math::pow(2.0, exponent_bits - 1) - 1;
	int exponent = static_cast<int>(buffer.read_bits(bit_offset, exponent_bits)) - static_cast<int>(bias);
	bit_offset += exponent_bits;

	// Convert integer mantissa into the floating point representation
	// When the index of the mantissa and exponent are 0, then this is a special case and the mantissa is 0.
	const double mantissa_scale = Math::pow(2.0, exponent <= 0 ? mantissa_bits - 1 : mantissa_bits);
	const double mantissa = exponent <= 0 ? integer_mantissa / mantissa_scale / Math::pow(2.0, exponent) : integer_mantissa / mantissa_scale + 0.5;

	return ldexp(sign ? -mantissa : mantissa, exponent);
}

real_t DataBuffer::add_positive_unit_real(real_t p_input, CompressionLevel p_compression_level) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V_MSG(p_input < 0 || p_input > 1, p_input, "Value must be between zero and one.");
#endif
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const int bits = get_bit_taken(DATA_TYPE_UNIT_REAL, p_compression_level);

	const double max_value = static_cast<double>(~(UINT64_MAX << bits));

	const uint64_t compressed_val = compress_unit_float(p_input, max_value);

	make_room_in_bits(bits);
	buffer.store_bits(bit_offset, compressed_val, bits);
	bit_offset += bits;

#ifdef DEBUG_ENABLED
	// Can't never happen because the buffer size is correctly handled.
	CRASH_COND((metadata_size + bit_size) > buffer.size_in_bits() && bit_offset > buffer.size_in_bits());
#endif

	return decompress_unit_float(compressed_val, max_value);
}

real_t DataBuffer::read_positive_unit_real(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, 0.0);

	const int bits = get_bit_taken(DATA_TYPE_UNIT_REAL, p_compression_level);

	const double max_value = static_cast<double>(~(UINT64_MAX << bits));

	const uint64_t compressed_val = buffer.read_bits(bit_offset, bits);
	bit_offset += bits;

	return decompress_unit_float(compressed_val, max_value);
}

real_t DataBuffer::add_unit_real(real_t p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const real_t added_real = add_positive_unit_real(ABS(p_input), p_compression_level);

	const int bits_for_sign = 1;
	const uint32_t is_negative = p_input < 0.0;
	make_room_in_bits(bits_for_sign);
	buffer.store_bits(bit_offset, is_negative, bits_for_sign);
	bit_offset += bits_for_sign;

#ifdef DEBUG_ENABLED
	// Can't never happen because the buffer size is correctly handled.
	CRASH_COND((metadata_size + bit_size) > buffer.size_in_bits() && bit_offset > buffer.size_in_bits());
#endif

	return is_negative ? -added_real : added_real;
}

real_t DataBuffer::read_unit_real(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, 0.0);

	const real_t value = read_positive_unit_real(p_compression_level);

	const int bits_for_sign = 1;
	const bool is_negative = buffer.read_bits(bit_offset, bits_for_sign);
	bit_offset += bits_for_sign;

	return is_negative ? -value : value;
}

Vector2 DataBuffer::add_vector2(Vector2 p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

#ifndef REAL_T_IS_DOUBLE
	// Fallback to compression level 1 if real_t is float
	if (p_compression_level == DataBuffer::COMPRESSION_LEVEL_0) {
		WARN_PRINT_ONCE("Compression level 0 is not supported for a binary compiled without REAL_T_IS_DOUBLE, falling back to compression level 0");
		p_compression_level = DataBuffer::COMPRESSION_LEVEL_1;
	}
#endif

	Vector2 r;
	r[0] = add_real(p_input[0], p_compression_level);
	r[1] = add_real(p_input[1], p_compression_level);
	return r;
}

Vector2 DataBuffer::read_vector2(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector2());

#ifndef REAL_T_IS_DOUBLE
	// Fallback to compression level 1 if real_t is float
	if (p_compression_level == DataBuffer::COMPRESSION_LEVEL_0) {
		WARN_PRINT_ONCE("Compression level 0 is not supported for a binary compiled without REAL_T_IS_DOUBLE, falling back to compression level 0");
		p_compression_level = DataBuffer::COMPRESSION_LEVEL_1;
	}
#endif

	Vector2 r;
	r[0] = read_real(p_compression_level);
	r[1] = read_real(p_compression_level);
	return r;
}

Vector2 DataBuffer::add_normalized_vector2(Vector2 p_input, CompressionLevel p_compression_level) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(p_input.is_normalized() == false, p_input);
#endif
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const int bits = get_bit_taken(DATA_TYPE_NORMALIZED_VECTOR2, p_compression_level);
	const int bits_for_the_angle = bits - 1;
	const int bits_for_zero = 1;

	const double angle = p_input.angle();
	const uint32_t is_not_zero = p_input.length_squared() > CMP_EPSILON;

	const double max_value = static_cast<double>(~(UINT64_MAX << bits_for_the_angle));

	const uint64_t compressed_angle = compress_unit_float((angle + Math_PI) / Math_TAU, max_value);

	make_room_in_bits(bits);
	buffer.store_bits(bit_offset, is_not_zero, bits_for_zero);
	buffer.store_bits(bit_offset + 1, compressed_angle, bits_for_the_angle);
	bit_offset += bits;

	const real_t decompressed_angle = (decompress_unit_float(compressed_angle, max_value) * Math_TAU) - Math_PI;
	const real_t x = Math::cos(decompressed_angle);
	const real_t y = Math::sin(decompressed_angle);

#ifdef DEBUG_ENABLED
	// Can't never happen because the buffer size is correctly handled.
	CRASH_COND((metadata_size + bit_size) > buffer.size_in_bits() && bit_offset > buffer.size_in_bits());
#endif

	return Vector2(x, y) * is_not_zero;
}

Vector2 DataBuffer::read_normalized_vector2(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector2());

	const int bits = get_bit_taken(DATA_TYPE_NORMALIZED_VECTOR2, p_compression_level);
	const int bits_for_the_angle = bits - 1;
	const int bits_for_zero = 1;

	const double max_value = static_cast<double>(~(UINT64_MAX << bits_for_the_angle));

	const real_t is_not_zero = buffer.read_bits(bit_offset, bits_for_zero);
	const uint64_t compressed_angle = buffer.read_bits(bit_offset + 1, bits_for_the_angle);
	bit_offset += bits;

	const real_t decompressed_angle = (decompress_unit_float(compressed_angle, max_value) * Math_TAU) - Math_PI;
	const real_t x = Math::cos(decompressed_angle);
	const real_t y = Math::sin(decompressed_angle);

	return Vector2(x, y) * is_not_zero;
}

Vector3 DataBuffer::add_vector3(Vector3 p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

#ifndef REAL_T_IS_DOUBLE
	// Fallback to compression level 1 if real_t is float
	if (p_compression_level == DataBuffer::COMPRESSION_LEVEL_0) {
		WARN_PRINT_ONCE("Compression level 0 is not supported for a binary compiled without REAL_T_IS_DOUBLE, falling back to compression level 0");
		p_compression_level = DataBuffer::COMPRESSION_LEVEL_1;
	}
#endif

	Vector3 r;
	r[0] = add_real(p_input[0], p_compression_level);
	r[1] = add_real(p_input[1], p_compression_level);
	r[2] = add_real(p_input[2], p_compression_level);
	return r;
}

Vector3 DataBuffer::read_vector3(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector3());

#ifndef REAL_T_IS_DOUBLE
	// Fallback to compression level 1 if real_t is float
	if (p_compression_level == DataBuffer::COMPRESSION_LEVEL_0) {
		WARN_PRINT_ONCE("Compression level 0 is not supported for a binary compiled without REAL_T_IS_DOUBLE, falling back to compression level 0");
		p_compression_level = DataBuffer::COMPRESSION_LEVEL_1;
	}
#endif

	Vector3 r;
	r[0] = read_real(p_compression_level);
	r[1] = read_real(p_compression_level);
	r[2] = read_real(p_compression_level);
	return r;
}

Vector3 DataBuffer::add_normalized_vector3(Vector3 p_input, CompressionLevel p_compression_level) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(p_input.is_normalized() == false, p_input);
#endif
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const real_t x_axis = add_unit_real(p_input.x, p_compression_level);
	const real_t y_axis = add_unit_real(p_input.y, p_compression_level);
	const real_t z_axis = add_unit_real(p_input.z, p_compression_level);

	return Vector3(x_axis, y_axis, z_axis);
}

Vector3 DataBuffer::read_normalized_vector3(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector3());

	const real_t x_axis = read_unit_real(p_compression_level);
	const real_t y_axis = read_unit_real(p_compression_level);
	const real_t z_axis = read_unit_real(p_compression_level);

	return Vector3(x_axis, y_axis, z_axis);
}

Variant DataBuffer::add_variant(const Variant &p_input) {
	// TODO consider to use a method similar to `_encode_and_compress_variant`
	// to compress the encoded data a bit.

	// Get the variant size.
	int len = 0;

	const Error len_err = encode_variant(
			p_input,
			nullptr,
			len,
			false);

	ERR_FAIL_COND_V_MSG(
			len_err != OK,
			Variant(),
			"Was not possible encode the variant.");

	// Variant encoding pads the data to byte, so doesn't make sense write it
	// unpadded.
	make_room_pad_to_next_byte();
	make_room_in_bits(len * 8);

#ifdef DEBUG_ENABLED
	// This condition is always false thanks to the `make_room_pad_to_next_byte`.
	// so it's safe to assume we are starting from the begin of the byte.
	CRASH_COND((bit_offset % 8) != 0);
#endif

	const Error write_err = encode_variant(
			p_input,
			buffer.get_bytes_mut().ptrw() + (bit_offset / 8),
			len,
			false);

	ERR_FAIL_COND_V_MSG(
			write_err != OK,
			Variant(),
			"Was not possible encode the variant.");

	bit_offset += len * 8;

	return p_input;
}

Variant DataBuffer::read_variant() {
	Variant ret;

	int len = 0;

	// The Variant is always written starting from the beginning of the byte.
	const bool success = pad_to_next_byte();
	ERR_FAIL_COND_V_MSG(success == false, Variant(), "Padding failed.");

#ifdef DEBUG_ENABLED
	// This condition is always false thanks to the `pad_to_next_byte`; So is
	// safe to assume we are starting from the begin of the byte.
	CRASH_COND((bit_offset % 8) != 0);
#endif

	const Error read_err = decode_variant(
			ret,
			buffer.get_bytes().ptr() + (bit_offset / 8),
			buffer.size_in_bytes() - (bit_offset / 8),
			&len,
			false);

	ERR_FAIL_COND_V_MSG(
			read_err != OK,
			Variant(),
			"Was not possible decode the variant.");

	bit_offset += len * 8;

	return ret;
}

void DataBuffer::zero() {
	buffer.zero();
}

void DataBuffer::skip_bool() {
	const int bits = get_bool_size();
	skip(bits);
}

void DataBuffer::skip_int(CompressionLevel p_compression) {
	const int bits = get_int_size(p_compression);
	skip(bits);
}

void DataBuffer::skip_real(CompressionLevel p_compression) {
	const int bits = get_real_size(p_compression);
	skip(bits);
}

void DataBuffer::skip_unit_real(CompressionLevel p_compression) {
	const int bits = get_unit_real_size(p_compression);
	skip(bits);
}

void DataBuffer::skip_vector2(CompressionLevel p_compression) {
	const int bits = get_vector2_size(p_compression);
	skip(bits);
}

void DataBuffer::skip_normalized_vector2(CompressionLevel p_compression) {
	const int bits = get_normalized_vector2_size(p_compression);
	skip(bits);
}

void DataBuffer::skip_vector3(CompressionLevel p_compression) {
	const int bits = get_vector3_size(p_compression);
	skip(bits);
}

void DataBuffer::skip_normalized_vector3(CompressionLevel p_compression) {
	const int bits = get_normalized_vector3_size(p_compression);
	skip(bits);
}

int DataBuffer::get_bool_size() const {
	return DataBuffer::get_bit_taken(DATA_TYPE_BOOL, COMPRESSION_LEVEL_0);
}

int DataBuffer::get_int_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_INT, p_compression);
}

int DataBuffer::get_real_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_REAL, p_compression);
}

int DataBuffer::get_unit_real_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_UNIT_REAL, p_compression);
}

int DataBuffer::get_vector2_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_VECTOR2, p_compression);
}

int DataBuffer::get_normalized_vector2_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_NORMALIZED_VECTOR2, p_compression);
}

int DataBuffer::get_vector3_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_VECTOR3, p_compression);
}

int DataBuffer::get_normalized_vector3_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_NORMALIZED_VECTOR3, p_compression);
}

int DataBuffer::read_bool_size() {
	const int bits = get_bool_size();
	skip(bits);
	return bits;
}

int DataBuffer::read_int_size(CompressionLevel p_compression) {
	const int bits = get_int_size(p_compression);
	skip(bits);
	return bits;
}

int DataBuffer::read_real_size(CompressionLevel p_compression) {
	const int bits = get_real_size(p_compression);
	skip(bits);
	return bits;
}

int DataBuffer::read_unit_real_size(CompressionLevel p_compression) {
	const int bits = get_unit_real_size(p_compression);
	skip(bits);
	return bits;
}

int DataBuffer::read_vector2_size(CompressionLevel p_compression) {
	const int bits = get_vector2_size(p_compression);
	skip(bits);
	return bits;
}

int DataBuffer::read_normalized_vector2_size(CompressionLevel p_compression) {
	const int bits = get_normalized_vector2_size(p_compression);
	skip(bits);
	return bits;
}

int DataBuffer::read_vector3_size(CompressionLevel p_compression) {
	const int bits = get_vector3_size(p_compression);
	skip(bits);
	return bits;
}

int DataBuffer::read_normalized_vector3_size(CompressionLevel p_compression) {
	const int bits = get_normalized_vector3_size(p_compression);
	skip(bits);
	return bits;
}

int DataBuffer::read_variant_size() {
	int len = 0;

	Variant ret;

	// The Variant is always written starting from the beginning of the byte.
	const bool success = pad_to_next_byte();
	ERR_FAIL_COND_V_MSG(success == false, Variant(), "Padding failed.");

#ifdef DEBUG_ENABLED
	// This condition is always false thanks to the `pad_to_next_byte`; So is
	// safe to assume we are starting from the begin of the byte.
	CRASH_COND((bit_offset % 8) != 0);
#endif

	const Error read_err = decode_variant(
			ret,
			buffer.get_bytes().ptr() + (bit_offset / 8),
			buffer.size_in_bytes() - (bit_offset / 8),
			&len,
			false);

	ERR_FAIL_COND_V_MSG(
			read_err != OK,
			0,
			"Was not possible to decode the variant, error: " + itos(read_err));

	bit_offset += len * 8;

	return len * 8;
}

int DataBuffer::get_bit_taken(DataType p_data_type, CompressionLevel p_compression) {
	switch (p_data_type) {
		case DATA_TYPE_BOOL:
			// No matter what, 1 bit.
			return 1;
		case DATA_TYPE_INT: {
			switch (p_compression) {
				case COMPRESSION_LEVEL_0:
					return 64;
				case COMPRESSION_LEVEL_1:
					return 32;
				case COMPRESSION_LEVEL_2:
					return 16;
				case COMPRESSION_LEVEL_3:
					return 8;
				default:
					// Unreachable
					CRASH_NOW_MSG("Compression level not supported!");
			}
		} break;
		case DATA_TYPE_REAL: {
			return get_mantissa_bits(p_compression) +
				   get_exponent_bits(p_compression);
		} break;
		case DATA_TYPE_POSITIVE_UNIT_REAL: {
			switch (p_compression) {
				case COMPRESSION_LEVEL_0:
					return 10;
				case COMPRESSION_LEVEL_1:
					return 8;
				case COMPRESSION_LEVEL_2:
					return 6;
				case COMPRESSION_LEVEL_3:
					return 4;
				default:
					// Unreachable
					CRASH_NOW_MSG("Compression level not supported!");
			}
		} break;
		case DATA_TYPE_UNIT_REAL: {
			return get_bit_taken(DATA_TYPE_POSITIVE_UNIT_REAL, p_compression) + 1;
		} break;
		case DATA_TYPE_VECTOR2: {
			return get_bit_taken(DATA_TYPE_REAL, p_compression) * 2;
		} break;
		case DATA_TYPE_NORMALIZED_VECTOR2: {
			// +1 bit to know if the vector is 0 or a direction
			switch (p_compression) {
				case CompressionLevel::COMPRESSION_LEVEL_0:
					return 11 + 1;
				case CompressionLevel::COMPRESSION_LEVEL_1:
					return 10 + 1;
				case CompressionLevel::COMPRESSION_LEVEL_2:
					return 9 + 1;
				case CompressionLevel::COMPRESSION_LEVEL_3:
					return 8 + 1;
			}
		} break;
		case DATA_TYPE_VECTOR3: {
			return get_bit_taken(DATA_TYPE_REAL, p_compression) * 3;
		} break;
		case DATA_TYPE_NORMALIZED_VECTOR3: {
			switch (p_compression) {
				case CompressionLevel::COMPRESSION_LEVEL_0:
					return 11 * 3;
				case CompressionLevel::COMPRESSION_LEVEL_1:
					return 10 * 3;
				case CompressionLevel::COMPRESSION_LEVEL_2:
					return 8 * 3;
				case CompressionLevel::COMPRESSION_LEVEL_3:
					return 6 * 3;
			}
		} break;
		case DATA_TYPE_VARIANT: {
			ERR_FAIL_V_MSG(0, "The variant size is dynamic and can't be know at compile time.");
		}
		default:
			// Unreachable
			CRASH_NOW_MSG("Input type not supported!");
	}

	// Unreachable
	CRASH_NOW_MSG("It was not possible to obtain the bit taken by this input data.");
	return 0; // Useless, but MS CI is too noisy.
}

int DataBuffer::get_mantissa_bits(CompressionLevel p_compression) {
	// https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
	switch (p_compression) {
		case CompressionLevel::COMPRESSION_LEVEL_0:
			return 53; // Binary64 format
		case CompressionLevel::COMPRESSION_LEVEL_1:
			return 24; // Binary32 format
		case CompressionLevel::COMPRESSION_LEVEL_2:
			return 11; // Binary16 format
		case CompressionLevel::COMPRESSION_LEVEL_3:
			return 4; // https://en.wikipedia.org/wiki/Minifloat
	}

	// Unreachable
	CRASH_NOW_MSG("Unknown compression level.");
	return 0; // Useless, but MS CI is too noisy.
}

int DataBuffer::get_exponent_bits(CompressionLevel p_compression) {
	// https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
	switch (p_compression) {
		case CompressionLevel::COMPRESSION_LEVEL_0:
			return 11; // Binary64 format
		case CompressionLevel::COMPRESSION_LEVEL_1:
			return 8; // Binary32 format
		case CompressionLevel::COMPRESSION_LEVEL_2:
			return 5; // Binary16 format
		case CompressionLevel::COMPRESSION_LEVEL_3:
			return 4; // https://en.wikipedia.org/wiki/Minifloat
	}

	// Unreachable
	CRASH_NOW_MSG("Unknown compression level.");
	return 0; // Useless, but MS CI is too noisy.
}

uint64_t DataBuffer::compress_unit_float(double p_value, double p_scale_factor) {
	return Math::round(MIN(p_value * p_scale_factor, p_scale_factor));
}

double DataBuffer::decompress_unit_float(uint64_t p_value, double p_scale_factor) {
	return static_cast<double>(p_value) / p_scale_factor;
}

void DataBuffer::make_room_in_bits(int p_dim) {
	const int array_min_dim = bit_offset + p_dim;
	if (array_min_dim > buffer.size_in_bits()) {
		buffer.resize_in_bits(array_min_dim);
	}

	if (array_min_dim > metadata_size) {
		const int new_bit_size = array_min_dim - metadata_size;
		if (new_bit_size > bit_size) {
			bit_size = new_bit_size;
		}
	}
}

void DataBuffer::make_room_pad_to_next_byte() {
	const int bits_to_next_byte = ((bit_offset + 7) & ~7) - bit_offset;
	make_room_in_bits(bits_to_next_byte);
	bit_offset += bits_to_next_byte;
}

bool DataBuffer::pad_to_next_byte() {
	const int bits_to_next_byte = ((bit_offset + 7) & ~7) - bit_offset;
	ERR_FAIL_COND_V_MSG(
			bit_offset + bits_to_next_byte > buffer.size_in_bits(),
			false,
			"");
	bit_offset += bits_to_next_byte;
	return true;
}
