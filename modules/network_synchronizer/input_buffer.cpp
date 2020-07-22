/*************************************************************************/
/*  player_protocol.cpp                                                  */
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

/**
	@author AndreaCatania
*/

#include "input_buffer.h"

#include "core/io/marshalls.h"

void DataBuffer::_bind_methods() {
	BIND_CONSTANT(DATA_TYPE_BOOL);
	BIND_CONSTANT(DATA_TYPE_INT);
	BIND_CONSTANT(DATA_TYPE_REAL);
	BIND_CONSTANT(DATA_TYPE_PRECISE_REAL);
	BIND_CONSTANT(DATA_TYPE_UNIT_REAL);
	BIND_CONSTANT(DATA_TYPE_VECTOR2);
	BIND_CONSTANT(DATA_TYPE_PRECISE_VECTOR2);
	BIND_CONSTANT(DATA_TYPE_NORMALIZED_VECTOR2);
	BIND_CONSTANT(DATA_TYPE_VECTOR3);
	BIND_CONSTANT(DATA_TYPE_PRECISE_VECTOR3);
	BIND_CONSTANT(DATA_TYPE_NORMALIZED_VECTOR3);

	BIND_CONSTANT(COMPRESSION_LEVEL_0);
	BIND_CONSTANT(COMPRESSION_LEVEL_1);
	BIND_CONSTANT(COMPRESSION_LEVEL_2);
	BIND_CONSTANT(COMPRESSION_LEVEL_3);

	ClassDB::bind_method(D_METHOD("add_bool", "value"), &DataBuffer::add_bool);
	ClassDB::bind_method(D_METHOD("add_int", "value", "compression_level"), &DataBuffer::add_int, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_real", "value", "compression_level"), &DataBuffer::add_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_precise_real", "value", "compression_level"), &DataBuffer::add_precise_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_unit_real", "value", "compression_level"), &DataBuffer::add_unit_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_vector2", "value", "compression_level"), &DataBuffer::add_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_precise_vector2", "value", "compression_level"), &DataBuffer::add_precise_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_normalized_vector2", "value", "compression_level"), &DataBuffer::add_normalized_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_vector3", "value", "compression_level"), &DataBuffer::add_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_precise_vector3", "value", "compression_level"), &DataBuffer::add_precise_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("add_normalized_vector3", "value", "compression_level"), &DataBuffer::add_normalized_vector3, DEFVAL(COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("read_bool"), &DataBuffer::read_bool);
	ClassDB::bind_method(D_METHOD("read_int", "compression_level"), &DataBuffer::read_int, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_real", "compression_level"), &DataBuffer::read_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_precise_real", "compression_level"), &DataBuffer::read_precise_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_unit_real", "compression_level"), &DataBuffer::read_unit_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_vector2", "compression_level"), &DataBuffer::read_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_precise_vector2", "compression_level"), &DataBuffer::read_precise_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_normalized_vector2", "compression_level"), &DataBuffer::read_normalized_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_vector3", "compression_level"), &DataBuffer::read_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_precise_vector3", "compression_level"), &DataBuffer::read_precise_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_normalized_vector3", "compression_level"), &DataBuffer::read_normalized_vector3, DEFVAL(COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("skip_bool"), &DataBuffer::skip_bool);
	ClassDB::bind_method(D_METHOD("skip_int", "compression_level"), &DataBuffer::skip_int, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_real", "compression_level"), &DataBuffer::skip_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_precise_real", "compression_level"), &DataBuffer::skip_precise_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_unit_real", "compression_level"), &DataBuffer::skip_unit_real, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_vector2", "compression_level"), &DataBuffer::skip_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_precise_vector2", "compression_level"), &DataBuffer::skip_precise_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_normalized_vector2", "compression_level"), &DataBuffer::skip_normalized_vector2, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_vector3", "compression_level"), &DataBuffer::skip_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_precise_vector3", "compression_level"), &DataBuffer::skip_precise_vector3, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_normalized_vector3", "compression_level"), &DataBuffer::skip_normalized_vector3, DEFVAL(COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("get_bool_size"), &DataBuffer::get_bool_size);
	ClassDB::bind_method(D_METHOD("get_int_size", "compression_level"), &DataBuffer::get_int_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_real_size", "compression_level"), &DataBuffer::get_real_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_precise_real_size", "compression_level"), &DataBuffer::get_precise_real_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_unit_real_size", "compression_level"), &DataBuffer::get_unit_real_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_vector2_size", "compression_level"), &DataBuffer::get_vector2_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_precise_vector2_size", "compression_level"), &DataBuffer::get_precise_vector2_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_normalized_vector2_size", "compression_level"), &DataBuffer::get_normalized_vector2_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_vector3_size", "compression_level"), &DataBuffer::get_vector3_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_precise_vector3_size", "compression_level"), &DataBuffer::get_precise_vector3_size, DEFVAL(COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_normalized_vector3_size", "compression_level"), &DataBuffer::get_normalized_vector3_size, DEFVAL(COMPRESSION_LEVEL_1));
}

DataBuffer::DataBuffer() :
		bit_offset(0),
		is_reading(true) {}

DataBuffer::DataBuffer(const BitArray &p_buffer) :
		bit_offset(0),
		is_reading(true),
		buffer(p_buffer) {
}

int DataBuffer::get_buffer_size() const {
	return buffer.get_bytes().size();
}

void DataBuffer::begin_write() {
	bit_offset = 0;
	is_reading = false;
}

void DataBuffer::dry() {
	buffer.resize_in_bits(bit_offset);
}

void DataBuffer::seek(int p_bits) {
	ERR_FAIL_COND(buffer.size_in_bits() < p_bits);
	bit_offset = p_bits;
}

void DataBuffer::skip(int p_bits) {
	ERR_FAIL_COND(buffer.size_in_bits() < bit_offset + p_bits);
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

	if (bits == 8) {
		value = MAX(MIN(value, INT8_MAX), INT8_MIN) & UINT8_MAX;
	} else if (bits == 16) {
		value = MAX(MIN(value, INT16_MAX), INT16_MIN) & UINT16_MAX;
	} else if (bits == 32) {
		value = MAX(MIN(value, INT32_MAX), INT32_MIN) & UINT32_MAX;
	} else {
		// Nothing to do here
	}

	make_room_in_bits(bits);
	buffer.store_bits(bit_offset, value, bits);
	bit_offset += bits;

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

real_t DataBuffer::add_real(real_t p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const real_t integral = Math::floor(p_input);
	const real_t fractional = p_input - integral;

	return real_t(add_int(integral, p_compression_level)) +
		   add_unit_real(fractional, COMPRESSION_LEVEL_1);
}

real_t DataBuffer::read_real(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, 0.0);

	const real_t integral = read_int(p_compression_level);
	const real_t fractional = read_unit_real(COMPRESSION_LEVEL_1);

	return integral + fractional;
}

real_t DataBuffer::add_precise_real(real_t p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const real_t integral = Math::floor(p_input);
	const real_t fractional = p_input - integral;

	const real_t ri = add_int(integral, p_compression_level);
	const real_t rf = add_unit_real(fractional, COMPRESSION_LEVEL_0);

	return ri + rf;
}

real_t DataBuffer::read_precise_real(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, 0.0);

	const real_t integral = read_int(p_compression_level);
	const real_t fractional = read_unit_real(COMPRESSION_LEVEL_0);

	return integral + fractional;
}

real_t DataBuffer::add_unit_real(real_t p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const int bits = get_bit_taken(DATA_TYPE_UNIT_REAL, p_compression_level);

	const double max_value = static_cast<double>(~(UINT64_MAX << bits));

	const uint64_t compressed_val = compress_unit_float(p_input, max_value);

	make_room_in_bits(bits);
	buffer.store_bits(bit_offset, compressed_val, bits);
	bit_offset += bits;

	return decompress_unit_float(compressed_val, max_value);
}

real_t DataBuffer::read_unit_real(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, 0.0);

	const int bits = get_bit_taken(DATA_TYPE_UNIT_REAL, p_compression_level);

	const double max_value = static_cast<double>(~(UINT64_MAX << bits));

	const uint64_t compressed_val = buffer.read_bits(bit_offset, bits);
	bit_offset += bits;

	return decompress_unit_float(compressed_val, max_value);
}

Vector2 DataBuffer::add_vector2(Vector2 p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	Vector2 r;
	r[0] = add_real(p_input[0], p_compression_level);
	r[1] = add_real(p_input[1], p_compression_level);
	return r;
}

Vector2 DataBuffer::read_vector2(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector2());

	Vector2 r;
	r[0] = read_real(p_compression_level);
	r[1] = read_real(p_compression_level);
	return r;
}

Vector2 DataBuffer::add_precise_vector2(Vector2 p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	Vector2 r;
	r[0] = add_precise_real(p_input[0], p_compression_level);
	r[1] = add_precise_real(p_input[1], p_compression_level);
	return r;
}

Vector2 DataBuffer::read_precise_vector2(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector2());

	Vector2 r;
	r[0] = read_precise_real(p_compression_level);
	r[1] = read_precise_real(p_compression_level);
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

	Vector3 r;
	r[0] = add_real(p_input[0], p_compression_level);
	r[1] = add_real(p_input[1], p_compression_level);
	r[2] = add_real(p_input[2], p_compression_level);
	return r;
}

Vector3 DataBuffer::read_vector3(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector3());

	Vector3 r;
	r[0] = read_real(p_compression_level);
	r[1] = read_real(p_compression_level);
	r[2] = read_real(p_compression_level);
	return r;
}

Vector3 DataBuffer::add_precise_vector3(Vector3 p_input, CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == true, p_input);

	Vector3 r;
	r[0] = add_precise_real(p_input[0], p_compression_level);
	r[1] = add_precise_real(p_input[1], p_compression_level);
	r[2] = add_precise_real(p_input[2], p_compression_level);
	return r;
}

Vector3 DataBuffer::read_precise_vector3(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector3());

	Vector3 r;
	r[0] = read_precise_real(p_compression_level);
	r[1] = read_precise_real(p_compression_level);
	r[2] = read_precise_real(p_compression_level);
	return r;
}

Vector3 DataBuffer::add_normalized_vector3(Vector3 p_input, CompressionLevel p_compression_level) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(p_input.is_normalized() == false, p_input);
#endif
	ERR_FAIL_COND_V(is_reading == true, p_input);

	const int bits = get_bit_taken(DATA_TYPE_NORMALIZED_VECTOR3, p_compression_level);
	const int bits_for_the_axis = bits / 3;

	const double max_value = static_cast<double>(~(UINT64_MAX << bits_for_the_axis));

	const uint64_t compressed_x_axis = compress_unit_float(p_input[0], max_value);
	const uint64_t compressed_y_axis = compress_unit_float(p_input[1], max_value);
	const uint64_t compressed_z_axis = compress_unit_float(p_input[2], max_value);

	make_room_in_bits(bits);

	buffer.store_bits(bit_offset, compressed_x_axis, bits_for_the_axis);
	bit_offset += bits_for_the_axis;

	buffer.store_bits(bit_offset, compressed_y_axis, bits_for_the_axis);
	bit_offset += bits_for_the_axis;

	buffer.store_bits(bit_offset, compressed_z_axis, bits_for_the_axis);
	bit_offset += bits_for_the_axis;

	const real_t decompressed_x_axis = decompress_unit_float(compressed_x_axis, max_value);
	const real_t decompressed_y_axis = decompress_unit_float(compressed_y_axis, max_value);
	const real_t decompressed_z_axis = decompress_unit_float(compressed_z_axis, max_value);

	return Vector3(decompressed_x_axis, decompressed_y_axis, decompressed_z_axis);
}

Vector3 DataBuffer::read_normalized_vector3(CompressionLevel p_compression_level) {
	ERR_FAIL_COND_V(is_reading == false, Vector3());

	const int bits = get_bit_taken(DATA_TYPE_NORMALIZED_VECTOR3, p_compression_level);
	const int bits_for_the_axis = bits / 3;

	const double max_value = static_cast<double>(~(UINT64_MAX << bits_for_the_axis));

	const real_t decompressed_x_axis = decompress_unit_float(buffer.read_bits(bit_offset, bits_for_the_axis), max_value);
	bit_offset += bits_for_the_axis;
	const real_t decompressed_y_axis = decompress_unit_float(buffer.read_bits(bit_offset, bits_for_the_axis), max_value);
	bit_offset += bits_for_the_axis;
	const real_t decompressed_z_axis = decompress_unit_float(buffer.read_bits(bit_offset, bits_for_the_axis), max_value);
	bit_offset += bits_for_the_axis;

	return Vector3(decompressed_x_axis, decompressed_y_axis, decompressed_z_axis);
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

void DataBuffer::skip_precise_real(CompressionLevel p_compression) {
	const int bits = get_precise_real_size(p_compression);
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

void DataBuffer::skip_precise_vector2(CompressionLevel p_compression) {
	const int bits = get_precise_vector2_size(p_compression);
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

void DataBuffer::skip_precise_vector3(CompressionLevel p_compression) {
	const int bits = get_precise_vector3_size(p_compression);
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

int DataBuffer::get_precise_real_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_PRECISE_REAL, p_compression);
}

int DataBuffer::get_unit_real_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_UNIT_REAL, p_compression);
}

int DataBuffer::get_vector2_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_VECTOR2, p_compression);
}

int DataBuffer::get_precise_vector2_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_PRECISE_VECTOR2, p_compression);
}

int DataBuffer::get_normalized_vector2_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_NORMALIZED_VECTOR2, p_compression);
}

int DataBuffer::get_vector3_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_VECTOR3, p_compression);
}

int DataBuffer::get_precise_vector3_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_PRECISE_VECTOR3, p_compression);
}

int DataBuffer::get_normalized_vector3_size(CompressionLevel p_compression) const {
	return DataBuffer::get_bit_taken(DATA_TYPE_NORMALIZED_VECTOR3, p_compression);
}

// TODO please add an unit test to make sure the returned data are right.
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
			return get_bit_taken(DATA_TYPE_INT, p_compression) +
				   get_bit_taken(DATA_TYPE_UNIT_REAL, COMPRESSION_LEVEL_1);
		} break;
		case DATA_TYPE_PRECISE_REAL: {
			return get_bit_taken(DATA_TYPE_INT, p_compression) +
				   get_bit_taken(DATA_TYPE_UNIT_REAL, COMPRESSION_LEVEL_0);
		} break;
		case DATA_TYPE_UNIT_REAL: {
			switch (p_compression) {
				case COMPRESSION_LEVEL_0:
					// Max loss ~0.09%
					return 10;
				case COMPRESSION_LEVEL_1:
					// Max loss ~0.3%
					return 8;
				case COMPRESSION_LEVEL_2:
					// Max loss ~3.2%
					return 6;
				case COMPRESSION_LEVEL_3:
					// Max loss ~6%
					return 4;
				default:
					// Unreachable
					CRASH_NOW_MSG("Compression level not supported!");
			}
		} break;
		case DATA_TYPE_VECTOR2: {
			return get_bit_taken(DATA_TYPE_REAL, p_compression) * 2;
		} break;
		case DATA_TYPE_PRECISE_VECTOR2: {
			return get_bit_taken(DATA_TYPE_PRECISE_REAL, p_compression) * 2;
		} break;
		case DATA_TYPE_NORMALIZED_VECTOR2: {
			// +1 bit to know if the vector is 0 or a direction
			switch (p_compression) {
				case CompressionLevel::COMPRESSION_LEVEL_0:
					// Max loss 0.17°
					return 11 + 1;
				case CompressionLevel::COMPRESSION_LEVEL_1:
					// Max loss 0.35°
					return 10 + 1;
				case CompressionLevel::COMPRESSION_LEVEL_2:
					// Max loss 0.7°
					return 9 + 1;
				case CompressionLevel::COMPRESSION_LEVEL_3:
					// Max loss 1.1°
					return 8 + 1;
			}
		} break;
		case DATA_TYPE_VECTOR3: {
			return get_bit_taken(DATA_TYPE_REAL, p_compression) * 3;
		} break;
		case DATA_TYPE_PRECISE_VECTOR3: {
			return get_bit_taken(DATA_TYPE_PRECISE_REAL, p_compression) * 3;
		} break;
		case DATA_TYPE_NORMALIZED_VECTOR3: {
			switch (p_compression) {
				case CompressionLevel::COMPRESSION_LEVEL_0:
					// Max loss 0.02% per axis
					return 11 * 3;
				case CompressionLevel::COMPRESSION_LEVEL_1:
					// Max loss 0.09% per axis
					return 10 * 3;
				case CompressionLevel::COMPRESSION_LEVEL_2:
					// Max loss 0.3% per axis
					return 8 * 3;
				case CompressionLevel::COMPRESSION_LEVEL_3:
					// Max loss 3.2% per axis;
					return 6 * 3;
			}
		} break;
		default:
			// Unreachable
			CRASH_NOW_MSG("Input type not supported!");
	}

	// Unreachable
	CRASH_NOW_MSG("It was not possible to obtain the bit taken by this input data.");
	return 0; // Useless, but MS CI is too noisy.
}

uint64_t DataBuffer::compress_unit_float(double p_value, double p_scale_factor) {
	return MIN(p_value * p_scale_factor, p_scale_factor);
}

double DataBuffer::decompress_unit_float(uint64_t p_value, double p_scale_factor) {
	return static_cast<double>(p_value) / p_scale_factor;
}

void DataBuffer::make_room_in_bits(int p_dim) {
	const int array_min_dim = bit_offset + p_dim;
	if (array_min_dim > buffer.size_in_bits())
		buffer.resize_in_bits(array_min_dim);
}
