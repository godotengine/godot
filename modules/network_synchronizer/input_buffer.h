/*************************************************************************/
/*  player_protocol.h                                                    */
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

#include "scene/main/node.h"

#include "bit_array.h"

#ifndef INPUT_BUFFER_H
#define INPUT_BUFFER_H

class InputsBuffer {
public:
	enum DataType {
		DATA_TYPE_BOOL,
		DATA_TYPE_INT,
		DATA_TYPE_UNIT_REAL,
		DATA_TYPE_NORMALIZED_VECTOR2,
		DATA_TYPE_NORMALIZED_VECTOR3
	};

	/// Compression level for the stored input data.
	///
	/// Depending on the data type and the compression level used the amount of
	/// bits used and loss change.
	///
	/// ## Bool
	/// Always use 1 bit
	///
	/// ## Int
	/// COMPRESSION_LEVEL_0: 64 bits are used - Stores integers -9223372036854775808 / 9223372036854775807
	/// COMPRESSION_LEVEL_1: 32 bits are used - Stores integers -2147483648 / 2147483647
	/// COMPRESSION_LEVEL_2: 16 bits are used - Stores integers -32768 / 32767
	/// COMPRESSION_LEVEL_3: 8 bits are used - Stores integers -128 / 127
	///
	///
	/// ## Unit real
	/// COMPRESSION_LEVEL_0: 10 bits are used - Max loss 0.09%
	/// COMPRESSION_LEVEL_1: 8 bits are used - Max loss 0.3%
	/// COMPRESSION_LEVEL_2: 6 bits are used - Max loss 3.2%
	/// COMPRESSION_LEVEL_3: 4 bits are used - Max loss 6%
	///
	///
	/// ## Vector2
	/// COMPRESSION_LEVEL_0: 11 bits are used - Max loss 0.17°
	/// COMPRESSION_LEVEL_1: 10 bits are used - Max loss 0.35°
	/// COMPRESSION_LEVEL_2: 9 bits are used - Max loss 0.7°
	/// COMPRESSION_LEVEL_3: 8 bits are used - Max loss 1.1°
	///
	/// ## Vector3
	/// COMPRESSION_LEVEL_0: 11 * 3 bits are used - Max loss 0.02 per axis
	/// COMPRESSION_LEVEL_1: 10 * 3 bits are used - Max loss 0.09% per axis
	/// COMPRESSION_LEVEL_2: 8 * 3 bits are used - Max loss 0.3 per axis
	/// COMPRESSION_LEVEL_3: 6 * 3 bits are used - Max loss 3.2% per axis
	enum CompressionLevel {
		COMPRESSION_LEVEL_0,
		COMPRESSION_LEVEL_1,
		COMPRESSION_LEVEL_2,
		COMPRESSION_LEVEL_3
	};

private:
	int bit_offset;
	bool is_reading;
	BitArray buffer;

public:
	InputsBuffer();

	const BitArray &get_buffer() const {
		return buffer;
	}

	BitArray &get_buffer_mut() {
		return buffer;
	}

	// Returns the buffer size in bytes
	int get_buffer_size() const;

	/// Begin write.
	void begin_write();

	/// Make sure the buffer takes less space possible.
	void dry();

	/// Seek to bit.
	void seek(int p_bits);

	/// Skip n bits.
	void skip(int p_bits);

	/// Begin read.
	void begin_read();

	/// Add a boolean to the buffer.
	/// Returns the same data.
	bool add_bool(bool p_input);

	/// Parse the next data as boolean.
	bool read_bool();

	/// Add the next data as int.
	int64_t add_int(int64_t p_input, CompressionLevel p_compression_level);

	/// Parse the next data as int.
	int64_t read_int(CompressionLevel p_compression_level);

	/// Add a unit real into the buffer.
	///
	/// **Note:** Not unitary values lead to unexpected behaviour.
	///
	/// Returns the compressed value so both the client and the peers can use
	/// the same data.
	real_t add_unit_real(real_t p_input, CompressionLevel p_compression_level);

	/// Returns the unit real.
	real_t read_unit_real(CompressionLevel p_compression_level);

	/// Add a normalized vector2 into the buffer.
	/// Note: The compression algorithm rely on the fact that this is a
	/// normalized vector. The behaviour is unexpected for not normalized vectors.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector2 add_normalized_vector2(Vector2 p_input, CompressionLevel p_compression_level);

	/// Parse next data as normalized vector from the input buffer.
	Vector2 read_normalized_vector2(CompressionLevel p_compression_level);

	/// Add a normalized vector3 into the buffer.
	/// Note: The compression algorithm rely on the fact that this is a
	/// normalized vector. The behaviour is unexpected for not normalized vectors.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector3 add_normalized_vector3(Vector3 p_input, CompressionLevel p_compression_level);

	/// Parse next data as normalized vector from the input buffer.
	Vector3 read_normalized_vector3(CompressionLevel p_compression_level);

	// Puts all the bytes to 0.
	void zero();

	static int get_bit_taken(DataType p_data_type, CompressionLevel p_compression);

private:
	static uint64_t compress_unit_float(double p_value, double p_scale_factor);
	static double decompress_unit_float(uint64_t p_value, double p_scale_factor);

	void make_room_in_bits(int p_dim);
};

#endif
