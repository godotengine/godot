/*************************************************************************/
/*  data_buffer.h                                                        */
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

#include "core/object/object.h"
#include "core/object/class_db.h"

#include "bit_array.h"

#ifndef INPUT_BUFFER_H
#define INPUT_BUFFER_H

class DataBuffer : public Object {
	GDCLASS(DataBuffer, Object);

public:
	enum DataType {
		DATA_TYPE_BOOL,
		DATA_TYPE_INT,
		DATA_TYPE_REAL,
		DATA_TYPE_PRECISE_REAL,
		DATA_TYPE_UNIT_REAL,
		DATA_TYPE_VECTOR2,
		DATA_TYPE_PRECISE_VECTOR2,
		DATA_TYPE_NORMALIZED_VECTOR2,
		DATA_TYPE_VECTOR3,
		DATA_TYPE_PRECISE_VECTOR3,
		DATA_TYPE_NORMALIZED_VECTOR3,
		// The only dynamic sized value.
		DATA_TYPE_VARIANT
	};

	/// Compression level for the stored input data.
	///
	/// Depending on the data type and the compression level used the amount of
	/// bits used and loss change.
	///
	///
	/// ## Bool
	/// Always use 1 bit
	///
	///
	/// ## Int
	/// COMPRESSION_LEVEL_0: 64 bits are used - Stores integers -9223372036854775808 / 9223372036854775807
	/// COMPRESSION_LEVEL_1: 32 bits are used - Stores integers -2147483648 / 2147483647
	/// COMPRESSION_LEVEL_2: 16 bits are used - Stores integers -32768 / 32767
	/// COMPRESSION_LEVEL_3: 8 bits are used - Stores integers -128 / 127
	///
	///
	/// ## Real
	/// The floating point part has a precision of ~0.3%
	/// COMPRESSION_LEVEL_0: 72 bits are used - The integral part has a range of -9223372036854775808 / 9223372036854775807
	/// COMPRESSION_LEVEL_1: 40 bits are used - The integral part has a range of -2147483648 / 2147483647
	/// COMPRESSION_LEVEL_2: 24 bits are used - The integral part has a range of -32768 / 32767
	/// COMPRESSION_LEVEL_3: 16 bits are used - The integral part has a range of -128 / 127
	///
	///
	/// ## Precise Real
	/// The floating point part has a precision of ~0.09%
	/// COMPRESSION_LEVEL_0: 74 bits are used - The integral part has a range of -9223372036854775808 / 9223372036854775807
	/// COMPRESSION_LEVEL_1: 42 bits are used - The integral part has a range of -2147483648 / 2147483647
	/// COMPRESSION_LEVEL_2: 26 bits are used - The integral part has a range of -32768 / 32767
	/// COMPRESSION_LEVEL_3: 18 bits are used - The integral part has a range of -128 / 127
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
	/// The floating point part has a precision of ~0.3% per axis.
	/// COMPRESSION_LEVEL_0: 72 * 2 bits are used - Max vector size -9223372036854775808 / 9223372036854775807
	/// COMPRESSION_LEVEL_1: 40 * 2 bits are used - Max vector size -2147483648 / 2147483647
	/// COMPRESSION_LEVEL_2: 24 * 2 bits are used - Max vector size -32768 / 32767
	/// COMPRESSION_LEVEL_3: 16 * 2 bits are used - Max vector size -128 / 127
	///
	///
	/// ## Precise Vector2
	/// The floating point part has a precision of ~0.09% per axis.
	/// COMPRESSION_LEVEL_0: 74 * 2 bits are used - Max vector size -9223372036854775808 / 9223372036854775807
	/// COMPRESSION_LEVEL_1: 42 * 2 bits are used - Max vector size -2147483648 / 2147483647
	/// COMPRESSION_LEVEL_2: 28 * 2 bits are used - Max vector size -32768 / 32767
	/// COMPRESSION_LEVEL_3: 18 * 2 bits are used - Max vector size -128 / 127
	///
	///
	/// ## Normalized Vector2
	/// COMPRESSION_LEVEL_0: 11 bits are used - Max loss 0.17°
	/// COMPRESSION_LEVEL_1: 10 bits are used - Max loss 0.35°
	/// COMPRESSION_LEVEL_2: 9 bits are used - Max loss 0.7°
	/// COMPRESSION_LEVEL_3: 8 bits are used - Max loss 1.1°
	///
	///
	/// ## Vector3
	/// The floating point part has a precision of ~0.3% per axis.
	/// COMPRESSION_LEVEL_0: 72 * 3 bits are used - Max vector size -9223372036854775808 / 9223372036854775807
	/// COMPRESSION_LEVEL_1: 40 * 3 bits are used - Max vector size -2147483648 / 2147483647
	/// COMPRESSION_LEVEL_2: 24 * 3 bits are used - Max vector size -32768 / 32767
	/// COMPRESSION_LEVEL_3: 16 * 3 bits are used - Max vector size -128 / 127
	///
	///
	/// ## Precise Vector3
	/// The floating point part has a precision of ~0.09% per axis.
	/// COMPRESSION_LEVEL_0: 74 * 3 bits are used - Max vector size -9223372036854775808 / 9223372036854775807
	/// COMPRESSION_LEVEL_1: 42 * 3 bits are used - Max vector size -2147483648 / 2147483647
	/// COMPRESSION_LEVEL_2: 28 * 3 bits are used - Max vector size -32768 / 32767
	/// COMPRESSION_LEVEL_3: 18 * 3 bits are used - Max vector size -128 / 127
	///
	///
	/// ## Normalized Vector3
	/// COMPRESSION_LEVEL_0: 11 * 3 bits are used - Max loss 0.02 per axis
	/// COMPRESSION_LEVEL_1: 10 * 3 bits are used - Max loss 0.09% per axis
	/// COMPRESSION_LEVEL_2: 8 * 3 bits are used - Max loss 0.3 per axis
	/// COMPRESSION_LEVEL_3: 6 * 3 bits are used - Max loss 3.2% per axis
	///
	/// ## Variant
	/// It's dynamic sized. It's not possible to compress it.
	enum CompressionLevel {
		COMPRESSION_LEVEL_0,
		COMPRESSION_LEVEL_1,
		COMPRESSION_LEVEL_2,
		COMPRESSION_LEVEL_3
	};

private:
	int metadata_size = 0;
	int bit_offset = 0;
	int bit_size = 0;
	bool is_reading = false;
	BitArray buffer;

public:
	static void _bind_methods();

	DataBuffer();
	DataBuffer(const DataBuffer &p_other);
	DataBuffer(const BitArray &p_buffer);

	const BitArray &get_buffer() const {
		return buffer;
	}

	BitArray &get_buffer_mut() {
		return buffer;
	}

	/// Begin write.
	void begin_write(int p_metadata_size);

	/// Make sure the buffer takes least space possible.
	void dry();

	/// Seek the offset to a specific bit. Seek to a bit greater than the actual
	/// size is not allowed.
	void seek(int p_bits);

	/// Set the bit size and the metadata size.
	void force_set_size(int p_metadata_bit_size, int p_bit_size);

	/// Returns the metadata size in bits.
	int get_metadata_size() const;
	/// Returns the buffer size in bits
	int size() const;
	/// Total size in bits.
	int total_size() const;

	/// Returns the bit offset.
	int get_bit_offset() const;

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

	/// Add a real into the buffer. Depending on the compression level is possible
	/// to store different range level.
	/// The fractional part has a precision of ~0.3%
	///
	/// Returns the compressed value so both the client and the peers can use
	/// the same data.
	real_t add_real(real_t p_input, CompressionLevel p_compression_level);

	/// Parse the following data as a real.
	real_t read_real(CompressionLevel p_compression_level);

	/// Add a real into the buffer. Depending on the compression level is possible
	/// to store different range level.
	/// The fractional part has a precision of ~0.09%
	///
	/// Returns the compressed value so both the client and the peers can use
	/// the same data.
	real_t add_precise_real(real_t p_input, CompressionLevel p_compression_level);

	/// Parse the following data as a precise real.
	real_t read_precise_real(CompressionLevel p_compression_level);

	/// Add a unit real into the buffer.
	///
	/// **Note:** Not unitary values lead to unexpected behaviour.
	///
	/// Returns the compressed value so both the client and the peers can use
	/// the same data.
	real_t add_unit_real(real_t p_input, CompressionLevel p_compression_level);

	/// Parse the following data as an unit real.
	real_t read_unit_real(CompressionLevel p_compression_level);

	/// Add a vector2 into the buffer.
	/// Note: This kind of vector occupies more space than the normalized verison.
	/// Consider use a normalized vector to save bandwidth if possible.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector2 add_vector2(Vector2 p_input, CompressionLevel p_compression_level);

	/// Parse next data as vector from the input buffer.
	Vector2 read_vector2(CompressionLevel p_compression_level);

	/// Add a precise vector2 into the buffer.
	/// Note: This kind of vector occupies more space than the normalized verison.
	/// Consider use a normalized vector to save bandwidth if possible.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector2 add_precise_vector2(Vector2 p_input, CompressionLevel p_compression_level);

	/// Parse next data as precise vector from the input buffer.
	Vector2 read_precise_vector2(CompressionLevel p_compression_level);

	/// Add a normalized vector2 into the buffer.
	/// Note: The compression algorithm rely on the fact that this is a
	/// normalized vector. The behaviour is unexpected for not normalized vectors.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector2 add_normalized_vector2(Vector2 p_input, CompressionLevel p_compression_level);

	/// Parse next data as normalized vector from the input buffer.
	Vector2 read_normalized_vector2(CompressionLevel p_compression_level);

	/// Add a vector3 into the buffer.
	/// Note: This kind of vector occupies more space than the normalized verison.
	/// Consider use a normalized vector to save bandwidth if possible.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector3 add_vector3(Vector3 p_input, CompressionLevel p_compression_level);

	/// Parse next data as vector3 from the input buffer.
	Vector3 read_vector3(CompressionLevel p_compression_level);

	/// Add a precise vector3 into the buffer.
	/// Note: This kind of vector occupies more space than the normalized verison.
	/// Consider use a normalized vector to save bandwidth if possible.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector3 add_precise_vector3(Vector3 p_input, CompressionLevel p_compression_level);

	/// Parse next data as precise vector3 from the input buffer.
	Vector3 read_precise_vector3(CompressionLevel p_compression_level);

	/// Add a normalized vector3 into the buffer.
	/// Note: The compression algorithm rely on the fact that this is a
	/// normalized vector. The behaviour is unexpected for not normalized vectors.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector3 add_normalized_vector3(Vector3 p_input, CompressionLevel p_compression_level);

	/// Parse next data as normalized vector3 from the input buffer.
	Vector3 read_normalized_vector3(CompressionLevel p_compression_level);

	/// Add a variant. This is the only supported dynamic sized value.
	Variant add_variant(Variant p_input);

	/// Parse the next data as Variant and returns it.
	Variant read_variant();

	/// Puts all the bytes to 0.
	void zero();

	/** Skips the amount of bits a type takes. */

	void skip_bool();
	void skip_int(CompressionLevel p_compression);
	void skip_real(CompressionLevel p_compression);
	void skip_precise_real(CompressionLevel p_compression);
	void skip_unit_real(CompressionLevel p_compression);
	void skip_vector2(CompressionLevel p_compression);
	void skip_precise_vector2(CompressionLevel p_compression);
	void skip_normalized_vector2(CompressionLevel p_compression);
	void skip_vector3(CompressionLevel p_compression);
	void skip_precise_vector3(CompressionLevel p_compression);
	void skip_normalized_vector3(CompressionLevel p_compression);

	/** Just returns the size of a specific type. */

	int get_bool_size() const;
	int get_int_size(CompressionLevel p_compression) const;
	int get_real_size(CompressionLevel p_compression) const;
	int get_precise_real_size(CompressionLevel p_compression) const;
	int get_unit_real_size(CompressionLevel p_compression) const;
	int get_vector2_size(CompressionLevel p_compression) const;
	int get_precise_vector2_size(CompressionLevel p_compression) const;
	int get_normalized_vector2_size(CompressionLevel p_compression) const;
	int get_vector3_size(CompressionLevel p_compression) const;
	int get_precise_vector3_size(CompressionLevel p_compression) const;
	int get_normalized_vector3_size(CompressionLevel p_compression) const;

	/** Read the size and pass to the next parameter. */

	int read_bool_size();
	int read_int_size(CompressionLevel p_compression);
	int read_real_size(CompressionLevel p_compression);
	int read_precise_real_size(CompressionLevel p_compression);
	int read_unit_real_size(CompressionLevel p_compression);
	int read_vector2_size(CompressionLevel p_compression);
	int read_precise_vector2_size(CompressionLevel p_compression);
	int read_normalized_vector2_size(CompressionLevel p_compression);
	int read_vector3_size(CompressionLevel p_compression);
	int read_precise_vector3_size(CompressionLevel p_compression);
	int read_normalized_vector3_size(CompressionLevel p_compression);
	int read_variant_size();

	static int get_bit_taken(DataType p_data_type, CompressionLevel p_compression);

private:
	static uint64_t compress_unit_float(double p_value, double p_scale_factor);
	static double decompress_unit_float(uint64_t p_value, double p_scale_factor);

	void make_room_in_bits(int p_dim);
	void make_room_pad_to_next_byte();
	bool pad_to_next_byte();
};

VARIANT_ENUM_CAST(DataBuffer::DataType)
VARIANT_ENUM_CAST(DataBuffer::CompressionLevel)

#endif
