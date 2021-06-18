/*************************************************************************/
/*  test_data_buffer.h                                                   */
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

#ifndef TEST_DATA_BUFFER_H
#define TEST_DATA_BUFFER_H

#include "modules/network_synchronizer/data_buffer.h"
#include "modules/network_synchronizer/scene_synchronizer.h"

#include "tests/test_macros.h"

namespace TestDataBuffer {

inline Vector<double> real_values(DataBuffer::CompressionLevel p_compression_level) {
	Vector<double> values;
	values.append(Math_PI);
	values.append(0.0);
	values.append(-3.04);
	values.append(3.04);
	values.append(0.5);
	values.append(-0.5);
	values.append(1);
	values.append(-1);
	values.append(0.9);
	values.append(-0.9);
	values.append(3.9);
	values.append(-3.9);
	values.append(8);

	switch (p_compression_level) {
		case DataBuffer::COMPRESSION_LEVEL_3: {
			values.append(-15'360);
			values.append(15'360);
		} break;
		case DataBuffer::COMPRESSION_LEVEL_2: {
			// https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples
			values.append(-65'504);
			values.append(65'504);
			values.append(Math::pow(2.0, -14) / 1024);
			values.append(Math::pow(2.0, -14) * 1023 / 1024);
			values.append(Math::pow(2.0, -1) * (1 + 1023.0 / 1024));
			values.append((1 + 1.0 / 1024));
		} break;
		case DataBuffer::COMPRESSION_LEVEL_1: {
			// https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Single-precision_examples
			values.append(FLT_MIN);
			values.append(-FLT_MAX);
			values.append(FLT_MAX);
			values.append(Math::pow(2.0, -149));
			values.append(Math::pow(2.0, -126) * (1 - Math::pow(2.0, -23)));
			values.append(1 - Math::pow(2.0, -24));
			values.append(1 + Math::pow(2.0, -23));
		} break;
		case DataBuffer::COMPRESSION_LEVEL_0: {
			// https://en.wikipedia.org/wiki/Double-precision_floating-point_format#Double-precision_examples
			values.append(DBL_MIN);
			values.append(DBL_MAX);
			values.append(-DBL_MAX);
			values.append(1.0000000000000002);
			values.append(4.9406564584124654 * Math::pow(10.0, -324));
			values.append(2.2250738585072009 * Math::pow(10.0, -308));
		} break;
	}

	return values;
}

TEST_CASE("[Modules][DataBuffer] Bool") {
	bool value = {};

	SUBCASE("[Modules][DataBuffer] false") {
		value = false;
	}
	SUBCASE("[Modules][DataBuffer] true") {
		value = true;
	}

	DataBuffer buffer;
	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_bool(value) == value, "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_bool() == value, "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Int") {
	DataBuffer::CompressionLevel compression_level = {};
	int64_t value = {};

	DataBuffer buffer;
	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 127;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -128;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 32767;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -32768;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 2147483647;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -2147483648LL;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 2147483647;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -9223372036854775807LL;
		}
	}

	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_int(value, compression_level) == value, "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_int(compression_level) == value, "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Real") {
	DataBuffer::CompressionLevel compression_level = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3 (Minifloat)") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2 (Half perception)") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1 (Single perception)") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0 (Double perception)") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
	}

	DataBuffer buffer;
	const Vector<double> values = real_values(compression_level);
	const double epsilon = Math::pow(2.0, DataBuffer::get_mantissa_bits(compression_level) - 1);
	for (int i = 0; i < values.size(); ++i) {
		buffer.begin_write(0);
		const double value = values[i];
		CHECK_MESSAGE(buffer.add_real(value, compression_level) == doctest::Approx(value).epsilon(epsilon), "Should return the same value");

		buffer.begin_read();
		CHECK_MESSAGE(buffer.read_real(compression_level) == doctest::Approx(value).epsilon(epsilon), "Should read the same value");
	}
}

TEST_CASE("[Modules][DataBuffer] Positive unit real") {
	DataBuffer::CompressionLevel compression_level = {};
	double epsilon = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
		epsilon = 0.033335;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;
		epsilon = 0.007935;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;
		epsilon = 0.00196;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
		epsilon = 0.00049;
	}

	DataBuffer buffer;
	const Vector<double> values = real_values(compression_level);
	for (int i = 0; i < values.size(); ++i) {
		const double value = values[i];
		if (value < 0) {
			// Skip negative values
			continue;
		}
		double value_integral;
		const double value_unit = modf(values[i], &value_integral);
		buffer.begin_write(0);
		CHECK_MESSAGE(buffer.add_positive_unit_real(value_unit, compression_level) == doctest::Approx(value_unit).epsilon(epsilon), "Should return the same value");

		buffer.begin_read();
		CHECK_MESSAGE(buffer.read_positive_unit_real(compression_level) == doctest::Approx(value_unit).epsilon(epsilon), "Should read the same value");
	}
}

TEST_CASE("[Modules][DataBuffer] Unit real") {
	DataBuffer::CompressionLevel compression_level = {};
	double epsilon = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
		epsilon = 0.033335;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;
		epsilon = 0.007935;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;
		epsilon = 0.00196;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
		epsilon = 0.00049;
	}

	DataBuffer buffer;
	const Vector<double> values = real_values(compression_level);
	for (int i = 0; i < values.size(); ++i) {
		double value_integral;
		const double value_unit = modf(values[i], &value_integral);
		buffer.begin_write(0);
		CHECK_MESSAGE(buffer.add_unit_real(value_unit, compression_level) == doctest::Approx(value_unit).epsilon(epsilon), "Should return the same value");

		buffer.begin_read();
		CHECK_MESSAGE(buffer.read_unit_real(compression_level) == doctest::Approx(value_unit).epsilon(epsilon), "Should read the same value");
	}
}

TEST_CASE("[Modules][DataBuffer] Vector2") {
	DataBuffer::CompressionLevel compression_level = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
	}

	DataBuffer buffer;
	const double epsilon = Math::pow(2.0, DataBuffer::get_mantissa_bits(compression_level) - 1);
	const Vector<double> values = real_values(compression_level);
	for (int i = 0; i < values.size(); ++i) {
#ifdef REAL_T_IS_DOUBLE
		const Vector2 value = Vector2(values[i], values[i]);
#else
		const real_t clamped_value = CLAMP(values[i], -FLT_MIN, FLT_MAX);
		const Vector2 value = Vector2(clamped_value, clamped_value);
#endif
		buffer.begin_write(0);
		const Vector2 added_value = buffer.add_vector2(value, compression_level);
		CHECK_MESSAGE(added_value.x == doctest::Approx(value.x).epsilon(epsilon), "Added Vector2 should have the same x axis");
		CHECK_MESSAGE(added_value.y == doctest::Approx(value.y).epsilon(epsilon), "Added Vector2 should have the same y axis");

		buffer.begin_read();
		const Vector2 read_value = buffer.read_vector2(compression_level);
		CHECK_MESSAGE(read_value.x == doctest::Approx(value.x).epsilon(epsilon), "Read Vector2 should have the same x axis");
		CHECK_MESSAGE(read_value.y == doctest::Approx(value.y).epsilon(epsilon), "Read Vector2 should have the same y axis");
	}
}

TEST_CASE("[Modules][DataBuffer] Vector3") {
	DataBuffer::CompressionLevel compression_level = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
	}

	DataBuffer buffer;
	const Vector<double> values = real_values(compression_level);
	const double epsilon = Math::pow(2.0, DataBuffer::get_mantissa_bits(compression_level) - 1);
	for (int i = 0; i < values.size(); ++i) {
#ifdef REAL_T_IS_DOUBLE
		const Vector3 value = Vector3(values[i], values[i], values[i]);
#else
		const real_t clamped_value = CLAMP(values[i], -FLT_MIN, FLT_MAX);
		const Vector3 value = Vector3(clamped_value, clamped_value, clamped_value);
#endif
		buffer.begin_write(0);
		const Vector3 added_value = buffer.add_vector3(value, compression_level);
		CHECK_MESSAGE(added_value.x == doctest::Approx(value.x).epsilon(epsilon), "Added Vector3 should have the same x axis");
		CHECK_MESSAGE(added_value.y == doctest::Approx(value.y).epsilon(epsilon), "Added Vector3 should have the same y axis");
		CHECK_MESSAGE(added_value.z == doctest::Approx(value.z).epsilon(epsilon), "Added Vector3 should have the same z axis");

		buffer.begin_read();
		const Vector3 read_value = buffer.read_vector3(compression_level);
		CHECK_MESSAGE(read_value.x == doctest::Approx(value.x).epsilon(epsilon), "Read Vector3 should have the same x axis");
		CHECK_MESSAGE(read_value.y == doctest::Approx(value.y).epsilon(epsilon), "Read Vector3 should have the same y axis");
		CHECK_MESSAGE(read_value.z == doctest::Approx(value.z).epsilon(epsilon), "Read Vector3 should have the same z axis");
	}
}

TEST_CASE("[Modules][DataBuffer] Normalized Vector3") {
	DataBuffer::CompressionLevel compression_level = {};
	double epsilon = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
		epsilon = 0.033335;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;
		epsilon = 0.007935;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;
		epsilon = 0.00196;
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
		epsilon = 0.00049;
	}

	DataBuffer buffer;
	const Vector<double> values = real_values(compression_level);
	for (int i = 0; i < values.size(); ++i) {
		Vector3 value = Vector3(values[i], values[i], values[i]).normalized();
		if (!value.is_normalized()) {
			// Normalization fails for some numbers, probably a bug!
			continue;
		}
		buffer.begin_write(0);
		const Vector3 added_value = buffer.add_normalized_vector3(value, compression_level);
		CHECK_MESSAGE(added_value.x == doctest::Approx(value.x).epsilon(epsilon), "Added Vector3 should have the same x axis");
		CHECK_MESSAGE(added_value.y == doctest::Approx(value.y).epsilon(epsilon), "Added Vector3 should have the same y axis");
		CHECK_MESSAGE(added_value.z == doctest::Approx(value.z).epsilon(epsilon), "Added Vector3 should have the same z axis");

		buffer.begin_read();
		const Vector3 read_value = buffer.read_normalized_vector3(compression_level);
		CHECK_MESSAGE(read_value.x == doctest::Approx(value.x).epsilon(epsilon), "Read Vector3 should have the same x axis");
		CHECK_MESSAGE(read_value.y == doctest::Approx(value.y).epsilon(epsilon), "Read Vector3 should have the same y axis");
		CHECK_MESSAGE(read_value.z == doctest::Approx(value.z).epsilon(epsilon), "Read Vector3 should have the same z axis");
	}
}

TEST_CASE("[Modules][DataBuffer] Variant") {
	Variant value = {};

	SUBCASE("[Modules][DataBuffer] Invalid value") {
		value = {};
	}
	SUBCASE("[Modules][DataBuffer] String") {
		value = "VariantString";
	}
	SUBCASE("[Modules][DataBuffer] Vector") {
		value = sarray("VariantString1", "VariantString2", "VariantString3");
	}
	SUBCASE("[Modules][DataBuffer] Dictionary") {
		Dictionary dictionary;
		dictionary[1] = "Value";
		dictionary["Key"] = -1;
		value = dictionary;
	}
	SUBCASE("[Modules][DataBuffer] Array") {
		Array array;
		array.append("VariantString");
		array.append(0);
		array.append(-1.2);
		value = array;
	}

	DataBuffer buffer;
	buffer.begin_write(0);
	CHECK_MESSAGE(SceneSynchronizer::compare(buffer.add_variant(value), value, DBL_EPSILON), "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(SceneSynchronizer::compare(buffer.read_variant(), value, DBL_EPSILON), "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Seek") {
	DataBuffer buffer;
	buffer.begin_write(0);
	buffer.add_bool(true);
	buffer.add_bool(false);
	buffer.begin_read();

	ERR_PRINT_OFF
	buffer.seek(-1);
	CHECK_MESSAGE(buffer.get_bit_offset() == 0, "Bit offset should fail for negative values");
	ERR_PRINT_ON

	buffer.seek(1);
	CHECK_MESSAGE(buffer.get_bit_offset() == 1, "Bit offset should be 1 after seek to 1");
	CHECK_MESSAGE(buffer.read_bool() == false, "Should read false at position 1");

	buffer.seek(0);
	CHECK_MESSAGE(buffer.get_bit_offset() == 0, "Bit offset should be 0 after seek to 0");
	CHECK_MESSAGE(buffer.read_bool() == true, "Should read true at position 0");
}

TEST_CASE("[Modules][DataBuffer] Metadata") {
	bool value = {};
	bool metadata = {};

	SUBCASE("[Modules][DataBuffer] True") {
		metadata = true;
		value = false;
	}

	SUBCASE("[Modules][DataBuffer] False") {
		metadata = false;
		value = true;
	}

	const int metadata_size = DataBuffer::get_bit_taken(DataBuffer::DATA_TYPE_BOOL, DataBuffer::COMPRESSION_LEVEL_0);
	DataBuffer buffer;
	buffer.begin_write(metadata_size);
	buffer.add_bool(metadata);
	buffer.add_bool(value);
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_bool() == metadata, "Should return correct metadata");
	CHECK_MESSAGE(buffer.read_bool() == value, "Should return correct value after metadata");
	CHECK_MESSAGE(buffer.get_metadata_size() == metadata_size, "Metadata size should be equal to expected");
	CHECK_MESSAGE(buffer.size() == DataBuffer::get_bit_taken(DataBuffer::DATA_TYPE_BOOL, DataBuffer::COMPRESSION_LEVEL_0), "Size should be equal to expected");
	CHECK_MESSAGE(buffer.total_size() == DataBuffer::get_bit_taken(DataBuffer::DATA_TYPE_BOOL, DataBuffer::COMPRESSION_LEVEL_0) + metadata_size, "Total size should be equal to expected");
}

TEST_CASE("[Modules][DataBuffer] Zero") {
	constexpr DataBuffer::CompressionLevel compression = DataBuffer::COMPRESSION_LEVEL_0;
	DataBuffer buffer;
	buffer.begin_write(0);
	buffer.add_int(-1, compression);
	buffer.zero();
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_int(compression) == 0, "Should return 0");
}

TEST_CASE("[Modules][DataBuffer] Shrinking") {
	DataBuffer buffer;
	buffer.begin_write(0);
	for (int i = 0; i < 2; ++i) {
		buffer.add_real(3.14, DataBuffer::COMPRESSION_LEVEL_0);
	}
	const int original_size = buffer.total_size();

	ERR_PRINT_OFF;
	buffer.shrink_to(0, original_size + 1);
	ERR_PRINT_ON;
	CHECK_MESSAGE(buffer.total_size() == original_size, "Shrinking to a larger size should fail.");

	ERR_PRINT_OFF;
	buffer.shrink_to(0, -1);
	ERR_PRINT_ON;
	CHECK_MESSAGE(buffer.total_size() == original_size, "Shrinking with a negative bits size should fail.");

	buffer.shrink_to(0, original_size - 8);
	CHECK_MESSAGE(buffer.total_size() == original_size - 8, "Shrinking by 1 byte should succeed.");
	CHECK_MESSAGE(buffer.get_buffer().size_in_bits() == original_size, "Buffer size after shrinking by 1 byte should be the same.");

	buffer.dry();
	CHECK_MESSAGE(buffer.get_buffer().size_in_bits() == original_size - 8, "Buffer size after dry should changed to the smallest posiible.");
}

TEST_CASE("[Modules][DataBuffer] Skip") {
	const bool value = true;

	DataBuffer buffer;
	buffer.add_bool(!value);
	buffer.add_bool(value);

	buffer.begin_read();
	buffer.seek(DataBuffer::get_bit_taken(DataBuffer::DATA_TYPE_BOOL, DataBuffer::COMPRESSION_LEVEL_0));
	CHECK_MESSAGE(buffer.read_bool() == value, "Should read the same value");
}
} // namespace TestDataBuffer

#endif // TEST_DATA_BUFFER_H
