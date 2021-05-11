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

#include "tests/test_macros.h"

namespace TestDataBuffer {

TEST_CASE("[Modules][DataBuffer] Bool") {
	DataBuffer buffer;
	bool value = {};

	SUBCASE("[Modules][DataBuffer] false") {
		value = true;
	}
	SUBCASE("[Modules][DataBuffer] true") {
		value = false;
	}

	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_bool(value) == value, "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_bool() == value, "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Int") {
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	int64_t value = {};

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
	constexpr real_t epsilon = 0.00196;
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	real_t value = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 127.55;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -128.55;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 32767.55;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -32768.55;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 2147483647.55;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -2147483648.55;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 922337203685477.55;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -9223372036854775808.55;
		}
	}

	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_real(value, compression_level) == doctest::Approx(value).epsilon(epsilon), "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_real(compression_level) == doctest::Approx(value).epsilon(epsilon), "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Precise real") {
	constexpr real_t epsilon = 0.00049;
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	real_t value = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 127.555;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -128.555;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 32767.555;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -32768.555;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 2147483647.555;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -2147483648.555;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 922337203685477.555;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -9223372036854775808.555;
		}
	}

	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_precise_real(value, compression_level) == doctest::Approx(value).epsilon(epsilon), "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_precise_real(compression_level) == doctest::Approx(value).epsilon(epsilon), "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Positive unit real") {
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	real_t value = {};
	real_t epsilon = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
		epsilon = 0.033335;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 0.1;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;
		epsilon = 0.007935;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 0.05;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;
		epsilon = 0.00196;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 0.001;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
		epsilon = 0.00049;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 0.001;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
	}

	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_positive_unit_real(value, compression_level) == doctest::Approx(value).epsilon(epsilon), "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_positive_unit_real(compression_level) == doctest::Approx(value).epsilon(epsilon), "Should read the same value");
}
TEST_CASE("[Modules][DataBuffer] Unit real") {
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	real_t value = {};
	real_t epsilon = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
		epsilon = 0.033335;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 0.1;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -0.1;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;
		epsilon = 0.007935;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 0.05;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -0.05;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;
		epsilon = 0.00196;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 0.01;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -0.01;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
		epsilon = 0.00049;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 0.001;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -0.001;
		}
	}

	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_unit_real(value, compression_level) == doctest::Approx(value).epsilon(epsilon), "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_unit_real(compression_level) == doctest::Approx(value).epsilon(epsilon), "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Vector2") {
	constexpr real_t epsilon = 0.00196;
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	Vector2 value = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector2(127.55, 127.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector2(0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector2(-128.55, -128.55);
		}
		SUBCASE("[Modules][DataBuffer] Positive and negative") {
			value = Vector2(127.55, -128.55);
		}
		SUBCASE("[Modules][DataBuffer] Positive and zero") {
			value = Vector2(127.55, 0);
		}
		SUBCASE("[Modules][DataBuffer] Negative and zero") {
			value = Vector2(-128.55, 0);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector2(32767.55, 32767.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector2(0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector2(-32768.55, -32768.55);
		}
		SUBCASE("[Modules][DataBuffer] Positive and negative") {
			value = Vector2(32767.55, -32768.55);
		}
		SUBCASE("[Modules][DataBuffer] Positive and zero") {
			value = Vector2(32767.55, 0);
		}
		SUBCASE("[Modules][DataBuffer] Negative and zero") {
			value = Vector2(-32768.55, 0);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector2(2147483647.55, 2147483647.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector2(0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector2(-2147483648.55, -2147483648.55);
		}
		SUBCASE("[Modules][DataBuffer] Positive and negative") {
			value = Vector2(2147483647.55, -2147483648.55);
		}
		SUBCASE("[Modules][DataBuffer] Positive and zero") {
			value = Vector2(2147483647.55, 0);
		}
		SUBCASE("[Modules][DataBuffer] Negative and zero") {
			value = Vector2(-2147483648.55, 0);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector2(922337203685477.55, 922337203685477.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector2(0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector2(-9223372036854775808.55, -9223372036854775808.55);
		}
		SUBCASE("[Modules][DataBuffer] Positive and negative") {
			value = Vector2(922337203685477.55, -9223372036854775808.55);
		}
		SUBCASE("[Modules][DataBuffer] Positive and zero") {
			value = Vector2(922337203685477.55, 0);
		}
		SUBCASE("[Modules][DataBuffer] Negative and zero") {
			value = Vector2(-9223372036854775808.55, 0);
		}
	}

	buffer.begin_write(0);
	Vector2 added_value = buffer.add_vector2(value, compression_level);
	CHECK_MESSAGE(added_value.x == doctest::Approx(value.x).epsilon(epsilon), "Added Vector2 should have the same x axis");
	CHECK_MESSAGE(added_value.y == doctest::Approx(value.y).epsilon(epsilon), "Added Vector2 should have the same y axis");
	buffer.begin_read();
	Vector2 read_value = buffer.read_vector2(compression_level);
	CHECK_MESSAGE(read_value.x == doctest::Approx(value.x).epsilon(epsilon), "Read Vector2 should have the same x axis");
	CHECK_MESSAGE(read_value.y == doctest::Approx(value.y).epsilon(epsilon), "Read Vector2 should have the same y axis");
}

TEST_CASE("[Modules][DataBuffer] Precise Vector2") {
	constexpr real_t epsilon = 0.00049;
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	Vector2 value = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector2(127.555, 127.555);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector2(0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector2(-128.555, -128.555);
		}
		SUBCASE("[Modules][DataBuffer] Positive and negative") {
			value = Vector2(127.555, -128.555);
		}
		SUBCASE("[Modules][DataBuffer] Positive and zero") {
			value = Vector2(127.555, 0);
		}
		SUBCASE("[Modules][DataBuffer] Negative and zero") {
			value = Vector2(-128.555, 0);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector2(32767.555, 32767.555);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector2(0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector2(-32768.555, -32768.555);
		}
		SUBCASE("[Modules][DataBuffer] Positive and negative") {
			value = Vector2(32767.555, -32768.555);
		}
		SUBCASE("[Modules][DataBuffer] Positive and zero") {
			value = Vector2(32767.555, 0);
		}
		SUBCASE("[Modules][DataBuffer] Negative and zero") {
			value = Vector2(-32768.555, 0);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector2(2147483647.555, 2147483647.555);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector2(0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector2(-2147483648.555, -2147483648.555);
		}
		SUBCASE("[Modules][DataBuffer] Positive and negative") {
			value = Vector2(2147483647.555, -2147483648.555);
		}
		SUBCASE("[Modules][DataBuffer] Positive and zero") {
			value = Vector2(2147483647.555, 0);
		}
		SUBCASE("[Modules][DataBuffer] Negative and zero") {
			value = Vector2(-2147483648.555, 0);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector2(922337203685477.55, 922337203685477.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector2(0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector2(-9223372036854775808.555, -9223372036854775808.555);
		}
		SUBCASE("[Modules][DataBuffer] Positive and negative") {
			value = Vector2(922337203685477.55, -9223372036854775808.555);
		}
		SUBCASE("[Modules][DataBuffer] Positive and zero") {
			value = Vector2(922337203685477.55, 0);
		}
		SUBCASE("[Modules][DataBuffer] Negative and zero") {
			value = Vector2(-9223372036854775808.555, 0);
		}
	}

	buffer.begin_write(0);
	Vector2 added_value = buffer.add_precise_vector2(value, compression_level);
	CHECK_MESSAGE(added_value.x == doctest::Approx(value.x).epsilon(epsilon), "Added Vector2 should have the same x axis");
	CHECK_MESSAGE(added_value.y == doctest::Approx(value.y).epsilon(epsilon), "Added Vector2 should have the same y axis");
	buffer.begin_read();
	Vector2 read_value = buffer.read_precise_vector2(compression_level);
	CHECK_MESSAGE(read_value.x == doctest::Approx(value.x).epsilon(epsilon), "Read Vector2 should have the same x axis");
	CHECK_MESSAGE(read_value.y == doctest::Approx(value.y).epsilon(epsilon), "Read Vector2 should have the same y axis");
}

TEST_CASE("[Modules][DataBuffer] Vector3") {
	constexpr real_t epsilon = 0.00196;
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	Vector3 value = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector3(127.55, 127.55, 127.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector3(0.0, 0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector3(-128.55, -128.55, -128.55);
		}
		SUBCASE("[Modules][DataBuffer] Different axles") {
			value = Vector3(127.55, 0, -128.55);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector3(32767.55, 32767.55, 32767.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector3(0.0, 0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector3(-32768.55, -32768.55, -32768.55);
		}
		SUBCASE("[Modules][DataBuffer] Different axles") {
			value = Vector3(32767.55, 0, -32768.55);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector3(2147483647.55, 2147483647.55, 2147483647.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector3(0.0, 0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector3(-2147483648.55, -2147483648.55, -2147483648.55);
		}
		SUBCASE("[Modules][DataBuffer] Different axles") {
			value = Vector3(2147483647.55, 0, -2147483648.55);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector3(922337203685477.55, 922337203685477.55, 922337203685477.55);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector3(0.0, 0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector3(-9223372036854775808.55, -9223372036854775808.55, -9223372036854775807.55);
		}
		SUBCASE("[Modules][DataBuffer] Different axles") {
			value = Vector3(922337203685477.55, 0, -9223372036854775808.55);
		}
	}

	buffer.begin_write(0);
	Vector3 added_value = buffer.add_vector3(value, compression_level);
	CHECK_MESSAGE(added_value.x == doctest::Approx(value.x).epsilon(epsilon), "Added Vector3 should have the same x axis");
	CHECK_MESSAGE(added_value.y == doctest::Approx(value.y).epsilon(epsilon), "Added Vector3 should have the same y axis");
	CHECK_MESSAGE(added_value.z == doctest::Approx(value.z).epsilon(epsilon), "Added Vector3 should have the same z axis");
	buffer.begin_read();
	Vector3 read_value = buffer.read_vector3(compression_level);
	CHECK_MESSAGE(read_value.x == doctest::Approx(value.x).epsilon(epsilon), "Read Vector3 should have the same x axis");
	CHECK_MESSAGE(read_value.y == doctest::Approx(value.y).epsilon(epsilon), "Read Vector3 should have the same y axis");
	CHECK_MESSAGE(read_value.z == doctest::Approx(value.z).epsilon(epsilon), "Read Vector3 should have the same z axis");
}

TEST_CASE("[Modules][DataBuffer] Precise Vector3") {
	constexpr real_t epsilon = 0.00049;
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	Vector3 value = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector3(127.555, 127.555, 127.555);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector3(0.0, 0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector3(-128.555, -128.555, -128.555);
		}
		SUBCASE("[Modules][DataBuffer] Different axles") {
			value = Vector3(127.55, 0, -128.55);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector3(32767.555, 32767.555, 32767.555);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector3(0.0, 0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector3(-32768.555, -32768.555, -32768.555);
		}
		SUBCASE("[Modules][DataBuffer] Different axles") {
			value = Vector3(32767.555, 0, -32768.555);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector3(2147483647.555, 2147483647.555, 2147483647.555);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector3(0.0, 0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector3(-2147483648.555, -2147483648.555, -2147483648.555);
		}
		SUBCASE("[Modules][DataBuffer] Different axles") {
			value = Vector3(2147483647.555, 0, -2147483648.555);
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = Vector3(922337203685477.555, 922337203685477.555, 922337203685477.555);
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = Vector3(0.0, 0.0, 0.0);
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = Vector3(-9223372036854775808.555, -9223372036854775808.555, -9223372036854775808.555);
		}
		SUBCASE("[Modules][DataBuffer] Different axles") {
			value = Vector3(922337203685477.555, 0, -9223372036854775808.555);
		}
	}

	buffer.begin_write(0);
	Vector3 added_value = buffer.add_precise_vector3(value, compression_level);
	CHECK_MESSAGE(added_value.x == doctest::Approx(value.x).epsilon(epsilon), "Added Vector3 should have the same x axis");
	CHECK_MESSAGE(added_value.y == doctest::Approx(value.y).epsilon(epsilon), "Added Vector3 should have the same y axis");
	CHECK_MESSAGE(added_value.z == doctest::Approx(value.z).epsilon(epsilon), "Added Vector3 should have the same z axis");
	buffer.begin_read();
	Vector3 read_value = buffer.read_precise_vector3(compression_level);
	CHECK_MESSAGE(read_value.x == doctest::Approx(value.x).epsilon(epsilon), "Read Vector3 should have the same x axis");
	CHECK_MESSAGE(read_value.y == doctest::Approx(value.y).epsilon(epsilon), "Read Vector3 should have the same y axis");
	CHECK_MESSAGE(read_value.z == doctest::Approx(value.z).epsilon(epsilon), "Read Vector3 should have the same z axis");
}

TEST_CASE("[Modules][DataBuffer] Normalized Vector3") {
	const Vector3 value = Vector3(0.014278, 0.878079, 0.478303);
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	real_t epsilon = {};

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

	buffer.begin_write(0);
	Vector3 added_value = buffer.add_normalized_vector3(value, compression_level);
	CHECK_MESSAGE(added_value.x == doctest::Approx(value.x).epsilon(epsilon), "Added Vector3 should have the same x axis");
	CHECK_MESSAGE(added_value.y == doctest::Approx(value.y).epsilon(epsilon), "Added Vector3 should have the same y axis");
	CHECK_MESSAGE(added_value.z == doctest::Approx(value.z).epsilon(epsilon), "Added Vector3 should have the same z axis");
	buffer.begin_read();
	Vector3 read_value = buffer.read_normalized_vector3(compression_level);
	CHECK_MESSAGE(read_value.x == doctest::Approx(value.x).epsilon(epsilon), "Read Vector3 should have the same x axis");
	CHECK_MESSAGE(read_value.y == doctest::Approx(value.y).epsilon(epsilon), "Read Vector3 should have the same y axis");
	CHECK_MESSAGE(read_value.z == doctest::Approx(value.z).epsilon(epsilon), "Read Vector3 should have the same z axis");
}
} // namespace TestDataBuffer

#endif // TEST_DATA_BUFFER_H
