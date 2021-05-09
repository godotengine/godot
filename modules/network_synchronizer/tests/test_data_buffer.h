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
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	real_t value = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 127.15;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -128.15;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 32767.15;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -32768.15;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 2147483647.15;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -2147483648.15;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 2147483647.15;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -9223372036854775807.15;
		}
	}

	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_real(value, compression_level) == doctest::Approx(value).epsilon(0.03), "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_real(compression_level) == doctest::Approx(value).epsilon(0.03), "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Real") {
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	real_t value = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 127.015;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -128.015;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 2") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_2;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 32767.015;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -32768.015;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 1") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_1;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 2147483647.015;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -2147483648.015;
		}
	}

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;

		SUBCASE("[Modules][DataBuffer] Positive") {
			value = 2147483647.015;
		}
		SUBCASE("[Modules][DataBuffer] Zero") {
			value = 0.0;
		}
		SUBCASE("[Modules][DataBuffer] Negative") {
			value = -9223372036854775807.015;
		}
	}

	buffer.begin_write(0);
	CHECK_MESSAGE(buffer.add_precise_real(value, compression_level) == doctest::Approx(value).epsilon(0.009), "Should return the same value");
	buffer.begin_read();
	CHECK_MESSAGE(buffer.read_precise_real(compression_level) == doctest::Approx(value).epsilon(0.009), "Should read the same value");
}

TEST_CASE("[Modules][DataBuffer] Unit real") {
	DataBuffer buffer;
	DataBuffer::CompressionLevel compression_level = {};
	real_t value = {};
	real_t epsilon = {};

	SUBCASE("[Modules][DataBuffer] Compression level 3") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_3;
		epsilon = 0.06667;

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
		epsilon = 0.01587;

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
		epsilon = 0.00392;

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

	SUBCASE("[Modules][DataBuffer] Compression level 0") {
		compression_level = DataBuffer::COMPRESSION_LEVEL_0;
		epsilon = 0.00098;

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
} // namespace TestDataBuffer

#endif // TEST_DATA_BUFFER_H
