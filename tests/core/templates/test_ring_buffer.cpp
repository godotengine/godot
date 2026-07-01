/**************************************************************************/
/*  test_ring_buffer.cpp                                                  */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_ring_buffer)

#include "core/templates/ring_buffer.h"

namespace TestRingBuffer {

TEST_CASE("[RingBuffer] Initialization") {
	constexpr int RB_SIZE_POW = 4;
	RingBuffer<int> rb = RB_SIZE_POW;

	CHECK(rb.size() == (1 << RB_SIZE_POW));
	CHECK(rb.data_left() == 0);
	CHECK(rb.space_left() == rb.size() - 1);

	for (int i = 1; i < rb.size(); i++) {
		rb.write(i);

		CHECK(rb.size() == (1 << RB_SIZE_POW));
		CHECK(rb.data_left() == i);
		CHECK(rb.space_left() == (rb.size() - 1 - i));
	}

	for (int i = 1; rb.data_left() > 0; i++) {
		CHECK(rb.read() == i);
		CHECK(rb.data_left() == (rb.size() - 1 - i));
		CHECK(rb.space_left() == i);
	}
}

TEST_CASE("[RingBuffer] Clone 1") {
	constexpr int RB_SIZE_POW = 4;
	RingBuffer<int> rb1 = RB_SIZE_POW;
	RingBuffer<int> rb2 = rb1;

	CHECK(rb1.size() == rb2.size());
	CHECK(rb1.data_left() == rb2.data_left());
	CHECK(rb1.space_left() == rb2.space_left());

	for (int i = 1; i < rb1.size(); i++) {
		rb1.write(i);
	}

	CHECK(rb1.data_left() != rb2.data_left());
	CHECK(rb1.space_left() != rb2.space_left());

	for (int i = 1; i < rb2.size(); i++) {
		// rb1 and rb2 have separate heap allocated buffers. These writes should not effect rb1.
		rb2.write(i + rb2.size());
	}

	CHECK(rb1.size() == rb2.size());
	CHECK(rb1.data_left() == rb2.data_left());
	CHECK(rb1.space_left() == rb2.space_left());
	while (rb1.data_left() > 0) {
		CHECK(rb1.size() == rb2.size());
		CHECK(rb1.data_left() == rb2.data_left());
		CHECK(rb1.space_left() == rb2.space_left());
		CHECK(rb1.read() != rb2.read());
	}
}

TEST_CASE("[RingBuffer] Clone 2") {
	constexpr int RB_SIZE_POW = 4;
	RingBuffer<int> rb1 = RB_SIZE_POW;

	for (int i = 1; i < rb1.size(); i++) {
		rb1.write(i);
	}
	RingBuffer<int> rb2 = rb1;

	CHECK(rb1.size() == rb2.size());
	CHECK(rb1.data_left() == rb2.data_left());
	CHECK(rb1.space_left() == rb2.space_left());
	while (rb1.data_left() > 0) {
		CHECK(rb1.size() == rb2.size());
		CHECK(rb1.data_left() == rb2.data_left());
		CHECK(rb1.space_left() == rb2.space_left());
		CHECK(rb1.read() == rb2.read());
	}
}

TEST_CASE("[RingBuffer] Clone 3") {
	constexpr int RB_SIZE_POW = 4;
	RingBuffer<int> rb1 = RB_SIZE_POW;

	for (int i = 1; i < rb1.size(); i++) {
		rb1.write(i);
	}
	RingBuffer<int> rb2 = rb1;

	rb2.read();
	CHECK(rb1.data_left() > rb2.data_left());
	CHECK(rb1.space_left() < rb2.space_left());
}

TEST_CASE("[RingBuffer] Read 1") {
	RingBuffer<int> rb1 = 2;
	int buf[3] = { 7, 7, 7 };

	// Looping to test wrapping
	for (int _i = 0; _i < 4; _i++) {
		for (int i = 0; i < 3; i++) {
			rb1.write(i);
		}

		int r = rb1.read(buf, 3);
		CHECK(r == 3);

		for (int i = 0; i < 3; i++) {
			CHECK(buf[i] == i);
		}
	}
}

TEST_CASE("[RingBuffer] Read 2") {
	RingBuffer<int> rb1 = 2;
	int buf1[3] = { 7, 7, 7 };
	int buf2[3] = { 7, 7, 7 };

	for (int i = 0; i < 3; i++) {
		rb1.write(i);
	}

	int r1 = rb1.read(buf1, 3, false);
	CHECK(r1 == 3);
	CHECK(rb1.data_left() == 3);
	int r2 = rb1.read(buf2, 3);
	CHECK(r2 == 3);
	CHECK(rb1.data_left() == 0);

	for (int i = 0; i < 3; i++) {
		CHECK(buf1[i] == i);
		CHECK(buf1[i] == buf2[i]);
	}
}

TEST_CASE("[RingBuffer] Write 1") {
	RingBuffer<int> rb1 = 4;
	int buf1[6] = { 72, 48, 49, 93, 64, 74 };

	rb1.write(buf1, 6);
	CHECK(rb1.data_left() == 6);
	CHECK(rb1.space_left() == rb1.size() - 1 - rb1.data_left());

	for (int i = 0; i < 6; i++) {
		CHECK(rb1.read() == buf1[i]);
	}
}

TEST_CASE("[RingBuffer] Copy 1") {
	RingBuffer<int> rb1 = 2;
	int buf[3] = { 7, 7, 7 };

	for (int i = 0; i < 3; i++) {
		rb1.write(i);
	}

	int r = rb1.copy(buf, 0, 3);
	CHECK(r == 3);

	for (int i = 0; i < 3; i++) {
		CHECK(buf[i] == i);
	}
}

TEST_CASE("[RingBuffer] Copy 2") {
	RingBuffer<int> rb1 = 4;
	int buf[6] = { 7, 7, 7, 7, 7, 7 };

	for (int i = 0; i < 6; i++) {
		rb1.write(i);
	}

	int r = rb1.copy(buf, 4, 6);
	CHECK(r == 2);

	for (int i = 0; i < 2; i++) {
		CHECK(buf[i] == i + 4);
	}

	for (int i = 2; i < 6; i++) {
		CHECK(buf[i] == 7);
	}
}

TEST_CASE("[RingBuffer] Find 1") {
	RingBuffer<int> rb1 = 4;

	for (int i = 0; i < 6; i++) {
		rb1.write(i);
	}

	for (int i = 0; i < 6; i++) {
		int r = rb1.find(i, 0, 6);
		CHECK(r == i);
	}

	int r = rb1.find(7, 0, 6);
	CHECK(r == -1);
}

TEST_CASE("[RingBuffer] Advance read") {
	RingBuffer<int> rb1 = 4;

	for (int i = 0; i < 9; i++) {
		for (int ii = 0; ii < 8; ii++) {
			rb1.write(ii);
		}

		int len = rb1.data_left();
		int r = rb1.advance_read(i);
		CHECK(r == MIN(len, i));
		CHECK(rb1.data_left() == MAX(len - i, 0));

		for (int ii = i; ii < 8; ii++) {
			CHECK(rb1.read() == ii);
		}
	}
}

TEST_CASE("[RingBuffer] Decrease write") {
	RingBuffer<int> rb1 = 4;

	for (int i = 0; i < 9; i++) {
		for (int ii = 0; ii < 8; ii++) {
			rb1.write(ii);
		}

		int len = rb1.data_left();
		int r = rb1.decrease_write(i);
		CHECK(r == MIN(len, i));
		CHECK(rb1.data_left() == MAX(len - i, 0));

		for (int ii = 0; ii < 8 - i; ii++) {
			CHECK(rb1.read() == ii);
		}
	}
}

TEST_CASE("[RingBuffer] Resize 1") {
	RingBuffer<int> rb1 = 2;
	CHECK(rb1.size() == 1 << 2);
	CHECK(rb1.space_left() == rb1.size() - 1);

	rb1.resize(5);
	CHECK(rb1.size() == 1 << 5);
	CHECK(rb1.space_left() == rb1.size() - 1);

	rb1.resize(1);
	CHECK(rb1.size() == 1 << 1);
	CHECK(rb1.space_left() == rb1.size() - 1);
}

TEST_CASE("[RingBuffer] Resize 2") {
	RingBuffer<int> rb1 = 4;
	CHECK(rb1.size() == 1 << 4);
	CHECK(rb1.data_left() == 0);
	CHECK(rb1.space_left() == (1 << 4) - 1);

	for (int i = 0; i < rb1.size() - 1; i++) {
		rb1.write(i);
	}
	CHECK(rb1.data_left() == (1 << 4) - 1);
	CHECK(rb1.space_left() == 0);

	rb1.resize(5);
	CHECK(rb1.size() == 1 << 5);
	CHECK(rb1.data_left() == (1 << 4) - 1);
	CHECK(rb1.space_left() == ((1 << 5) - 1) - ((1 << 4) - 1));

	rb1.resize(1);
	CHECK(rb1.size() == 1 << 1);
	CHECK(rb1.data_left() == (1 << 1) - 1);
	CHECK(rb1.space_left() == 0);
}

} //namespace TestRingBuffer
