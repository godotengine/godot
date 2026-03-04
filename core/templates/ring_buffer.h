/**************************************************************************/
/*  ring_buffer.h                                                         */
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

#include "core/templates/local_vector.h"

template <typename T>
class RingBuffer {
	LocalVector<T> data;
	uint32_t read_pos = 0;
	uint32_t write_pos = 0;
	uint32_t size_mask;

	inline uint32_t inc(uint32_t &p_var, uint32_t p_size) const {
		const uint32_t ret = p_var;
		p_var += p_size;
		p_var = p_var & size_mask;
		return ret;
	}

public:
	T read() {
		ERR_FAIL_COND_V(space_left() < 1, T());
		return data.ptr()[inc(read_pos, 1)];
	}

	uint32_t read(T *p_buf, uint32_t p_size, bool p_advance = true) {
		const uint32_t left = data_left();

		p_size = MIN(left, p_size);
		uint32_t pos = read_pos;
		uint32_t to_read = p_size;
		uint32_t dst = 0;
		while (to_read) {
			uint32_t end = pos + to_read;
			end = MIN(end, size());
			uint32_t total = end - pos;
			const T *read = data.ptr();
			for (uint32_t i = 0; i < total; i++) {
				p_buf[dst++] = read[pos + i];
			}
			to_read -= total;
			pos = 0;
		}
		if (p_advance) {
			inc(read_pos, p_size);
		}
		return p_size;
	}

	uint32_t copy(T *p_buf, uint32_t p_offset, uint32_t p_size) const {
		const uint32_t left = data_left();
		if ((p_offset + p_size) > left) {
			p_size -= left - p_offset;
			if (p_size <= 0) {
				return 0;
			}
		}
		p_size = MIN(left, p_size);
		uint32_t pos = read_pos;
		inc(pos, p_offset);
		uint32_t to_read = p_size;
		uint32_t dst = 0;
		while (to_read) {
			uint32_t end = pos + to_read;
			end = MIN(end, size());
			uint32_t total = end - pos;
			for (uint32_t i = 0; i < total; i++) {
				p_buf[dst++] = data[pos + i];
			}
			to_read -= total;
			pos = 0;
		}
		return p_size;
	}

	int32_t find(const T &t, uint32_t p_offset, uint32_t p_max_size) const {
		const uint32_t left = data_left();
		if ((p_offset + p_max_size) > left) {
			p_max_size -= left - p_offset;
			if (p_max_size <= 0) {
				return 0;
			}
		}
		p_max_size = MIN(left, p_max_size);
		uint32_t pos = read_pos;
		inc(pos, p_offset);
		uint32_t to_read = p_max_size;
		while (to_read) {
			uint32_t end = pos + to_read;
			end = MIN(end, size());
			uint32_t total = end - pos;
			for (uint32_t i = 0; i < total; i++) {
				if (data[pos + i] == t) {
					return i + (p_max_size - to_read);
				}
			}
			to_read -= total;
			pos = 0;
		}
		return -1;
	}

	inline uint32_t advance_read(uint32_t p_n) {
		p_n = MIN(p_n, data_left());
		inc(read_pos, p_n);
		return p_n;
	}

	inline uint32_t decrease_write(uint32_t p_n) {
		p_n = MIN(p_n, data_left());
		inc(write_pos, size_mask + 1 - p_n);
		return p_n;
	}

	Error write(const T &p_v) {
		ERR_FAIL_COND_V(space_left() < 1, FAILED);
		data[inc(write_pos, 1)] = p_v;
		return OK;
	}

	uint32_t write(const T *p_buf, uint32_t p_size) {
		uint32_t left = space_left();
		p_size = MIN(left, p_size);

		uint32_t pos = write_pos;
		uint32_t to_write = p_size;
		uint32_t src = 0;
		while (to_write) {
			uint32_t end = pos + to_write;
			end = MIN(end, size());
			uint32_t total = end - pos;
			for (uint32_t i = 0; i < total; i++) {
				data[pos + i] = p_buf[src++];
			}
			to_write -= total;
			pos = 0;
		}

		inc(write_pos, p_size);
		return p_size;
	}

	inline uint32_t space_left() const {
		int32_t left = int32_t(read_pos) - int32_t(write_pos);
		if (left < 0) {
			return uint32_t(size() + left - 1);
		}
		if (left == 0) {
			return uint32_t(size() - 1);
		}
		return uint32_t(left - 1);
	}

	inline uint32_t data_left() const {
		return size() - space_left() - 1;
	}

	inline uint32_t size() const {
		return data.size();
	}

	inline void clear() {
		read_pos = 0;
		write_pos = 0;
	}

	void resize(uint32_t p_power) {
		uint32_t old_size = size();
		uint32_t new_size = 1 << p_power;
		uint32_t mask = new_size - 1;
		data.resize(uint32_t(1) << uint32_t(p_power));
		if (old_size < new_size && read_pos > write_pos) {
			for (uint32_t i = 0; i < write_pos; i++) {
				data[(old_size + i) & mask] = data[i];
			}
			write_pos = (old_size + write_pos) & mask;
		} else {
			read_pos = read_pos & mask;
			write_pos = write_pos & mask;
		}

		size_mask = mask;
	}

	RingBuffer(uint32_t p_power = 0) {
		resize(p_power);
	}
};
