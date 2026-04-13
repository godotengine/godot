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
	int _read_pos = 0;
	int _count = 0;
	int _size_mask;

	inline int _write_pos() const {
		return _wrapped_pos(_read_pos + _count);
	}

	inline int _inc_read(int p_amount = 1) {
		const int ret = _read_pos;

		_read_pos = _wrapped_pos(_read_pos + p_amount);
		_count -= p_amount;
		return ret;
	}

	inline int _wrapped_pos(int p_val) const {
		return p_val & _size_mask;
	}

public:
	T read() {
		ERR_FAIL_COND_V(data_left() < 1, T());
		return data.ptr()[_inc_read(1)];
	}

	int read(T *p_buf, int p_size, bool p_advance = true) {
		const int left = data_left();
		p_size = MIN(left, p_size);

		int pos = _read_pos;
		int to_read = p_size;
		int dst = 0;
		while (to_read) {
			int end = pos + to_read;
			end = MIN(end, size());
			int total = end - pos;
			const T *read = data.ptr();
			for (int i = 0; i < total; i++) {
				p_buf[dst++] = read[pos + i];
			}
			to_read -= total;
			pos = 0;
		}

		if (p_advance) {
			_inc_read(p_size);
		}

		return p_size;
	}

	int copy(T *p_buf, int p_offset, int p_size) const {
		const int left = data_left();
		if (p_offset > left) {
			return 0;
		}

		p_size = MIN(left - p_offset, p_size);
		int pos = _wrapped_pos(_read_pos + p_offset);
		int to_read = p_size;
		int dst = 0;
		while (to_read) {
			int end = pos + to_read;
			end = MIN(end, size());
			int total = end - pos;
			for (int i = 0; i < total; i++) {
				p_buf[dst++] = data[pos + i];
			}
			to_read -= total;
			pos = 0;
		}

		return p_size;
	}

	int find(const T &t, int p_offset, int p_max_size) const {
		const int left = data_left();
		if (p_offset > left) {
			return 0;
		}

		p_max_size = MIN(left - p_offset, p_max_size);
		int pos = _wrapped_pos(_read_pos + p_offset);
		int to_read = p_max_size;
		while (to_read) {
			int end = pos + to_read;
			end = MIN(end, size());
			int total = end - pos;
			for (int i = 0; i < total; i++) {
				if (data[pos + i] == t) {
					return i + (p_max_size - to_read);
				}
			}
			to_read -= total;
			pos = 0;
		}

		return -1;
	}

	inline int advance_read(int p_n) {
		p_n = MIN(p_n, data_left());
		_inc_read(p_n);
		return p_n;
	}

	inline int decrease_write(int p_n) {
		p_n = MIN(p_n, data_left());
		_count -= p_n;
		return p_n;
	}

	Error write(const T &p_v) {
		ERR_FAIL_COND_V(space_left() < 1, FAILED);
		data[_write_pos()] = p_v;
		_count += 1;
		return OK;
	}

	int write(const T *p_buf, int p_size) {
		const int left = space_left();
		p_size = MIN(left, p_size);

		int pos = _write_pos();
		int to_write = p_size;
		int src = 0;
		while (to_write) {
			int end = pos + to_write;
			end = MIN(end, size());
			int total = end - pos;
			for (int i = 0; i < total; i++) {
				data[pos + i] = p_buf[src++];
			}
			to_write -= total;
			pos = 0;
		}

		_count += p_size;
		return p_size;
	}

	inline int space_left() const {
		return size() - data_left() - 1;
	}
	inline int data_left() const {
		return _count;
	}

	inline int size() const {
		return data.size();
	}

	inline void clear() {
		_read_pos = 0;
		_count = 0;
	}

	void resize(int p_power) {
		const int old_size = size();
		const int new_size = uint32_t(1) << uint32_t(p_power);
		const int mask = new_size - 1;

		if (old_size < new_size) {
			data.resize(new_size);
			// Make data contiguous
			if (_read_pos > _write_pos()) {
				for (int i = 0; i < _write_pos(); i++) {
					data[_wrapped_pos(old_size + i)] = data[i];
				}
			}
		} else if (old_size > new_size) {
			LocalVector<T> new_data;
			new_data.resize(new_size);
			const int new_count = MIN(new_size - 1, data_left());
			for (int i = 0; i < new_count; i++) {
				new_data[i] = data[_wrapped_pos(_read_pos + i)];
			}
			_read_pos = 0;
			_count = new_count;
			data = new_data;
		}

		_size_mask = mask;
	}

	RingBuffer(int p_power = 0) {
		resize(p_power);
	}
};
