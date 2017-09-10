/*************************************************************************/
/*  file_access_buffered.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "file_access_buffered.h"

#include <string.h>

#include "error_macros.h"

Error FileAccessBuffered::set_error(Error p_error) const {

	return (last_error = p_error);
};

void FileAccessBuffered::set_cache_size(int p_size) {

	cache_size = p_size;
};

int FileAccessBuffered::get_cache_size() {

	return cache_size;
};

int FileAccessBuffered::cache_data_left() const {

	if (file.offset >= file.size) {
		return 0;
	};

	if (cache.offset == -1 || file.offset < cache.offset || file.offset >= cache.offset + cache.buffer.size()) {

		return read_data_block(file.offset, cache_size);

	} else {

		return cache.buffer.size() - (file.offset - cache.offset);
	};

	return 0;
};

void FileAccessBuffered::seek(size_t p_position) {

	file.offset = p_position;
};

void FileAccessBuffered::seek_end(int64_t p_position) {

	file.offset = file.size + p_position;
};

size_t FileAccessBuffered::get_position() const {

	return file.offset;
};

size_t FileAccessBuffered::get_len() const {

	return file.size;
};

bool FileAccessBuffered::eof_reached() const {

	return file.offset > file.size;
};

uint8_t FileAccessBuffered::get_8() const {

	ERR_FAIL_COND_V(!file.open, 0);

	uint8_t byte = 0;
	if (cache_data_left() >= 1) {

		byte = cache.buffer[file.offset - cache.offset];
	};

	++file.offset;

	return byte;
};

int FileAccessBuffered::get_buffer(uint8_t *p_dest, int p_length) const {

	ERR_FAIL_COND_V(!file.open, -1);

	if (p_length > cache_size) {

		int total_read = 0;

		if (!(cache.offset == -1 || file.offset < cache.offset || file.offset >= cache.offset + cache.buffer.size())) {

			int size = (cache.buffer.size() - (file.offset - cache.offset));
			size = size - (size % 4);
			//PoolVector<uint8_t>::Read read = cache.buffer.read();
			//memcpy(p_dest, read.ptr() + (file.offset - cache.offset), size);
			memcpy(p_dest, cache.buffer.ptr() + (file.offset - cache.offset), size);
			p_dest += size;
			p_length -= size;
			file.offset += size;
			total_read += size;
		};

		int err = read_data_block(file.offset, p_length, p_dest);
		if (err >= 0) {
			total_read += err;
			file.offset += err;
		};

		return total_read;
	};

	int to_read = p_length;
	int total_read = 0;
	while (to_read > 0) {

		int left = cache_data_left();
		if (left == 0) {
			if (to_read > 0) {
				file.offset += to_read;
			};
			return total_read;
		};
		if (left < 0) {
			return left;
		};

		int r = MIN(left, to_read);
		//PoolVector<uint8_t>::Read read = cache.buffer.read();
		//memcpy(p_dest+total_read, &read.ptr()[file.offset - cache.offset], r);
		memcpy(p_dest + total_read, cache.buffer.ptr() + (file.offset - cache.offset), r);

		file.offset += r;
		total_read += r;
		to_read -= r;
	};

	return p_length;
};

bool FileAccessBuffered::is_open() const {

	return file.open;
};

Error FileAccessBuffered::get_error() const {

	return last_error;
};

FileAccessBuffered::FileAccessBuffered() {

	cache_size = DEFAULT_CACHE_SIZE;
};

FileAccessBuffered::~FileAccessBuffered() {
}
