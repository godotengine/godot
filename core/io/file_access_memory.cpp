/**************************************************************************/
/*  file_access_memory.cpp                                                */
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

#include "file_access_memory.h"

#include "core/config/project_settings.h"

Ref<FileAccess> FileAccessMemory::create() {
	return memnew(FileAccessMemory);
}

bool FileAccessMemory::file_exists(const String &p_name) {
	return false;
}

Error FileAccessMemory::open_custom(const uint8_t *p_data, uint64_t p_len) {
	data = (uint8_t *)p_data;
	length = p_len;
	pos = 0;
	return OK;
}

Error FileAccessMemory::open_internal(const String &p_path, int p_mode_flags) {
	return ERR_FILE_NOT_FOUND;
}

bool FileAccessMemory::is_open() const {
	return data != nullptr;
}

void FileAccessMemory::seek(uint64_t p_position) {
	ERR_FAIL_NULL(data);
	pos = p_position;
}

void FileAccessMemory::seek_end(int64_t p_position) {
	ERR_FAIL_NULL(data);
	pos = length + p_position;
}

uint64_t FileAccessMemory::get_position() const {
	ERR_FAIL_NULL_V(data, 0);
	return pos;
}

uint64_t FileAccessMemory::get_length() const {
	ERR_FAIL_NULL_V(data, 0);
	return length;
}

bool FileAccessMemory::eof_reached() const {
	return pos >= length;
}

uint64_t FileAccessMemory::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	if (!p_length) {
		return 0;
	}

	ERR_FAIL_NULL_V(p_dst, -1);
	ERR_FAIL_NULL_V(data, -1);

	uint64_t left = length - pos;
	uint64_t read = MIN(p_length, left);

	if (read < p_length) {
		WARN_PRINT("Reading less data than requested");
	}

	memcpy(p_dst, &data[pos], read);
	pos += read;

	return read;
}

Error FileAccessMemory::get_error() const {
	return pos >= length ? ERR_FILE_EOF : OK;
}

void FileAccessMemory::flush() {
	ERR_FAIL_NULL(data);
}

bool FileAccessMemory::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	if (!p_length) {
		return true;
	}

	ERR_FAIL_NULL_V(p_src, false);

	uint64_t left = length - pos;
	uint64_t write = MIN(p_length, left);

	memcpy(&data[pos], p_src, write);
	pos += write;

	ERR_FAIL_COND_V_MSG(write < p_length, false, "Writing less data than requested.");

	return true;
}
