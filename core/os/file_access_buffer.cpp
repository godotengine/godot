/*************************************************************************/
/*  file_access_buffer.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "file_access_buffer.h"

#include "core/error_macros.h"

uint8_t *FileAccessBuffer::get_current_data() {
	DEV_ASSERT(readahead_pointer < BUFFER_SIZE);
	return &data[readahead_pointer];
}

const uint8_t *FileAccessBuffer::get_current_data() const {
	DEV_ASSERT(readahead_pointer < BUFFER_SIZE);
	return &data[readahead_pointer];
}

uint32_t FileAccessBuffer::get_buffer_offset() const {
	DEV_ASSERT(readahead_filled >= readahead_pointer);
	return readahead_filled - readahead_pointer;
}

bool FileAccessBuffer::is_buffer_empty() const {
	return get_buffer_offset() == 0;
}

void FileAccessBuffer::invalidate(bool p_eof) {
	readahead_filled = 0;
	readahead_pointer = 0;
	eof = p_eof;
}
