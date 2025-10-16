/**************************************************************************/
/*  delta_patch_file_interface.cpp                                        */
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

#include "delta_patch_file_interface.h"

bool DeltaPatchFileInterface::Read(void *p_data, size_t p_size, size_t *p_bytes_read) {
	*p_bytes_read = file->get_buffer(static_cast<uint8_t *>(p_data), p_size);
	return true;
}

bool DeltaPatchFileInterface::Write(const void *p_data, size_t p_count, size_t *p_bytes_written) {
	if (!file->store_buffer(static_cast<const uint8_t *>(p_data), p_count)) {
		*p_bytes_written = 0;
		return false;
	}

	*p_bytes_written = p_count;
	return true;
}

bool DeltaPatchFileInterface::Seek(int64_t p_pos) {
	file->seek(p_pos);
	return true;
}

bool DeltaPatchFileInterface::Close() {
	file->close();
	return true;
}

bool DeltaPatchFileInterface::GetSize(uint64_t *p_size) {
	*p_size = file->get_length();
	return true;
}

DeltaPatchFileInterface::DeltaPatchFileInterface(const Ref<FileAccess> &p_file) :
		file(p_file) {
}
