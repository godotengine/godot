/*************************************************************************/
/*  file_access_memory.cpp                                               */
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

#include "file_access_memory.h"

#include "core/map.h"
#include "core/os/dir_access.h"
#include "core/project_settings.h"

static Map<String, Vector<uint8_t>> *files = nullptr;

void FileAccessMemory::register_file(String p_name, Vector<uint8_t> p_data) {
	if (!files) {
		files = memnew((Map<String, Vector<uint8_t>>));
	}

	String name;
	if (ProjectSettings::get_singleton()) {
		name = ProjectSettings::get_singleton()->globalize_path(p_name);
	} else {
		name = p_name;
	}
	//name = DirAccess::normalize_path(name);

	(*files)[name] = p_data;
}

void FileAccessMemory::cleanup() {
	if (!files) {
		return;
	}

	memdelete(files);
}

FileAccess *FileAccessMemory::create() {
	return memnew(FileAccessMemory);
}

bool FileAccessMemory::file_exists(const String &p_name) {
	String name = fix_path(p_name);
	//name = DirAccess::normalize_path(name);

	return files && (files->find(name) != nullptr);
}

Error FileAccessMemory::open_custom(const uint8_t *p_data, uint64_t p_len) {
	data = (uint8_t *)p_data;
	length = p_len;
	pos = 0;
	return OK;
}

Error FileAccessMemory::_open(const String &p_path, int p_mode_flags) {
	ERR_FAIL_COND_V(!files, ERR_FILE_NOT_FOUND);

	String name = fix_path(p_path);
	//name = DirAccess::normalize_path(name);

	Map<String, Vector<uint8_t>>::Element *E = files->find(name);
	ERR_FAIL_COND_V_MSG(!E, ERR_FILE_NOT_FOUND, "Can't find file '" + p_path + "'.");

	data = E->get().ptrw();
	length = E->get().size();
	pos = 0;

	return OK;
}

void FileAccessMemory::close() {
	data = nullptr;
}

bool FileAccessMemory::is_open() const {
	return data != nullptr;
}

void FileAccessMemory::seek(uint64_t p_position) {
	ERR_FAIL_COND(!data);
	pos = p_position;
}

void FileAccessMemory::seek_end(int64_t p_position) {
	ERR_FAIL_COND(!data);
	pos = length + p_position;
}

uint64_t FileAccessMemory::get_position() const {
	ERR_FAIL_COND_V(!data, 0);
	return pos;
}

uint64_t FileAccessMemory::get_len() const {
	ERR_FAIL_COND_V(!data, 0);
	return length;
}

bool FileAccessMemory::eof_reached() const {
	return pos > length;
}

uint8_t FileAccessMemory::get_8() const {
	uint8_t ret = 0;
	if (pos < length) {
		ret = data[pos];
	}
	++pos;

	return ret;
}

uint64_t FileAccessMemory::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);
	ERR_FAIL_COND_V(!data, -1);

	uint64_t left = length - pos;
	uint64_t read = MIN(p_length, left);

	if (read < p_length) {
		WARN_PRINT("Reading less data than requested");
	}

	memcpy(p_dst, &data[pos], read);
	pos += p_length;

	return read;
}

Error FileAccessMemory::get_error() const {
	return pos >= length ? ERR_FILE_EOF : OK;
}

void FileAccessMemory::flush() {
	ERR_FAIL_COND(!data);
}

void FileAccessMemory::store_8(uint8_t p_byte) {
	ERR_FAIL_COND(!data);
	ERR_FAIL_COND(pos >= length);
	data[pos++] = p_byte;
}

void FileAccessMemory::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	ERR_FAIL_COND(!p_src && p_length > 0);
	uint64_t left = length - pos;
	uint64_t write = MIN(p_length, left);
	if (write < p_length) {
		WARN_PRINT("Writing less data than requested");
	}

	memcpy(&data[pos], p_src, write);
	pos += p_length;
}

FileAccessMemory::FileAccessMemory() {
	data = nullptr;
}
