/**************************************************************************/
/*  pck_reader.cpp                                                        */
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

#include "pck_reader.h"

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/file_access_pack.h"

Error PCKReader::open(const String &p_path, uint64_t p_offset, const PackedByteArray &p_key) {
	if (file.is_open()) {
		close();
	}

	return file.try_open_pack(p_path, p_offset, p_key) ? OK : FAILED;
}

Error PCKReader::close() {
	ERR_FAIL_COND_V_MSG(!file.is_open(), FAILED, "PCKReader cannot be closed because it is not open.");

	file.close();
	return OK;
}

PackedStringArray PCKReader::get_files() {
	ERR_FAIL_COND_V_MSG(!file.is_open(), PackedStringArray(), "PCKReader must be opened before use.");

	PackedStringArray result;
	for (const Pair<String, PackedData::PackedFile> &pair : file.get_files()) {
		result.append(pair.first);
	}

	return result;
}

PackedByteArray PCKReader::read_file(const String &p_path, bool p_case_sensitive) {
	ERR_FAIL_COND_V_MSG(!file.is_open(), PackedByteArray(), "PCKReader must be opened before use.");

	Ref<FileAccess> f = file.get_file(p_path, p_case_sensitive);
	ERR_FAIL_COND_V_MSG(f.is_null(), PackedByteArray(), "Failed to get file from PCK.");

	PackedByteArray res;
	res.resize(f->get_length());
	f->get_buffer((uint8_t *)res.ptr(), res.size());

	return res;
}

bool PCKReader::file_exists(const String &p_path, bool p_case_sensitive) {
	ERR_FAIL_COND_V_MSG(!file.is_open(), false, "PCKReader must be opened before use.");

	return file.file_exists(p_path, p_case_sensitive);
}

void PCKReader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path", "offset", "key"), &PCKReader::open, DEFVAL(0), DEFVAL(PackedByteArray()));
	ClassDB::bind_method(D_METHOD("close"), &PCKReader::close);
	ClassDB::bind_method(D_METHOD("get_files"), &PCKReader::get_files);
	ClassDB::bind_method(D_METHOD("read_file", "path", "case_sensitive"), &PCKReader::read_file, DEFVAL(Variant(true)));
	ClassDB::bind_method(D_METHOD("file_exists", "path", "case_sensitive"), &PCKReader::file_exists, DEFVAL(Variant(true)));
}

PCKReader::PCKReader() {}

PCKReader::~PCKReader() {
	if (file.is_open()) {
		close();
	}
}
