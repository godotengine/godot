/**************************************************************************/
/*  zip_reader.cpp                                                        */
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

#include "zip_reader.h"

#include "core/error/error_macros.h"
#include "core/io/zip_io.h"

// Callback functions for in-memory operations.

void *ZIPReader::_zipio_mem_open(voidpf p_opaque, const char *p_fname, int p_mode) {
	ZIPReader *self = (ZIPReader *)p_opaque;
	DEV_ASSERT(self != nullptr);
	self->source_cursor = 0;
	return p_opaque;
}

uLong ZIPReader::_zipio_mem_read(voidpf p_opaque, voidpf p_stream, void *p_buf, uLong p_size) {
	ZIPReader *self = (ZIPReader *)p_opaque;
	DEV_ASSERT(self != nullptr);

	uLong bytes_to_read = MIN(p_size, self->source.size() - self->source_cursor);
	if (bytes_to_read > 0) {
		memcpy(p_buf, self->source.ptr() + self->source_cursor, bytes_to_read);
		self->source_cursor += bytes_to_read;
	}
	return bytes_to_read;
}

uLong ZIPReader::_zipio_mem_write(voidpf p_opaque, voidpf p_stream, const void *p_buf, uLong p_size) {
	return 0; // Writing is not supported in ZIPReader.
}

long ZIPReader::_zipio_mem_tell(voidpf p_opaque, voidpf p_stream) {
	ZIPReader *self = (ZIPReader *)p_opaque;
	DEV_ASSERT(self != nullptr);
	return self->source_cursor;
}

long ZIPReader::_zipio_mem_seek(voidpf p_opaque, voidpf p_stream, uLong p_offset, int p_origin) {
	ZIPReader *self = (ZIPReader *)p_opaque;
	DEV_ASSERT(self != nullptr);

	uint64_t new_cursor = self->source_cursor;
	switch (p_origin) {
		case ZLIB_FILEFUNC_SEEK_CUR: {
			new_cursor += p_offset;
		} break;
		case ZLIB_FILEFUNC_SEEK_END: {
			new_cursor = self->source.size() + p_offset;
		} break;
		case ZLIB_FILEFUNC_SEEK_SET: {
			new_cursor = p_offset;
		} break;
	}

	if (new_cursor > (uint64_t)self->source.size()) {
		return -1;
	}

	self->source_cursor = new_cursor;
	return 0;
}

int ZIPReader::_zipio_mem_close(voidpf p_opaque, voidpf p_stream) {
	ZIPReader *self = (ZIPReader *)p_opaque;
	DEV_ASSERT(self != nullptr);

	self->source_cursor = 0;
	self->source.clear();
	return 0;
}

int ZIPReader::_zipio_mem_testerror(voidpf p_opaque, voidpf p_stream) {
	return 0;
}

// Implementation of ZIPReader methods.

Error ZIPReader::open(const String &p_path) {
	if (uzf) {
		close();
	}

	zlib_filefunc_def io = zipio_create_io(&fa);
	uzf = unzOpen2(p_path.utf8().get_data(), &io);
	return uzf != nullptr ? OK : FAILED;
}

Error ZIPReader::open_buffer(const Vector<uint8_t> &p_buffer) {
	if (uzf) {
		close();
	}

	source = p_buffer;

	zlib_filefunc_def io;
	io.opaque = (void *)this;
	io.zopen_file = _zipio_mem_open;
	io.zread_file = _zipio_mem_read;
	io.zwrite_file = _zipio_mem_write;
	io.ztell_file = _zipio_mem_tell;
	io.zseek_file = _zipio_mem_seek;
	io.zclose_file = _zipio_mem_close;
	io.zerror_file = _zipio_mem_testerror;
	io.alloc_mem = zipio_alloc;
	io.free_mem = zipio_free;

	uzf = unzOpen2(nullptr, &io);
	return uzf != nullptr ? OK : FAILED;
}

Error ZIPReader::close() {
	ERR_FAIL_NULL_V_MSG(uzf, FAILED, "ZIPReader cannot be closed because it is not open.");

	Error err = unzClose(uzf) == UNZ_OK ? OK : FAILED;
	if (err == OK) {
		DEV_ASSERT(fa.is_null());
		DEV_ASSERT(source.is_empty());
		uzf = nullptr;
	}

	return err;
}

PackedStringArray ZIPReader::get_files() {
	ERR_FAIL_NULL_V_MSG(uzf, PackedStringArray(), "ZIPReader must be opened before use.");

	unz_global_info gi;
	int err = unzGetGlobalInfo(uzf, &gi);
	ERR_FAIL_COND_V(err != UNZ_OK, PackedStringArray());
	if (gi.number_entry == 0) {
		return PackedStringArray();
	}

	err = unzGoToFirstFile(uzf);
	ERR_FAIL_COND_V(err != UNZ_OK, PackedStringArray());

	List<String> s;
	do {
		unz_file_info64 file_info;
		String filepath;

		err = godot_unzip_get_current_file_info(uzf, file_info, filepath);
		if (err == UNZ_OK) {
			s.push_back(filepath);
		}
	} while (unzGoToNextFile(uzf) == UNZ_OK);

	PackedStringArray arr;
	arr.resize(s.size());
	int idx = 0;
	for (const List<String>::Element *E = s.front(); E; E = E->next()) {
		arr.set(idx++, E->get());
	}
	return arr;
}

PackedByteArray ZIPReader::read_file(const String &p_path, bool p_case_sensitive) {
	ERR_FAIL_NULL_V_MSG(uzf, PackedByteArray(), "ZIPReader must be opened before use.");

	int err = UNZ_OK;

	// Locate and open the file.
	err = godot_unzip_locate_file(uzf, p_path, p_case_sensitive);
	ERR_FAIL_COND_V_MSG(err != UNZ_OK, PackedByteArray(), "File does not exist in zip archive: " + p_path);
	err = unzOpenCurrentFile(uzf);
	ERR_FAIL_COND_V_MSG(err != UNZ_OK, PackedByteArray(), "Could not open file within zip archive.");

	// Read the file info.
	unz_file_info info;
	err = unzGetCurrentFileInfo(uzf, &info, nullptr, 0, nullptr, 0, nullptr, 0);
	ERR_FAIL_COND_V_MSG(err != UNZ_OK, PackedByteArray(), "Unable to read file information from zip archive.");
	ERR_FAIL_COND_V_MSG(info.uncompressed_size > INT_MAX, PackedByteArray(), "File contents too large to read from zip archive (>2 GB).");

	// Read the file data.
	PackedByteArray data;
	data.resize(info.uncompressed_size);
	uint8_t *buffer = data.ptrw();
	int to_read = data.size();
	while (to_read > 0) {
		int bytes_read = unzReadCurrentFile(uzf, buffer, to_read);
		ERR_FAIL_COND_V_MSG(bytes_read < 0, PackedByteArray(), "IO/zlib error reading file from zip archive.");
		ERR_FAIL_COND_V_MSG(bytes_read == UNZ_EOF && to_read != 0, PackedByteArray(), "Incomplete file read from zip archive.");
		DEV_ASSERT(bytes_read <= to_read);
		buffer += bytes_read;
		to_read -= bytes_read;
	}

	// Verify the data and return.
	err = unzCloseCurrentFile(uzf);
	ERR_FAIL_COND_V_MSG(err != UNZ_OK, PackedByteArray(), "CRC error reading file from zip archive.");
	return data;
}

bool ZIPReader::file_exists(const String &p_path, bool p_case_sensitive) {
	ERR_FAIL_NULL_V_MSG(uzf, false, "ZIPReader must be opened before use.");

	int cs = p_case_sensitive ? 1 : 2;
	if (unzLocateFile(uzf, p_path.utf8().get_data(), cs) != UNZ_OK) {
		return false;
	}
	if (unzOpenCurrentFile(uzf) != UNZ_OK) {
		return false;
	}

	unzCloseCurrentFile(uzf);
	return true;
}

int ZIPReader::get_compression_level(const String &p_path, bool p_case_sensitive) {
	ERR_FAIL_NULL_V_MSG(uzf, -1, "ZIPReader must be opened before use.");

	int cs = p_case_sensitive ? 1 : 2;
	if (unzLocateFile(uzf, p_path.utf8().get_data(), cs) != UNZ_OK) {
		return -1;
	}

	int method;
	int level;
	if (unzOpenCurrentFile2(uzf, &method, &level, 1) != UNZ_OK) {
		return -1;
	}

	unzCloseCurrentFile(uzf);

	return level;
}

ZIPReader::~ZIPReader() {
	if (uzf) {
		close();
	}
}

void ZIPReader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path"), &ZIPReader::open);
	ClassDB::bind_method(D_METHOD("open_buffer", "buffer"), &ZIPReader::open_buffer);
	ClassDB::bind_method(D_METHOD("close"), &ZIPReader::close);
	ClassDB::bind_method(D_METHOD("get_files"), &ZIPReader::get_files);
	ClassDB::bind_method(D_METHOD("read_file", "path", "case_sensitive"), &ZIPReader::read_file, DEFVAL(Variant(true)));
	ClassDB::bind_method(D_METHOD("file_exists", "path", "case_sensitive"), &ZIPReader::file_exists, DEFVAL(Variant(true)));
	ClassDB::bind_method(D_METHOD("get_compression_level", "path", "case_sensitive"), &ZIPReader::get_compression_level, DEFVAL(Variant(true)));
}
