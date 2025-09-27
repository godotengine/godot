/**************************************************************************/
/*  zip_packer.cpp                                                        */
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

#include "zip_packer.h"

#include "core/io/zip_io.h"
#include "core/os/os.h"

// Callback functions for in-memory operations.

void *ZIPPacker::_zipio_mem_open(voidpf p_opaque, const char *p_fname, int p_mode) {
	ZIPPacker *self = (ZIPPacker *)p_opaque;
	DEV_ASSERT(self != nullptr);
	self->sink_cursor = 0;
	return p_opaque;
}

uLong ZIPPacker::_zipio_mem_read(voidpf p_opaque, voidpf p_stream, void *p_buf, uLong p_size) {
	ZIPPacker *self = (ZIPPacker *)p_opaque;
	DEV_ASSERT(self != nullptr);

	uLong bytes_to_read = MIN(p_size, self->sink.size() - self->sink_cursor);
	if (bytes_to_read > 0) {
		memcpy(p_buf, self->sink.ptr() + self->sink_cursor, bytes_to_read);
		self->sink_cursor += bytes_to_read;
	}
	return bytes_to_read;
}

uLong ZIPPacker::_zipio_mem_write(voidpf p_opaque, voidpf p_stream, const void *p_buf, uLong p_size) {
	ZIPPacker *self = (ZIPPacker *)p_opaque;
	DEV_ASSERT(self != nullptr);

	const uLong size_required = self->sink_cursor + p_size;
	if (size_required > (uint64_t)self->sink.size()) {
		self->sink.resize(size_required);
	}

	memcpy(self->sink.ptrw() + self->sink_cursor, p_buf, p_size);
	self->sink_cursor += p_size;

	return p_size;
}

long ZIPPacker::_zipio_mem_tell(voidpf p_opaque, voidpf p_stream) {
	ZIPPacker *self = (ZIPPacker *)p_opaque;
	DEV_ASSERT(self != nullptr);
	return self->sink_cursor;
}

long ZIPPacker::_zipio_mem_seek(voidpf p_opaque, voidpf p_stream, uLong p_offset, int p_origin) {
	ZIPPacker *self = (ZIPPacker *)p_opaque;
	DEV_ASSERT(self != nullptr);

	uint64_t new_cursor = self->sink_cursor;
	switch (p_origin) {
		case ZLIB_FILEFUNC_SEEK_CUR: {
			new_cursor += p_offset;
		} break;
		case ZLIB_FILEFUNC_SEEK_END: {
			new_cursor = self->sink.size() + p_offset;
		} break;
		case ZLIB_FILEFUNC_SEEK_SET: {
			new_cursor = p_offset;
		} break;
	}

	if (new_cursor > (uint64_t)self->sink.size()) {
		self->sink.resize(new_cursor);
	}

	self->sink_cursor = new_cursor;
	return 0;
}

int ZIPPacker::_zipio_mem_close(voidpf p_opaque, voidpf p_stream) {
	ZIPPacker *self = (ZIPPacker *)p_opaque;
	DEV_ASSERT(self != nullptr);

	self->sink_cursor = 0;
	// Keep `sink` as-is so that it can be retrieved.
	return 0;
}

int ZIPPacker::_zipio_mem_testerror(voidpf p_opaque, voidpf p_stream) {
	return 0;
}

// Implementation of ZIPPacker methods.

Error ZIPPacker::open(const String &p_path, ZipAppend p_append) {
	if (zf) {
		close();
	}

	zlib_filefunc_def io = zipio_create_io(&fa);
	zf = zipOpen2(p_path.utf8().get_data(), p_append, nullptr, &io);
	return zf != nullptr ? OK : FAILED;
}

Error ZIPPacker::open_buffer(const Vector<uint8_t> &p_init, ZipAppend p_append) {
	if (zf) {
		close();
	}

	sink = p_init;

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

	zf = zipOpen2(nullptr, p_append, nullptr, &io);
	return zf != nullptr ? OK : FAILED;
}

Error ZIPPacker::close() {
	ERR_FAIL_NULL_V_MSG(zf, FAILED, "ZIPPacker cannot be closed because it is not open.");

	Error err = zipClose(zf, nullptr) == ZIP_OK ? OK : FAILED;
	if (err == OK) {
		DEV_ASSERT(fa.is_null());
		zf = nullptr;
	}

	return err;
}

Vector<uint8_t> ZIPPacker::get_buffer() const {
	ERR_FAIL_COND_V_MSG(zf, PackedByteArray(), "ZIPPacker is still open. Call close() before retrieving the buffer.");
	return sink;
}

void ZIPPacker::set_compression_level(int p_compression_level) {
	ERR_FAIL_COND_MSG(p_compression_level < Z_DEFAULT_COMPRESSION || p_compression_level > Z_BEST_COMPRESSION, "Invalid compression level.");
	compression_level = p_compression_level;
}

int ZIPPacker::get_compression_level() const {
	return compression_level;
}

Error ZIPPacker::start_file(const String &p_path) {
	ERR_FAIL_NULL_V_MSG(zf, FAILED, "ZIPPacker must be opened before use.");

	zip_fileinfo zipfi;

	OS::DateTime time = OS::get_singleton()->get_datetime();

	zipfi.tmz_date.tm_sec = time.second;
	zipfi.tmz_date.tm_min = time.minute;
	zipfi.tmz_date.tm_hour = time.hour;
	zipfi.tmz_date.tm_mday = time.day;
	zipfi.tmz_date.tm_mon = time.month - 1;
	zipfi.tmz_date.tm_year = time.year;
	zipfi.dosDate = 0;
	zipfi.internal_fa = 0;
	zipfi.external_fa = 0;

	int err = zipOpenNewFileInZip4(zf,
			p_path.utf8().get_data(),
			&zipfi,
			nullptr,
			0,
			nullptr,
			0,
			nullptr,
			Z_DEFLATED,
			compression_level,
			0,
			-MAX_WBITS,
			DEF_MEM_LEVEL,
			Z_DEFAULT_STRATEGY,
			nullptr,
			0,
			0, // "version made by", indicates the compatibility of the file attribute information (the `external_fa` field above).
			1 << 11); // Bit 11 is the language encoding flag. When set, filename and comment fields must be encoded using UTF-8.
	return err == ZIP_OK ? OK : FAILED;
}

Error ZIPPacker::write_file(const Vector<uint8_t> &p_data) {
	ERR_FAIL_NULL_V_MSG(zf, FAILED, "ZIPPacker must be opened before use.");

	return zipWriteInFileInZip(zf, p_data.ptr(), p_data.size()) == ZIP_OK ? OK : FAILED;
}

Error ZIPPacker::close_file() {
	ERR_FAIL_NULL_V_MSG(zf, FAILED, "ZIPPacker must be opened before use.");

	return zipCloseFileInZip(zf) == ZIP_OK ? OK : FAILED;
}

void ZIPPacker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path", "append"), &ZIPPacker::open, DEFVAL(Variant(APPEND_CREATE)));
	ClassDB::bind_method(D_METHOD("open_buffer", "init", "append"), &ZIPPacker::open_buffer, DEFVAL(PackedByteArray()), DEFVAL(Variant(APPEND_CREATE)));
	ClassDB::bind_method(D_METHOD("get_buffer"), &ZIPPacker::get_buffer);
	ClassDB::bind_method(D_METHOD("set_compression_level", "compression_level"), &ZIPPacker::set_compression_level);
	ClassDB::bind_method(D_METHOD("get_compression_level"), &ZIPPacker::get_compression_level);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "compression_level"), "set_compression_level", "get_compression_level");
	ClassDB::bind_method(D_METHOD("start_file", "path"), &ZIPPacker::start_file);
	ClassDB::bind_method(D_METHOD("write_file", "data"), &ZIPPacker::write_file);
	ClassDB::bind_method(D_METHOD("close_file"), &ZIPPacker::close_file);
	ClassDB::bind_method(D_METHOD("close"), &ZIPPacker::close);

	BIND_ENUM_CONSTANT(APPEND_CREATE);
	BIND_ENUM_CONSTANT(APPEND_CREATEAFTER);
	BIND_ENUM_CONSTANT(APPEND_ADDINZIP);

	BIND_ENUM_CONSTANT(COMPRESSION_DEFAULT);
	BIND_ENUM_CONSTANT(COMPRESSION_NONE);
	BIND_ENUM_CONSTANT(COMPRESSION_FAST);
	BIND_ENUM_CONSTANT(COMPRESSION_BEST);
}

ZIPPacker::~ZIPPacker() {
	if (zf) {
		close();
	}
}
