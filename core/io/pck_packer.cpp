/*************************************************************************/
/*  pck_packer.cpp                                                       */
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

#include "pck_packer.h"

#include "core/io/file_access_pack.h" // PACK_HEADER_MAGIC, PACK_FORMAT_VERSION
#include "core/os/file_access.h"
#include "core/version.h"

static uint64_t _align(uint64_t p_n, int p_alignment) {
	if (p_alignment == 0) {
		return p_n;
	}

	uint64_t rest = p_n % p_alignment;
	if (rest == 0) {
		return p_n;
	} else {
		return p_n + (p_alignment - rest);
	}
};

static void _pad(FileAccess *p_file, int p_bytes) {
	for (int i = 0; i < p_bytes; i++) {
		p_file->store_8(0);
	};
};

void PCKPacker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("pck_start", "pck_name", "alignment"), &PCKPacker::pck_start, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("add_file", "pck_path", "source_path"), &PCKPacker::add_file);
	ClassDB::bind_method(D_METHOD("flush", "verbose"), &PCKPacker::flush, DEFVAL(false));
};

Error PCKPacker::pck_start(const String &p_file, int p_alignment) {
	if (file != nullptr) {
		memdelete(file);
	}

	file = FileAccess::open(p_file, FileAccess::WRITE);

	ERR_FAIL_COND_V_MSG(!file, ERR_CANT_CREATE, "Can't open file to write: " + String(p_file) + ".");

	alignment = p_alignment;

	file->store_32(PACK_HEADER_MAGIC);
	file->store_32(PACK_FORMAT_VERSION);
	file->store_32(VERSION_MAJOR);
	file->store_32(VERSION_MINOR);
	file->store_32(VERSION_PATCH);

	for (int i = 0; i < 16; i++) {
		file->store_32(0); // reserved
	};

	files.clear();

	return OK;
};

Error PCKPacker::add_file(const String &p_file, const String &p_src) {
	ERR_FAIL_COND_V_MSG(!file, ERR_INVALID_PARAMETER, "File must be opened before use.");

	FileAccess *f = FileAccess::open(p_src, FileAccess::READ);
	if (!f) {
		return ERR_FILE_CANT_OPEN;
	};

	File pf;
	pf.path = p_file;
	pf.src_path = p_src;
	pf.size = f->get_len();
	pf.offset_offset = 0;

	files.push_back(pf);

	f->close();
	memdelete(f);

	return OK;
};

Error PCKPacker::flush(bool p_verbose) {
	ERR_FAIL_COND_V_MSG(!file, ERR_INVALID_PARAMETER, "File must be opened before use.");

	// write the index

	file->store_32(files.size());

	for (int i = 0; i < files.size(); i++) {
		file->store_pascal_string(files[i].path);
		files.write[i].offset_offset = file->get_position();
		file->store_64(0); // offset
		file->store_64(files[i].size); // size

		// # empty md5
		file->store_32(0);
		file->store_32(0);
		file->store_32(0);
		file->store_32(0);
	};

	uint64_t ofs = file->get_position();
	ofs = _align(ofs, alignment);

	_pad(file, ofs - file->get_position());

	const uint32_t buf_max = 65536;
	uint8_t *buf = memnew_arr(uint8_t, buf_max);

	int count = 0;
	for (int i = 0; i < files.size(); i++) {
		FileAccess *src = FileAccess::open(files[i].src_path, FileAccess::READ);
		uint64_t to_write = files[i].size;
		while (to_write > 0) {
			uint64_t read = src->get_buffer(buf, MIN(to_write, buf_max));
			file->store_buffer(buf, read);
			to_write -= read;
		};

		uint64_t pos = file->get_position();
		file->seek(files[i].offset_offset); // go back to store the file's offset
		file->store_64(ofs);
		file->seek(pos);

		ofs = _align(ofs + files[i].size, alignment);
		_pad(file, ofs - pos);

		src->close();
		memdelete(src);
		count += 1;
		const int file_num = files.size();
		if (p_verbose && (file_num > 0)) {
			print_line(vformat("[%d/%d - %d%%] PCKPacker flush: %s -> %s", count, file_num, float(count) / file_num * 100, files[i].src_path, files[i].path));
		}
	}

	if (p_verbose) {
		printf("\n");
	}

	file->close();
	memdelete(file);
	file = nullptr;

	memdelete_arr(buf);

	return OK;
};

PCKPacker::PCKPacker() {
	file = nullptr;
};

PCKPacker::~PCKPacker() {
	if (file != nullptr) {
		memdelete(file);
	};
	file = nullptr;
};
