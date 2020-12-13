/*************************************************************************/
/*  pck_packer.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/crypto/crypto_core.h"
#include "core/io/file_access_encrypted.h"
#include "core/io/file_access_pack.h" // PACK_HEADER_MAGIC, PACK_FORMAT_VERSION
#include "core/os/file_access.h"
#include "core/version.h"

static int _get_pad(int p_alignment, int p_n) {
	int rest = p_n % p_alignment;
	int pad = 0;
	if (rest > 0) {
		pad = p_alignment - rest;
	}

	return pad;
}

void PCKPacker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("pck_start", "pck_name", "alignment", "key", "encrypt_directory"), &PCKPacker::pck_start, DEFVAL(0), DEFVAL(String()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_file", "pck_path", "source_path", "encrypt"), &PCKPacker::add_file, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("flush", "verbose"), &PCKPacker::flush, DEFVAL(false));
}

Error PCKPacker::pck_start(const String &p_file, int p_alignment, const String &p_key, bool p_encrypt_directory) {
	ERR_FAIL_COND_V_MSG((p_key.empty() || !p_key.is_valid_hex_number(false) || p_key.length() != 64), ERR_CANT_CREATE, "Invalid Encryption Key (must be 64 characters long).");

	String _key = p_key.to_lower();
	key.resize(32);
	for (int i = 0; i < 32; i++) {
		int v = 0;
		if (i * 2 < _key.length()) {
			char32_t ct = _key[i * 2];
			if (ct >= '0' && ct <= '9') {
				ct = ct - '0';
			} else if (ct >= 'a' && ct <= 'f') {
				ct = 10 + ct - 'a';
			}
			v |= ct << 4;
		}

		if (i * 2 + 1 < _key.length()) {
			char32_t ct = _key[i * 2 + 1];
			if (ct >= '0' && ct <= '9') {
				ct = ct - '0';
			} else if (ct >= 'a' && ct <= 'f') {
				ct = 10 + ct - 'a';
			}
			v |= ct;
		}
		key.write[i] = v;
	}
	enc_dir = p_encrypt_directory;

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

	uint32_t pack_flags = 0;
	if (enc_dir) {
		pack_flags |= PACK_DIR_ENCRYPTED;
	}
	file->store_32(pack_flags); // flags

	files.clear();
	ofs = 0;

	return OK;
}

Error PCKPacker::add_file(const String &p_file, const String &p_src, bool p_encrypt) {
	FileAccess *f = FileAccess::open(p_src, FileAccess::READ);
	if (!f) {
		return ERR_FILE_CANT_OPEN;
	}

	File pf;
	pf.path = p_file;
	pf.src_path = p_src;
	pf.ofs = ofs;
	pf.size = f->get_len();

	Vector<uint8_t> data = FileAccess::get_file_as_array(p_src);
	{
		unsigned char hash[16];
		CryptoCore::md5(data.ptr(), data.size(), hash);
		pf.md5.resize(16);
		for (int i = 0; i < 16; i++) {
			pf.md5.write[i] = hash[i];
		}
	}
	pf.encrypted = p_encrypt;

	uint64_t _size = pf.size;
	if (p_encrypt) { // Add encryption overhead.
		if (_size % 16) { // Pad to encryption block size.
			_size += 16 - (_size % 16);
		}
		_size += 16; // hash
		_size += 8; // data size
		_size += 16; // iv
	}

	int pad = _get_pad(alignment, ofs + _size);
	ofs = ofs + _size + pad;

	files.push_back(pf);

	f->close();
	memdelete(f);

	return OK;
}

Error PCKPacker::flush(bool p_verbose) {
	ERR_FAIL_COND_V_MSG(!file, ERR_INVALID_PARAMETER, "File must be opened before use.");

	int64_t file_base_ofs = file->get_position();
	file->store_64(0); // files base

	for (int i = 0; i < 16; i++) {
		file->store_32(0); // reserved
	}

	// write the index
	file->store_32(files.size());

	FileAccessEncrypted *fae = nullptr;
	FileAccess *fhead = file;

	if (enc_dir) {
		fae = memnew(FileAccessEncrypted);
		ERR_FAIL_COND_V(!fae, ERR_CANT_CREATE);

		Error err = fae->open_and_parse(file, key, FileAccessEncrypted::MODE_WRITE_AES256, false);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);

		fhead = fae;
	}

	for (int i = 0; i < files.size(); i++) {
		int string_len = files[i].path.utf8().length();
		int pad = _get_pad(4, string_len);

		fhead->store_32(string_len + pad);
		fhead->store_buffer((const uint8_t *)files[i].path.utf8().get_data(), string_len);
		for (int j = 0; j < pad; j++) {
			fhead->store_8(0);
		}

		fhead->store_64(files[i].ofs);
		fhead->store_64(files[i].size); // pay attention here, this is where file is
		fhead->store_buffer(files[i].md5.ptr(), 16); //also save md5 for file

		uint32_t flags = 0;
		if (files[i].encrypted) {
			flags |= PACK_FILE_ENCRYPTED;
		}
		fhead->store_32(flags);
	}

	if (fae) {
		fae->release();
		memdelete(fae);
	}

	int header_padding = _get_pad(alignment, file->get_position());
	for (int i = 0; i < header_padding; i++) {
		file->store_8(Math::rand() % 256);
	}

	int64_t file_base = file->get_position();
	file->seek(file_base_ofs);
	file->store_64(file_base); // update files base
	file->seek(file_base);

	const uint32_t buf_max = 65536;
	uint8_t *buf = memnew_arr(uint8_t, buf_max);

	int count = 0;
	for (int i = 0; i < files.size(); i++) {
		FileAccess *src = FileAccess::open(files[i].src_path, FileAccess::READ);
		uint64_t to_write = files[i].size;

		fae = nullptr;
		FileAccess *ftmp = file;
		if (files[i].encrypted) {
			fae = memnew(FileAccessEncrypted);
			ERR_FAIL_COND_V(!fae, ERR_CANT_CREATE);

			Error err = fae->open_and_parse(file, key, FileAccessEncrypted::MODE_WRITE_AES256, false);
			ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
			ftmp = fae;
		}

		while (to_write > 0) {
			int read = src->get_buffer(buf, MIN(to_write, buf_max));
			ftmp->store_buffer(buf, read);
			to_write -= read;
		}

		if (fae) {
			fae->release();
			memdelete(fae);
		}

		int pad = _get_pad(alignment, file->get_position());
		for (int j = 0; j < pad; j++) {
			file->store_8(Math::rand() % 256);
		}

		src->close();
		memdelete(src);
		count += 1;
		const int file_num = files.size();
		if (p_verbose && (file_num > 0)) {
			if (count % 100 == 0) {
				printf("%i/%i (%.2f)\r", count, file_num, float(count) / file_num * 100);
				fflush(stdout);
			}
		}
	}

	if (p_verbose) {
		printf("\n");
	}

	file->close();
	memdelete_arr(buf);

	return OK;
}

PCKPacker::~PCKPacker() {
	if (file != nullptr) {
		memdelete(file);
	}
	file = nullptr;
}
