/*************************************************************************/
/*  file_access_encrypted.cpp                                            */
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
#include "file_access_encrypted.h"

#include "core/variant.h"
#include "os/copymem.h"
#include "print_string.h"

#include "thirdparty/misc/aes256.h"
#include "thirdparty/misc/md5.h"

#include <stdio.h>

#define COMP_MAGIC 0x43454447

Error FileAccessEncrypted::open_and_parse(FileAccess *p_base, const Vector<uint8_t> &p_key, Mode p_mode) {

	//print_line("open and parse!");
	ERR_FAIL_COND_V(file != NULL, ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(p_key.size() != 32, ERR_INVALID_PARAMETER);

	pos = 0;
	eofed = false;

	if (p_mode == MODE_WRITE_AES256) {

		data.clear();
		writing = true;
		file = p_base;
		mode = p_mode;
		key = p_key;

	} else if (p_mode == MODE_READ) {

		writing = false;
		key = p_key;
		uint32_t magic = p_base->get_32();
		ERR_FAIL_COND_V(magic != COMP_MAGIC, ERR_FILE_UNRECOGNIZED);

		mode = Mode(p_base->get_32());
		ERR_FAIL_INDEX_V(mode, MODE_MAX, ERR_FILE_CORRUPT);
		ERR_FAIL_COND_V(mode == 0, ERR_FILE_CORRUPT);

		unsigned char md5d[16];
		p_base->get_buffer(md5d, 16);
		length = p_base->get_64();
		base = p_base->get_position();
		ERR_FAIL_COND_V(p_base->get_len() < base + length, ERR_FILE_CORRUPT);
		uint32_t ds = length;
		if (ds % 16) {
			ds += 16 - (ds % 16);
		}

		data.resize(ds);

		uint32_t blen = p_base->get_buffer(data.ptrw(), ds);
		ERR_FAIL_COND_V(blen != ds, ERR_FILE_CORRUPT);

		aes256_context ctx;
		aes256_init(&ctx, key.ptrw());

		for (size_t i = 0; i < ds; i += 16) {

			aes256_decrypt_ecb(&ctx, &data[i]);
		}

		aes256_done(&ctx);

		data.resize(length);

		MD5_CTX md5;
		MD5Init(&md5);
		MD5Update(&md5, (uint8_t *)data.ptr(), data.size());
		MD5Final(&md5);

		ERR_FAIL_COND_V(String::md5(md5.digest) != String::md5(md5d), ERR_FILE_CORRUPT);

		file = p_base;
	}

	return OK;
}

Error FileAccessEncrypted::open_and_parse_password(FileAccess *p_base, const String &p_key, Mode p_mode) {

	String cs = p_key.md5_text();
	ERR_FAIL_COND_V(cs.length() != 32, ERR_INVALID_PARAMETER);
	Vector<uint8_t> key;
	key.resize(32);
	for (int i = 0; i < 32; i++) {

		key[i] = cs[i];
	}

	return open_and_parse(p_base, key, p_mode);
}

Error FileAccessEncrypted::_open(const String &p_path, int p_mode_flags) {

	return OK;
}
void FileAccessEncrypted::close() {

	if (!file)
		return;

	if (writing) {

		Vector<uint8_t> compressed;
		size_t len = data.size();
		if (len % 16) {
			len += 16 - (len % 16);
		}

		MD5_CTX md5;
		MD5Init(&md5);
		MD5Update(&md5, (uint8_t *)data.ptr(), data.size());
		MD5Final(&md5);

		compressed.resize(len);
		zeromem(compressed.ptrw(), len);
		for (int i = 0; i < data.size(); i++) {
			compressed[i] = data[i];
		}

		aes256_context ctx;
		aes256_init(&ctx, key.ptrw());

		for (size_t i = 0; i < len; i += 16) {

			aes256_encrypt_ecb(&ctx, &compressed[i]);
		}

		aes256_done(&ctx);

		file->store_32(COMP_MAGIC);
		file->store_32(mode);

		file->store_buffer(md5.digest, 16);
		file->store_64(data.size());

		file->store_buffer(compressed.ptr(), compressed.size());
		file->close();
		memdelete(file);
		file = NULL;
		data.clear();

	} else {

		file->close();
		memdelete(file);
		data.clear();
		file = NULL;
	}
}

bool FileAccessEncrypted::is_open() const {

	return file != NULL;
}

void FileAccessEncrypted::seek(size_t p_position) {

	if (p_position > (size_t)data.size())
		p_position = data.size();

	pos = p_position;
	eofed = false;
}

void FileAccessEncrypted::seek_end(int64_t p_position) {

	seek(data.size() + p_position);
}
size_t FileAccessEncrypted::get_position() const {

	return pos;
}
size_t FileAccessEncrypted::get_len() const {

	return data.size();
}

bool FileAccessEncrypted::eof_reached() const {

	return eofed;
}

uint8_t FileAccessEncrypted::get_8() const {

	ERR_FAIL_COND_V(writing, 0);
	if (pos >= data.size()) {
		eofed = true;
		return 0;
	}

	uint8_t b = data[pos];
	pos++;
	return b;
}
int FileAccessEncrypted::get_buffer(uint8_t *p_dst, int p_length) const {

	ERR_FAIL_COND_V(writing, 0);

	int to_copy = MIN(p_length, data.size() - pos);
	for (int i = 0; i < to_copy; i++) {

		p_dst[i] = data[pos++];
	}

	if (to_copy < p_length) {
		eofed = true;
	}

	return to_copy;
}

Error FileAccessEncrypted::get_error() const {

	return eofed ? ERR_FILE_EOF : OK;
}

void FileAccessEncrypted::store_buffer(const uint8_t *p_src, int p_length) {

	ERR_FAIL_COND(!writing);

	if (pos < data.size()) {

		for (int i = 0; i < p_length; i++) {

			store_8(p_src[i]);
		}
	} else if (pos == data.size()) {

		data.resize(pos + p_length);
		for (int i = 0; i < p_length; i++) {

			data[pos + i] = p_src[i];
		}
		pos += p_length;
	}
}

void FileAccessEncrypted::flush() {
	ERR_FAIL_COND(!writing);

	// encrypted files keep data in memory till close()
}

void FileAccessEncrypted::store_8(uint8_t p_dest) {

	ERR_FAIL_COND(!writing);

	if (pos < data.size()) {
		data[pos] = p_dest;
		pos++;
	} else if (pos == data.size()) {
		data.push_back(p_dest);
		pos++;
	}
}

bool FileAccessEncrypted::file_exists(const String &p_name) {

	FileAccess *fa = FileAccess::open(p_name, FileAccess::READ);
	if (!fa)
		return false;
	memdelete(fa);
	return true;
}

uint64_t FileAccessEncrypted::_get_modified_time(const String &p_file) {

	return 0;
}

FileAccessEncrypted::FileAccessEncrypted() {

	file = NULL;
	pos = 0;
	eofed = false;
	mode = MODE_MAX;
	writing = false;
}

FileAccessEncrypted::~FileAccessEncrypted() {

	if (file)
		close();
}
