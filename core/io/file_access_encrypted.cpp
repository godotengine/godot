/*************************************************************************/
/*  file_access_encrypted.cpp                                            */
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

#include "file_access_encrypted.h"

#include "core/crypto/crypto_core.h"
#include "core/os/copymem.h"
#include "core/print_string.h"
#include "core/variant.h"

#include <stdio.h>

Error FileAccessEncrypted::open_and_parse(FileAccess *p_base, const Vector<uint8_t> &p_key, Mode p_mode, bool p_with_magic) {
	ERR_FAIL_COND_V_MSG(file != nullptr, ERR_ALREADY_IN_USE, "Can't open file while another file from path '" + file->get_path_absolute() + "' is open.");
	ERR_FAIL_COND_V(p_key.size() != 32, ERR_INVALID_PARAMETER);

	pos = 0;
	eofed = false;
	use_magic = p_with_magic;

	if (p_mode == MODE_WRITE_AES256) {
		data.clear();
		writing = true;
		file = p_base;
		key = p_key;

	} else if (p_mode == MODE_READ) {
		writing = false;
		key = p_key;

		if (use_magic) {
			uint32_t magic = p_base->get_32();
			ERR_FAIL_COND_V(magic != ENCRYPTED_HEADER_MAGIC, ERR_FILE_UNRECOGNIZED);
		}

		unsigned char md5d[16];
		p_base->get_buffer(md5d, 16);
		length = p_base->get_64();

		unsigned char iv[16];
		for (int i = 0; i < 16; i++) {
			iv[i] = p_base->get_8();
		}

		base = p_base->get_position();
		ERR_FAIL_COND_V(p_base->get_len() < base + length, ERR_FILE_CORRUPT);
		uint32_t ds = length;
		if (ds % 16) {
			ds += 16 - (ds % 16);
		}
		data.resize(ds);

		uint32_t blen = p_base->get_buffer(data.ptrw(), ds);
		ERR_FAIL_COND_V(blen != ds, ERR_FILE_CORRUPT);

		{
			CryptoCore::AESContext ctx;

			ctx.set_encode_key(key.ptrw(), 256); // Due to the nature of CFB, same key schedule is used for both encryption and decryption!
			ctx.decrypt_cfb(ds, iv, data.ptrw(), data.ptrw());
		}

		data.resize(length);

		unsigned char hash[16];
		ERR_FAIL_COND_V(CryptoCore::md5(data.ptr(), data.size(), hash) != OK, ERR_BUG);

		ERR_FAIL_COND_V_MSG(String::md5(hash) != String::md5(md5d), ERR_FILE_CORRUPT, "The MD5 sum of the decrypted file does not match the expected value. It could be that the file is corrupt, or that the provided decryption key is invalid.");

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
		key.write[i] = cs[i];
	}

	return open_and_parse(p_base, key, p_mode);
}

Error FileAccessEncrypted::_open(const String &p_path, int p_mode_flags) {
	return OK;
}

void FileAccessEncrypted::close() {
	if (!file) {
		return;
	}

	_release();

	file->close();
	memdelete(file);

	file = nullptr;
}

void FileAccessEncrypted::release() {
	if (!file) {
		return;
	}

	_release();

	file = nullptr;
}

void FileAccessEncrypted::_release() {
	if (writing) {
		Vector<uint8_t> compressed;
		size_t len = data.size();
		if (len % 16) {
			len += 16 - (len % 16);
		}

		unsigned char hash[16];
		ERR_FAIL_COND(CryptoCore::md5(data.ptr(), data.size(), hash) != OK); // Bug?

		compressed.resize(len);
		zeromem(compressed.ptrw(), len);
		for (int i = 0; i < data.size(); i++) {
			compressed.write[i] = data[i];
		}

		CryptoCore::AESContext ctx;
		ctx.set_encode_key(key.ptrw(), 256);

		if (use_magic) {
			file->store_32(ENCRYPTED_HEADER_MAGIC);
		}

		file->store_buffer(hash, 16);
		file->store_64(data.size());

		unsigned char iv[16];
		for (int i = 0; i < 16; i++) {
			iv[i] = Math::rand() % 256;
			file->store_8(iv[i]);
		}

		ctx.encrypt_cfb(len, iv, compressed.ptrw(), compressed.ptrw());

		file->store_buffer(compressed.ptr(), compressed.size());
		data.clear();
	}
}

bool FileAccessEncrypted::is_open() const {
	return file != nullptr;
}

String FileAccessEncrypted::get_path() const {
	if (file) {
		return file->get_path();
	} else {
		return "";
	}
}

String FileAccessEncrypted::get_path_absolute() const {
	if (file) {
		return file->get_path_absolute();
	} else {
		return "";
	}
}

void FileAccessEncrypted::seek(size_t p_position) {
	if (p_position > (size_t)data.size()) {
		p_position = data.size();
	}

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
	ERR_FAIL_COND_V_MSG(writing, 0, "File has not been opened in read mode.");
	if (pos >= data.size()) {
		eofed = true;
		return 0;
	}

	uint8_t b = data[pos];
	pos++;
	return b;
}

int FileAccessEncrypted::get_buffer(uint8_t *p_dst, int p_length) const {
	ERR_FAIL_COND_V_MSG(writing, 0, "File has not been opened in read mode.");

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
	ERR_FAIL_COND_MSG(!writing, "File has not been opened in read mode.");

	if (pos < data.size()) {
		for (int i = 0; i < p_length; i++) {
			store_8(p_src[i]);
		}
	} else if (pos == data.size()) {
		data.resize(pos + p_length);
		for (int i = 0; i < p_length; i++) {
			data.write[pos + i] = p_src[i];
		}
		pos += p_length;
	}
}

void FileAccessEncrypted::flush() {
	ERR_FAIL_COND_MSG(!writing, "File has not been opened in read mode.");

	// encrypted files keep data in memory till close()
}

void FileAccessEncrypted::store_8(uint8_t p_dest) {
	ERR_FAIL_COND_MSG(!writing, "File has not been opened in read mode.");

	if (pos < data.size()) {
		data.write[pos] = p_dest;
		pos++;
	} else if (pos == data.size()) {
		data.push_back(p_dest);
		pos++;
	}
}

bool FileAccessEncrypted::file_exists(const String &p_name) {
	FileAccess *fa = FileAccess::open(p_name, FileAccess::READ);
	if (!fa) {
		return false;
	}
	memdelete(fa);
	return true;
}

uint64_t FileAccessEncrypted::_get_modified_time(const String &p_file) {
	return 0;
}

uint32_t FileAccessEncrypted::_get_unix_permissions(const String &p_file) {
	return 0;
}

Error FileAccessEncrypted::_set_unix_permissions(const String &p_file, uint32_t p_permissions) {
	ERR_PRINT("Setting UNIX permissions on encrypted files is not implemented yet.");
	return ERR_UNAVAILABLE;
}

FileAccessEncrypted::~FileAccessEncrypted() {
	if (file) {
		close();
	}
}
