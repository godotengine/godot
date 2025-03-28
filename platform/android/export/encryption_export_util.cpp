/**************************************************************************/
/*  encryption_export_util.cpp                                            */
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

#include "encryption_export_util.h"

#include "core/crypto/crypto_core.h"
#include "core/io/file_access_encrypted.h"
#include "core/io/file_access_memory.h"
#include "core/io/file_access_pack.h"
#include "core/math/random_pcg.h"
#include "core/version.h"
#include "editor/export/editor_export_platform.h"

constexpr int64_t DIRECTORY_HEADER_SIZE = 23 * sizeof(uint64_t); // Directory header size.
constexpr int64_t ENCRYPTION_OVERHEAD = 64; // Hash + IV + data size.

String encrypt_file(Vector<uint8_t> &r_enc_data, HashSet<String> r_ids, HashMap<String, String> &r_directory, const String &p_path, const Vector<uint8_t> &p_data, const Vector<uint8_t> &p_key, uint64_t p_seed) {
	String path = p_path;
	if (path.begins_with("/")) {
		path = path.substr(1, path.length());
	} else if (path.begins_with("res://")) {
		path = path.substr(6, path.length());
	}

	String id;
	do {
		// Generate random ID.
		CharString cs = path.utf8();
		for (int i = 0; i < 16; i++) {
			cs += char(32 + Math::rand() % 220);
		}
		unsigned char hash[32];
		CryptoCore::sha256((unsigned char *)cs.ptr(), cs.length(), hash);
		id = String::hex_encode_buffer(hash, 32);
	} while (r_ids.has(id));

	r_ids.insert(id);

	{
		uint64_t len = p_data.size();
		len += ENCRYPTION_OVERHEAD; // Encryption overhead (hash + iv + size).
		if (len % 16) {
			len += 16 - (len % 16);
		}
		r_enc_data.resize(len);
	}

	Vector<uint8_t> iv;
	if (p_seed != 0) {
		uint64_t seed = p_seed;

		const uint8_t *ptr = p_data.ptr();
		uint64_t len = p_data.size();
		for (uint64_t i = 0; i < len; i++) {
			seed = ((seed << 5) + seed) ^ ptr[i];
		}

		RandomPCG rng = RandomPCG(seed);
		iv.resize(16);
		for (int i = 0; i < 16; i++) {
			iv.write[i] = rng.rand() % 256;
		}
	}

	Ref<FileAccessMemory> fmem;
	fmem.instantiate();
	ERR_FAIL_COND_V(fmem.is_null(), String());
	Error err = fmem->open_custom(r_enc_data.ptrw(), r_enc_data.size());
	ERR_FAIL_COND_V(err != OK, String());

	Ref<FileAccessEncrypted> fae;
	fae.instantiate();
	ERR_FAIL_COND_V(fae.is_null(), String());
	err = fae->open_and_parse(fmem, p_key, FileAccessEncrypted::MODE_WRITE_AES256, false, iv);
	ERR_FAIL_COND_V(err != OK, String());

	// Store file content.
	fae->store_buffer(p_data.ptr(), p_data.size());

	fae.unref();
	fmem.unref();

	r_directory[path] = id;

	return id;
}

Error encrypt_directory(const HashMap<String, String> &p_directory, const String &p_script_key, Vector<uint8_t> &r_dir_data) {
	uint64_t len = DIRECTORY_HEADER_SIZE; // Header size.
	for (const KeyValue<String, String> &E : p_directory) {
		uint32_t string_len = E.key.utf8().length();
		uint32_t pad = EditorExportPlatform::_get_pad(sizeof(uint32_t), string_len);
		len += (sizeof(uint32_t) + string_len + pad);

		string_len = E.value.utf8().length();
		pad = EditorExportPlatform::_get_pad(sizeof(uint32_t), string_len);
		len += (sizeof(uint32_t) + string_len + pad);
	}
	len += ENCRYPTION_OVERHEAD; // Encryption overhead.
	if (len % 16) {
		len += 16 - (len % 16); // Alignment.
	}
	r_dir_data.resize(len);
	memset(r_dir_data.ptrw(), 0, len);

	Ref<FileAccessMemory> fmem;
	fmem.instantiate();
	ERR_FAIL_COND_V(fmem.is_null(), ERR_CANT_CREATE);
	Error err = fmem->open_custom(r_dir_data.ptrw(), r_dir_data.size());
	ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);

	fmem->store_32(DIR_HEADER_MAGIC);
	fmem->store_32(PACK_FORMAT_VERSION);
	fmem->store_32(VERSION_MAJOR);
	fmem->store_32(VERSION_MINOR);
	fmem->store_32(VERSION_PATCH);
	fmem->store_32(PACK_DIR_ENCRYPTED); // Flags.

	for (int i = 0; i < 16; i++) {
		// Reserved.
		fmem->store_32(0);
	}

	fmem->store_32(p_directory.size()); // Amount of files.

	Ref<FileAccessEncrypted> fae;
	Ref<FileAccess> fhead = fmem;
	Vector<uint8_t> key;
	key.resize(32);
	if (p_script_key.length() == 64) {
		for (int i = 0; i < 32; i++) {
			int v = 0;
			if (i * 2 < p_script_key.length()) {
				char32_t ct = p_script_key[i * 2];
				if (is_digit(ct)) {
					ct = ct - '0';
				} else if (ct >= 'a' && ct <= 'f') {
					ct = 10 + ct - 'a';
				}
				v |= ct << 4;
			}

			if (i * 2 + 1 < p_script_key.length()) {
				char32_t ct = p_script_key[i * 2 + 1];
				if (is_digit(ct)) {
					ct = ct - '0';
				} else if (ct >= 'a' && ct <= 'f') {
					ct = 10 + ct - 'a';
				}
				v |= ct;
			}
			key.write[i] = v;
		}
	}
	fae.instantiate();
	if (fae.is_null()) {
		return ERR_CANT_CREATE;
	}

	err = fae->open_and_parse(fmem, key, FileAccessEncrypted::MODE_WRITE_AES256, false);
	if (err != OK) {
		return err;
	}
	fhead = fae;

	for (const KeyValue<String, String> &E : p_directory) {
		uint32_t string_len = E.key.utf8().length();
		uint32_t pad = EditorExportPlatform::_get_pad(sizeof(uint32_t), string_len);

		fhead->store_32(string_len + pad);
		fhead->store_buffer((const uint8_t *)E.key.utf8().get_data(), string_len);
		for (uint32_t j = 0; j < pad; j++) {
			fhead->store_8(0);
		}

		string_len = E.value.utf8().length();
		pad = EditorExportPlatform::_get_pad(sizeof(uint32_t), string_len);

		fhead->store_32(string_len + pad);
		fhead->store_buffer((const uint8_t *)E.value.utf8().get_data(), string_len);
		for (uint32_t j = 0; j < pad; j++) {
			fhead->store_8(0);
		}
	}

	if (fae.is_valid()) {
		fhead.unref();
		fae.unref();
	}

	fmem.unref();

	return OK;
}
