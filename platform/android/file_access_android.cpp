/**************************************************************************/
/*  file_access_android.cpp                                               */
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

#include "file_access_android.h"

#include "core/object/script_language.h"
#include "core/string/print_string.h"
#include "core/version.h"

AAssetManager *FileAccessAndroid::asset_manager = nullptr;
HashMap<String, String> FileAccessAndroid::directory;
bool FileAccessAndroid::dir_loaded = false;

String FileAccessAndroid::get_path() const {
	return path_src;
}

String FileAccessAndroid::get_path_absolute() const {
	return absolute_path;
}

void FileAccessAndroid::_load_encrypted_directory() {
	if (!dir_loaded) {
		dir_loaded = true;

		const String &enc_path = ".encrypted/directory";

		Ref<FileAccessAndroid> f;
		f.instantiate();
		ERR_FAIL_COND_MSG(f.is_null(), "Can't open encrypted file directory.");

		f->asset = AAssetManager_open(asset_manager, enc_path.utf8().get_data(), AASSET_MODE_STREAMING);
		if (f->asset) {
			f->path_src = enc_path;
			f->absolute_path = enc_path;
			f->len = AAsset_getLength(f->asset);
			f->pos = 0;
			f->eof = false;

			uint32_t magic = f->get_32();
			if (magic == DIR_HEADER_MAGIC) {
				uint32_t version = f->get_32();
				uint32_t ver_major = f->get_32();
				uint32_t ver_minor = f->get_32();
				f->get_32(); // patch number, not used for validation.

				ERR_FAIL_COND_MSG(version != PACK_FORMAT_VERSION, "Directory version unsupported: " + itos(version) + ".");
				ERR_FAIL_COND_MSG(ver_major > VERSION_MAJOR || (ver_major == VERSION_MAJOR && ver_minor > VERSION_MINOR), "Directory created with a newer version of the engine: " + itos(ver_major) + "." + itos(ver_minor) + ".");

				uint32_t pack_flags = f->get_32();

				bool enc_directory = (pack_flags & PACK_DIR_ENCRYPTED);

				for (int i = 0; i < 16; i++) {
					//reserved
					f->get_32();
				}

				int file_count = f->get_32();

				Ref<FileAccess> fhead = f;
				if (enc_directory) {
					Ref<FileAccessEncrypted> fae;
					fae.instantiate();
					ERR_FAIL_COND_MSG(fae.is_null(), "Can't open encrypted directory.");

					Vector<uint8_t> key;
					key.resize(32);
					for (int i = 0; i < key.size(); i++) {
						key.write[i] = script_encryption_key[i];
					}

					Error err = fae->open_and_parse(f, key, FileAccessEncrypted::MODE_READ, false);
					ERR_FAIL_COND_MSG(err, "Can't open encrypted directory.");
					fhead = fae;
				}

				for (int i = 0; i < file_count; i++) {
					uint32_t sl = fhead->get_32();
					CharString cs;
					cs.resize(sl + 1);
					fhead->get_buffer((uint8_t *)cs.ptr(), sl);
					cs[sl] = 0;

					String path;
					path.parse_utf8(cs.ptr());

					sl = fhead->get_32();
					cs.resize(sl + 1);
					fhead->get_buffer((uint8_t *)cs.ptr(), sl);
					cs[sl] = 0;

					String enc_path;
					enc_path.parse_utf8(cs.ptr());

					directory[path] = enc_path;
				}
			}
		}
	}
}

Error FileAccessAndroid::open_internal(const String &p_path, int p_mode_flags) {
	_close();

	path_src = p_path;
	String path = fix_path(p_path).simplify_path();
	absolute_path = path;
	if (path.begins_with("/")) {
		path = path.substr(1, path.length());
	} else if (path.begins_with("res://")) {
		path = path.substr(6, path.length());
	}

	ERR_FAIL_COND_V(p_mode_flags & FileAccess::WRITE, ERR_UNAVAILABLE); //can't write on android..

	_load_encrypted_directory();
	if (directory.has(path)) {
		const String &enc_path = ".encrypted/" + directory[path];

		Ref<FileAccessAndroid> f;
		f.instantiate();
		ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Can't open encrypted file '" + path + "'.");

		f->asset = AAssetManager_open(asset_manager, enc_path.utf8().get_data(), AASSET_MODE_STREAMING);
		if (!f->asset) {
			return ERR_CANT_OPEN;
		}
		f->path_src = enc_path;
		f->absolute_path = enc_path;
		f->len = AAsset_getLength(f->asset);
		f->pos = 0;
		f->eof = false;

		fae.instantiate();
		ERR_FAIL_COND_V_MSG(fae.is_null(), ERR_CANT_OPEN, "Can't open encrypted file '" + path + "'.");

		Vector<uint8_t> key;
		key.resize(32);
		for (int i = 0; i < key.size(); i++) {
			key.write[i] = script_encryption_key[i];
		}

		Error err = fae->open_and_parse(f, key, FileAccessEncrypted::MODE_READ, false);
		if (err != OK) {
			if (fae.is_valid()) {
				fae.unref();
			}
			ERR_FAIL_V_MSG(err, "Can't open encrypted file '" + path + "'.");
		}
	} else {
		asset = AAssetManager_open(asset_manager, path.utf8().get_data(), AASSET_MODE_STREAMING);
		if (!asset) {
			return ERR_CANT_OPEN;
		}
		len = AAsset_getLength(asset);
		pos = 0;
		eof = false;
	}

	return OK;
}

void FileAccessAndroid::_close() {
	if (fae.is_valid()) {
		fae.unref();
	}

	if (!asset) {
		return;
	}
	AAsset_close(asset);
	asset = nullptr;
}

bool FileAccessAndroid::is_open() const {
	return asset != nullptr;
}

void FileAccessAndroid::seek(uint64_t p_position) {
	if (fae.is_valid()) {
		fae->seek(p_position);
	} else {
		ERR_FAIL_NULL(asset);

		AAsset_seek(asset, p_position, SEEK_SET);
		pos = p_position;
		if (pos > len) {
			pos = len;
			eof = true;
		} else {
			eof = false;
		}
	}
}

void FileAccessAndroid::seek_end(int64_t p_position) {
	if (fae.is_valid()) {
		fae->seek_end(p_position);
	} else {
		ERR_FAIL_NULL(asset);

		AAsset_seek(asset, p_position, SEEK_END);
		pos = len + p_position;
	}
}

uint64_t FileAccessAndroid::get_position() const {
	if (fae.is_valid()) {
		return fae->get_position();
	} else {
		return pos;
	}
}

uint64_t FileAccessAndroid::get_length() const {
	if (fae.is_valid()) {
		return fae->get_length();
	} else {
		return len;
	}
}

bool FileAccessAndroid::eof_reached() const {
	if (fae.is_valid()) {
		return fae->eof_reached();
	} else {
		return eof;
	}
}

uint8_t FileAccessAndroid::get_8() const {
	if (fae.is_valid()) {
		return fae->get_8();
	} else {
		if (pos >= len) {
			eof = true;
			return 0;
		}

		uint8_t byte;
		AAsset_read(asset, &byte, 1);
		pos++;
		return byte;
	}
}

uint64_t FileAccessAndroid::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);

	if (fae.is_valid()) {
		return fae->get_buffer(p_dst, p_length);
	} else {
		int r = AAsset_read(asset, p_dst, p_length);

		if (pos + p_length > len) {
			eof = true;
		}

		if (r >= 0) {
			pos += r;
			if (pos > len) {
				pos = len;
			}
		}
		return r;
	}
}

Error FileAccessAndroid::get_error() const {
	if (fae.is_valid()) {
		return fae->get_error();
	} else {
		return eof ? ERR_FILE_EOF : OK; // not sure what else it may happen
	}
}

void FileAccessAndroid::flush() {
	ERR_FAIL();
}

void FileAccessAndroid::store_8(uint8_t p_dest) {
	ERR_FAIL();
}

bool FileAccessAndroid::file_exists(const String &p_path) {
	String path = fix_path(p_path).simplify_path();
	if (path.begins_with("/")) {
		path = path.substr(1, path.length());
	} else if (path.begins_with("res://")) {
		path = path.substr(6, path.length());
	}

	_load_encrypted_directory();
	if (directory.has(path)) {
		path = ".encrypted/" + directory[path];
	}

	AAsset *at = AAssetManager_open(asset_manager, path.utf8().get_data(), AASSET_MODE_STREAMING);

	if (!at) {
		return false;
	}

	AAsset_close(at);
	return true;
}

void FileAccessAndroid::close() {
	_close();
}

FileAccessAndroid::~FileAccessAndroid() {
	_close();
}
