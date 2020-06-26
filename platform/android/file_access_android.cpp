/*************************************************************************/
/*  file_access_android.cpp                                              */
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

#include "file_access_android.h"
#include "core/print_string.h"

AAssetManager *FileAccessAndroid::asset_manager = nullptr;

/*void FileAccessAndroid::make_default() {

	create_func=create_android;
}*/

FileAccess *FileAccessAndroid::create_android() {
	return memnew(FileAccessAndroid);
}

Error FileAccessAndroid::_open(const String &p_path, int p_mode_flags) {
	String path = fix_path(p_path).simplify_path();
	if (path.begins_with("/"))
		path = path.substr(1, path.length());
	else if (path.begins_with("res://"))
		path = path.substr(6, path.length());

	ERR_FAIL_COND_V(p_mode_flags & FileAccess::WRITE, ERR_UNAVAILABLE); //can't write on android..
	a = AAssetManager_open(asset_manager, path.utf8().get_data(), AASSET_MODE_STREAMING);
	if (!a)
		return ERR_CANT_OPEN;
	//ERR_FAIL_COND_V(!a,ERR_FILE_NOT_FOUND);
	len = AAsset_getLength(a);
	pos = 0;
	eof = false;

	return OK;
}

void FileAccessAndroid::close() {
	if (!a)
		return;
	AAsset_close(a);
	a = nullptr;
}

bool FileAccessAndroid::is_open() const {
	return a != nullptr;
}

void FileAccessAndroid::seek(size_t p_position) {
	ERR_FAIL_COND(!a);
	AAsset_seek(a, p_position, SEEK_SET);
	pos = p_position;
	if (pos > len) {
		pos = len;
		eof = true;
	} else {
		eof = false;
	}
}

void FileAccessAndroid::seek_end(int64_t p_position) {
	ERR_FAIL_COND(!a);
	AAsset_seek(a, p_position, SEEK_END);
	pos = len + p_position;
}

size_t FileAccessAndroid::get_position() const {
	return pos;
}

size_t FileAccessAndroid::get_len() const {
	return len;
}

bool FileAccessAndroid::eof_reached() const {
	return eof;
}

uint8_t FileAccessAndroid::get_8() const {
	if (pos >= len) {
		eof = true;
		return 0;
	}

	uint8_t byte;
	AAsset_read(a, &byte, 1);
	pos++;
	return byte;
}

int FileAccessAndroid::get_buffer(uint8_t *p_dst, int p_length) const {
	off_t r = AAsset_read(a, p_dst, p_length);

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

Error FileAccessAndroid::get_error() const {
	return eof ? ERR_FILE_EOF : OK; //not sure what else it may happen
}

void FileAccessAndroid::flush() {
	ERR_FAIL();
}

void FileAccessAndroid::store_8(uint8_t p_dest) {
	ERR_FAIL();
}

bool FileAccessAndroid::file_exists(const String &p_path) {
	String path = fix_path(p_path).simplify_path();
	if (path.begins_with("/"))
		path = path.substr(1, path.length());
	else if (path.begins_with("res://"))
		path = path.substr(6, path.length());

	AAsset *at = AAssetManager_open(asset_manager, path.utf8().get_data(), AASSET_MODE_STREAMING);

	if (!at)
		return false;

	AAsset_close(at);
	return true;
}

FileAccessAndroid::FileAccessAndroid() {
	a = nullptr;
	eof = false;
}

FileAccessAndroid::~FileAccessAndroid() {
	close();
}
