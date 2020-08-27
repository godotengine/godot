/*************************************************************************/
/*  file_access_jandroid.cpp                                             */
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

#include "file_access_jandroid.h"
#include "core/os/os.h"
#include "thread_jandroid.h"
#include <unistd.h>

jobject FileAccessJAndroid::io = nullptr;
jclass FileAccessJAndroid::cls;
jmethodID FileAccessJAndroid::_file_open = 0;
jmethodID FileAccessJAndroid::_file_get_size = 0;
jmethodID FileAccessJAndroid::_file_seek = 0;
jmethodID FileAccessJAndroid::_file_read = 0;
jmethodID FileAccessJAndroid::_file_tell = 0;
jmethodID FileAccessJAndroid::_file_eof = 0;
jmethodID FileAccessJAndroid::_file_close = 0;

FileAccess *FileAccessJAndroid::create_jandroid() {
	return memnew(FileAccessJAndroid);
}

Error FileAccessJAndroid::_open(const String &p_path, int p_mode_flags) {
	if (is_open())
		close();

	String path = fix_path(p_path).simplify_path();
	if (path.begins_with("/"))
		path = path.substr(1, path.length());
	else if (path.begins_with("res://"))
		path = path.substr(6, path.length());

	JNIEnv *env = ThreadAndroid::get_env();

	jstring js = env->NewStringUTF(path.utf8().get_data());
	int res = env->CallIntMethod(io, _file_open, js, (p_mode_flags & WRITE) ? true : false);
	env->DeleteLocalRef(js);

	OS::get_singleton()->print("fopen: '%s' ret %i\n", path.utf8().get_data(), res);

	if (res <= 0)
		return ERR_FILE_CANT_OPEN;
	id = res;

	return OK;
}

void FileAccessJAndroid::close() {
	if (!is_open())
		return;

	JNIEnv *env = ThreadAndroid::get_env();

	env->CallVoidMethod(io, _file_close, id);
	id = 0;
}

bool FileAccessJAndroid::is_open() const {
	return id != 0;
}

void FileAccessJAndroid::seek(size_t p_position) {
	JNIEnv *env = ThreadAndroid::get_env();

	ERR_FAIL_COND_MSG(!is_open(), "File must be opened before use.");
	env->CallVoidMethod(io, _file_seek, id, p_position);
}

void FileAccessJAndroid::seek_end(int64_t p_position) {
	ERR_FAIL_COND_MSG(!is_open(), "File must be opened before use.");

	seek(get_len());
}

size_t FileAccessJAndroid::get_position() const {
	JNIEnv *env = ThreadAndroid::get_env();
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "File must be opened before use.");
	return env->CallIntMethod(io, _file_tell, id);
}

size_t FileAccessJAndroid::get_len() const {
	JNIEnv *env = ThreadAndroid::get_env();
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "File must be opened before use.");
	return env->CallIntMethod(io, _file_get_size, id);
}

bool FileAccessJAndroid::eof_reached() const {
	JNIEnv *env = ThreadAndroid::get_env();
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "File must be opened before use.");
	return env->CallIntMethod(io, _file_eof, id);
}

uint8_t FileAccessJAndroid::get_8() const {
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "File must be opened before use.");
	uint8_t byte;
	get_buffer(&byte, 1);
	return byte;
}

int FileAccessJAndroid::get_buffer(uint8_t *p_dst, int p_length) const {
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "File must be opened before use.");
	if (p_length == 0)
		return 0;
	JNIEnv *env = ThreadAndroid::get_env();

	jbyteArray jca = (jbyteArray)env->CallObjectMethod(io, _file_read, id, p_length);

	int len = env->GetArrayLength(jca);
	env->GetByteArrayRegion(jca, 0, len, (jbyte *)p_dst);
	env->DeleteLocalRef((jobject)jca);

	return len;
}

Error FileAccessJAndroid::get_error() const {
	if (eof_reached())
		return ERR_FILE_EOF;
	return OK;
}

void FileAccessJAndroid::flush() {
}

void FileAccessJAndroid::store_8(uint8_t p_dest) {
}

bool FileAccessJAndroid::file_exists(const String &p_path) {
	JNIEnv *env = ThreadAndroid::get_env();

	String path = fix_path(p_path).simplify_path();
	if (path.begins_with("/"))
		path = path.substr(1, path.length());
	else if (path.begins_with("res://"))
		path = path.substr(6, path.length());

	jstring js = env->NewStringUTF(path.utf8().get_data());
	int res = env->CallIntMethod(io, _file_open, js, false);
	if (res <= 0) {
		env->DeleteLocalRef(js);
		return false;
	}
	env->CallVoidMethod(io, _file_close, res);
	env->DeleteLocalRef(js);
	return true;
}

void FileAccessJAndroid::setup(jobject p_io) {
	io = p_io;
	JNIEnv *env = ThreadAndroid::get_env();

	jclass c = env->GetObjectClass(io);
	cls = (jclass)env->NewGlobalRef(c);

	_file_open = env->GetMethodID(cls, "file_open", "(Ljava/lang/String;Z)I");
	_file_get_size = env->GetMethodID(cls, "file_get_size", "(I)I");
	_file_tell = env->GetMethodID(cls, "file_tell", "(I)I");
	_file_eof = env->GetMethodID(cls, "file_eof", "(I)Z");
	_file_seek = env->GetMethodID(cls, "file_seek", "(II)V");
	_file_read = env->GetMethodID(cls, "file_read", "(II)[B");
	_file_close = env->GetMethodID(cls, "file_close", "(I)V");
}

FileAccessJAndroid::FileAccessJAndroid() {
	id = 0;
}

FileAccessJAndroid::~FileAccessJAndroid() {
	if (is_open())
		close();
}
