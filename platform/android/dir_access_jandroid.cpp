/*************************************************************************/
/*  dir_access_jandroid.cpp                                              */
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

#include "dir_access_jandroid.h"
#include "core/string/print_string.h"
#include "file_access_android.h"
#include "string_android.h"
#include "thread_jandroid.h"

jobject DirAccessJAndroid::io = nullptr;
jclass DirAccessJAndroid::cls = nullptr;
jmethodID DirAccessJAndroid::_dir_open = nullptr;
jmethodID DirAccessJAndroid::_dir_next = nullptr;
jmethodID DirAccessJAndroid::_dir_close = nullptr;
jmethodID DirAccessJAndroid::_dir_is_dir = nullptr;

DirAccess *DirAccessJAndroid::create_fs() {
	return memnew(DirAccessJAndroid);
}

Error DirAccessJAndroid::list_dir_begin() {
	list_dir_end();
	JNIEnv *env = get_jni_env();

	jstring js = env->NewStringUTF(current_dir.utf8().get_data());
	int res = env->CallIntMethod(io, _dir_open, js);
	if (res <= 0)
		return ERR_CANT_OPEN;

	id = res;

	return OK;
}

String DirAccessJAndroid::get_next() {
	ERR_FAIL_COND_V(id == 0, "");

	JNIEnv *env = get_jni_env();
	jstring str = (jstring)env->CallObjectMethod(io, _dir_next, id);
	if (!str)
		return "";

	String ret = jstring_to_string((jstring)str, env);
	env->DeleteLocalRef((jobject)str);
	return ret;
}

bool DirAccessJAndroid::current_is_dir() const {
	JNIEnv *env = get_jni_env();

	return env->CallBooleanMethod(io, _dir_is_dir, id);
}

bool DirAccessJAndroid::current_is_hidden() const {
	return current != "." && current != ".." && current.begins_with(".");
}

void DirAccessJAndroid::list_dir_end() {
	if (id == 0)
		return;

	JNIEnv *env = get_jni_env();
	env->CallVoidMethod(io, _dir_close, id);
	id = 0;
}

int DirAccessJAndroid::get_drive_count() {
	return 0;
}

String DirAccessJAndroid::get_drive(int p_drive) {
	return "";
}

Error DirAccessJAndroid::change_dir(String p_dir) {
	JNIEnv *env = get_jni_env();

	if (p_dir.is_empty() || p_dir == "." || (p_dir == ".." && current_dir.is_empty()))
		return OK;

	String new_dir;

	if (p_dir != "res://" && p_dir.length() > 1 && p_dir.ends_with("/"))
		p_dir = p_dir.substr(0, p_dir.length() - 1);

	if (p_dir.begins_with("/"))
		new_dir = p_dir.substr(1, p_dir.length());
	else if (p_dir.begins_with("res://"))
		new_dir = p_dir.substr(6, p_dir.length());
	else if (current_dir.is_empty())
		new_dir = p_dir;
	else
		new_dir = current_dir.plus_file(p_dir);

	//test if newdir exists
	new_dir = new_dir.simplify_path();

	jstring js = env->NewStringUTF(new_dir.utf8().get_data());
	int res = env->CallIntMethod(io, _dir_open, js);
	env->DeleteLocalRef(js);
	if (res <= 0)
		return ERR_INVALID_PARAMETER;

	env->CallVoidMethod(io, _dir_close, res);

	current_dir = new_dir;

	return OK;
}

String DirAccessJAndroid::get_current_dir(bool p_include_drive) {
	return "res://" + current_dir;
}

bool DirAccessJAndroid::file_exists(String p_file) {
	String sd;
	if (current_dir.is_empty())
		sd = p_file;
	else
		sd = current_dir.plus_file(p_file);

	FileAccessAndroid *f = memnew(FileAccessAndroid);
	bool exists = f->file_exists(sd);
	memdelete(f);

	return exists;
}

bool DirAccessJAndroid::dir_exists(String p_dir) {
	JNIEnv *env = get_jni_env();

	String sd;

	if (current_dir.is_empty())
		sd = p_dir;
	else {
		if (p_dir.is_relative_path())
			sd = current_dir.plus_file(p_dir);
		else
			sd = fix_path(p_dir);
	}

	String path = sd.simplify_path();

	if (path.begins_with("/"))
		path = path.substr(1, path.length());
	else if (path.begins_with("res://"))
		path = path.substr(6, path.length());

	jstring js = env->NewStringUTF(path.utf8().get_data());
	int res = env->CallIntMethod(io, _dir_open, js);
	env->DeleteLocalRef(js);
	if (res <= 0)
		return false;

	env->CallVoidMethod(io, _dir_close, res);

	return true;
}

Error DirAccessJAndroid::make_dir(String p_dir) {
	ERR_FAIL_V(ERR_UNAVAILABLE);
}

Error DirAccessJAndroid::rename(String p_from, String p_to) {
	ERR_FAIL_V(ERR_UNAVAILABLE);
}

Error DirAccessJAndroid::remove(String p_name) {
	ERR_FAIL_V(ERR_UNAVAILABLE);
}

String DirAccessJAndroid::get_filesystem_type() const {
	return "APK";
}

uint64_t DirAccessJAndroid::get_space_left() {
	return 0;
}

void DirAccessJAndroid::setup(jobject p_io) {
	JNIEnv *env = get_jni_env();
	io = p_io;

	jclass c = env->GetObjectClass(io);
	cls = (jclass)env->NewGlobalRef(c);

	_dir_open = env->GetMethodID(cls, "dir_open", "(Ljava/lang/String;)I");
	_dir_next = env->GetMethodID(cls, "dir_next", "(I)Ljava/lang/String;");
	_dir_close = env->GetMethodID(cls, "dir_close", "(I)V");
	_dir_is_dir = env->GetMethodID(cls, "dir_is_dir", "(I)Z");

	//(*env)->CallVoidMethod(env,obj,aMethodID, myvar);
}

DirAccessJAndroid::DirAccessJAndroid() {
	id = 0;
}

DirAccessJAndroid::~DirAccessJAndroid() {
	list_dir_end();
}
