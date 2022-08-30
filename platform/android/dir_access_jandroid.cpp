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
#include "core/print_string.h"
#include "string_android.h"
#include "thread_jandroid.h"

jobject DirAccessJAndroid::dir_access_handler = nullptr;
jclass DirAccessJAndroid::cls = nullptr;
jmethodID DirAccessJAndroid::_dir_open = nullptr;
jmethodID DirAccessJAndroid::_dir_next = nullptr;
jmethodID DirAccessJAndroid::_dir_close = nullptr;
jmethodID DirAccessJAndroid::_dir_is_dir = nullptr;
jmethodID DirAccessJAndroid::_dir_exists = nullptr;
jmethodID DirAccessJAndroid::_file_exists = nullptr;
jmethodID DirAccessJAndroid::_get_drive_count = nullptr;
jmethodID DirAccessJAndroid::_get_drive = nullptr;
jmethodID DirAccessJAndroid::_make_dir = nullptr;
jmethodID DirAccessJAndroid::_get_space_left = nullptr;
jmethodID DirAccessJAndroid::_rename = nullptr;
jmethodID DirAccessJAndroid::_remove = nullptr;
jmethodID DirAccessJAndroid::_current_is_hidden = nullptr;

Error DirAccessJAndroid::list_dir_begin() {
	list_dir_end();
	int res = dir_open(current_dir);
	if (res <= 0) {
		return ERR_CANT_OPEN;
	}

	id = res;

	return OK;
}

String DirAccessJAndroid::get_next() {
	ERR_FAIL_COND_V(id == 0, "");
	if (_dir_next) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, "");
		jstring str = (jstring)env->CallObjectMethod(dir_access_handler, _dir_next, get_access_type(), id);
		if (!str) {
			return "";
		}

		String ret = jstring_to_string((jstring)str, env);
		env->DeleteLocalRef((jobject)str);
		return ret;
	} else {
		return "";
	}
}

bool DirAccessJAndroid::current_is_dir() const {
	if (_dir_is_dir) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, false);
		return env->CallBooleanMethod(dir_access_handler, _dir_is_dir, get_access_type(), id);
	} else {
		return false;
	}
}

bool DirAccessJAndroid::current_is_hidden() const {
	if (_current_is_hidden) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, false);
		return env->CallBooleanMethod(dir_access_handler, _current_is_hidden, get_access_type(), id);
	}
	return false;
}

void DirAccessJAndroid::list_dir_end() {
	if (id == 0) {
		return;
	}

	dir_close(id);
	id = 0;
}

int DirAccessJAndroid::get_drive_count() {
	if (_get_drive_count) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, 0);
		return env->CallIntMethod(dir_access_handler, _get_drive_count, get_access_type());
	} else {
		return 0;
	}
}

String DirAccessJAndroid::get_drive(int p_drive) {
	if (_get_drive) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, "");
		jstring j_drive = (jstring)env->CallObjectMethod(dir_access_handler, _get_drive, get_access_type(), p_drive);
		if (!j_drive) {
			return "";
		}

		String drive = jstring_to_string(j_drive, env);
		env->DeleteLocalRef(j_drive);
		return drive;
	} else {
		return "";
	}
}

String DirAccessJAndroid::_get_root_string() const {
	if (get_access_type() == ACCESS_FILESYSTEM) {
		return "/";
	}
	return DirAccessUnix::_get_root_string();
}

String DirAccessJAndroid::get_current_dir() {
	String base = _get_root_path();
	String bd = current_dir;
	if (base != "") {
		bd = current_dir.replace_first(base, "");
	}

	String root_string = _get_root_string();
	if (bd.begins_with(root_string)) {
		return bd;
	} else if (bd.begins_with("/")) {
		return root_string + bd.substr(1, bd.length());
	} else {
		return root_string + bd;
	}
}

Error DirAccessJAndroid::change_dir(String p_dir) {
	String new_dir = get_absolute_path(p_dir);
	if (new_dir == current_dir) {
		return OK;
	}

	if (!dir_exists(new_dir)) {
		return ERR_INVALID_PARAMETER;
	}

	current_dir = new_dir;
	return OK;
}

String DirAccessJAndroid::get_absolute_path(String p_path) {
	if (current_dir != "" && p_path == current_dir) {
		return current_dir;
	}

	if (p_path.is_rel_path()) {
		p_path = get_current_dir().plus_file(p_path);
	}

	p_path = fix_path(p_path);
	p_path = p_path.simplify_path();
	return p_path;
}

bool DirAccessJAndroid::file_exists(String p_file) {
	if (_file_exists) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, false);

		String path = get_absolute_path(p_file);
		jstring j_path = env->NewStringUTF(path.utf8().get_data());
		bool result = env->CallBooleanMethod(dir_access_handler, _file_exists, get_access_type(), j_path);
		env->DeleteLocalRef(j_path);
		return result;
	} else {
		return false;
	}
}

bool DirAccessJAndroid::dir_exists(String p_dir) {
	if (_dir_exists) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, false);

		String path = get_absolute_path(p_dir);
		jstring j_path = env->NewStringUTF(path.utf8().get_data());
		bool result = env->CallBooleanMethod(dir_access_handler, _dir_exists, get_access_type(), j_path);
		env->DeleteLocalRef(j_path);
		return result;
	} else {
		return false;
	}
}

Error DirAccessJAndroid::make_dir_recursive(String p_dir) {
	// Check if the directory exists already
	if (dir_exists(p_dir)) {
		return ERR_ALREADY_EXISTS;
	}

	if (_make_dir) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, ERR_UNCONFIGURED);

		String path = get_absolute_path(p_dir);
		jstring j_dir = env->NewStringUTF(path.utf8().get_data());
		bool result = env->CallBooleanMethod(dir_access_handler, _make_dir, get_access_type(), j_dir);
		env->DeleteLocalRef(j_dir);
		if (result) {
			return OK;
		} else {
			return FAILED;
		}
	} else {
		return ERR_UNCONFIGURED;
	}
}

Error DirAccessJAndroid::make_dir(String p_dir) {
	return make_dir_recursive(p_dir);
}

Error DirAccessJAndroid::rename(String p_from, String p_to) {
	if (_rename) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, ERR_UNCONFIGURED);

		String from_path = get_absolute_path(p_from);
		jstring j_from = env->NewStringUTF(from_path.utf8().get_data());

		String to_path = get_absolute_path(p_to);
		jstring j_to = env->NewStringUTF(to_path.utf8().get_data());

		bool result = env->CallBooleanMethod(dir_access_handler, _rename, get_access_type(), j_from, j_to);
		env->DeleteLocalRef(j_from);
		env->DeleteLocalRef(j_to);
		if (result) {
			return OK;
		} else {
			return FAILED;
		}
	} else {
		return ERR_UNCONFIGURED;
	}
}

Error DirAccessJAndroid::remove(String p_name) {
	if (_remove) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, ERR_UNCONFIGURED);

		String path = get_absolute_path(p_name);
		jstring j_name = env->NewStringUTF(path.utf8().get_data());
		bool result = env->CallBooleanMethod(dir_access_handler, _remove, get_access_type(), j_name);
		env->DeleteLocalRef(j_name);
		if (result) {
			return OK;
		} else {
			return FAILED;
		}
	} else {
		return ERR_UNCONFIGURED;
	}
}

uint64_t DirAccessJAndroid::get_space_left() {
	if (_get_space_left) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, 0);
		return env->CallLongMethod(dir_access_handler, _get_space_left, get_access_type());
	} else {
		return 0;
	}
}

void DirAccessJAndroid::setup(jobject p_dir_access_handler) {
	JNIEnv *env = get_jni_env();
	dir_access_handler = env->NewGlobalRef(p_dir_access_handler);

	jclass c = env->GetObjectClass(dir_access_handler);
	cls = (jclass)env->NewGlobalRef(c);

	_dir_open = env->GetMethodID(cls, "dirOpen", "(ILjava/lang/String;)I");
	_dir_next = env->GetMethodID(cls, "dirNext", "(II)Ljava/lang/String;");
	_dir_close = env->GetMethodID(cls, "dirClose", "(II)V");
	_dir_is_dir = env->GetMethodID(cls, "dirIsDir", "(II)Z");
	_dir_exists = env->GetMethodID(cls, "dirExists", "(ILjava/lang/String;)Z");
	_file_exists = env->GetMethodID(cls, "fileExists", "(ILjava/lang/String;)Z");
	_get_drive_count = env->GetMethodID(cls, "getDriveCount", "(I)I");
	_get_drive = env->GetMethodID(cls, "getDrive", "(II)Ljava/lang/String;");
	_make_dir = env->GetMethodID(cls, "makeDir", "(ILjava/lang/String;)Z");
	_get_space_left = env->GetMethodID(cls, "getSpaceLeft", "(I)J");
	_rename = env->GetMethodID(cls, "rename", "(ILjava/lang/String;Ljava/lang/String;)Z");
	_remove = env->GetMethodID(cls, "remove", "(ILjava/lang/String;)Z");
	_current_is_hidden = env->GetMethodID(cls, "isCurrentHidden", "(II)Z");
}

DirAccessJAndroid::DirAccessJAndroid() {
}

DirAccessJAndroid::~DirAccessJAndroid() {
	list_dir_end();
}

int DirAccessJAndroid::dir_open(String p_path) {
	if (_dir_open) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND_V(env == nullptr, 0);

		String path = get_absolute_path(p_path);
		jstring js = env->NewStringUTF(path.utf8().get_data());
		int dirId = env->CallIntMethod(dir_access_handler, _dir_open, get_access_type(), js);
		env->DeleteLocalRef(js);
		return dirId;
	} else {
		return 0;
	}
}

void DirAccessJAndroid::dir_close(int p_id) {
	if (_dir_close) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND(env == nullptr);
		env->CallVoidMethod(dir_access_handler, _dir_close, get_access_type(), p_id);
	}
}
