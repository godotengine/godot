/*************************************************************************/
/*  java_godot_view_wrapper.cpp                                          */
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

#include "java_godot_view_wrapper.h"

#include "thread_jandroid.h"

GodotJavaViewWrapper::GodotJavaViewWrapper(jobject godot_view) {
	JNIEnv *env = get_jni_env();
	ERR_FAIL_COND(env == nullptr);

	_godot_view = env->NewGlobalRef(godot_view);

	_cls = (jclass)env->NewGlobalRef(env->GetObjectClass(godot_view));

	if (android_get_device_api_level() >= __ANDROID_API_O__) {
		_request_pointer_capture = env->GetMethodID(_cls, "requestPointerCapture", "()V");
		_release_pointer_capture = env->GetMethodID(_cls, "releasePointerCapture", "()V");
		_set_pointer_icon = env->GetMethodID(_cls, "setPointerIcon", "(I)V");
	}
}

void GodotJavaViewWrapper::request_pointer_capture() {
	if (_request_pointer_capture != 0) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND(env == nullptr);

		env->CallVoidMethod(_godot_view, _request_pointer_capture);
	}
}

void GodotJavaViewWrapper::release_pointer_capture() {
	if (_request_pointer_capture != 0) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND(env == nullptr);

		env->CallVoidMethod(_godot_view, _release_pointer_capture);
	}
}

void GodotJavaViewWrapper::set_pointer_icon(int pointer_type) {
	if (_set_pointer_icon != 0) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_COND(env == nullptr);

		env->CallVoidMethod(_godot_view, _set_pointer_icon, pointer_type);
	}
}

GodotJavaViewWrapper::~GodotJavaViewWrapper() {
	JNIEnv *env = get_jni_env();
	ERR_FAIL_COND(env == nullptr);

	env->DeleteGlobalRef(_godot_view);
	env->DeleteGlobalRef(_cls);
}
