/*************************************************************************/
/*  java_godot_io_wrapper.cpp                                            */
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

#include "java_godot_io_wrapper.h"
#include "core/error_list.h"

// JNIEnv is only valid within the thread it belongs to, in a multi threading environment
// we can't cache it.
// For GodotIO we call all access methods from our thread and we thus get a valid JNIEnv
// from ThreadAndroid.

GodotIOJavaWrapper::GodotIOJavaWrapper(JNIEnv *p_env, jobject p_godot_io_instance) {
	godot_io_instance = p_env->NewGlobalRef(p_godot_io_instance);
	if (godot_io_instance) {
		cls = p_env->GetObjectClass(godot_io_instance);
		if (cls) {
			cls = (jclass)p_env->NewGlobalRef(cls);
		} else {
			// this is a pretty serious fail.. bail... pointers will stay 0
			return;
		}

		_open_URI = p_env->GetMethodID(cls, "openURI", "(Ljava/lang/String;)I");
		_get_data_dir = p_env->GetMethodID(cls, "getDataDir", "()Ljava/lang/String;");
		_get_locale = p_env->GetMethodID(cls, "getLocale", "()Ljava/lang/String;");
		_get_model = p_env->GetMethodID(cls, "getModel", "()Ljava/lang/String;");
		_get_screen_DPI = p_env->GetMethodID(cls, "getScreenDPI", "()I");
		_get_window_safe_area = p_env->GetMethodID(cls, "getWindowSafeArea", "()[I"),
		_get_unique_id = p_env->GetMethodID(cls, "getUniqueID", "()Ljava/lang/String;");
		_show_keyboard = p_env->GetMethodID(cls, "showKeyboard", "(Ljava/lang/String;ZIII)V");
		_hide_keyboard = p_env->GetMethodID(cls, "hideKeyboard", "()V");
		_set_screen_orientation = p_env->GetMethodID(cls, "setScreenOrientation", "(I)V");
		_get_screen_orientation = p_env->GetMethodID(cls, "getScreenOrientation", "()I");
		_get_system_dir = p_env->GetMethodID(cls, "getSystemDir", "(I)Ljava/lang/String;");
		_play_video = p_env->GetMethodID(cls, "playVideo", "(Ljava/lang/String;)V");
		_is_video_playing = p_env->GetMethodID(cls, "isVideoPlaying", "()Z");
		_pause_video = p_env->GetMethodID(cls, "pauseVideo", "()V");
		_stop_video = p_env->GetMethodID(cls, "stopVideo", "()V");
	}
}

GodotIOJavaWrapper::~GodotIOJavaWrapper() {
	// nothing to do here for now
}

jobject GodotIOJavaWrapper::get_instance() {
	return godot_io_instance;
}

Error GodotIOJavaWrapper::open_uri(const String &p_uri) {
	if (_open_URI) {
		JNIEnv *env = ThreadAndroid::get_env();
		jstring jStr = env->NewStringUTF(p_uri.utf8().get_data());
		return env->CallIntMethod(godot_io_instance, _open_URI, jStr) ? ERR_CANT_OPEN : OK;
	} else {
		return ERR_UNAVAILABLE;
	}
}

String GodotIOJavaWrapper::get_user_data_dir() {
	if (_get_data_dir) {
		JNIEnv *env = ThreadAndroid::get_env();
		jstring s = (jstring)env->CallObjectMethod(godot_io_instance, _get_data_dir);
		return jstring_to_string(s, env);
	} else {
		return String();
	}
}

String GodotIOJavaWrapper::get_locale() {
	if (_get_locale) {
		JNIEnv *env = ThreadAndroid::get_env();
		jstring s = (jstring)env->CallObjectMethod(godot_io_instance, _get_locale);
		return jstring_to_string(s, env);
	} else {
		return String();
	}
}

String GodotIOJavaWrapper::get_model() {
	if (_get_model) {
		JNIEnv *env = ThreadAndroid::get_env();
		jstring s = (jstring)env->CallObjectMethod(godot_io_instance, _get_model);
		return jstring_to_string(s, env);
	} else {
		return String();
	}
}

int GodotIOJavaWrapper::get_screen_dpi() {
	if (_get_screen_DPI) {
		JNIEnv *env = ThreadAndroid::get_env();
		return env->CallIntMethod(godot_io_instance, _get_screen_DPI);
	} else {
		return 160;
	}
}

void GodotIOJavaWrapper::get_window_safe_area(int (&p_rect_xywh)[4]) {
	if (_get_window_safe_area) {
		JNIEnv *env = ThreadAndroid::get_env();
		jintArray returnArray = (jintArray)env->CallObjectMethod(godot_io_instance, _get_window_safe_area);
		ERR_FAIL_COND(env->GetArrayLength(returnArray) != 4);
		jint *arrayBody = env->GetIntArrayElements(returnArray, JNI_FALSE);
		for (int i = 0; i < 4; i++) {
			p_rect_xywh[i] = arrayBody[i];
		}
		env->ReleaseIntArrayElements(returnArray, arrayBody, 0);
	}
}

String GodotIOJavaWrapper::get_unique_id() {
	if (_get_unique_id) {
		JNIEnv *env = ThreadAndroid::get_env();
		jstring s = (jstring)env->CallObjectMethod(godot_io_instance, _get_unique_id);
		return jstring_to_string(s, env);
	} else {
		return String();
	}
}

bool GodotIOJavaWrapper::has_vk() {
	return (_show_keyboard != 0) && (_hide_keyboard != 0);
}

void GodotIOJavaWrapper::show_vk(const String &p_existing, bool p_multiline, int p_max_input_length, int p_cursor_start, int p_cursor_end) {
	if (_show_keyboard) {
		JNIEnv *env = ThreadAndroid::get_env();
		jstring jStr = env->NewStringUTF(p_existing.utf8().get_data());
		env->CallVoidMethod(godot_io_instance, _show_keyboard, jStr, p_multiline, p_max_input_length, p_cursor_start, p_cursor_end);
	}
}

void GodotIOJavaWrapper::hide_vk() {
	if (_hide_keyboard) {
		JNIEnv *env = ThreadAndroid::get_env();
		env->CallVoidMethod(godot_io_instance, _hide_keyboard);
	}
}

void GodotIOJavaWrapper::set_screen_orientation(int p_orient) {
	if (_set_screen_orientation) {
		JNIEnv *env = ThreadAndroid::get_env();
		env->CallVoidMethod(godot_io_instance, _set_screen_orientation, p_orient);
	}
}

int GodotIOJavaWrapper::get_screen_orientation() const {
	if (_get_screen_orientation) {
		JNIEnv *env = ThreadAndroid::get_env();
		return env->CallIntMethod(godot_io_instance, _get_screen_orientation);
	} else {
		return 0;
	}
}

String GodotIOJavaWrapper::get_system_dir(int p_dir) {
	if (_get_system_dir) {
		JNIEnv *env = ThreadAndroid::get_env();
		jstring s = (jstring)env->CallObjectMethod(godot_io_instance, _get_system_dir, p_dir);
		return jstring_to_string(s, env);
	} else {
		return String(".");
	}
}

void GodotIOJavaWrapper::play_video(const String &p_path) {
	// Why is this not here?!?!
}

bool GodotIOJavaWrapper::is_video_playing() {
	if (_is_video_playing) {
		JNIEnv *env = ThreadAndroid::get_env();
		return env->CallBooleanMethod(godot_io_instance, _is_video_playing);
	} else {
		return false;
	}
}

void GodotIOJavaWrapper::pause_video() {
	if (_pause_video) {
		JNIEnv *env = ThreadAndroid::get_env();
		env->CallVoidMethod(godot_io_instance, _pause_video);
	}
}

void GodotIOJavaWrapper::stop_video() {
	if (_stop_video) {
		JNIEnv *env = ThreadAndroid::get_env();
		env->CallVoidMethod(godot_io_instance, _stop_video);
	}
}

// volatile because it can be changed from non-main thread and we need to
// ensure the change is immediately visible to other threads.
static volatile int virtual_keyboard_height;

int GodotIOJavaWrapper::get_vk_height() {
	return virtual_keyboard_height;
}

void GodotIOJavaWrapper::set_vk_height(int p_height) {
	virtual_keyboard_height = p_height;
}
