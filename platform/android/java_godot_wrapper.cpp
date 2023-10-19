/**************************************************************************/
/*  java_godot_wrapper.cpp                                                */
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

#include "java_godot_wrapper.h"

// JNIEnv is only valid within the thread it belongs to, in a multi threading environment
// we can't cache it.
// For Godot we call most access methods from our thread and we thus get a valid JNIEnv
// from get_jni_env(). For one or two we expect to pass the environment

// TODO we could probably create a base class for this...

GodotJavaWrapper::GodotJavaWrapper(JNIEnv *p_env, jobject p_activity, jobject p_godot_instance) {
	godot_instance = p_env->NewGlobalRef(p_godot_instance);
	activity = p_env->NewGlobalRef(p_activity);

	// get info about our Godot class so we can get pointers and stuff...
	godot_class = p_env->FindClass("org/godotengine/godot/Godot");
	if (godot_class) {
		godot_class = (jclass)p_env->NewGlobalRef(godot_class);
	} else {
		// this is a pretty serious fail.. bail... pointers will stay 0
		return;
	}
	activity_class = p_env->FindClass("android/app/Activity");
	if (activity_class) {
		activity_class = (jclass)p_env->NewGlobalRef(activity_class);
	} else {
		// this is a pretty serious fail.. bail... pointers will stay 0
		return;
	}

	// get some Godot method pointers...
	_restart = p_env->GetMethodID(godot_class, "restart", "()V");
	_finish = p_env->GetMethodID(godot_class, "forceQuit", "(I)Z");
	_set_keep_screen_on = p_env->GetMethodID(godot_class, "setKeepScreenOn", "(Z)V");
	_alert = p_env->GetMethodID(godot_class, "alert", "(Ljava/lang/String;Ljava/lang/String;)V");
	_is_dark_mode_supported = p_env->GetMethodID(godot_class, "isDarkModeSupported", "()Z");
	_is_dark_mode = p_env->GetMethodID(godot_class, "isDarkMode", "()Z");
	_get_clipboard = p_env->GetMethodID(godot_class, "getClipboard", "()Ljava/lang/String;");
	_set_clipboard = p_env->GetMethodID(godot_class, "setClipboard", "(Ljava/lang/String;)V");
	_has_clipboard = p_env->GetMethodID(godot_class, "hasClipboard", "()Z");
	_request_permission = p_env->GetMethodID(godot_class, "requestPermission", "(Ljava/lang/String;)Z");
	_request_permissions = p_env->GetMethodID(godot_class, "requestPermissions", "()Z");
	_get_granted_permissions = p_env->GetMethodID(godot_class, "getGrantedPermissions", "()[Ljava/lang/String;");
	_get_ca_certificates = p_env->GetMethodID(godot_class, "getCACertificates", "()Ljava/lang/String;");
	_init_input_devices = p_env->GetMethodID(godot_class, "initInputDevices", "()V");
	_vibrate = p_env->GetMethodID(godot_class, "vibrate", "(I)V");
	_get_input_fallback_mapping = p_env->GetMethodID(godot_class, "getInputFallbackMapping", "()Ljava/lang/String;");
	_on_godot_setup_completed = p_env->GetMethodID(godot_class, "onGodotSetupCompleted", "()V");
	_on_godot_main_loop_started = p_env->GetMethodID(godot_class, "onGodotMainLoopStarted", "()V");
	_create_new_godot_instance = p_env->GetMethodID(godot_class, "createNewGodotInstance", "([Ljava/lang/String;)I");
	_get_render_view = p_env->GetMethodID(godot_class, "getRenderView", "()Lorg/godotengine/godot/GodotRenderView;");
	_begin_benchmark_measure = p_env->GetMethodID(godot_class, "nativeBeginBenchmarkMeasure", "(Ljava/lang/String;)V");
	_end_benchmark_measure = p_env->GetMethodID(godot_class, "nativeEndBenchmarkMeasure", "(Ljava/lang/String;)V");
	_dump_benchmark = p_env->GetMethodID(godot_class, "nativeDumpBenchmark", "(Ljava/lang/String;)V");
	_get_gdextension_list_config_file = p_env->GetMethodID(godot_class, "getGDExtensionConfigFiles", "()[Ljava/lang/String;");
	_has_feature = p_env->GetMethodID(godot_class, "hasFeature", "(Ljava/lang/String;)Z");
}

GodotJavaWrapper::~GodotJavaWrapper() {
	if (godot_view) {
		delete godot_view;
	}

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);
	env->DeleteGlobalRef(godot_instance);
	env->DeleteGlobalRef(godot_class);
	env->DeleteGlobalRef(activity);
	env->DeleteGlobalRef(activity_class);
}

jobject GodotJavaWrapper::get_activity() {
	return activity;
}

GodotJavaViewWrapper *GodotJavaWrapper::get_godot_view() {
	if (godot_view != nullptr) {
		return godot_view;
	}
	if (_get_render_view) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, nullptr);
		jobject godot_render_view = env->CallObjectMethod(godot_instance, _get_render_view);
		if (!env->IsSameObject(godot_render_view, nullptr)) {
			godot_view = new GodotJavaViewWrapper(godot_render_view);
		}
	}
	return godot_view;
}

void GodotJavaWrapper::on_godot_setup_completed(JNIEnv *p_env) {
	if (_on_godot_setup_completed) {
		if (p_env == nullptr) {
			p_env = get_jni_env();
		}
		p_env->CallVoidMethod(godot_instance, _on_godot_setup_completed);
	}
}

void GodotJavaWrapper::on_godot_main_loop_started(JNIEnv *p_env) {
	if (_on_godot_main_loop_started) {
		if (p_env == nullptr) {
			p_env = get_jni_env();
		}
		ERR_FAIL_NULL(p_env);
		p_env->CallVoidMethod(godot_instance, _on_godot_main_loop_started);
	}
}

void GodotJavaWrapper::restart(JNIEnv *p_env) {
	if (_restart) {
		if (p_env == nullptr) {
			p_env = get_jni_env();
		}
		ERR_FAIL_NULL(p_env);
		p_env->CallVoidMethod(godot_instance, _restart);
	}
}

bool GodotJavaWrapper::force_quit(JNIEnv *p_env, int p_instance_id) {
	if (_finish) {
		if (p_env == nullptr) {
			p_env = get_jni_env();
		}
		ERR_FAIL_NULL_V(p_env, false);
		return p_env->CallBooleanMethod(godot_instance, _finish, p_instance_id);
	}
	return false;
}

void GodotJavaWrapper::set_keep_screen_on(bool p_enabled) {
	if (_set_keep_screen_on) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);
		env->CallVoidMethod(godot_instance, _set_keep_screen_on, p_enabled);
	}
}

void GodotJavaWrapper::alert(const String &p_message, const String &p_title) {
	if (_alert) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);
		jstring jStrMessage = env->NewStringUTF(p_message.utf8().get_data());
		jstring jStrTitle = env->NewStringUTF(p_title.utf8().get_data());
		env->CallVoidMethod(godot_instance, _alert, jStrMessage, jStrTitle);
	}
}

bool GodotJavaWrapper::is_dark_mode_supported() {
	if (_is_dark_mode_supported) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, false);
		return env->CallBooleanMethod(godot_instance, _is_dark_mode_supported);
	} else {
		return false;
	}
}

bool GodotJavaWrapper::is_dark_mode() {
	if (_is_dark_mode) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, false);
		return env->CallBooleanMethod(godot_instance, _is_dark_mode);
	} else {
		return false;
	}
}

bool GodotJavaWrapper::has_get_clipboard() {
	return _get_clipboard != nullptr;
}

String GodotJavaWrapper::get_clipboard() {
	if (_get_clipboard) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, String());
		jstring s = (jstring)env->CallObjectMethod(godot_instance, _get_clipboard);
		return jstring_to_string(s, env);
	} else {
		return String();
	}
}

String GodotJavaWrapper::get_input_fallback_mapping() {
	if (_get_input_fallback_mapping) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, String());
		jstring fallback_mapping = (jstring)env->CallObjectMethod(godot_instance, _get_input_fallback_mapping);
		return jstring_to_string(fallback_mapping, env);
	} else {
		return String();
	}
}

bool GodotJavaWrapper::has_set_clipboard() {
	return _set_clipboard != nullptr;
}

void GodotJavaWrapper::set_clipboard(const String &p_text) {
	if (_set_clipboard) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);
		jstring jStr = env->NewStringUTF(p_text.utf8().get_data());
		env->CallVoidMethod(godot_instance, _set_clipboard, jStr);
	}
}

bool GodotJavaWrapper::has_has_clipboard() {
	return _has_clipboard != nullptr;
}

bool GodotJavaWrapper::has_clipboard() {
	if (_has_clipboard) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, false);
		return env->CallBooleanMethod(godot_instance, _has_clipboard);
	} else {
		return false;
	}
}

bool GodotJavaWrapper::request_permission(const String &p_name) {
	if (_request_permission) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, false);
		jstring jStrName = env->NewStringUTF(p_name.utf8().get_data());
		return env->CallBooleanMethod(godot_instance, _request_permission, jStrName);
	} else {
		return false;
	}
}

bool GodotJavaWrapper::request_permissions() {
	if (_request_permissions) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, false);
		return env->CallBooleanMethod(godot_instance, _request_permissions);
	} else {
		return false;
	}
}

Vector<String> GodotJavaWrapper::get_granted_permissions() const {
	Vector<String> permissions_list;
	if (_get_granted_permissions) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, permissions_list);
		jobject permissions_object = env->CallObjectMethod(godot_instance, _get_granted_permissions);
		jobjectArray *arr = reinterpret_cast<jobjectArray *>(&permissions_object);

		jsize len = env->GetArrayLength(*arr);
		for (int i = 0; i < len; i++) {
			jstring jstr = (jstring)env->GetObjectArrayElement(*arr, i);
			String str = jstring_to_string(jstr, env);
			permissions_list.push_back(str);
			env->DeleteLocalRef(jstr);
		}
	}
	return permissions_list;
}

Vector<String> GodotJavaWrapper::get_gdextension_list_config_file() const {
	Vector<String> config_file_list;
	if (_get_gdextension_list_config_file) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, config_file_list);
		jobject config_file_list_object = env->CallObjectMethod(godot_instance, _get_gdextension_list_config_file);
		jobjectArray *arr = reinterpret_cast<jobjectArray *>(&config_file_list_object);

		jsize len = env->GetArrayLength(*arr);
		for (int i = 0; i < len; i++) {
			jstring j_config_file = (jstring)env->GetObjectArrayElement(*arr, i);
			String config_file = jstring_to_string(j_config_file, env);
			config_file_list.push_back(config_file);
			env->DeleteLocalRef(j_config_file);
		}
	}
	return config_file_list;
}

String GodotJavaWrapper::get_ca_certificates() const {
	if (_get_ca_certificates) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, String());
		jstring s = (jstring)env->CallObjectMethod(godot_instance, _get_ca_certificates);
		return jstring_to_string(s, env);
	} else {
		return String();
	}
}

void GodotJavaWrapper::init_input_devices() {
	if (_init_input_devices) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);
		env->CallVoidMethod(godot_instance, _init_input_devices);
	}
}

void GodotJavaWrapper::vibrate(int p_duration_ms) {
	if (_vibrate) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);
		env->CallVoidMethod(godot_instance, _vibrate, p_duration_ms);
	}
}

int GodotJavaWrapper::create_new_godot_instance(List<String> args) {
	if (_create_new_godot_instance) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, 0);
		jobjectArray jargs = env->NewObjectArray(args.size(), env->FindClass("java/lang/String"), env->NewStringUTF(""));
		for (int i = 0; i < args.size(); i++) {
			env->SetObjectArrayElement(jargs, i, env->NewStringUTF(args[i].utf8().get_data()));
		}
		return env->CallIntMethod(godot_instance, _create_new_godot_instance, jargs);
	} else {
		return 0;
	}
}

void GodotJavaWrapper::begin_benchmark_measure(const String &p_label) {
	if (_begin_benchmark_measure) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);
		jstring j_label = env->NewStringUTF(p_label.utf8().get_data());
		env->CallVoidMethod(godot_instance, _begin_benchmark_measure, j_label);
	}
}

void GodotJavaWrapper::end_benchmark_measure(const String &p_label) {
	if (_end_benchmark_measure) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);
		jstring j_label = env->NewStringUTF(p_label.utf8().get_data());
		env->CallVoidMethod(godot_instance, _end_benchmark_measure, j_label);
	}
}

void GodotJavaWrapper::dump_benchmark(const String &benchmark_file) {
	if (_dump_benchmark) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);
		jstring j_benchmark_file = env->NewStringUTF(benchmark_file.utf8().get_data());
		env->CallVoidMethod(godot_instance, _dump_benchmark, j_benchmark_file);
	}
}

bool GodotJavaWrapper::has_feature(const String &p_feature) const {
	if (_has_feature) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, false);

		jstring j_feature = env->NewStringUTF(p_feature.utf8().get_data());
		return env->CallBooleanMethod(godot_instance, _has_feature, j_feature);
	} else {
		return false;
	}
}
