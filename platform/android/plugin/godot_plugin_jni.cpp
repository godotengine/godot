/**************************************************************************/
/*  godot_plugin_jni.cpp                                                  */
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

#include "godot_plugin_jni.h"

#include <core/engine.h>
#include <core/error_macros.h>
#include <core/project_settings.h>
#include <platform/android/api/jni_singleton.h>
#include <platform/android/jni_utils.h>
#include <platform/android/string_android.h>

static HashMap<String, JNISingleton *> jni_singletons;

extern "C" {

JNIEXPORT void JNICALL Java_org_godotengine_godot_plugin_GodotPlugin_nativeRegisterSingleton(JNIEnv *env, jclass clazz, jstring name, jobject obj) {
	String singname = jstring_to_string(name, env);
	JNISingleton *s = (JNISingleton *)ClassDB::instance("JNISingleton");
	s->set_instance(env->NewGlobalRef(obj));
	jni_singletons[singname] = s;

	Engine::get_singleton()->add_singleton(Engine::Singleton(singname, s));
	ProjectSettings::get_singleton()->set(singname, s);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_plugin_GodotPlugin_nativeRegisterMethod(JNIEnv *env, jclass clazz, jstring sname, jstring name, jstring ret, jobjectArray args) {
	String singname = jstring_to_string(sname, env);

	ERR_FAIL_COND(!jni_singletons.has(singname));

	JNISingleton *s = jni_singletons.get(singname);

	String mname = jstring_to_string(name, env);
	String retval = jstring_to_string(ret, env);
	Vector<Variant::Type> types;
	String cs = "(";

	int stringCount = env->GetArrayLength(args);

	for (int i = 0; i < stringCount; i++) {
		jstring string = (jstring)env->GetObjectArrayElement(args, i);
		const String rawString = jstring_to_string(string, env);
		types.push_back(get_jni_type(rawString));
		cs += get_jni_sig(rawString);
	}

	cs += ")";
	cs += get_jni_sig(retval);
	jclass cls = env->GetObjectClass(s->get_instance());
	jmethodID mid = env->GetMethodID(cls, mname.ascii().get_data(), cs.ascii().get_data());
	if (!mid) {
		print_line("Failed getting method ID " + mname);
	}

	s->add_method(mname, mid, types, get_jni_type(retval));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_plugin_GodotPlugin_nativeRegisterSignal(JNIEnv *env, jclass clazz, jstring j_plugin_name, jstring j_signal_name, jobjectArray j_signal_param_types) {
	String singleton_name = jstring_to_string(j_plugin_name, env);

	ERR_FAIL_COND(!jni_singletons.has(singleton_name));

	JNISingleton *singleton = jni_singletons.get(singleton_name);

	String signal_name = jstring_to_string(j_signal_name, env);
	Vector<Variant::Type> types;

	int stringCount = env->GetArrayLength(j_signal_param_types);

	for (int i = 0; i < stringCount; i++) {
		jstring j_signal_param_type = (jstring)env->GetObjectArrayElement(j_signal_param_types, i);
		const String signal_param_type = jstring_to_string(j_signal_param_type, env);
		types.push_back(get_jni_type(signal_param_type));
	}

	singleton->add_signal(signal_name, types);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_plugin_GodotPlugin_nativeEmitSignal(JNIEnv *env, jclass clazz, jstring j_plugin_name, jstring j_signal_name, jobjectArray j_signal_params) {
	String singleton_name = jstring_to_string(j_plugin_name, env);

	ERR_FAIL_COND(!jni_singletons.has(singleton_name));

	JNISingleton *singleton = jni_singletons.get(singleton_name);

	String signal_name = jstring_to_string(j_signal_name, env);

	int count = env->GetArrayLength(j_signal_params);
	ERR_FAIL_COND_MSG(count > VARIANT_ARG_MAX, "Maximum argument count exceeded!");

	Variant variant_params[VARIANT_ARG_MAX];
	const Variant *args[VARIANT_ARG_MAX];

	for (int i = 0; i < count; i++) {
		jobject j_param = env->GetObjectArrayElement(j_signal_params, i);
		variant_params[i] = _jobject_to_variant(env, j_param);
		args[i] = &variant_params[i];
		env->DeleteLocalRef(j_param);
	}

	singleton->emit_signal(signal_name, args, count);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_plugin_GodotPlugin_nativeRegisterGDNativeLibraries(JNIEnv *env, jclass clazz, jobjectArray gdnlib_paths) {
	int gdnlib_count = env->GetArrayLength(gdnlib_paths);
	if (gdnlib_count == 0) {
		return;
	}

	// Retrieve the current list of gdnative libraries.
	Array singletons = Array();
	if (ProjectSettings::get_singleton()->has_setting("gdnative/singletons")) {
		singletons = ProjectSettings::get_singleton()->get("gdnative/singletons");
	}

	// Insert the libraries provided by the plugin
	for (int i = 0; i < gdnlib_count; i++) {
		jstring relative_path = (jstring)env->GetObjectArrayElement(gdnlib_paths, i);

		String path = "res://" + jstring_to_string(relative_path, env);
		if (!singletons.has(path)) {
			singletons.push_back(path);
		}
		env->DeleteLocalRef(relative_path);
	}

	// Insert the updated list back into project settings.
	ProjectSettings::get_singleton()->set("gdnative/singletons", singletons);
}
}
