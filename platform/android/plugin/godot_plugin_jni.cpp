/*************************************************************************/
/*  godot_plugin_jni.cpp                                                 */
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

#include "godot_plugin_jni.h"

#include <core/engine.h>
#include <core/error_macros.h>
#include <core/project_settings.h>
#include <platform/android/jni_utils.h>
#include <platform/android/string_android.h>

class JNISingleton : public Object {

	GDCLASS(JNISingleton, Object);

	struct MethodData {

		jmethodID method;
		Variant::Type ret_type;
		Vector<Variant::Type> argtypes;
	};

	jobject instance;
	Map<StringName, MethodData> method_map;

public:
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

		ERR_FAIL_COND_V(!instance, Variant());

		r_error.error = Variant::CallError::CALL_OK;

		Map<StringName, MethodData>::Element *E = method_map.find(p_method);
		if (!E) {

			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			return Variant();
		}

		int ac = E->get().argtypes.size();
		if (ac < p_argcount) {

			r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument = ac;
			return Variant();
		}

		if (ac > p_argcount) {

			r_error.error = Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			r_error.argument = ac;
			return Variant();
		}

		for (int i = 0; i < p_argcount; i++) {

			if (!Variant::can_convert(p_args[i]->get_type(), E->get().argtypes[i])) {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = i;
				r_error.expected = E->get().argtypes[i];
			}
		}

		jvalue *v = NULL;

		if (p_argcount) {

			v = (jvalue *)alloca(sizeof(jvalue) * p_argcount);
		}

		JNIEnv *env = ThreadAndroid::get_env();

		int res = env->PushLocalFrame(16);

		ERR_FAIL_COND_V(res != 0, Variant());

		List<jobject> to_erase;
		for (int i = 0; i < p_argcount; i++) {

			jvalret vr = _variant_to_jvalue(env, E->get().argtypes[i], p_args[i]);
			v[i] = vr.val;
			if (vr.obj)
				to_erase.push_back(vr.obj);
		}

		Variant ret;

		switch (E->get().ret_type) {

			case Variant::NIL: {

				env->CallVoidMethodA(instance, E->get().method, v);
			} break;
			case Variant::BOOL: {

				ret = env->CallBooleanMethodA(instance, E->get().method, v) == JNI_TRUE;
			} break;
			case Variant::INT: {

				ret = env->CallIntMethodA(instance, E->get().method, v);
			} break;
			case Variant::REAL: {

				ret = env->CallFloatMethodA(instance, E->get().method, v);
			} break;
			case Variant::STRING: {

				jobject o = env->CallObjectMethodA(instance, E->get().method, v);
				ret = jstring_to_string((jstring)o, env);
				env->DeleteLocalRef(o);
			} break;
			case Variant::POOL_STRING_ARRAY: {

				jobjectArray arr = (jobjectArray)env->CallObjectMethodA(instance, E->get().method, v);

				ret = _jobject_to_variant(env, arr);

				env->DeleteLocalRef(arr);
			} break;
			case Variant::POOL_INT_ARRAY: {

				jintArray arr = (jintArray)env->CallObjectMethodA(instance, E->get().method, v);

				int fCount = env->GetArrayLength(arr);
				PoolVector<int> sarr;
				sarr.resize(fCount);

				PoolVector<int>::Write w = sarr.write();
				env->GetIntArrayRegion(arr, 0, fCount, w.ptr());
				w.release();
				ret = sarr;
				env->DeleteLocalRef(arr);
			} break;
			case Variant::POOL_REAL_ARRAY: {

				jfloatArray arr = (jfloatArray)env->CallObjectMethodA(instance, E->get().method, v);

				int fCount = env->GetArrayLength(arr);
				PoolVector<float> sarr;
				sarr.resize(fCount);

				PoolVector<float>::Write w = sarr.write();
				env->GetFloatArrayRegion(arr, 0, fCount, w.ptr());
				w.release();
				ret = sarr;
				env->DeleteLocalRef(arr);
			} break;

			case Variant::DICTIONARY: {

				jobject obj = env->CallObjectMethodA(instance, E->get().method, v);
				ret = _jobject_to_variant(env, obj);
				env->DeleteLocalRef(obj);

			} break;
			default: {

				env->PopLocalFrame(NULL);
				ERR_FAIL_V(Variant());
			} break;
		}

		while (to_erase.size()) {
			env->DeleteLocalRef(to_erase.front()->get());
			to_erase.pop_front();
		}

		env->PopLocalFrame(NULL);

		return ret;
	}

	jobject get_instance() const {

		return instance;
	}
	void set_instance(jobject p_instance) {

		instance = p_instance;
	}

	void add_method(const StringName &p_name, jmethodID p_method, const Vector<Variant::Type> &p_args, Variant::Type p_ret_type) {

		MethodData md;
		md.method = p_method;
		md.argtypes = p_args;
		md.ret_type = p_ret_type;
		method_map[p_name] = md;
	}

	JNISingleton() {
		instance = NULL;
	}
};

static HashMap<String, JNISingleton *> jni_singletons;

extern "C" {

JNIEXPORT void JNICALL Java_org_godotengine_godot_plugin_GodotPlugin_nativeRegisterSingleton(JNIEnv *env, jclass clazz, jstring name, jobject obj) {

	String singname = jstring_to_string(name, env);
	JNISingleton *s = memnew(JNISingleton);
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

JNIEXPORT void JNICALL Java_org_godotengine_godot_plugin_GodotPlugin_nativeRegisterGDNativeLibraries(JNIEnv *env, jobject obj, jobjectArray gdnlib_paths) {
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
