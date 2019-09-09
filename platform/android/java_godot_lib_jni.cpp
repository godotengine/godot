/*************************************************************************/
/*  java_godot_lib_jni.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "java_godot_lib_jni.h"
#include "java_godot_io_wrapper.h"
#include "java_godot_wrapper.h"

#include "android/asset_manager_jni.h"
#include "audio_driver_jandroid.h"
#include "core/engine.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "dir_access_jandroid.h"
#include "file_access_android.h"
#include "file_access_jandroid.h"
#include "java_class_wrapper.h"
#include "main/input_default.h"
#include "main/main.h"
#include "os_android.h"
#include "string_android.h"
#include "thread_jandroid.h"
#include <unistd.h>

static JavaClassWrapper *java_class_wrapper = NULL;
static OS_Android *os_android = NULL;
static GodotJavaWrapper *godot_java = NULL;
static GodotIOJavaWrapper *godot_io_java = NULL;

struct jvalret {

	jobject obj;
	jvalue val;
	jvalret() { obj = NULL; }
};

jvalret _variant_to_jvalue(JNIEnv *env, Variant::Type p_type, const Variant *p_arg, bool force_jobject = false) {

	jvalret v;

	switch (p_type) {

		case Variant::BOOL: {

			if (force_jobject) {
				jclass bclass = env->FindClass("java/lang/Boolean");
				jmethodID ctor = env->GetMethodID(bclass, "<init>", "(Z)V");
				jvalue val;
				val.z = (bool)(*p_arg);
				jobject obj = env->NewObjectA(bclass, ctor, &val);
				v.val.l = obj;
				v.obj = obj;
				env->DeleteLocalRef(bclass);
			} else {
				v.val.z = *p_arg;
			};
		} break;
		case Variant::INT: {

			if (force_jobject) {

				jclass bclass = env->FindClass("java/lang/Integer");
				jmethodID ctor = env->GetMethodID(bclass, "<init>", "(I)V");
				jvalue val;
				val.i = (int)(*p_arg);
				jobject obj = env->NewObjectA(bclass, ctor, &val);
				v.val.l = obj;
				v.obj = obj;
				env->DeleteLocalRef(bclass);

			} else {
				v.val.i = *p_arg;
			};
		} break;
		case Variant::REAL: {

			if (force_jobject) {

				jclass bclass = env->FindClass("java/lang/Double");
				jmethodID ctor = env->GetMethodID(bclass, "<init>", "(D)V");
				jvalue val;
				val.d = (double)(*p_arg);
				jobject obj = env->NewObjectA(bclass, ctor, &val);
				v.val.l = obj;
				v.obj = obj;
				env->DeleteLocalRef(bclass);

			} else {
				v.val.f = *p_arg;
			};
		} break;
		case Variant::STRING: {

			String s = *p_arg;
			jstring jStr = env->NewStringUTF(s.utf8().get_data());
			v.val.l = jStr;
			v.obj = jStr;
		} break;
		case Variant::POOL_STRING_ARRAY: {

			PoolVector<String> sarray = *p_arg;
			jobjectArray arr = env->NewObjectArray(sarray.size(), env->FindClass("java/lang/String"), env->NewStringUTF(""));

			for (int j = 0; j < sarray.size(); j++) {

				jstring str = env->NewStringUTF(sarray[j].utf8().get_data());
				env->SetObjectArrayElement(arr, j, str);
				env->DeleteLocalRef(str);
			}
			v.val.l = arr;
			v.obj = arr;

		} break;

		case Variant::DICTIONARY: {

			Dictionary dict = *p_arg;
			jclass dclass = env->FindClass("org/godotengine/godot/Dictionary");
			jmethodID ctor = env->GetMethodID(dclass, "<init>", "()V");
			jobject jdict = env->NewObject(dclass, ctor);

			Array keys = dict.keys();

			jobjectArray jkeys = env->NewObjectArray(keys.size(), env->FindClass("java/lang/String"), env->NewStringUTF(""));
			for (int j = 0; j < keys.size(); j++) {
				jstring str = env->NewStringUTF(String(keys[j]).utf8().get_data());
				env->SetObjectArrayElement(jkeys, j, str);
				env->DeleteLocalRef(str);
			};

			jmethodID set_keys = env->GetMethodID(dclass, "set_keys", "([Ljava/lang/String;)V");
			jvalue val;
			val.l = jkeys;
			env->CallVoidMethodA(jdict, set_keys, &val);
			env->DeleteLocalRef(jkeys);

			jobjectArray jvalues = env->NewObjectArray(keys.size(), env->FindClass("java/lang/Object"), NULL);

			for (int j = 0; j < keys.size(); j++) {
				Variant var = dict[keys[j]];
				jvalret v = _variant_to_jvalue(env, var.get_type(), &var, true);
				env->SetObjectArrayElement(jvalues, j, v.val.l);
				if (v.obj) {
					env->DeleteLocalRef(v.obj);
				}
			};

			jmethodID set_values = env->GetMethodID(dclass, "set_values", "([Ljava/lang/Object;)V");
			val.l = jvalues;
			env->CallVoidMethodA(jdict, set_values, &val);
			env->DeleteLocalRef(jvalues);
			env->DeleteLocalRef(dclass);

			v.val.l = jdict;
			v.obj = jdict;
		} break;

		case Variant::POOL_INT_ARRAY: {

			PoolVector<int> array = *p_arg;
			jintArray arr = env->NewIntArray(array.size());
			PoolVector<int>::Read r = array.read();
			env->SetIntArrayRegion(arr, 0, array.size(), r.ptr());
			v.val.l = arr;
			v.obj = arr;

		} break;
		case Variant::POOL_BYTE_ARRAY: {
			PoolVector<uint8_t> array = *p_arg;
			jbyteArray arr = env->NewByteArray(array.size());
			PoolVector<uint8_t>::Read r = array.read();
			env->SetByteArrayRegion(arr, 0, array.size(), reinterpret_cast<const signed char *>(r.ptr()));
			v.val.l = arr;
			v.obj = arr;

		} break;
		case Variant::POOL_REAL_ARRAY: {

			PoolVector<float> array = *p_arg;
			jfloatArray arr = env->NewFloatArray(array.size());
			PoolVector<float>::Read r = array.read();
			env->SetFloatArrayRegion(arr, 0, array.size(), r.ptr());
			v.val.l = arr;
			v.obj = arr;

		} break;
		default: {

			v.val.i = 0;
		} break;
	}
	return v;
}

String _get_class_name(JNIEnv *env, jclass cls, bool *array) {

	jclass cclass = env->FindClass("java/lang/Class");
	jmethodID getName = env->GetMethodID(cclass, "getName", "()Ljava/lang/String;");
	jstring clsName = (jstring)env->CallObjectMethod(cls, getName);

	if (array) {
		jmethodID isArray = env->GetMethodID(cclass, "isArray", "()Z");
		jboolean isarr = env->CallBooleanMethod(cls, isArray);
		(*array) = isarr ? true : false;
	}
	String name = jstring_to_string(clsName, env);
	env->DeleteLocalRef(clsName);

	return name;
}

Variant _jobject_to_variant(JNIEnv *env, jobject obj) {

	if (obj == NULL) {
		return Variant();
	}

	jclass c = env->GetObjectClass(obj);
	bool array;
	String name = _get_class_name(env, c, &array);

	if (name == "java.lang.String") {

		return jstring_to_string((jstring)obj, env);
	};

	if (name == "[Ljava.lang.String;") {

		jobjectArray arr = (jobjectArray)obj;
		int stringCount = env->GetArrayLength(arr);
		PoolVector<String> sarr;

		for (int i = 0; i < stringCount; i++) {
			jstring string = (jstring)env->GetObjectArrayElement(arr, i);
			sarr.push_back(jstring_to_string(string, env));
			env->DeleteLocalRef(string);
		}

		return sarr;
	};

	if (name == "java.lang.Boolean") {

		jmethodID boolValue = env->GetMethodID(c, "booleanValue", "()Z");
		bool ret = env->CallBooleanMethod(obj, boolValue);
		return ret;
	};

	if (name == "java.lang.Integer" || name == "java.lang.Long") {

		jclass nclass = env->FindClass("java/lang/Number");
		jmethodID longValue = env->GetMethodID(nclass, "longValue", "()J");
		jlong ret = env->CallLongMethod(obj, longValue);
		return ret;
	};

	if (name == "[I") {

		jintArray arr = (jintArray)obj;
		int fCount = env->GetArrayLength(arr);
		PoolVector<int> sarr;
		sarr.resize(fCount);

		PoolVector<int>::Write w = sarr.write();
		env->GetIntArrayRegion(arr, 0, fCount, w.ptr());
		w.release();
		return sarr;
	};

	if (name == "[B") {

		jbyteArray arr = (jbyteArray)obj;
		int fCount = env->GetArrayLength(arr);
		PoolVector<uint8_t> sarr;
		sarr.resize(fCount);

		PoolVector<uint8_t>::Write w = sarr.write();
		env->GetByteArrayRegion(arr, 0, fCount, reinterpret_cast<signed char *>(w.ptr()));
		w.release();
		return sarr;
	};

	if (name == "java.lang.Float" || name == "java.lang.Double") {

		jclass nclass = env->FindClass("java/lang/Number");
		jmethodID doubleValue = env->GetMethodID(nclass, "doubleValue", "()D");
		double ret = env->CallDoubleMethod(obj, doubleValue);
		return ret;
	};

	if (name == "[D") {

		jdoubleArray arr = (jdoubleArray)obj;
		int fCount = env->GetArrayLength(arr);
		PoolRealArray sarr;
		sarr.resize(fCount);

		PoolRealArray::Write w = sarr.write();

		for (int i = 0; i < fCount; i++) {

			double n;
			env->GetDoubleArrayRegion(arr, i, 1, &n);
			w.ptr()[i] = n;
		};
		return sarr;
	};

	if (name == "[F") {

		jfloatArray arr = (jfloatArray)obj;
		int fCount = env->GetArrayLength(arr);
		PoolRealArray sarr;
		sarr.resize(fCount);

		PoolRealArray::Write w = sarr.write();

		for (int i = 0; i < fCount; i++) {

			float n;
			env->GetFloatArrayRegion(arr, i, 1, &n);
			w.ptr()[i] = n;
		};
		return sarr;
	};

	if (name == "[Ljava.lang.Object;") {

		jobjectArray arr = (jobjectArray)obj;
		int objCount = env->GetArrayLength(arr);
		Array varr;

		for (int i = 0; i < objCount; i++) {
			jobject jobj = env->GetObjectArrayElement(arr, i);
			Variant v = _jobject_to_variant(env, jobj);
			varr.push_back(v);
			env->DeleteLocalRef(jobj);
		}

		return varr;
	};

	if (name == "java.util.HashMap" || name == "org.godotengine.godot.Dictionary") {

		Dictionary ret;
		jclass oclass = c;
		jmethodID get_keys = env->GetMethodID(oclass, "get_keys", "()[Ljava/lang/String;");
		jobjectArray arr = (jobjectArray)env->CallObjectMethod(obj, get_keys);

		PoolStringArray keys = _jobject_to_variant(env, arr);
		env->DeleteLocalRef(arr);

		jmethodID get_values = env->GetMethodID(oclass, "get_values", "()[Ljava/lang/Object;");
		arr = (jobjectArray)env->CallObjectMethod(obj, get_values);

		Array vals = _jobject_to_variant(env, arr);
		env->DeleteLocalRef(arr);

		for (int i = 0; i < keys.size(); i++) {

			ret[keys[i]] = vals[i];
		};

		return ret;
	};

	env->DeleteLocalRef(c);

	return Variant();
}

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

struct TST {

	int a;
	TST() {

		a = 5;
	}
};

TST tst;

static bool initialized = false;
static int step = 0;

static Size2 new_size;
static Vector3 accelerometer;
static Vector3 gravity;
static Vector3 magnetometer;
static Vector3 gyroscope;
static HashMap<String, JNISingleton *> jni_singletons;

// virtual Error native_video_play(String p_path);
// virtual bool native_video_is_playing();
// virtual void native_video_pause();
// virtual void native_video_stop();

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_setVirtualKeyboardHeight(JNIEnv *env, jobject obj, jint p_height) {
	if (godot_io_java) {
		godot_io_java->set_vk_height(p_height);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_initialize(JNIEnv *env, jobject obj, jobject activity, jobject p_asset_manager, jboolean p_use_apk_expansion) {

	initialized = true;

	JavaVM *jvm;
	env->GetJavaVM(&jvm);

	// create our wrapper classes
	godot_java = new GodotJavaWrapper(env, activity); // our activity is our godot instance is our activity..
	godot_io_java = new GodotIOJavaWrapper(env, godot_java->get_member_object("io", "Lorg/godotengine/godot/GodotIO;", env));

	ThreadAndroid::make_default(jvm);
#ifdef USE_JAVA_FILE_ACCESS
	FileAccessJAndroid::setup(godot_io_java->get_instance());
#else

	jobject amgr = env->NewGlobalRef(p_asset_manager);

	FileAccessAndroid::asset_manager = AAssetManager_fromJava(env, amgr);
#endif

	DirAccessJAndroid::setup(godot_io_java->get_instance());
	AudioDriverAndroid::setup(godot_io_java->get_instance());

	os_android = new OS_Android(godot_java, godot_io_java, p_use_apk_expansion);

	char wd[500];
	getcwd(wd, 500);

	godot_java->on_video_init(env);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_ondestroy(JNIEnv *env, jobject obj, jobject activity) {
	// lets cleanup
	if (godot_io_java) {
		delete godot_io_java;
	}
	if (godot_java) {
		delete godot_java;
	}
	if (os_android) {
		delete os_android;
	}
}

static void _initialize_java_modules() {

	if (!ProjectSettings::get_singleton()->has_setting("android/modules")) {
		return;
	}

	String modules = ProjectSettings::get_singleton()->get("android/modules");
	modules = modules.strip_edges();
	if (modules == String()) {
		return;
	}
	Vector<String> mods = modules.split(",", false);

	if (mods.size()) {
		jobject cls = godot_java->get_class_loader();

		// TODO create wrapper for class loader

		JNIEnv *env = ThreadAndroid::get_env();
		jclass classLoader = env->FindClass("java/lang/ClassLoader");
		jmethodID findClass = env->GetMethodID(classLoader, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");

		for (int i = 0; i < mods.size(); i++) {

			String m = mods[i];

			print_line("Loading Android module: " + m);
			jstring strClassName = env->NewStringUTF(m.utf8().get_data());
			jclass singletonClass = (jclass)env->CallObjectMethod(cls, findClass, strClassName);
			ERR_CONTINUE_MSG(!singletonClass, "Couldn't find singleton for class: " + m + ".");

			jmethodID initialize = env->GetStaticMethodID(singletonClass, "initialize", "(Landroid/app/Activity;)Lorg/godotengine/godot/Godot$SingletonBase;");
			ERR_CONTINUE_MSG(!initialize, "Couldn't find proper initialize function 'public static Godot.SingletonBase Class::initialize(Activity p_activity)' initializer for singleton class: " + m + ".");

			jobject obj = env->CallStaticObjectMethod(singletonClass, initialize, godot_java->get_activity());
			env->NewGlobalRef(obj);
		}
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_setup(JNIEnv *env, jobject obj, jobjectArray p_cmdline) {
	ThreadAndroid::setup_thread();

	const char **cmdline = NULL;
	jstring *j_cmdline = NULL;
	int cmdlen = 0;
	if (p_cmdline) {
		cmdlen = env->GetArrayLength(p_cmdline);
		if (cmdlen) {
			cmdline = (const char **)malloc((cmdlen + 1) * sizeof(const char *));
			cmdline[cmdlen] = NULL;
			j_cmdline = (jstring *)malloc(cmdlen * sizeof(jstring));

			for (int i = 0; i < cmdlen; i++) {

				jstring string = (jstring)env->GetObjectArrayElement(p_cmdline, i);
				const char *rawString = env->GetStringUTFChars(string, 0);

				cmdline[i] = rawString;
				j_cmdline[i] = string;
			}
		}
	}

	Error err = Main::setup("apk", cmdlen, (char **)cmdline, false);
	if (cmdline) {
		if (j_cmdline) {
			for (int i = 0; i < cmdlen; ++i) {
				env->ReleaseStringUTFChars(j_cmdline[i], cmdline[i]);
			}
			free(j_cmdline);
		}
		free(cmdline);
	}

	if (err != OK) {
		return; //should exit instead and print the error
	}

	java_class_wrapper = memnew(JavaClassWrapper(godot_java->get_activity()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("JavaClassWrapper", java_class_wrapper));
	_initialize_java_modules();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_resize(JNIEnv *env, jobject obj, jint width, jint height) {

	if (os_android)
		os_android->set_display_size(Size2(width, height));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_newcontext(JNIEnv *env, jobject obj, bool p_32_bits) {

	if (os_android) {
		if (step == 0) {
			// During startup
			os_android->set_context_is_16_bits(!p_32_bits);
		} else {
			// GL context recreated because it was lost; restart app to let it reload everything
			os_android->main_loop_end();
			godot_java->restart(env);
			step = -1; // Ensure no further steps are attempted
		}
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_back(JNIEnv *env, jobject obj) {
	if (step == 0)
		return;

	os_android->main_loop_request_go_back();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_step(JNIEnv *env, jobject obj) {
	if (step == -1)
		return;

	if (step == 0) {

		// Since Godot is initialized on the UI thread, _main_thread_id was set to that thread's id,
		// but for Godot purposes, the main thread is the one running the game loop
		Main::setup2(Thread::get_caller_id());
		++step;
		return;
	}

	if (step == 1) {
		if (!Main::start()) {
			return; //should exit instead and print the error
		}

		os_android->main_loop_begin();
		++step;
	}

	os_android->process_accelerometer(accelerometer);
	os_android->process_gravity(gravity);
	os_android->process_magnetometer(magnetometer);
	os_android->process_gyroscope(gyroscope);

	if (os_android->main_loop_iterate()) {

		godot_java->force_quit(env);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_touch(JNIEnv *env, jobject obj, jint ev, jint pointer, jint count, jintArray positions) {

	if (step == 0)
		return;

	Vector<OS_Android::TouchPos> points;
	for (int i = 0; i < count; i++) {

		jint p[3];
		env->GetIntArrayRegion(positions, i * 3, 3, p);
		OS_Android::TouchPos tp;
		tp.pos = Point2(p[1], p[2]);
		tp.id = p[0];
		points.push_back(tp);
	}

	os_android->process_touch(ev, pointer, points);

	/*
	if (os_android)
		os_android->process_touch(ev,pointer,points);
	*/
}

/*
 * Android Key codes.
 */
enum {
	AKEYCODE_UNKNOWN = 0,
	AKEYCODE_SOFT_LEFT = 1,
	AKEYCODE_SOFT_RIGHT = 2,
	AKEYCODE_HOME = 3,
	AKEYCODE_BACK = 4,
	AKEYCODE_CALL = 5,
	AKEYCODE_ENDCALL = 6,
	AKEYCODE_0 = 7,
	AKEYCODE_1 = 8,
	AKEYCODE_2 = 9,
	AKEYCODE_3 = 10,
	AKEYCODE_4 = 11,
	AKEYCODE_5 = 12,
	AKEYCODE_6 = 13,
	AKEYCODE_7 = 14,
	AKEYCODE_8 = 15,
	AKEYCODE_9 = 16,
	AKEYCODE_STAR = 17,
	AKEYCODE_POUND = 18,
	AKEYCODE_DPAD_UP = 19,
	AKEYCODE_DPAD_DOWN = 20,
	AKEYCODE_DPAD_LEFT = 21,
	AKEYCODE_DPAD_RIGHT = 22,
	AKEYCODE_DPAD_CENTER = 23,
	AKEYCODE_VOLUME_UP = 24,
	AKEYCODE_VOLUME_DOWN = 25,
	AKEYCODE_POWER = 26,
	AKEYCODE_CAMERA = 27,
	AKEYCODE_CLEAR = 28,
	AKEYCODE_A = 29,
	AKEYCODE_B = 30,
	AKEYCODE_C = 31,
	AKEYCODE_D = 32,
	AKEYCODE_E = 33,
	AKEYCODE_F = 34,
	AKEYCODE_G = 35,
	AKEYCODE_H = 36,
	AKEYCODE_I = 37,
	AKEYCODE_J = 38,
	AKEYCODE_K = 39,
	AKEYCODE_L = 40,
	AKEYCODE_M = 41,
	AKEYCODE_N = 42,
	AKEYCODE_O = 43,
	AKEYCODE_P = 44,
	AKEYCODE_Q = 45,
	AKEYCODE_R = 46,
	AKEYCODE_S = 47,
	AKEYCODE_T = 48,
	AKEYCODE_U = 49,
	AKEYCODE_V = 50,
	AKEYCODE_W = 51,
	AKEYCODE_X = 52,
	AKEYCODE_Y = 53,
	AKEYCODE_Z = 54,
	AKEYCODE_COMMA = 55,
	AKEYCODE_PERIOD = 56,
	AKEYCODE_ALT_LEFT = 57,
	AKEYCODE_ALT_RIGHT = 58,
	AKEYCODE_SHIFT_LEFT = 59,
	AKEYCODE_SHIFT_RIGHT = 60,
	AKEYCODE_TAB = 61,
	AKEYCODE_SPACE = 62,
	AKEYCODE_SYM = 63,
	AKEYCODE_EXPLORER = 64,
	AKEYCODE_ENVELOPE = 65,
	AKEYCODE_ENTER = 66,
	AKEYCODE_DEL = 67,
	AKEYCODE_GRAVE = 68,
	AKEYCODE_MINUS = 69,
	AKEYCODE_EQUALS = 70,
	AKEYCODE_LEFT_BRACKET = 71,
	AKEYCODE_RIGHT_BRACKET = 72,
	AKEYCODE_BACKSLASH = 73,
	AKEYCODE_SEMICOLON = 74,
	AKEYCODE_APOSTROPHE = 75,
	AKEYCODE_SLASH = 76,
	AKEYCODE_AT = 77,
	AKEYCODE_NUM = 78,
	AKEYCODE_HEADSETHOOK = 79,
	AKEYCODE_FOCUS = 80, // *Camera* focus
	AKEYCODE_PLUS = 81,
	AKEYCODE_MENU = 82,
	AKEYCODE_NOTIFICATION = 83,
	AKEYCODE_SEARCH = 84,
	AKEYCODE_MEDIA_PLAY_PAUSE = 85,
	AKEYCODE_MEDIA_STOP = 86,
	AKEYCODE_MEDIA_NEXT = 87,
	AKEYCODE_MEDIA_PREVIOUS = 88,
	AKEYCODE_MEDIA_REWIND = 89,
	AKEYCODE_MEDIA_FAST_FORWARD = 90,
	AKEYCODE_MUTE = 91,
	AKEYCODE_PAGE_UP = 92,
	AKEYCODE_PAGE_DOWN = 93,
	AKEYCODE_PICTSYMBOLS = 94,
	AKEYCODE_SWITCH_CHARSET = 95,
	AKEYCODE_BUTTON_A = 96,
	AKEYCODE_BUTTON_B = 97,
	AKEYCODE_BUTTON_C = 98,
	AKEYCODE_BUTTON_X = 99,
	AKEYCODE_BUTTON_Y = 100,
	AKEYCODE_BUTTON_Z = 101,
	AKEYCODE_BUTTON_L1 = 102,
	AKEYCODE_BUTTON_R1 = 103,
	AKEYCODE_BUTTON_L2 = 104,
	AKEYCODE_BUTTON_R2 = 105,
	AKEYCODE_BUTTON_THUMBL = 106,
	AKEYCODE_BUTTON_THUMBR = 107,
	AKEYCODE_BUTTON_START = 108,
	AKEYCODE_BUTTON_SELECT = 109,
	AKEYCODE_BUTTON_MODE = 110,

	// NOTE: If you add a new keycode here you must also add it to several other files.
	//       Refer to frameworks/base/core/java/android/view/KeyEvent.java for the full list.
};

struct _WinTranslatePair {

	unsigned int keysym;
	unsigned int keycode;
};

static _WinTranslatePair _ak_to_keycode[] = {
	{ KEY_TAB, AKEYCODE_TAB },
	{ KEY_ENTER, AKEYCODE_ENTER },
	{ KEY_SHIFT, AKEYCODE_SHIFT_LEFT },
	{ KEY_SHIFT, AKEYCODE_SHIFT_RIGHT },
	{ KEY_ALT, AKEYCODE_ALT_LEFT },
	{ KEY_ALT, AKEYCODE_ALT_RIGHT },
	{ KEY_MENU, AKEYCODE_MENU },
	{ KEY_PAUSE, AKEYCODE_MEDIA_PLAY_PAUSE },
	{ KEY_ESCAPE, AKEYCODE_BACK },
	{ KEY_SPACE, AKEYCODE_SPACE },
	{ KEY_PAGEUP, AKEYCODE_PAGE_UP },
	{ KEY_PAGEDOWN, AKEYCODE_PAGE_DOWN },
	{ KEY_HOME, AKEYCODE_HOME }, //(0x24)
	{ KEY_LEFT, AKEYCODE_DPAD_LEFT },
	{ KEY_UP, AKEYCODE_DPAD_UP },
	{ KEY_RIGHT, AKEYCODE_DPAD_RIGHT },
	{ KEY_DOWN, AKEYCODE_DPAD_DOWN },
	{ KEY_PERIODCENTERED, AKEYCODE_DPAD_CENTER },
	{ KEY_BACKSPACE, AKEYCODE_DEL },
	{ KEY_0, AKEYCODE_0 }, ////0 key
	{ KEY_1, AKEYCODE_1 }, ////1 key
	{ KEY_2, AKEYCODE_2 }, ////2 key
	{ KEY_3, AKEYCODE_3 }, ////3 key
	{ KEY_4, AKEYCODE_4 }, ////4 key
	{ KEY_5, AKEYCODE_5 }, ////5 key
	{ KEY_6, AKEYCODE_6 }, ////6 key
	{ KEY_7, AKEYCODE_7 }, ////7 key
	{ KEY_8, AKEYCODE_8 }, ////8 key
	{ KEY_9, AKEYCODE_9 }, ////9 key
	{ KEY_A, AKEYCODE_A }, ////A key
	{ KEY_B, AKEYCODE_B }, ////B key
	{ KEY_C, AKEYCODE_C }, ////C key
	{ KEY_D, AKEYCODE_D }, ////D key
	{ KEY_E, AKEYCODE_E }, ////E key
	{ KEY_F, AKEYCODE_F }, ////F key
	{ KEY_G, AKEYCODE_G }, ////G key
	{ KEY_H, AKEYCODE_H }, ////H key
	{ KEY_I, AKEYCODE_I }, ////I key
	{ KEY_J, AKEYCODE_J }, ////J key
	{ KEY_K, AKEYCODE_K }, ////K key
	{ KEY_L, AKEYCODE_L }, ////L key
	{ KEY_M, AKEYCODE_M }, ////M key
	{ KEY_N, AKEYCODE_N }, ////N key
	{ KEY_O, AKEYCODE_O }, ////O key
	{ KEY_P, AKEYCODE_P }, ////P key
	{ KEY_Q, AKEYCODE_Q }, ////Q key
	{ KEY_R, AKEYCODE_R }, ////R key
	{ KEY_S, AKEYCODE_S }, ////S key
	{ KEY_T, AKEYCODE_T }, ////T key
	{ KEY_U, AKEYCODE_U }, ////U key
	{ KEY_V, AKEYCODE_V }, ////V key
	{ KEY_W, AKEYCODE_W }, ////W key
	{ KEY_X, AKEYCODE_X }, ////X key
	{ KEY_Y, AKEYCODE_Y }, ////Y key
	{ KEY_Z, AKEYCODE_Z }, ////Z key
	{ KEY_HOMEPAGE, AKEYCODE_EXPLORER },
	{ KEY_LAUNCH0, AKEYCODE_BUTTON_A },
	{ KEY_LAUNCH1, AKEYCODE_BUTTON_B },
	{ KEY_LAUNCH2, AKEYCODE_BUTTON_C },
	{ KEY_LAUNCH3, AKEYCODE_BUTTON_X },
	{ KEY_LAUNCH4, AKEYCODE_BUTTON_Y },
	{ KEY_LAUNCH5, AKEYCODE_BUTTON_Z },
	{ KEY_LAUNCH6, AKEYCODE_BUTTON_L1 },
	{ KEY_LAUNCH7, AKEYCODE_BUTTON_R1 },
	{ KEY_LAUNCH8, AKEYCODE_BUTTON_L2 },
	{ KEY_LAUNCH9, AKEYCODE_BUTTON_R2 },
	{ KEY_LAUNCHA, AKEYCODE_BUTTON_THUMBL },
	{ KEY_LAUNCHB, AKEYCODE_BUTTON_THUMBR },
	{ KEY_LAUNCHC, AKEYCODE_BUTTON_START },
	{ KEY_LAUNCHD, AKEYCODE_BUTTON_SELECT },
	{ KEY_LAUNCHE, AKEYCODE_BUTTON_MODE },
	{ KEY_VOLUMEMUTE, AKEYCODE_MUTE },
	{ KEY_VOLUMEDOWN, AKEYCODE_VOLUME_DOWN },
	{ KEY_VOLUMEUP, AKEYCODE_VOLUME_UP },
	{ KEY_BACK, AKEYCODE_MEDIA_REWIND },
	{ KEY_FORWARD, AKEYCODE_MEDIA_FAST_FORWARD },
	{ KEY_MEDIANEXT, AKEYCODE_MEDIA_NEXT },
	{ KEY_MEDIAPREVIOUS, AKEYCODE_MEDIA_PREVIOUS },
	{ KEY_MEDIASTOP, AKEYCODE_MEDIA_STOP },
	{ KEY_PLUS, AKEYCODE_PLUS },
	{ KEY_EQUAL, AKEYCODE_EQUALS }, // the '+' key
	{ KEY_COMMA, AKEYCODE_COMMA }, // the ',' key
	{ KEY_MINUS, AKEYCODE_MINUS }, // the '-' key
	{ KEY_SLASH, AKEYCODE_SLASH }, // the '/?' key
	{ KEY_BACKSLASH, AKEYCODE_BACKSLASH },
	{ KEY_BRACKETLEFT, AKEYCODE_LEFT_BRACKET },
	{ KEY_BRACKETRIGHT, AKEYCODE_RIGHT_BRACKET },
	{ KEY_UNKNOWN, 0 }
};
/*
TODO: map these android key:
    AKEYCODE_SOFT_LEFT       = 1,
    AKEYCODE_SOFT_RIGHT      = 2,
    AKEYCODE_CALL            = 5,
    AKEYCODE_ENDCALL         = 6,
    AKEYCODE_STAR            = 17,
    AKEYCODE_POUND           = 18,
    AKEYCODE_POWER           = 26,
    AKEYCODE_CAMERA          = 27,
    AKEYCODE_CLEAR           = 28,
    AKEYCODE_SYM             = 63,
    AKEYCODE_ENVELOPE        = 65,
    AKEYCODE_GRAVE           = 68,
    AKEYCODE_SEMICOLON       = 74,
    AKEYCODE_APOSTROPHE      = 75,
    AKEYCODE_AT              = 77,
    AKEYCODE_NUM             = 78,
    AKEYCODE_HEADSETHOOK     = 79,
    AKEYCODE_FOCUS           = 80,   // *Camera* focus
    AKEYCODE_NOTIFICATION    = 83,
    AKEYCODE_SEARCH          = 84,
    AKEYCODE_PICTSYMBOLS     = 94,
    AKEYCODE_SWITCH_CHARSET  = 95,
*/

static unsigned int android_get_keysym(unsigned int p_code) {
	for (int i = 0; _ak_to_keycode[i].keysym != KEY_UNKNOWN; i++) {

		if (_ak_to_keycode[i].keycode == p_code) {

			return _ak_to_keycode[i].keysym;
		}
	}

	return KEY_UNKNOWN;
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joybutton(JNIEnv *env, jobject obj, jint p_device, jint p_button, jboolean p_pressed) {
	if (step == 0)
		return;

	OS_Android::JoypadEvent jevent;
	jevent.device = p_device;
	jevent.type = OS_Android::JOY_EVENT_BUTTON;
	jevent.index = p_button;
	jevent.pressed = p_pressed;

	os_android->process_joy_event(jevent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyaxis(JNIEnv *env, jobject obj, jint p_device, jint p_axis, jfloat p_value) {
	if (step == 0)
		return;

	OS_Android::JoypadEvent jevent;
	jevent.device = p_device;
	jevent.type = OS_Android::JOY_EVENT_AXIS;
	jevent.index = p_axis;
	jevent.value = p_value;

	os_android->process_joy_event(jevent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyhat(JNIEnv *env, jobject obj, jint p_device, jint p_hat_x, jint p_hat_y) {
	if (step == 0)
		return;

	OS_Android::JoypadEvent jevent;
	jevent.device = p_device;
	jevent.type = OS_Android::JOY_EVENT_HAT;
	int hat = 0;
	if (p_hat_x != 0) {
		if (p_hat_x < 0)
			hat |= InputDefault::HAT_MASK_LEFT;
		else
			hat |= InputDefault::HAT_MASK_RIGHT;
	}
	if (p_hat_y != 0) {
		if (p_hat_y < 0)
			hat |= InputDefault::HAT_MASK_UP;
		else
			hat |= InputDefault::HAT_MASK_DOWN;
	}
	jevent.hat = hat;

	os_android->process_joy_event(jevent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyconnectionchanged(JNIEnv *env, jobject obj, jint p_device, jboolean p_connected, jstring p_name) {
	if (os_android) {
		String name = jstring_to_string(p_name, env);
		os_android->joy_connection_changed(p_device, p_connected, name);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_key(JNIEnv *env, jobject obj, jint p_scancode, jint p_unicode_char, jboolean p_pressed) {
	if (step == 0)
		return;

	Ref<InputEventKey> ievent;
	ievent.instance();
	int val = p_unicode_char;
	int scancode = android_get_keysym(p_scancode);
	ievent->set_scancode(scancode);
	ievent->set_unicode(val);
	ievent->set_pressed(p_pressed);

	if (val == '\n') {
		ievent->set_scancode(KEY_ENTER);
	} else if (val == 61448) {
		ievent->set_scancode(KEY_BACKSPACE);
		ievent->set_unicode(KEY_BACKSPACE);
	} else if (val == 61453) {
		ievent->set_scancode(KEY_ENTER);
		ievent->set_unicode(KEY_ENTER);
	} else if (p_scancode == 4) {

		os_android->main_loop_request_go_back();
	}

	os_android->process_event(ievent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_accelerometer(JNIEnv *env, jobject obj, jfloat x, jfloat y, jfloat z) {
	accelerometer = Vector3(x, y, z);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_gravity(JNIEnv *env, jobject obj, jfloat x, jfloat y, jfloat z) {
	gravity = Vector3(x, y, z);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_magnetometer(JNIEnv *env, jobject obj, jfloat x, jfloat y, jfloat z) {
	magnetometer = Vector3(x, y, z);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_gyroscope(JNIEnv *env, jobject obj, jfloat x, jfloat y, jfloat z) {
	gyroscope = Vector3(x, y, z);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_focusin(JNIEnv *env, jobject obj) {

	if (step == 0)
		return;

	os_android->main_loop_focusin();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_focusout(JNIEnv *env, jobject obj) {

	if (step == 0)
		return;

	os_android->main_loop_focusout();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_audio(JNIEnv *env, jobject obj) {

	ThreadAndroid::setup_thread();
	AudioDriverAndroid::thread_func(env);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_singleton(JNIEnv *env, jobject obj, jstring name, jobject p_object) {

	String singname = jstring_to_string(name, env);
	JNISingleton *s = memnew(JNISingleton);
	s->set_instance(env->NewGlobalRef(p_object));
	jni_singletons[singname] = s;

	Engine::get_singleton()->add_singleton(Engine::Singleton(singname, s));
	ProjectSettings::get_singleton()->set(singname, s);
}

static Variant::Type get_jni_type(const String &p_type) {

	static struct {
		const char *name;
		Variant::Type type;
	} _type_to_vtype[] = {
		{ "void", Variant::NIL },
		{ "boolean", Variant::BOOL },
		{ "int", Variant::INT },
		{ "float", Variant::REAL },
		{ "double", Variant::REAL },
		{ "java.lang.String", Variant::STRING },
		{ "[I", Variant::POOL_INT_ARRAY },
		{ "[B", Variant::POOL_BYTE_ARRAY },
		{ "[F", Variant::POOL_REAL_ARRAY },
		{ "[Ljava.lang.String;", Variant::POOL_STRING_ARRAY },
		{ "org.godotengine.godot.Dictionary", Variant::DICTIONARY },
		{ NULL, Variant::NIL }
	};

	int idx = 0;

	while (_type_to_vtype[idx].name) {

		if (p_type == _type_to_vtype[idx].name)
			return _type_to_vtype[idx].type;

		idx++;
	}

	return Variant::NIL;
}

static const char *get_jni_sig(const String &p_type) {

	static struct {
		const char *name;
		const char *sig;
	} _type_to_vtype[] = {
		{ "void", "V" },
		{ "boolean", "Z" },
		{ "int", "I" },
		{ "float", "F" },
		{ "double", "D" },
		{ "java.lang.String", "Ljava/lang/String;" },
		{ "org.godotengine.godot.Dictionary", "Lorg/godotengine/godot/Dictionary;" },
		{ "[I", "[I" },
		{ "[B", "[B" },
		{ "[F", "[F" },
		{ "[Ljava.lang.String;", "[Ljava/lang/String;" },
		{ NULL, "V" }
	};

	int idx = 0;

	while (_type_to_vtype[idx].name) {

		if (p_type == _type_to_vtype[idx].name)
			return _type_to_vtype[idx].sig;

		idx++;
	}

	return "Ljava/lang/Object;";
}

JNIEXPORT jstring JNICALL Java_org_godotengine_godot_GodotLib_getGlobal(JNIEnv *env, jobject obj, jstring path) {

	String js = jstring_to_string(path, env);

	return env->NewStringUTF(ProjectSettings::get_singleton()->get(js).operator String().utf8().get_data());
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_method(JNIEnv *env, jobject obj, jstring sname, jstring name, jstring ret, jobjectArray args) {

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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_callobject(JNIEnv *env, jobject p_obj, jint ID, jstring method, jobjectArray params) {

	Object *obj = ObjectDB::get_instance(ID);
	ERR_FAIL_COND(!obj);

	int res = env->PushLocalFrame(16);
	ERR_FAIL_COND(res != 0);

	String str_method = jstring_to_string(method, env);

	int count = env->GetArrayLength(params);
	Variant *vlist = (Variant *)alloca(sizeof(Variant) * count);
	Variant **vptr = (Variant **)alloca(sizeof(Variant *) * count);
	for (int i = 0; i < count; i++) {

		jobject obj = env->GetObjectArrayElement(params, i);
		Variant v;
		if (obj)
			v = _jobject_to_variant(env, obj);
		memnew_placement(&vlist[i], Variant);
		vlist[i] = v;
		vptr[i] = &vlist[i];
		env->DeleteLocalRef(obj);
	};

	Variant::CallError err;
	obj->call(str_method, (const Variant **)vptr, count, err);
	// something

	env->PopLocalFrame(NULL);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_calldeferred(JNIEnv *env, jobject p_obj, jint ID, jstring method, jobjectArray params) {

	Object *obj = ObjectDB::get_instance(ID);
	ERR_FAIL_COND(!obj);

	int res = env->PushLocalFrame(16);
	ERR_FAIL_COND(res != 0);

	String str_method = jstring_to_string(method, env);

	int count = env->GetArrayLength(params);
	Variant args[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(count, VARIANT_ARG_MAX); i++) {

		jobject obj = env->GetObjectArrayElement(params, i);
		if (obj)
			args[i] = _jobject_to_variant(env, obj);
		env->DeleteLocalRef(obj);
	};

	obj->call_deferred(str_method, args[0], args[1], args[2], args[3], args[4]);
	// something
	env->PopLocalFrame(NULL);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_requestPermissionResult(JNIEnv *env, jobject p_obj, jstring p_permission, jboolean p_result) {
	String permission = jstring_to_string(p_permission, env);
	if (permission == "android.permission.RECORD_AUDIO" && p_result) {
		AudioDriver::get_singleton()->capture_start();
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_onRendererResumed(JNIEnv *env, jclass clazz) {
	if (step == 0)
		return;

	if (os_android->get_main_loop()) {
		os_android->get_main_loop()->notification(MainLoop::NOTIFICATION_APP_RESUMED);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_onRendererPaused(JNIEnv *env, jclass clazz) {
	if (step == 0)
		return;

	if (os_android->get_main_loop()) {
		os_android->get_main_loop()->notification(MainLoop::NOTIFICATION_APP_PAUSED);
	}
}
