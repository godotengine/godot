/*************************************************************************/
/*  java_godot_lib_jni.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "android_keys_utils.h"
#include "api/java_class_wrapper.h"
#include "api/jni_singleton.h"
#include "audio_driver_jandroid.h"
#include "core/engine.h"
#include "core/project_settings.h"
#include "dir_access_jandroid.h"
#include "file_access_android.h"
#include "jni_utils.h"
#include "main/input_default.h"
#include "main/main.h"
#include "net_socket_android.h"
#include "os_android.h"
#include "string_android.h"
#include "thread_jandroid.h"

#include <android/input.h>
#include <unistd.h>

static JavaClassWrapper *java_class_wrapper = NULL;
static OS_Android *os_android = NULL;
static GodotJavaWrapper *godot_java = NULL;
static GodotIOJavaWrapper *godot_io_java = NULL;

static bool initialized = false;
static int step = 0;

static Size2 new_size;
static Vector3 accelerometer;
static Vector3 gravity;
static Vector3 magnetometer;
static Vector3 gyroscope;

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

		JNIEnv *env = get_jni_env();
		jclass classLoader = env->FindClass("java/lang/ClassLoader");
		jmethodID findClass = env->GetMethodID(classLoader, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");

		for (int i = 0; i < mods.size(); i++) {

			String m = mods[i];

			// Deprecated in Godot 3.2.2, it's now a plugin to enable in export preset.
			if (m == "org/godotengine/godot/GodotPaymentV3") {
				WARN_PRINT("GodotPaymentV3 is deprecated and is replaced by the 'GodotPayment' plugin, which should be enabled in the Android export preset.");
				print_line("Skipping Android module: " + m);
				continue;
			}

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

extern "C" {

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_setVirtualKeyboardHeight(JNIEnv *env, jclass clazz, jint p_height) {
	if (godot_io_java) {
		godot_io_java->set_vk_height(p_height);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_initialize(JNIEnv *env, jclass clazz, jobject activity, jobject godot_instance, jobject p_asset_manager, jboolean p_use_apk_expansion) {

	initialized = true;

	JavaVM *jvm;
	env->GetJavaVM(&jvm);

	// create our wrapper classes
	godot_java = new GodotJavaWrapper(env, activity, godot_instance);
	godot_io_java = new GodotIOJavaWrapper(env, godot_java->get_member_object("io", "Lorg/godotengine/godot/GodotIO;", env));

	init_thread_jandroid(jvm, env);

	jobject amgr = env->NewGlobalRef(p_asset_manager);

	FileAccessAndroid::asset_manager = AAssetManager_fromJava(env, amgr);

	DirAccessJAndroid::setup(godot_io_java->get_instance());
	AudioDriverAndroid::setup(godot_io_java->get_instance());
	NetSocketAndroid::setup(godot_java->get_member_object("netUtils", "Lorg/godotengine/godot/utils/GodotNetUtils;", env));

	os_android = new OS_Android(godot_java, godot_io_java, p_use_apk_expansion);

	char wd[500];
	getcwd(wd, 500);

	godot_java->on_video_init(env);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_ondestroy(JNIEnv *env, jclass clazz) {
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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_setup(JNIEnv *env, jclass clazz, jobjectArray p_cmdline) {
	setup_android_thread();

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
	ClassDB::register_class<JNISingleton>();
	_initialize_java_modules();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_resize(JNIEnv *env, jclass clazz, jint width, jint height) {

	if (os_android)
		os_android->set_display_size(Size2(width, height));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_newcontext(JNIEnv *env, jclass clazz, jboolean p_32_bits) {

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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_back(JNIEnv *env, jclass clazz) {
	if (step == 0)
		return;

	os_android->main_loop_request_go_back();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_step(JNIEnv *env, jclass clazz) {
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
		godot_java->on_godot_main_loop_started(env);
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

void touch_preprocessing(JNIEnv *env, jclass clazz, jint input_device, jint ev, jint pointer, jint pointer_count, jfloatArray positions, jint buttons_mask, jfloat vertical_factor, jfloat horizontal_factor) {
	if (step == 0)
		return;

	Vector<OS_Android::TouchPos> points;
	for (int i = 0; i < pointer_count; i++) {
		jfloat p[3];
		env->GetFloatArrayRegion(positions, i * 3, 3, p);
		OS_Android::TouchPos tp;
		tp.pos = Point2(p[1], p[2]);
		tp.id = (int)p[0];
		points.push_back(tp);
	}

	if ((input_device & AINPUT_SOURCE_MOUSE) == AINPUT_SOURCE_MOUSE) {
		os_android->process_mouse_event(ev, buttons_mask, points[0].pos, vertical_factor, horizontal_factor);
	} else {
		os_android->process_touch(ev, pointer, points);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_touch__IIII_3F(JNIEnv *env, jclass clazz, jint input_device, jint ev, jint pointer, jint pointer_count, jfloatArray position) {
	touch_preprocessing(env, clazz, input_device, ev, pointer, pointer_count, position);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_touch__IIII_3FI(JNIEnv *env, jclass clazz, jint input_device, jint ev, jint pointer, jint pointer_count, jfloatArray position, jint buttons_mask) {
	touch_preprocessing(env, clazz, input_device, ev, pointer, pointer_count, position, buttons_mask);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_touch__IIII_3FIFF(JNIEnv *env, jclass clazz, jint input_device, jint ev, jint pointer, jint pointer_count, jfloatArray position, jint buttons_mask, jfloat vertical_factor, jfloat horizontal_factor) {
	touch_preprocessing(env, clazz, input_device, ev, pointer, pointer_count, position, buttons_mask, vertical_factor, horizontal_factor);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_hover(JNIEnv *env, jclass clazz, jint p_type, jfloat p_x, jfloat p_y) {
	if (step == 0)
		return;

	os_android->process_hover(p_type, Point2(p_x, p_y));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_doubleTap(JNIEnv *env, jclass clazz, jint p_button_mask, jint p_x, jint p_y) {
	if (step == 0)
		return;

	os_android->process_double_tap(p_button_mask, Point2(p_x, p_y));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_scroll(JNIEnv *env, jclass clazz, jint p_x, jint p_y) {
	if (step == 0)
		return;

	os_android->process_scroll(Point2(p_x, p_y));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joybutton(JNIEnv *env, jclass clazz, jint p_device, jint p_button, jboolean p_pressed) {
	if (step == 0)
		return;

	OS_Android::JoypadEvent jevent;
	jevent.device = p_device;
	jevent.type = OS_Android::JOY_EVENT_BUTTON;
	jevent.index = p_button;
	jevent.pressed = p_pressed;

	os_android->process_joy_event(jevent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyaxis(JNIEnv *env, jclass clazz, jint p_device, jint p_axis, jfloat p_value) {
	if (step == 0)
		return;

	OS_Android::JoypadEvent jevent;
	jevent.device = p_device;
	jevent.type = OS_Android::JOY_EVENT_AXIS;
	jevent.index = p_axis;
	jevent.value = p_value;

	os_android->process_joy_event(jevent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyhat(JNIEnv *env, jclass clazz, jint p_device, jint p_hat_x, jint p_hat_y) {
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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyconnectionchanged(JNIEnv *env, jclass clazz, jint p_device, jboolean p_connected, jstring p_name) {
	if (os_android) {
		String name = jstring_to_string(p_name, env);
		os_android->joy_connection_changed(p_device, p_connected, name);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_key(JNIEnv *env, jclass clazz, jint p_scancode, jint p_unicode_char, jboolean p_pressed) {
	if (step == 0)
		return;

	os_android->process_key_event(p_scancode, p_unicode_char, p_pressed);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_accelerometer(JNIEnv *env, jclass clazz, jfloat x, jfloat y, jfloat z) {
	accelerometer = Vector3(x, y, z);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_gravity(JNIEnv *env, jclass clazz, jfloat x, jfloat y, jfloat z) {
	gravity = Vector3(x, y, z);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_magnetometer(JNIEnv *env, jclass clazz, jfloat x, jfloat y, jfloat z) {
	magnetometer = Vector3(x, y, z);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_gyroscope(JNIEnv *env, jclass clazz, jfloat x, jfloat y, jfloat z) {
	gyroscope = Vector3(x, y, z);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_focusin(JNIEnv *env, jclass clazz) {

	if (step == 0)
		return;

	os_android->main_loop_focusin();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_focusout(JNIEnv *env, jclass clazz) {

	if (step == 0)
		return;

	os_android->main_loop_focusout();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_audio(JNIEnv *env, jclass clazz) {

	setup_android_thread();
	AudioDriverAndroid::thread_func(env);
}

JNIEXPORT jstring JNICALL Java_org_godotengine_godot_GodotLib_getGlobal(JNIEnv *env, jclass clazz, jstring path) {

	String js = jstring_to_string(path, env);

	return env->NewStringUTF(ProjectSettings::get_singleton()->get(js).operator String().utf8().get_data());
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_callobject(JNIEnv *env, jclass clazz, jlong ID, jstring method, jobjectArray params) {
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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_calldeferred(JNIEnv *env, jclass clazz, jlong ID, jstring method, jobjectArray params) {
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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_requestPermissionResult(JNIEnv *env, jclass clazz, jstring p_permission, jboolean p_result) {
	String permission = jstring_to_string(p_permission, env);
	if (permission == "android.permission.RECORD_AUDIO" && p_result) {
		AudioDriver::get_singleton()->capture_start();
	}

	if (os_android->get_main_loop()) {
		os_android->get_main_loop()->emit_signal("on_request_permissions_result", permission, p_result == JNI_TRUE);
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
}
