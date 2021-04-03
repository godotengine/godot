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
#include "api/java_class_wrapper.h"
#include "api/jni_singleton.h"
#include "audio_driver_jandroid.h"
#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "dir_access_jandroid.h"
#include "display_server_android.h"
#include "file_access_android.h"
#include "jni_utils.h"
#include "main/main.h"
#include "net_socket_android.h"
#include "os_android.h"
#include "string_android.h"
#include "thread_jandroid.h"

#include <android/input.h>
#include <unistd.h>

#include <android/native_window_jni.h>

static JavaClassWrapper *java_class_wrapper = nullptr;
static OS_Android *os_android = nullptr;
static GodotJavaWrapper *godot_java = nullptr;
static GodotIOJavaWrapper *godot_io_java = nullptr;

static bool initialized = false;
static int step = 0;

static Size2 new_size;
static Vector3 accelerometer;
static Vector3 gravity;
static Vector3 magnetometer;
static Vector3 gyroscope;
static Vector2 location;
static float altitude;

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

	const char **cmdline = nullptr;
	jstring *j_cmdline = nullptr;
	int cmdlen = 0;
	if (p_cmdline) {
		cmdlen = env->GetArrayLength(p_cmdline);
		if (cmdlen) {
			cmdline = (const char **)memalloc((cmdlen + 1) * sizeof(const char *));
			ERR_FAIL_NULL_MSG(cmdline, "Out of memory.");
			cmdline[cmdlen] = nullptr;
			j_cmdline = (jstring *)memalloc(cmdlen * sizeof(jstring));
			ERR_FAIL_NULL_MSG(j_cmdline, "Out of memory.");

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
			memfree(j_cmdline);
		}
		memfree(cmdline);
	}

	if (err != OK) {
		return; // should exit instead and print the error
	}

	java_class_wrapper = memnew(JavaClassWrapper(godot_java->get_activity()));
	ClassDB::register_class<JNISingleton>();
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_resize(JNIEnv *env, jclass clazz, jobject p_surface, jint p_width, jint p_height) {
	if (os_android) {
		os_android->set_display_size(Size2i(p_width, p_height));

		// No need to reset the surface during startup
		if (step > 0) {
			if (p_surface) {
				ANativeWindow *native_window = ANativeWindow_fromSurface(env, p_surface);
				os_android->set_native_window(native_window);

				DisplayServerAndroid::get_singleton()->reset_window();
			}
		}
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_newcontext(JNIEnv *env, jclass clazz, jobject p_surface, jboolean p_32_bits) {
	if (os_android) {
		if (step == 0) {
			// During startup
			os_android->set_context_is_16_bits(!p_32_bits);
			if (p_surface) {
				ANativeWindow *native_window = ANativeWindow_fromSurface(env, p_surface);
				os_android->set_native_window(native_window);
			}
		} else {
			// Rendering context recreated because it was lost; restart app to let it reload everything
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
		// Since Godot is initialized on the UI thread, main_thread_id was set to that thread's id,
		// but for Godot purposes, the main thread is the one running the game loop
		Main::setup2(Thread::get_caller_id());
		++step;
		return;
	}

	if (step == 1) {
		if (!Main::start()) {
			return; // should exit instead and print the error
		}

		godot_java->on_godot_setup_completed(env);
		os_android->main_loop_begin();
		godot_java->on_godot_main_loop_started(env);
		++step;
	}

	DisplayServerAndroid::get_singleton()->process_accelerometer(accelerometer);
	DisplayServerAndroid::get_singleton()->process_gravity(gravity);
	DisplayServerAndroid::get_singleton()->process_magnetometer(magnetometer);
	DisplayServerAndroid::get_singleton()->process_gyroscope(gyroscope);
	DisplayServerAndroid::get_singleton()->process_location(location);
	DisplayServerAndroid::get_singleton()->process_altitude(altitude);

	if (os_android->main_loop_iterate()) {
		godot_java->force_quit(env);
	}
}

void touch_preprocessing(JNIEnv *env, jclass clazz, jint input_device, jint ev, jint pointer, jint pointer_count, jfloatArray positions, jint buttons_mask, jfloat vertical_factor, jfloat horizontal_factor) {
	if (step == 0)
		return;

	Vector<DisplayServerAndroid::TouchPos> points;
	for (int i = 0; i < pointer_count; i++) {
		jfloat p[3];
		env->GetFloatArrayRegion(positions, i * 3, 3, p);
		DisplayServerAndroid::TouchPos tp;
		tp.pos = Point2(p[1], p[2]);
		tp.id = (int)p[0];
		points.push_back(tp);
	}
	if ((input_device & AINPUT_SOURCE_MOUSE) == AINPUT_SOURCE_MOUSE || (input_device & AINPUT_SOURCE_MOUSE_RELATIVE) == AINPUT_SOURCE_MOUSE_RELATIVE) {
		DisplayServerAndroid::get_singleton()->process_mouse_event(input_device, ev, buttons_mask, points[0].pos, vertical_factor, horizontal_factor);
	} else {
		DisplayServerAndroid::get_singleton()->process_touch(ev, pointer, points);
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

	DisplayServerAndroid::get_singleton()->process_hover(p_type, Point2(p_x, p_y));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_doubleTap(JNIEnv *env, jclass clazz, jint p_button_mask, jint p_x, jint p_y) {
	if (step == 0)
		return;

	DisplayServerAndroid::get_singleton()->process_double_tap(p_button_mask, Point2(p_x, p_y));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_scroll(JNIEnv *env, jclass clazz, jint p_x, jint p_y) {
	if (step == 0)
		return;

	DisplayServerAndroid::get_singleton()->process_scroll(Point2(p_x, p_y));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joybutton(JNIEnv *env, jclass clazz, jint p_device, jint p_button, jboolean p_pressed) {
	if (step == 0)
		return;

	DisplayServerAndroid::JoypadEvent jevent;
	jevent.device = p_device;
	jevent.type = DisplayServerAndroid::JOY_EVENT_BUTTON;
	jevent.index = p_button;
	jevent.pressed = p_pressed;

	DisplayServerAndroid::get_singleton()->process_joy_event(jevent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyaxis(JNIEnv *env, jclass clazz, jint p_device, jint p_axis, jfloat p_value) {
	if (step == 0)
		return;

	DisplayServerAndroid::JoypadEvent jevent;
	jevent.device = p_device;
	jevent.type = DisplayServerAndroid::JOY_EVENT_AXIS;
	jevent.index = p_axis;
	jevent.value = p_value;

	DisplayServerAndroid::get_singleton()->process_joy_event(jevent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyhat(JNIEnv *env, jclass clazz, jint p_device, jint p_hat_x, jint p_hat_y) {
	if (step == 0)
		return;

	DisplayServerAndroid::JoypadEvent jevent;
	jevent.device = p_device;
	jevent.type = DisplayServerAndroid::JOY_EVENT_HAT;
	int hat = 0;
	if (p_hat_x != 0) {
		if (p_hat_x < 0)
			hat |= Input::HAT_MASK_LEFT;
		else
			hat |= Input::HAT_MASK_RIGHT;
	}
	if (p_hat_y != 0) {
		if (p_hat_y < 0)
			hat |= Input::HAT_MASK_UP;
		else
			hat |= Input::HAT_MASK_DOWN;
	}
	jevent.hat = hat;

	DisplayServerAndroid::get_singleton()->process_joy_event(jevent);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyconnectionchanged(JNIEnv *env, jclass clazz, jint p_device, jboolean p_connected, jstring p_name) {
	if (os_android) {
		String name = jstring_to_string(p_name, env);
		Input::get_singleton()->joy_connection_changed(p_device, p_connected, name);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_key(JNIEnv *env, jclass clazz, jint p_keycode, jint p_scancode, jint p_unicode_char, jboolean p_pressed) {
	if (step == 0)
		return;

	DisplayServerAndroid::get_singleton()->process_key_event(p_keycode, p_scancode, p_unicode_char, p_pressed);
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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_location(JNIEnv *env, jclass clazz, jfloat lat, jfloat lng) {
	location = Vector2(lat, lng);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_altitude(JNIEnv *env, jclass clazz, jfloat alt) {
	altitude = alt;
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
	Object *obj = ObjectDB::get_instance(ObjectID(ID));
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

	Callable::CallError err;
	obj->call(str_method, (const Variant **)vptr, count, err);
	// something

	env->PopLocalFrame(nullptr);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_calldeferred(JNIEnv *env, jclass clazz, jlong ID, jstring method, jobjectArray params) {
	Object *obj = ObjectDB::get_instance(ObjectID(ID));
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

	static_assert(VARIANT_ARG_MAX == 5, "This code needs to be updated if VARIANT_ARG_MAX != 5");
	obj->call_deferred(str_method, args[0], args[1], args[2], args[3], args[4]);
	// something
	env->PopLocalFrame(nullptr);
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
		os_android->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_RESUMED);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_onRendererPaused(JNIEnv *env, jclass clazz) {
	if (step == 0)
		return;

	if (os_android->get_main_loop()) {
		os_android->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_PAUSED);
	}
}
}
