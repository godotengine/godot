/*************************************************************************/
/*  java_godot_lib_jni.cpp                                               */
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

#include "java_godot_lib_jni.h"
#include "java_godot_io_wrapper.h"
#include "java_godot_wrapper.h"

#include "android/asset_manager_jni.h"
#include "api/java_class_wrapper.h"
#include "audio_driver_jandroid.h"
#include "core/engine.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "dir_access_jandroid.h"
#include "file_access_android.h"
#include "file_access_jandroid.h"
#include "jni_utils.h"
#include "main/input_default.h"
#include "main/main.h"
#include "net_socket_android.h"
#include "os_android.h"
#include "string_android.h"
#include "thread_jandroid.h"
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

extern "C" {

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_setVirtualKeyboardHeight(JNIEnv *env, jclass clazz, jint p_height) {
	if (godot_io_java) {
		godot_io_java->set_vk_height(p_height);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_initialize(JNIEnv *env, jclass clazz, jobject activity, jobject p_asset_manager, jboolean p_use_apk_expansion) {

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
	NetSocketAndroid::setup(godot_java->get_member_object("netUtils", "Lorg/godotengine/godot/utils/GodotNetUtils;", env));

	os_android = new OS_Android(godot_java, godot_io_java, p_use_apk_expansion);

	char wd[500];
	getcwd(wd, 500);

	godot_java->on_video_init(env);
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_ondestroy(JNIEnv *env, jclass clazz, jobject activity) {
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
		godot_java->on_gl_godot_main_loop_started(env);
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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_touch(JNIEnv *env, jclass clazz, jint ev, jint pointer, jint count, jintArray positions) {

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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_hover(JNIEnv *env, jclass clazz, jint p_type, jint p_x, jint p_y) {
	if (step == 0)
		return;

	os_android->process_hover(p_type, Point2(p_x, p_y));
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_doubletap(JNIEnv *env, jclass clazz, jint p_x, jint p_y) {
	if (step == 0)
		return;

	os_android->process_double_tap(Point2(p_x, p_y));
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

	ThreadAndroid::setup_thread();
	AudioDriverAndroid::thread_func(env);
}

JNIEXPORT jstring JNICALL Java_org_godotengine_godot_GodotLib_getGlobal(JNIEnv *env, jclass clazz, jstring path) {

	String js = jstring_to_string(path, env);

	return env->NewStringUTF(ProjectSettings::get_singleton()->get(js).operator String().utf8().get_data());
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_callobject(JNIEnv *env, jclass clazz, jint ID, jstring method, jobjectArray params) {

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

JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_calldeferred(JNIEnv *env, jclass clazz, jint ID, jstring method, jobjectArray params) {

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
