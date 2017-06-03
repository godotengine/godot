/*************************************************************************/
/*  godot_android.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifdef ANDROID_NATIVE_ACTIVITY

#include <errno.h>
#include <jni.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include "file_access_android.h"
#include "global_config.h"
#include "main/main.h"
#include "os_android.h"
#include <android/log.h>
#include <android/sensor.h>
#include <android/window.h>
#include <android_native_app_glue.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "godot", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "godot", __VA_ARGS__))

extern "C" {
JNIEXPORT void JNICALL Java_org_godotengine_godot_Godot_registerSingleton(JNIEnv *env, jobject obj, jstring name, jobject p_object);
JNIEXPORT void JNICALL Java_org_godotengine_godot_Godot_registerMethod(JNIEnv *env, jobject obj, jstring sname, jstring name, jstring ret, jobjectArray args);
JNIEXPORT jstring JNICALL Java_org_godotengine_godot_Godot_getGlobal(JNIEnv *env, jobject obj, jstring path);
};

class JNISingleton : public Object {

	GDCLASS(JNISingleton, Object);

	struct MethodData {

		jmethodID method;
		Variant::Type ret_type;
		Vector<Variant::Type> argtypes;
	};

	jobject instance;
	Map<StringName, MethodData> method_map;
	JNIEnv *env;

public:
	void update_env(JNIEnv *p_env) { env = p_env; }

	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

		print_line("attempt to call " + String(p_method));

		r_error.error = Variant::CallError::CALL_OK;

		Map<StringName, MethodData>::Element *E = method_map.find(p_method);
		if (!E) {

			print_line("no exists");
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			return Variant();
		}

		int ac = E->get().argtypes.size();
		if (ac < p_argcount) {

			print_line("fewargs");
			r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument = ac;
			return Variant();
		}

		if (ac > p_argcount) {

			print_line("manyargs");
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

		for (int i = 0; i < p_argcount; i++) {

			switch (E->get().argtypes[i]) {

				case Variant::BOOL: {

					v[i].z = *p_args[i];
				} break;
				case Variant::INT: {

					v[i].i = *p_args[i];
				} break;
				case Variant::REAL: {

					v[i].f = *p_args[i];
				} break;
				case Variant::STRING: {

					String s = *p_args[i];
					jstring jStr = env->NewStringUTF(s.utf8().get_data());
					v[i].l = jStr;
				} break;
				case Variant::STRING_ARRAY: {

					PoolVector<String> sarray = *p_args[i];
					jobjectArray arr = env->NewObjectArray(sarray.size(), env->FindClass("java/lang/String"), env->NewStringUTF(""));

					for (int j = 0; j < sarray.size(); j++) {

						env->SetObjectArrayElement(arr, j, env->NewStringUTF(sarray[i].utf8().get_data()));
					}
					v[i].l = arr;

				} break;
				case Variant::INT_ARRAY: {

					PoolVector<int> array = *p_args[i];
					jintArray arr = env->NewIntArray(array.size());
					PoolVector<int>::Read r = array.read();
					env->SetIntArrayRegion(arr, 0, array.size(), r.ptr());
					v[i].l = arr;

				} break;
				case Variant::REAL_ARRAY: {

					PoolVector<float> array = *p_args[i];
					jfloatArray arr = env->NewFloatArray(array.size());
					PoolVector<float>::Read r = array.read();
					env->SetFloatArrayRegion(arr, 0, array.size(), r.ptr());
					v[i].l = arr;

				} break;
				default: {

					ERR_FAIL_V(Variant());
				} break;
			}
		}

		print_line("calling method!!");

		Variant ret;

		switch (E->get().ret_type) {

			case Variant::NIL: {

				print_line("call void");
				env->CallVoidMethodA(instance, E->get().method, v);
			} break;
			case Variant::BOOL: {

				ret = env->CallBooleanMethodA(instance, E->get().method, v);
				print_line("call bool");
			} break;
			case Variant::INT: {

				ret = env->CallIntMethodA(instance, E->get().method, v);
				print_line("call int");
			} break;
			case Variant::REAL: {

				ret = env->CallFloatMethodA(instance, E->get().method, v);
			} break;
			case Variant::STRING: {

				jobject o = env->CallObjectMethodA(instance, E->get().method, v);
				String singname = env->GetStringUTFChars((jstring)o, NULL);
			} break;
			case Variant::STRING_ARRAY: {

				jobjectArray arr = (jobjectArray)env->CallObjectMethodA(instance, E->get().method, v);

				int stringCount = env->GetArrayLength(arr);
				PoolVector<String> sarr;

				for (int i = 0; i < stringCount; i++) {
					jstring string = (jstring)env->GetObjectArrayElement(arr, i);
					const char *rawString = env->GetStringUTFChars(string, 0);
					sarr.push_back(String(rawString));
				}

				ret = sarr;

			} break;
			case Variant::INT_ARRAY: {

				jintArray arr = (jintArray)env->CallObjectMethodA(instance, E->get().method, v);

				int fCount = env->GetArrayLength(arr);
				PoolVector<int> sarr;
				sarr.resize(fCount);

				PoolVector<int>::Write w = sarr.write();
				env->GetIntArrayRegion(arr, 0, fCount, w.ptr());
				w = PoolVector<int>::Write();
				ret = sarr;
			} break;
			case Variant::REAL_ARRAY: {

				jfloatArray arr = (jfloatArray)env->CallObjectMethodA(instance, E->get().method, v);

				int fCount = env->GetArrayLength(arr);
				PoolVector<float> sarr;
				sarr.resize(fCount);

				PoolVector<float>::Write w = sarr.write();
				env->GetFloatArrayRegion(arr, 0, fCount, w.ptr());
				w = PoolVector<float>::Write();
				ret = sarr;
			} break;
			default: {

				print_line("failure..");
				ERR_FAIL_V(Variant());
			} break;
		}

		print_line("success");

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

	JNISingleton() {}
};

//JNIEnv *JNISingleton::env=NULL;

static HashMap<String, JNISingleton *> jni_singletons;

struct engine {
	struct android_app *app;
	OS_Android *os;
	JNIEnv *jni;

	ASensorManager *sensorManager;
	const ASensor *accelerometerSensor;
	const ASensor *magnetometerSensor;
	const ASensor *gyroscopeSensor;
	ASensorEventQueue *sensorEventQueue;

	bool display_active;
	bool requested_quit;
	int animating;
	EGLDisplay display;
	EGLSurface surface;
	EGLContext context;
	int32_t width;
	int32_t height;
};

/**
 * Initialize an EGL context for the current display.
 */
static int engine_init_display(struct engine *engine, bool p_gl2) {
	// initialize OpenGL ES and EGL

	/*
     * Here specify the attributes of the desired configuration.
     * Below, we select an EGLConfig with at least 8 bits per color
     * component compatible with on-screen windows
     */
	const EGLint gl2_attribs[] = {
		//  EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_BLUE_SIZE, 4,
		EGL_GREEN_SIZE, 4,
		EGL_RED_SIZE, 4,
		EGL_ALPHA_SIZE, 0,
		EGL_DEPTH_SIZE, 16,
		EGL_STENCIL_SIZE, EGL_DONT_CARE,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_NONE
	};

	const EGLint gl1_attribs[] = {
		//  EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_BLUE_SIZE, 4,
		EGL_GREEN_SIZE, 4,
		EGL_RED_SIZE, 4,
		EGL_ALPHA_SIZE, 0,
		EGL_DEPTH_SIZE, 16,
		EGL_STENCIL_SIZE, EGL_DONT_CARE,
		EGL_NONE
	};

	const EGLint *attribs = p_gl2 ? gl2_attribs : gl1_attribs;

	EGLint w, h, dummy, format;
	EGLint numConfigs;
	EGLConfig config;
	EGLSurface surface;
	EGLContext context;

	EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

	eglInitialize(display, 0, 0);

	/* Here, the application chooses the configuration it desires. In this
     * sample, we have a very simplified selection process, where we pick
     * the first EGLConfig that matches our criteria */

	eglChooseConfig(display, attribs, &config, 1, &numConfigs);

	LOGI("Num configs: %i\n", numConfigs);

	/* EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
     * guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
     * As soon as we picked a EGLConfig, we can safely reconfigure the
     * ANativeWindow buffers to match, using EGL_NATIVE_VISUAL_ID. */
	eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);

	ANativeWindow_setBuffersGeometry(engine->app->window, 0, 0, format);
	//ANativeWindow_setFlags(engine->app->window, 0, 0, format|);

	surface = eglCreateWindowSurface(display, config, engine->app->window, NULL);

	const EGLint context_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_NONE
	};
	context = eglCreateContext(display, config, EGL_NO_CONTEXT, p_gl2 ? context_attribs : NULL);

	if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
		LOGW("Unable to eglMakeCurrent");
		return -1;
	}

	eglQuerySurface(display, surface, EGL_WIDTH, &w);
	eglQuerySurface(display, surface, EGL_HEIGHT, &h);
	print_line("INIT VIDEO MODE: " + itos(w) + "," + itos(h));

	//engine->os->set_egl_extensions(eglQueryString(display,EGL_EXTENSIONS));
	engine->os->init_video_mode(w, h);

	engine->display = display;
	engine->context = context;
	engine->surface = surface;
	engine->width = w;
	engine->height = h;
	engine->display_active = true;

	//engine->state.angle = 0;

	// Initialize GL state.
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
	glEnable(GL_CULL_FACE);
	//  glShadeModel(GL_SMOOTH);
	glDisable(GL_DEPTH_TEST);
	LOGI("GL Version: %s - %s %s\n", glGetString(GL_VERSION), glGetString(GL_VENDOR), glGetString(GL_RENDERER));

	return 0;
}

static void engine_draw_frame(struct engine *engine) {
	if (engine->display == NULL) {
		// No display.
		return;
	}

	// Just fill the screen with a color.
	//glClearColor(0,1,0,1);
	//glClear(GL_COLOR_BUFFER_BIT);
	if (engine->os && engine->os->main_loop_iterate() == true) {

		engine->requested_quit = true;
		return; //should exit instead
	}

	eglSwapBuffers(engine->display, engine->surface);
}

static void engine_term_display(struct engine *engine) {
	if (engine->display != EGL_NO_DISPLAY) {
		eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		if (engine->context != EGL_NO_CONTEXT) {
			eglDestroyContext(engine->display, engine->context);
		}
		if (engine->surface != EGL_NO_SURFACE) {
			eglDestroySurface(engine->display, engine->surface);
		}
		eglTerminate(engine->display);
	}

	engine->animating = 0;
	engine->display = EGL_NO_DISPLAY;
	engine->context = EGL_NO_CONTEXT;
	engine->surface = EGL_NO_SURFACE;
	engine->display_active = false;
}

/**
 * Process the next input event.
 */
static int32_t engine_handle_input(struct android_app *app, AInputEvent *event) {
	struct engine *engine = (struct engine *)app->userData;

	if (!engine->os)
		return 0;

	switch (AInputEvent_getType(event)) {

		case AINPUT_EVENT_TYPE_KEY: {

			int ac = AKeyEvent_getAction(event);
			switch (ac) {

				case AKEY_EVENT_ACTION_DOWN: {

					int32_t code = AKeyEvent_getKeyCode(event);
					if (code == AKEYCODE_BACK) {

						//AInputQueue_finishEvent(AInputQueue* queue, AInputEvent* event, int handled);
						if (engine->os)
							engine->os->main_loop_request_quit();
						return 1;
					}

				} break;
				case AKEY_EVENT_ACTION_UP: {

				} break;
			}

		} break;
		case AINPUT_EVENT_TYPE_MOTION: {

			Vector<OS_Android::TouchPos> touchvec;

			int pc = AMotionEvent_getPointerCount(event);

			touchvec.resize(pc);

			for (int i = 0; i < pc; i++) {

				touchvec[i].pos.x = AMotionEvent_getX(event, i);
				touchvec[i].pos.y = AMotionEvent_getY(event, i);
				touchvec[i].id = AMotionEvent_getPointerId(event, i);
			}

			//System.out.printf("gaction: %d\n",event.getAction());
			int pidx = (AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> 8;
			switch (AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK) {

				case AMOTION_EVENT_ACTION_DOWN: {
					engine->os->process_touch(0, 0, touchvec);

					//System.out.printf("action down at: %f,%f\n", event.getX(),event.getY());
				} break;
				case AMOTION_EVENT_ACTION_MOVE: {
					engine->os->process_touch(1, 0, touchvec);
					/*
					for(int i=0;i<event.getPointerCount();i++) {
						System.out.printf("%d - moved to: %f,%f\n",i, event.getX(i),event.getY(i));
					}
					*/
				} break;
				case AMOTION_EVENT_ACTION_POINTER_UP: {

					engine->os->process_touch(4, pidx, touchvec);
					//System.out.printf("%d - s.up at: %f,%f\n",pointer_idx, event.getX(pointer_idx),event.getY(pointer_idx));
				} break;
				case AMOTION_EVENT_ACTION_POINTER_DOWN: {
					engine->os->process_touch(3, pidx, touchvec);
					//System.out.printf("%d - s.down at: %f,%f\n",pointer_idx, event.getX(pointer_idx),event.getY(pointer_idx));
				} break;
				case AMOTION_EVENT_ACTION_CANCEL:
				case AMOTION_EVENT_ACTION_UP: {
					engine->os->process_touch(2, 0, touchvec);
					/*
					for(int i=0;i<event.getPointerCount();i++) {
						System.out.printf("%d - up! %f,%f\n",i, event.getX(i),event.getY(i));
					}
					*/
				} break;
			}

			return 1;
		} break;
	}

	return 0;
}

/**
 * Process the next main command.
 */

static void _gfx_init(void *ud, bool p_gl2) {

	struct engine *engine = (struct engine *)ud;
	engine_init_display(engine, p_gl2);
}

static void engine_handle_cmd(struct android_app *app, int32_t cmd) {
	struct engine *engine = (struct engine *)app->userData;
	// LOGI("**** CMD %i\n",cmd);
	switch (cmd) {
		case APP_CMD_SAVE_STATE:
			// The system has asked us to save our current state.  Do so.
			//engine->app->savedState = malloc(sizeof(struct saved_state));
			//*((struct saved_state*)engine->app->savedState) = engine->state;
			//engine->app->savedStateSize = sizeof(struct saved_state);
			break;
		case APP_CMD_CONFIG_CHANGED:
		case APP_CMD_WINDOW_RESIZED: {

#if 0
// android blows
		if (engine->display_active) {

			EGLint w,h;
			eglQuerySurface(engine->display, engine->surface, EGL_WIDTH, &w);
			eglQuerySurface(engine->display, engine->surface, EGL_HEIGHT, &h);
			engine->os->init_video_mode(w,h);
			//print_line("RESIZED VIDEO MODE: "+itos(w)+","+itos(h));
			engine_draw_frame(engine);

		}
#else

			if (engine->display_active) {

				EGLint w, h;
				eglQuerySurface(engine->display, engine->surface, EGL_WIDTH, &w);
				eglQuerySurface(engine->display, engine->surface, EGL_HEIGHT, &h);
				//  if (w==engine->os->get_video_mode().width && h==engine->os->get_video_mode().height)
				//    break;

				engine_term_display(engine);
			}

			engine->os->reload_gfx();
			engine_draw_frame(engine);
			engine->animating = 1;

/*
			    EGLint w,h;
			    eglQuerySurface(engine->display, engine->surface, EGL_WIDTH, &w);
			    eglQuerySurface(engine->display, engine->surface, EGL_HEIGHT, &h);
			    engine->os->init_video_mode(w,h);
			    //print_line("RESIZED VIDEO MODE: "+itos(w)+","+itos(h));

		    }*/

#endif

		} break;
		case APP_CMD_INIT_WINDOW:
			//The window is being shown, get it ready.
			//LOGI("INIT WINDOW");
			if (engine->app->window != NULL) {

				if (engine->os == NULL) {

					//do initialization here, when there's OpenGL! hackish but the only way
					engine->os = new OS_Android(_gfx_init, engine);

					//char *args[]={"-test","gui",NULL};
					__android_log_print(ANDROID_LOG_INFO, "godot", "pre asdasd setup...");
#if 0
				Error err  = Main::setup("apk",2,args);
#else
					Error err = Main::setup("apk", 0, NULL);

					String modules = GlobalConfig::get_singleton()->get("android/modules");
					Vector<String> mods = modules.split(",", false);
					mods.push_back("GodotOS");
					__android_log_print(ANDROID_LOG_INFO, "godot", "mod count: %i", mods.size());

					if (mods.size()) {

						jclass activityClass = engine->jni->FindClass("android/app/NativeActivity");

						jmethodID getClassLoader = engine->jni->GetMethodID(activityClass, "getClassLoader", "()Ljava/lang/ClassLoader;");

						jobject cls = engine->jni->CallObjectMethod(app->activity->clazz, getClassLoader);

						jclass classLoader = engine->jni->FindClass("java/lang/ClassLoader");

						jmethodID findClass = engine->jni->GetMethodID(classLoader, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");

						static JNINativeMethod methods[] = {
							{ "registerSingleton", "(Ljava/lang/String;Ljava/lang/Object;)V", (void *)&Java_org_godotengine_godot_Godot_registerSingleton },
							{ "registerMethod", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;)V", (void *)&Java_org_godotengine_godot_Godot_registerMethod },
							{ "getGlobal", "(Ljava/lang/String;)Ljava/lang/String;", (void *)&Java_org_godotengine_godot_Godot_getGlobal },
						};

						jstring gstrClassName = engine->jni->NewStringUTF("org/godotengine/godot/Godot");
						jclass GodotClass = (jclass)engine->jni->CallObjectMethod(cls, findClass, gstrClassName);

						__android_log_print(ANDROID_LOG_INFO, "godot", "godot ****^*^*?^*^*class data %x", GodotClass);

						engine->jni->RegisterNatives(GodotClass, methods, sizeof(methods) / sizeof(methods[0]));

						for (int i = 0; i < mods.size(); i++) {

							String m = mods[i];
							//jclass singletonClass = engine->jni->FindClass(m.utf8().get_data());

							jstring strClassName = engine->jni->NewStringUTF(m.utf8().get_data());
							jclass singletonClass = (jclass)engine->jni->CallObjectMethod(cls, findClass, strClassName);

							__android_log_print(ANDROID_LOG_INFO, "godot", "****^*^*?^*^*class data %x", singletonClass);
							jmethodID initialize = engine->jni->GetStaticMethodID(singletonClass, "initialize", "(Landroid/app/Activity;)Lorg/godotengine/godot/Godot$SingletonBase;");

							jobject obj = engine->jni->CallStaticObjectMethod(singletonClass, initialize, app->activity->clazz);
							__android_log_print(ANDROID_LOG_INFO, "godot", "****^*^*?^*^*class instance %x", obj);
							jobject gob = engine->jni->NewGlobalRef(obj);
						}
					}

#endif

					if (!Main::start())
						return; //should exit instead and print the error

					engine->os->main_loop_begin();
				} else {
					//i guess recreate resources?
					engine->os->reload_gfx();
				}

				engine->animating = 1;
				engine_draw_frame(engine);
			}
			break;
		case APP_CMD_TERM_WINDOW:
			// The window is being hidden or closed, clean it up.
			//LOGI("TERM WINDOW");
			engine_term_display(engine);
			break;
		case APP_CMD_GAINED_FOCUS:
			// When our app gains focus, we start monitoring the accelerometer.
			if (engine->accelerometerSensor != NULL) {
				ASensorEventQueue_enableSensor(engine->sensorEventQueue,
						engine->accelerometerSensor);
				// We'd like to get 60 events per second (in us).
				ASensorEventQueue_setEventRate(engine->sensorEventQueue,
						engine->accelerometerSensor, (1000L / 60) * 1000);
			}
			// Also start monitoring the magnetometer.
			if (engine->magnetometerSensor != NULL) {
				ASensorEventQueue_enableSensor(engine->sensorEventQueue,
						engine->magnetometerSensor);
				// We'd like to get 60 events per second (in us).
				ASensorEventQueue_setEventRate(engine->sensorEventQueue,
						engine->magnetometerSensor, (1000L / 60) * 1000);
			}
			// And the gyroscope.
			if (engine->gyroscopeSensor != NULL) {
				ASensorEventQueue_enableSensor(engine->sensorEventQueue,
						engine->gyroscopeSensor);
				// We'd like to get 60 events per second (in us).
				ASensorEventQueue_setEventRate(engine->sensorEventQueue,
						engine->gyroscopeSensor, (1000L / 60) * 1000);
			}
			engine->animating = 1;
			break;
		case APP_CMD_LOST_FOCUS:
			// When our app loses focus, we stop monitoring the sensors.
			// This is to avoid consuming battery while not being used.
			if (engine->accelerometerSensor != NULL) {
				ASensorEventQueue_disableSensor(engine->sensorEventQueue,
						engine->accelerometerSensor);
			}
			if (engine->magnetometerSensor != NULL) {
				ASensorEventQueue_disableSensor(engine->sensorEventQueue,
						engine->magnetometerSensor);
			}
			if (engine->gyroscopeSensor != NULL) {
				ASensorEventQueue_disableSensor(engine->sensorEventQueue,
						engine->gyroscopeSensor);
			}
			// Also stop animating.
			engine->animating = 0;
			engine_draw_frame(engine);
			break;
	}
}

void android_main(struct android_app *state) {
	struct engine engine;
	// Make sure glue isn't stripped.
	app_dummy();

	memset(&engine, 0, sizeof(engine));
	state->userData = &engine;
	state->onAppCmd = engine_handle_cmd;
	state->onInputEvent = engine_handle_input;
	engine.app = state;
	engine.requested_quit = false;
	engine.os = NULL;
	engine.display_active = false;

	FileAccessAndroid::asset_manager = state->activity->assetManager;

	// Prepare to monitor sensors
	engine.sensorManager = ASensorManager_getInstance();
	engine.accelerometerSensor = ASensorManager_getDefaultSensor(engine.sensorManager,
			ASENSOR_TYPE_ACCELEROMETER);
	engine.magnetometerSensor = ASensorManager_getDefaultSensor(engine.sensorManager,
			ASENSOR_TYPE_MAGNETIC_FIELD);
	engine.gyroscopeSensor = ASensorManager_getDefaultSensor(engine.sensorManager,
			ASENSOR_TYPE_GYROSCOPE);
	engine.sensorEventQueue = ASensorManager_createEventQueue(engine.sensorManager,
			state->looper, LOOPER_ID_USER, NULL, NULL);

	ANativeActivity_setWindowFlags(state->activity, AWINDOW_FLAG_FULLSCREEN | AWINDOW_FLAG_KEEP_SCREEN_ON, 0);

	state->activity->vm->AttachCurrentThread(&engine.jni, NULL);

	// loop waiting for stuff to do.

	while (1) {
		// Read all pending events.
		int ident;
		int events;
		struct android_poll_source *source;

		// If not animating, we will block forever waiting for events.
		// If animating, we loop until all events are read, then continue
		// to draw the next frame of animation.

		int nullmax = 50;
		while ((ident = ALooper_pollAll(engine.animating ? 0 : -1, NULL, &events,
						(void **)&source)) >= 0) {

			// Process this event.

			if (source != NULL) {
				// LOGI("process\n");
				source->process(state, source);
			} else {
				nullmax--;
				if (nullmax < 0)
					break;
			}

			// If a sensor has data, process it now.
			// LOGI("events\n");
			if (ident == LOOPER_ID_USER) {
				if (engine.accelerometerSensor != NULL || engine.magnetometerSensor != NULL || engine.gyroscopeSensor != NULL) {
					ASensorEvent event;
					while (ASensorEventQueue_getEvents(engine.sensorEventQueue,
								   &event, 1) > 0) {

						if (engine.os) {
							if (event.acceleration != NULL) {
								engine.os->process_accelerometer(Vector3(event.acceleration.x, event.acceleration.y,
										event.acceleration.z));
							}
							if (event.magnetic != NULL) {
								engine.os->process_magnetometer(Vector3(event.magnetic.x, event.magnetic.y,
										event.magnetic.z));
							}
							if (event.vector != NULL) {
								engine.os->process_gyroscope(Vector3(event.vector.x, event.vector.y,
										event.vector.z));
							}
						}
					}
				}
			}

			// Check if we are exiting.
			if (state->destroyRequested != 0) {
				if (engine.os) {
					engine.os->main_loop_request_quit();
				}
				state->destroyRequested = 0;
			}

			if (engine.requested_quit) {
				engine_term_display(&engine);
				exit(0);
				return;
			}

			//     LOGI("end\n");
		}

		// LOGI("engine animating? %i\n",engine.animating);

		if (engine.animating) {
			//do os render

			engine_draw_frame(&engine);
			//LOGI("TERM WINDOW");
		}
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_Godot_registerSingleton(JNIEnv *env, jobject obj, jstring name, jobject p_object) {

	String singname = env->GetStringUTFChars(name, NULL);
	JNISingleton *s = memnew(JNISingleton);
	s->update_env(env);
	s->set_instance(env->NewGlobalRef(p_object));
	jni_singletons[singname] = s;

	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton(singname, s));
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
		{ "java.lang.String", Variant::STRING },
		{ "[I", Variant::INT_ARRAY },
		{ "[F", Variant::REAL_ARRAY },
		{ "[Ljava.lang.String;", Variant::STRING_ARRAY },
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
		{ "java.lang.String", "Ljava/lang/String;" },
		{ "[I", "[I" },
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

	return "";
}

JNIEXPORT jstring JNICALL Java_org_godotengine_godot_Godot_getGlobal(JNIEnv *env, jobject obj, jstring path) {

	String js = env->GetStringUTFChars(path, NULL);

	return env->NewStringUTF(GlobalConfig::get_singleton()->get(js).operator String().utf8().get_data());
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_Godot_registerMethod(JNIEnv *env, jobject obj, jstring sname, jstring name, jstring ret, jobjectArray args) {

	String singname = env->GetStringUTFChars(sname, NULL);

	ERR_FAIL_COND(!jni_singletons.has(singname));

	JNISingleton *s = jni_singletons.get(singname);

	String mname = env->GetStringUTFChars(name, NULL);
	String retval = env->GetStringUTFChars(ret, NULL);
	Vector<Variant::Type> types;
	String cs = "(";

	int stringCount = env->GetArrayLength(args);

	print_line("Singl:  " + singname + " Method: " + mname + " RetVal: " + retval);
	for (int i = 0; i < stringCount; i++) {

		jstring string = (jstring)env->GetObjectArrayElement(args, i);
		const char *rawString = env->GetStringUTFChars(string, 0);
		types.push_back(get_jni_type(String(rawString)));
		cs += get_jni_sig(String(rawString));
	}

	cs += ")";
	cs += get_jni_sig(retval);
	jclass cls = env->GetObjectClass(s->get_instance());
	print_line("METHOD: " + mname + " sig: " + cs);
	jmethodID mid = env->GetMethodID(cls, mname.ascii().get_data(), cs.ascii().get_data());
	if (!mid) {

		print_line("FAILED GETTING METHOID " + mname);
	}

	s->add_method(mname, mid, types, get_jni_type(retval));
}

#endif
