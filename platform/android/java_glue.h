/*************************************************************************/
/*  java_glue.h                                                          */
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
#ifndef ANDROID_NATIVE_ACTIVITY

#ifndef JAVA_GLUE_H
#define JAVA_GLUE_H

#include <android/log.h>
#include <jni.h>

extern "C" {
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_initialize(JNIEnv *env, jobject obj, jobject activity, jboolean p_need_reload_hook, jobjectArray p_cmdline, jobject p_asset_manager);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_resize(JNIEnv *env, jobject obj, jint width, jint height, jboolean reload);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_newcontext(JNIEnv *env, jobject obj, bool p_32_bits);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_step(JNIEnv *env, jobject obj);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_quit(JNIEnv *env, jobject obj);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_touch(JNIEnv *env, jobject obj, jint ev, jint pointer, jint count, jintArray positions);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_key(JNIEnv *env, jobject obj, jint p_scancode, jint p_unicode_char, jboolean p_pressed);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joybutton(JNIEnv *env, jobject obj, jint p_device, jint p_button, jboolean p_pressed);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyaxis(JNIEnv *env, jobject obj, jint p_device, jint p_axis, jfloat p_value);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyhat(JNIEnv *env, jobject obj, jint p_device, jint p_hat_x, jint p_hat_y);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_joyconnectionchanged(JNIEnv *env, jobject obj, jint p_device, jboolean p_connected, jstring p_name);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_audio(JNIEnv *env, jobject obj);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_accelerometer(JNIEnv *env, jobject obj, jfloat x, jfloat y, jfloat z);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_magnetometer(JNIEnv *env, jobject obj, jfloat x, jfloat y, jfloat z);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_gyroscope(JNIEnv *env, jobject obj, jfloat x, jfloat y, jfloat z);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_focusin(JNIEnv *env, jobject obj);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_focusout(JNIEnv *env, jobject obj);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_singleton(JNIEnv *env, jobject obj, jstring name, jobject p_object);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_method(JNIEnv *env, jobject obj, jstring sname, jstring name, jstring ret, jobjectArray args);
JNIEXPORT jstring JNICALL Java_org_godotengine_godot_GodotLib_getGlobal(JNIEnv *env, jobject obj, jstring path);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_callobject(JNIEnv *env, jobject obj, jint ID, jstring method, jobjectArray params);
JNIEXPORT void JNICALL Java_org_godotengine_godot_GodotLib_calldeferred(JNIEnv *env, jobject obj, jint ID, jstring method, jobjectArray params);
}

#endif
#endif // JAVA_GLUE_H
