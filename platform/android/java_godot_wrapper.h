/*************************************************************************/
/*  java_godot_wrapper.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

// note, swapped java and godot around in the file name so all the java
// wrappers are together

#ifndef JAVA_GODOT_WRAPPER_H
#define JAVA_GODOT_WRAPPER_H

#include <android/log.h>
#include <jni.h>

#include "string_android.h"

// Class that makes functions in java/src/org/godotengine/godot/Godot.java callable from C++
class GodotJavaWrapper {
private:
	jobject godot_instance;
	jobject activity;
	jclass godot_class;
	jclass activity_class;

	jmethodID _on_video_init = 0;
	jmethodID _create_offscreen_gl = 0;
	jmethodID _destroy_offscreen_gl = 0;
	jmethodID _set_offscreen_gl_current = 0;
	jmethodID _restart = 0;
	jmethodID _finish = 0;
	jmethodID _set_keep_screen_on = 0;
	jmethodID _alert = 0;
	jmethodID _get_GLES_version_code = 0;
	jmethodID _get_clipboard = 0;
	jmethodID _set_clipboard = 0;
	jmethodID _request_permission = 0;
	jmethodID _request_permissions = 0;
	jmethodID _get_granted_permissions = 0;
	jmethodID _init_input_devices = 0;
	jmethodID _get_surface = 0;
	jmethodID _is_activity_resumed = 0;
	jmethodID _vibrate = 0;
	jmethodID _get_input_fallback_mapping = 0;
	jmethodID _on_godot_setup_completed = 0;
	jmethodID _on_godot_main_loop_started = 0;
	jmethodID _get_class_loader = 0;

public:
	GodotJavaWrapper(JNIEnv *p_env, jobject p_activity, jobject p_godot_instance);
	~GodotJavaWrapper();

	jobject get_activity();
	jobject get_member_object(const char *p_name, const char *p_class, JNIEnv *p_env = NULL);

	jobject get_class_loader();

	void gfx_init(bool gl2);
	bool create_offscreen_gl(JNIEnv *p_env);
	void destroy_offscreen_gl(JNIEnv *p_env);
	void set_offscreen_gl_current(JNIEnv *p_env, bool p_current);
	void on_video_init(JNIEnv *p_env = NULL);
	void on_godot_setup_completed(JNIEnv *p_env = NULL);
	void on_godot_main_loop_started(JNIEnv *p_env = NULL);
	void restart(JNIEnv *p_env = NULL);
	void force_quit(JNIEnv *p_env = NULL);
	void set_keep_screen_on(bool p_enabled);
	void alert(const String &p_message, const String &p_title);
	int get_gles_version_code();
	bool has_get_clipboard();
	String get_clipboard();
	bool has_set_clipboard();
	void set_clipboard(const String &p_text);
	bool request_permission(const String &p_name);
	bool request_permissions();
	Vector<String> get_granted_permissions() const;
	void init_input_devices();
	jobject get_surface();
	bool is_activity_resumed();
	void vibrate(int p_duration_ms);
	String get_input_fallback_mapping();
};

#endif /* !JAVA_GODOT_WRAPPER_H */
