/**************************************************************************/
/*  java_godot_wrapper.h                                                  */
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

#ifndef JAVA_GODOT_WRAPPER_H
#define JAVA_GODOT_WRAPPER_H

#include "java_godot_view_wrapper.h"
#include "string_android.h"

#include "core/templates/list.h"

#include <android/log.h>
#include <jni.h>

// Class that makes functions in java/src/org/godotengine/godot/Godot.kt callable from C++
class GodotJavaWrapper {
private:
	jobject godot_instance;
	jobject activity;
	jclass godot_class;
	jclass activity_class;

	GodotJavaViewWrapper *godot_view = nullptr;

	jmethodID _restart = nullptr;
	jmethodID _finish = nullptr;
	jmethodID _set_keep_screen_on = nullptr;
	jmethodID _alert = nullptr;
	jmethodID _is_dark_mode_supported = nullptr;
	jmethodID _is_dark_mode = nullptr;
	jmethodID _get_clipboard = nullptr;
	jmethodID _set_clipboard = nullptr;
	jmethodID _has_clipboard = nullptr;
	jmethodID _request_permission = nullptr;
	jmethodID _request_permissions = nullptr;
	jmethodID _get_granted_permissions = nullptr;
	jmethodID _get_gdextension_list_config_file = nullptr;
	jmethodID _get_ca_certificates = nullptr;
	jmethodID _init_input_devices = nullptr;
	jmethodID _vibrate = nullptr;
	jmethodID _get_input_fallback_mapping = nullptr;
	jmethodID _on_godot_setup_completed = nullptr;
	jmethodID _on_godot_main_loop_started = nullptr;
	jmethodID _create_new_godot_instance = nullptr;
	jmethodID _get_render_view = nullptr;
	jmethodID _begin_benchmark_measure = nullptr;
	jmethodID _end_benchmark_measure = nullptr;
	jmethodID _dump_benchmark = nullptr;
	jmethodID _has_feature = nullptr;

public:
	GodotJavaWrapper(JNIEnv *p_env, jobject p_activity, jobject p_godot_instance);
	~GodotJavaWrapper();

	jobject get_activity();

	GodotJavaViewWrapper *get_godot_view();

	void on_godot_setup_completed(JNIEnv *p_env = nullptr);
	void on_godot_main_loop_started(JNIEnv *p_env = nullptr);
	void restart(JNIEnv *p_env = nullptr);
	bool force_quit(JNIEnv *p_env = nullptr, int p_instance_id = 0);
	void set_keep_screen_on(bool p_enabled);
	void alert(const String &p_message, const String &p_title);
	bool is_dark_mode_supported();
	bool is_dark_mode();
	bool has_get_clipboard();
	String get_clipboard();
	bool has_set_clipboard();
	void set_clipboard(const String &p_text);
	bool has_has_clipboard();
	bool has_clipboard();
	bool request_permission(const String &p_name);
	bool request_permissions();
	Vector<String> get_granted_permissions() const;
	String get_ca_certificates() const;
	void init_input_devices();
	void vibrate(int p_duration_ms, float p_amplitude = -1.0);
	String get_input_fallback_mapping();
	int create_new_godot_instance(const List<String> &args);
	void begin_benchmark_measure(const String &p_context, const String &p_label);
	void end_benchmark_measure(const String &p_context, const String &p_label);
	void dump_benchmark(const String &benchmark_file);

	// Return the list of gdextensions config file.
	Vector<String> get_gdextension_list_config_file() const;

	// Return true if the given feature is supported.
	bool has_feature(const String &p_feature) const;
};

#endif // JAVA_GODOT_WRAPPER_H
