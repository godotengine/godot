/*************************************************************************/
/*  java_godot_io_wrapper.h                                              */
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

// note, swapped java and godot around in the file name so all the java
// wrappers are together

#ifndef JAVA_GODOT_IO_WRAPPER_H
#define JAVA_GODOT_IO_WRAPPER_H

#include <android/log.h>
#include <jni.h>

#include "string_android.h"

// Class that makes functions in java/src/org/godotengine/godot/GodotIO.java callable from C++
class GodotIOJavaWrapper {
private:
	jobject godot_io_instance;
	jclass cls;

	jmethodID _open_URI = 0;
	jmethodID _get_data_dir = 0;
	jmethodID _get_locale = 0;
	jmethodID _get_model = 0;
	jmethodID _get_screen_DPI = 0;
	jmethodID _get_window_safe_area = 0;
	jmethodID _get_unique_id = 0;
	jmethodID _show_keyboard = 0;
	jmethodID _hide_keyboard = 0;
	jmethodID _set_screen_orientation = 0;
	jmethodID _get_screen_orientation = 0;
	jmethodID _get_system_dir = 0;
	jmethodID _play_video = 0;
	jmethodID _is_video_playing = 0;
	jmethodID _pause_video = 0;
	jmethodID _stop_video = 0;

public:
	GodotIOJavaWrapper(JNIEnv *p_env, jobject p_godot_io_instance);
	~GodotIOJavaWrapper();

	jobject get_instance();

	Error open_uri(const String &p_uri);
	String get_user_data_dir();
	String get_locale();
	String get_model();
	int get_screen_dpi();
	void get_window_safe_area(int (&p_rect_xywh)[4]);
	String get_unique_id();
	bool has_vk();
	void show_vk(const String &p_existing, bool p_multiline, int p_max_input_length, int p_cursor_start, int p_cursor_end);
	void hide_vk();
	int get_vk_height();
	void set_vk_height(int p_height);
	void set_screen_orientation(int p_orient);
	int get_screen_orientation() const;
	String get_system_dir(int p_dir);
	void play_video(const String &p_path);
	bool is_video_playing();
	void pause_video();
	void stop_video();
};

#endif /* !JAVA_GODOT_IO_WRAPPER_H */
