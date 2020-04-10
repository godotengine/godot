/*************************************************************************/
/*  os_android.h                                                         */
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

#ifndef OS_ANDROID_H
#define OS_ANDROID_H

#include "audio_driver_jandroid.h"
#include "audio_driver_opensl.h"
#include "core/os/main_loop.h"
#include "drivers/unix/os_unix.h"
#include "servers/audio_server.h"

class GodotJavaWrapper;
class GodotIOJavaWrapper;

struct ANativeWindow;

class OS_Android : public OS_Unix {
  // Move to DisplayServerAndroid
public:
	// Touch pointer info constants.
	// Note: These values must match the one in org.godotengine.godot.input.GodotInputHandler.java
	enum {
		// Must match org.godotengine.godot.input.GodotInputHandler#POINTER_INFO_ID_OFFSET
		TOUCH_POINTER_INFO_ID_OFFSET = 0,
		// Must match org.godotengine.godot.input.GodotInputHandler#POINTER_INFO_TOOL_TYPE_OFFSET
		TOUCH_POINTER_INFO_TOOL_TYPE_OFFSET = 1,
		// Must match org.godotengine.godot.input.GodotInputHandler#POINTER_INFO_SIZE
		TOUCH_POINTER_INFO_SIZE = 2
	};

	// Android MotionEvent's ACTION_* constants.
	enum {
		ACTION_BUTTON_PRESS = 11,
		ACTION_BUTTON_RELEASE = 12,
		ACTION_CANCEL = 3,
		ACTION_DOWN = 0,
		ACTION_MOVE = 2,
		ACTION_POINTER_DOWN = 5,
		ACTION_POINTER_UP = 6,
		ACTION_UP = 1
	};

	// Android MotionEvent's TOOL_TYPE_* constants.
	enum {
		TOOL_TYPE_UNKNOWN = 0,
		TOOL_TYPE_FINGER = 1,
		TOOL_TYPE_STYLUS = 2,
		TOOL_TYPE_MOUSE = 3,
		TOOL_TYPE_ERASER = 4
	};

	// Android MotionEvent's BUTTON_* constants.
	enum {
		BUTTON_PRIMARY = 1,
		BUTTON_SECONDARY = 2,
		BUTTON_TERTIARY = 4,
		BUTTON_BACK = 8,
		BUTTON_FORWARD = 16,
		BUTTON_STYLUS_PRIMARY = 32,
		BUTTON_STYLUS_SECONDARY = 64
	};

	struct TouchPos {
		int id;
		Point2 pos;
		int tool_type;
	};

	enum {
		JOY_EVENT_BUTTON = 0,
		JOY_EVENT_AXIS = 1,
		JOY_EVENT_HAT = 2
	};

	struct JoypadEvent {

		int device;
		int type;
		int index;
		bool pressed;
		float value;
		int hat;
	};
	// end -move

private:
	Size2i display_size;

	// Move to DisplayServerAndroid
	void send_touch_event(TouchPos touch_pos, int android_motion_event_action_button);
	void release_touch_event(TouchPos touch_pos, int android_motion_event_action_button, bool update_last_mouse_buttons_mask = false);
	void release_touches(int android_motion_event_action_button, bool update_last_mouse_buttons_mask = false);
	int get_mouse_button_index(int android_motion_event_button_state);
	inline bool is_mouse_pointer(TouchPos touch_pos) const;
	inline bool is_mouse_pointer(int tool_type) const;

	Vector<TouchPos> touch;
	// Needed to calculate the relative position on hover events
	Point2 hover_prev_pos = Point2();
	// This is specific to the Godot engine and should not be confused with Android's MotionEvent#getButtonState()
	int last_mouse_buttons_mask = 0;
	Point2 last_mouse_position = Point2();
	// - end move

	bool use_apk_expansion;

#if defined(OPENGL_ENABLED)
	bool use_16bits_fbo;
	const char *gl_extensions;
#endif

#if defined(VULKAN_ENABLED)
	ANativeWindow *native_window;
#endif

	mutable String data_dir_cache;

	//AudioDriverAndroid audio_driver_android;
	AudioDriverOpenSL audio_driver_android;

	MainLoop *main_loop;

	GodotJavaWrapper *godot_java;
	GodotIOJavaWrapper *godot_io_java;

public:
	virtual void initialize_core();
	virtual void initialize();

	virtual void initialize_joypads();

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();

	typedef int64_t ProcessID;

	static OS_Android *get_singleton();
	GodotJavaWrapper *get_godot_java();
	GodotIOJavaWrapper *get_godot_io_java();

	virtual bool request_permission(const String &p_name);
	virtual bool request_permissions();
	virtual Vector<String> get_granted_permissions() const;

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);

	virtual String get_name() const;
	virtual MainLoop *get_main_loop() const;

	void main_loop_begin();
	bool main_loop_iterate();
	void main_loop_request_go_back();
	void main_loop_end();
	void main_loop_focusout();
	void main_loop_focusin();

	void set_display_size(const Size2i &p_size);
	Size2i get_display_size() const;

	void set_context_is_16_bits(bool p_is_16);
	void set_opengl_extensions(const char *p_gl_extensions);

	void set_native_window(ANativeWindow *p_native_window);
	ANativeWindow *get_native_window() const;

	virtual Error shell_open(String p_uri);
	virtual String get_user_data_dir() const;
	virtual String get_resource_dir() const;
	virtual String get_locale() const;
	virtual String get_model_name() const;

	virtual String get_unique_id() const;

	virtual String get_system_dir(SystemDir p_dir) const;

	// Move to DisplayServerAndroid
	void process_accelerometer(const Vector3 &p_accelerometer);
	void process_gravity(const Vector3 &p_gravity);
	void process_magnetometer(const Vector3 &p_magnetometer);
	void process_gyroscope(const Vector3 &p_gyroscope);
	void process_touch(int motion_event_action, int motion_event_action_button, int p_pointer, const Vector<TouchPos> &p_points);
	void process_hover(int tool_type, int p_type, Point2 p_pos);
	void process_double_tap(int tool_type, int android_motion_event_button_state, Point2 p_pos);
	void process_scroll(int tool_type, Point2 start, Point2 end, Vector2 scroll_delta);
	void process_joy_event(JoypadEvent p_event);
	void process_event(Ref<InputEvent> p_event);
	void init_video_mode(int p_video_width, int p_video_height);

	virtual Error native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track);
	virtual bool native_video_is_playing() const;
	virtual void native_video_pause();
	virtual void native_video_stop();

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;
	void joy_connection_changed(int p_device, bool p_connected, String p_name);
	// - end move
	void vibrate_handheld(int p_duration_ms);

	virtual bool _check_internal_feature_support(const String &p_feature);
	OS_Android(GodotJavaWrapper *p_godot_java, GodotIOJavaWrapper *p_godot_io_java, bool p_use_apk_expansion);
	~OS_Android();
};

#endif
