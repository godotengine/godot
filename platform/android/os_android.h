/**************************************************************************/
/*  os_android.h                                                          */
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

#ifndef OS_ANDROID_H
#define OS_ANDROID_H

#include "audio_driver_opensl.h"
#include "core/os/main_loop.h"
#include "drivers/unix/os_unix.h"
#include "main/input_default.h"
#include "servers/audio_server.h"
#include "servers/visual/rasterizer.h"

class GodotJavaWrapper;
class GodotIOJavaWrapper;

class OS_Android : public OS_Unix {
	// https://developer.android.com/reference/android/view/PointerIcon
	// mapping between Godot's cursor shape to Android's'
	int android_cursors[CURSOR_MAX] = {
		1000, //CURSOR_ARROW
		1008, //CURSOR_IBEAM
		1002, //CURSOR_POINTIN
		1007, //CURSOR_CROSS
		1004, //CURSOR_WAIT
		1004, //CURSOR_BUSY
		1021, //CURSOR_DRAG
		1021, //CURSOR_CAN_DRO
		1000, //CURSOR_FORBIDD (no corresponding icon in Android's icon so fallback to default)
		1015, //CURSOR_VSIZE
		1014, //CURSOR_HSIZE
		1017, //CURSOR_BDIAGSI
		1016, //CURSOR_FDIAGSI
		1020, //CURSOR_MOVE
		1015, //CURSOR_VSPLIT
		1014, //CURSOR_HSPLIT
		1003, //CURSOR_HELP
	};
	const int CURSOR_TYPE_NULL = 0;
	MouseMode mouse_mode = MouseMode::MOUSE_MODE_VISIBLE;
	CursorShape cursor_shape = CursorShape::CURSOR_ARROW;

	bool use_apk_expansion;
	bool secondary_gl_available = false;

	VisualServer *visual_server;

	mutable String data_dir_cache;
	mutable String cache_dir_cache;

	AudioDriverOpenSL audio_driver_android;

	const char *gl_extensions = nullptr;

	InputDefault *input;
	VideoMode default_videomode;
	MainLoop *main_loop = nullptr;

	GodotJavaWrapper *godot_java;
	GodotIOJavaWrapper *godot_io_java;

	int video_driver_index;

	bool transparency_enabled = false;

public:
	static const char *ANDROID_EXEC_PATH;

	virtual bool tts_is_speaking() const;
	virtual bool tts_is_paused() const;
	virtual Array tts_get_voices() const;

	virtual void tts_speak(const String &p_text, const String &p_voice, int p_volume = 50, float p_pitch = 1.f, float p_rate = 1.f, int p_utterance_id = 0, bool p_interrupt = false);
	virtual void tts_pause();
	virtual void tts_resume();
	virtual void tts_stop();

	// functions used by main to initialize/deinitialize the OS
	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;

	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual int get_current_video_driver() const;

	virtual void initialize_core();
	virtual Error initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();

	typedef int64_t ProcessID;

	GodotJavaWrapper *get_godot_java();
	GodotIOJavaWrapper *get_godot_io_java();

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");
	virtual bool request_permission(const String &p_name);
	virtual bool request_permissions();
	virtual Vector<String> get_granted_permissions() const;

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);

	virtual Point2 get_mouse_position() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual void set_keep_screen_on(bool p_enabled);

	virtual Size2 get_window_size() const;
	virtual Rect2 get_window_safe_area() const;
	virtual Array get_display_cutouts() const;

	virtual String get_name() const;
	virtual MainLoop *get_main_loop() const;

	virtual bool can_draw() const;

	void main_loop_begin();
	bool main_loop_iterate(bool *r_should_swap_buffers = nullptr);
	void main_loop_end();
	void main_loop_focusout();
	void main_loop_focusin();

	virtual bool has_touchscreen_ui_hint() const;

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), VirtualKeyboardType p_type = KEYBOARD_TYPE_DEFAULT, int p_max_input_length = -1, int p_cursor_start = -1, int p_cursor_end = -1);
	virtual void hide_virtual_keyboard();
	virtual int get_virtual_keyboard_height() const;

	void _set_cursor_shape_helper(CursorShape p_shape, bool force = false);
	virtual void set_cursor_shape(CursorShape p_shape);
	virtual CursorShape get_cursor_shape() const;
	virtual void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);

	virtual void set_mouse_mode(MouseMode p_mode);
	virtual MouseMode get_mouse_mode() const;

	void set_opengl_extensions(const char *p_gl_extensions);
	void set_display_size(Size2 p_size);

	void set_offscreen_gl_available(bool p_available);
	virtual bool is_offscreen_gl_available() const;
	virtual void set_offscreen_gl_current(bool p_current);

	virtual void set_screen_orientation(ScreenOrientation p_orientation);
	virtual ScreenOrientation get_screen_orientation() const;

	virtual Error shell_open(String p_uri);
	virtual String get_executable_path() const;
	virtual String get_user_data_dir() const;
	virtual String get_data_path() const;
	virtual String get_cache_path() const;
	virtual String get_resource_dir() const;
	virtual String get_locale() const;
	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;
	virtual bool has_clipboard() const;
	virtual String get_model_name() const;
	virtual int get_screen_dpi(int p_screen = 0) const;
	virtual float get_screen_scale(int p_screen = -1) const;
	virtual float get_screen_max_scale() const;
	virtual float get_screen_refresh_rate(int p_screen = 0) const;

	virtual bool get_window_per_pixel_transparency_enabled() const { return transparency_enabled; }
	virtual void set_window_per_pixel_transparency_enabled(bool p_enabled) { ERR_FAIL_MSG("Setting per-pixel transparency is not supported at runtime, please set it in project settings instead."); }

	virtual String get_unique_id() const;

	virtual String get_system_dir(SystemDir p_dir, bool p_shared_storage = true) const;

	virtual Error move_to_trash(const String &p_path);

	void process_accelerometer(const Vector3 &p_accelerometer);
	void process_gravity(const Vector3 &p_gravity);
	void process_magnetometer(const Vector3 &p_magnetometer);
	void process_gyroscope(const Vector3 &p_gyroscope);

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;
	void vibrate_handheld(int p_duration_ms);

	virtual String get_config_path() const;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking = true, ProcessID *r_child_id = nullptr, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr, bool p_open_console = false);
	virtual Error kill(const ProcessID &p_pid);

	virtual void benchmark_begin_measure(const String &p_what);
	virtual void benchmark_end_measure(const String &p_what);
	virtual void benchmark_dump();

	virtual bool _check_internal_feature_support(const String &p_feature);
	OS_Android(GodotJavaWrapper *p_godot_java, GodotIOJavaWrapper *p_godot_io_java, bool p_use_apk_expansion);
	~OS_Android();

private:
	Error create_instance(const List<String> &p_arguments, ProcessID *r_child_id);
};

#endif // OS_ANDROID_H
